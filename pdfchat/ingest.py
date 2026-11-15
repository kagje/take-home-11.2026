"""PDF ingestion: validate, extract text per page, classify page quality, and
decide how the document should be served to the model.

The key idea: PDFs vary in *size* and *quality*, and both are decided per
document (size -> routing) and per page (quality -> extraction method), because
real-world PDFs are often mixed (a digital report with a scanned appendix).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import StrEnum
from functools import cache, cached_property
from pathlib import Path

import pymupdf
import tiktoken

from pdfchat import config
from pdfchat.pagelabels import resolve_labels


class IngestError(Exception):
    """Raised for files we refuse to process; message is shown to the user."""


class PageKind(StrEnum):
    TEXT = "text"  # usable text layer
    SCANNED = "scanned"  # little/no text, page is mostly an image -> OCR
    GARBLED = "garbled"  # text layer exists but is junk (broken encoding) -> OCR
    EMPTY = "empty"  # nothing on it (blank / separator page)


class Strategy(StrEnum):
    FULL_CONTEXT = "full_context"  # whole document goes into the prompt
    RETRIEVAL = "retrieval"  # chunk + embed + hybrid search


@dataclass
class Page:
    number: int  # 1-based position in the PDF; what tools and the model use
    text: str  # cleaned text (running headers/footers removed); what gets indexed
    kind: PageKind
    image_coverage: float
    ocr: bool = False  # True once text came from vision OCR
    label: str | None = None  # printed page number, if known and != number (see pagelabels.py)
    raw_text: str = ""  # text layer or OCR output before cleanup; `text`/`label` derive from it

    @property
    def needs_ocr(self) -> bool:
        return self.kind in (PageKind.SCANNED, PageKind.GARBLED) and not self.ocr


@dataclass
class Document:
    id: str  # short content hash; stable across re-uploads of the same file
    name: str
    path: Path
    pages: list[Page]
    notes: list[str] = field(default_factory=list)  # warnings surfaced to the user
    pdf_labels: list[str] = field(default_factory=list)  # /PageLabels as read from the file
    label_source: str | None = None  # where printed page numbers came from, if any
    stripped_lines: int = 0  # running header/footer lines removed

    # Cached: tokenising a 500-page document takes ~0.3s and the prompt builder
    # asks every turn. Call refresh() after changing page text (e.g. OCR).
    @cached_property
    def text(self) -> str:
        return "\n\n".join(f"[page {p.number}]\n{p.text}" for p in self.pages if p.text)

    @cached_property
    def token_count(self) -> int:
        return count_tokens(self.text)

    def refresh(self) -> None:
        for attr in ("text", "token_count"):
            self.__dict__.pop(attr, None)

    @property
    def strategy(self) -> Strategy:
        if self.token_count <= config.FULL_CONTEXT_TOKEN_BUDGET:
            return Strategy.FULL_CONTEXT
        return Strategy.RETRIEVAL

    def pages_of_kind(self, kind: PageKind) -> list[Page]:
        return [p for p in self.pages if p.kind == kind]

    def page_ref(self, first: int, last: int | None = None) -> str:
        """A page (range) as shown to people: printed number first when known,
        e.g. 'p.9 (PDF p.13)', else just 'p.13'."""
        last = first if last is None else last
        pdf = str(first) if last == first else f"{first}-{last}"
        a, b = self.pages[first - 1].label, self.pages[last - 1].label
        if a and b:
            return f"p.{a if last == first else f'{a}-{b}'} (PDF p.{pdf})"
        return f"p.{pdf}"


@cache
def encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(encoder().encode(text, disallowed_special=()))


# --- Page quality heuristics --------------------------------------------------

_NORMAL_CHARS = re.compile(r"[\w\s.,;:!?()\[\]{}'\"%&/@#*+=<>|~^$€£-]")
_CID_GLYPH = re.compile(r"\(cid:\d+\)")  # PyMuPDF/pdfminer marker for unmapped glyphs


def printable_ratio(text: str) -> float:
    """Share of characters that look like normal prose. Broken font encodings
    produce text full of symbols, private-use codepoints, or (cid:NN) markers."""
    if not text:
        return 0.0
    cleaned = _CID_GLYPH.sub("�", text)
    return len(_NORMAL_CHARS.findall(cleaned)) / len(cleaned)


def image_coverage(page: pymupdf.Page) -> float:
    """Approximate share of the page area covered by raster images (capped at 1)."""
    area = page.rect.width * page.rect.height
    if not area:
        return 0.0
    covered = 0.0
    for info in page.get_image_info():
        bbox = pymupdf.Rect(info["bbox"]) & page.rect  # clip to page
        covered += bbox.width * bbox.height
    return min(covered / area, 1.0)


def classify_page(text: str, coverage: float) -> PageKind:
    meaningful = len(text.strip())
    if meaningful < config.MIN_TEXT_CHARS:
        # A tiny text layer on top of a full-page image is a typical scan
        # (often just a page number or a stamp); without an image it's blank.
        return PageKind.SCANNED if coverage >= config.SCAN_IMAGE_COVERAGE else PageKind.EMPTY
    if printable_ratio(text) < config.MIN_PRINTABLE_RATIO:
        return PageKind.GARBLED
    return PageKind.TEXT


# --- Running headers / footers --------------------------------------------------

_DIGITS = re.compile(r"\d+")


def _line_key(line: str) -> str:
    # "PAGE 150" and "PAGE 151" are the same running line.
    return _DIGITS.sub("#", " ".join(line.split()).lower())


def strip_running_lines(texts: list[str]) -> tuple[list[str], int]:
    """Remove running headers/footers: lines that sit in the top or bottom
    zone of at least half of the pages. Restricting to those zones keeps
    repeated *content* (e.g. "Related Controls: None.") intact.

    Without this, every chunk of a long report starts with the same title and
    URL, which dilutes embeddings and skews BM25."""
    if len(texts) < config.RUNNING_LINE_MIN_PAGES:
        return [t.strip() for t in texts], 0

    zone = config.RUNNING_LINE_ZONE

    def zone_keys(lines: list[str]) -> set[str]:
        return {_line_key(ln) for ln in lines[:zone] + lines[-zone:]}

    split = [[ln for ln in t.splitlines() if ln.strip()] for t in texts]
    counts: dict[str, int] = {}
    for lines in split:
        for key in zone_keys(lines):
            counts[key] = counts.get(key, 0) + 1
    running = {k for k, n in counts.items() if n >= len(texts) * config.RUNNING_LINE_MIN_SHARE}

    removed = 0
    out = []
    for lines in split:
        kept = [
            ln for i, ln in enumerate(lines) if not ((i < zone or i >= len(lines) - zone) and _line_key(ln) in running)
        ]
        removed += len(lines) - len(kept)
        out.append("\n".join(kept).strip())
    return out, removed


# --- Entry point --------------------------------------------------------------


def _validate(path: Path) -> bytes:
    size_mb = path.stat().st_size / 1_048_576
    if size_mb > config.MAX_FILE_MB:
        raise IngestError(f"File is {size_mb:.0f} MB; the limit is {config.MAX_FILE_MB} MB.")
    data = path.read_bytes()
    # PyMuPDF happily opens HTML/images/XPS as "documents", so check the magic
    # bytes ourselves (the spec allows junk before the header, hence the window).
    if b"%PDF-" not in data[:1024]:
        raise IngestError("This doesn't look like a PDF file.")
    return data


def ingest_pdf(path: str | Path, name: str | None = None) -> Document:
    """Parse a PDF into a Document. CPU-bound and blocking: call it off the
    event loop. Pages needing OCR are flagged but not OCR'd here (see ocr.py).
    Ingest itself never calls an API: it is fast, free and deterministic."""
    path = Path(path)
    data = _validate(path)

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # pymupdf raises various types for corrupt files
        raise IngestError(f"Could not open PDF: {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise IngestError("This PDF is password-protected.")
        if doc.page_count == 0:
            raise IngestError("This PDF has no pages.")
        if doc.page_count > config.MAX_PAGES:
            raise IngestError(f"PDF has {doc.page_count} pages; the limit is {config.MAX_PAGES}.")

        # Content-stream order. We deliberately skip sort=True: it is ~7x
        # slower and, by sorting blocks top-to-bottom, interleaves the columns
        # of two-column layouts (papers, reports). Layout-aware parsing
        # (Docling/Marker) is the real fix; see README.
        raw = [page.get_text("text") for page in doc]
        coverages = [image_coverage(page) for page in doc]
        pdf_labels = [page.get_label() for page in doc]

    # Classify on stripped text: a scan whose text layer is only a running
    # header is still a scan.
    stripped, _ = strip_running_lines(raw)
    pages = [
        Page(n, "", classify_page(text, cov), cov, raw_text=raw_text)
        for n, (raw_text, text, cov) in enumerate(zip(raw, stripped, coverages, strict=True), start=1)
    ]
    result = Document(
        id=hashlib.sha256(data).hexdigest()[:12],
        name=name or path.name,
        path=path,
        pages=pages,
        pdf_labels=pdf_labels,
    )
    clean_text(result)
    return result


def clean_text(doc: Document) -> None:
    """Derive every page's `text` and printed `label` from its `raw_text`.

    Runs at ingest and again after OCR, so OCR'd pages get exactly the same
    treatment as digital ones: running headers stripped, printed page numbers
    detected (scanned documents have page numbers too)."""
    raw = [p.raw_text for p in doc.pages]
    # Labels first: the printed page number is usually one of the running lines.
    labels, doc.label_source = resolve_labels(doc.pdf_labels, raw)
    texts, doc.stripped_lines = strip_running_lines(raw)
    for page, text, label in zip(doc.pages, texts, labels, strict=True):
        page.label = label
        # Junk text would only pollute the index; drop it until OCR replaces it.
        page.text = "" if page.kind == PageKind.GARBLED and not page.ocr else text
    doc.refresh()


def render_page(doc: Document, page: int, dpi: int) -> bytes:
    """PNG of one page (1-based). Used for view_page and OCR."""
    with pymupdf.open(doc.path) as pdf:
        return pdf[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def format_ranges(numbers: list[int]) -> str:
    """[1,2,3,7,9,10] -> '1-3, 7, 9-10'"""
    out: list[str] = []
    for n in numbers:
        if out and n == int(out[-1].split("-")[-1]) + 1:
            out[-1] = f"{out[-1].split('-')[0]}-{n}"
        else:
            out.append(str(n))
    return ", ".join(out)
