from pathlib import Path

import pymupdf
import pytest

from pdfchat import config
from pdfchat.ingest import (
    IngestError,
    PageKind,
    Strategy,
    classify_page,
    ingest_pdf,
    printable_ratio,
    strip_running_lines,
)

PROSE = "The quick brown fox jumps over the lazy dog. " * 10


def _image_bytes() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 50, 50), False)
    pix.clear_with(200)
    return pix.tobytes("png")


@pytest.fixture
def mixed_pdf(tmp_path: Path) -> Path:
    """Page 1: text. Page 2: blank. Page 3: full-page image (scan). Page 4: junk text."""
    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), PROSE)
    doc.new_page()
    scan = doc.new_page()
    scan.insert_image(scan.rect, stream=_image_bytes())
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), "§¶†‡•°±¤ " * 40)
    path = tmp_path / "mixed.pdf"
    doc.save(path)
    return path


def test_classifies_each_page_independently(mixed_pdf: Path) -> None:
    doc = ingest_pdf(mixed_pdf)
    assert [p.kind for p in doc.pages] == [PageKind.TEXT, PageKind.EMPTY, PageKind.SCANNED, PageKind.GARBLED]
    assert [p.number for p in doc.pages if p.needs_ocr] == [3, 4]


def test_garbled_text_is_dropped_so_it_never_reaches_the_index(mixed_pdf: Path) -> None:
    doc = ingest_pdf(mixed_pdf)
    assert doc.pages[3].text == ""
    assert "§" not in doc.text


def test_small_documents_use_full_context(mixed_pdf: Path) -> None:
    assert ingest_pdf(mixed_pdf).strategy == Strategy.FULL_CONTEXT


def test_large_documents_use_retrieval(mixed_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "FULL_CONTEXT_TOKEN_BUDGET", 10)
    assert ingest_pdf(mixed_pdf).strategy == Strategy.RETRIEVAL


def test_document_id_is_content_hash(mixed_pdf: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy.pdf"
    copy.write_bytes(mixed_pdf.read_bytes())
    assert ingest_pdf(mixed_pdf).id == ingest_pdf(copy).id


def test_rejects_non_pdf(tmp_path: Path) -> None:
    # Regression: an HTML error page downloaded as .pdf opened "fine" in PyMuPDF.
    fake = tmp_path / "fake.pdf"
    fake.write_text("<html><body>Access denied</body></html>")
    with pytest.raises(IngestError, match="doesn't look like a PDF"):
        ingest_pdf(fake)


def test_rejects_encrypted(tmp_path: Path) -> None:
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 50), "secret")
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="pw")
    with pytest.raises(IngestError, match="password"):
        ingest_pdf(path)


@pytest.mark.parametrize(
    ("text", "coverage", "expected"),
    [
        (PROSE, 0.0, PageKind.TEXT),
        (PROSE, 1.0, PageKind.TEXT),  # digital page with a big figure is still text
        ("12", 1.0, PageKind.SCANNED),  # scan with just a page-number text layer
        ("12", 0.1, PageKind.EMPTY),
        ("(cid:12)(cid:7)(cid:44) " * 20, 0.0, PageKind.GARBLED),
    ],
)
def test_classify_page(text: str, coverage: float, expected: PageKind) -> None:
    assert classify_page(text, coverage) == expected


def test_printable_ratio() -> None:
    assert printable_ratio(PROSE) > 0.99
    assert printable_ratio("") == 0.0


def _report_page(n: int) -> str:
    # Body lines must differ by letters, not digits: digits are normalised away.
    body = "\n".join(f"Paragraph {chr(65 + n)}{chr(65 + i)} with unique content." for i in range(12))
    footer = "Confidential - do not distribute"
    return f"ACME ANNUAL REPORT 2026\nPAGE {n}\n{body}\nRelated Controls: None.\n{body}\n{footer}"


def test_strips_running_headers_and_footers_but_keeps_repeated_content() -> None:
    texts, removed = strip_running_lines([_report_page(n) for n in range(1, 11)])
    assert removed == 30  # title + page number + footer on each of 10 pages
    assert all("ACME" not in t and "PAGE" not in t and "Confidential" not in t for t in texts)
    assert all("Related Controls: None." in t for t in texts)  # mid-page, not a header


def test_running_line_detection_needs_enough_pages() -> None:
    texts, removed = strip_running_lines([_report_page(1), _report_page(2)])
    assert removed == 0
    assert "ACME" in texts[0]
