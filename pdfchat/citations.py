"""Find citations in an answer, normalise them to `[file.pdf p.N]` (or
`[file.pdf p.9 (PDF p.13)]` when printed page numbers differ), and check them
against the pages the model was actually shown.

The prompt asks for that exact format, but models drift ("p. 3", "page 3",
"file.pdf, p3"). Normalising lets the UI turn every citation into a link that
opens the PDF at the cited page, and drops citations of documents or pages
that don't exist (a cheap hallucination guard: they stay as plain text).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pdfchat.ingest import Document
from pdfchat.library import Library

_CITATION = re.compile(
    r"""\[
        (?P<name>[^\[\]]+?) ,?\s+               # file name, optional comma
        (?:p\.?|pp\.?|page) \s* (?P<page>\d+)   # p.3 | p. 3 | p3 | pp.3 | page 3
        (?: \s*[-\u2013]\s* (?P<end>\d+) )?     # optional range end: -5 or en dash
    \]""",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True)
class Citation:
    doc: Document
    page: int  # PDF position: what the link opens
    end: int  # last page of a range (== page for a single page)
    verified: bool = True  # False: the model never saw this page's content

    @property
    def label(self) -> str:
        """Exactly the text shown inside the brackets, e.g.
        'Handbook.pdf p.9 (PDF p.13)'. Also the UI element name: Chainlit links
        every occurrence of an element's name in the message."""
        return f"{self.doc.name} {self.doc.page_ref(self.page, self.end)}"


def _resolve(library: Library, name: str) -> Document | None:
    name = name.strip().lower()
    for doc in library.docs.values():
        if name in (doc.name.lower(), doc.id):
            return doc
    return None


UNVERIFIED = "(unverified)"

Seen = frozenset[tuple[str, int]] | set[tuple[str, int]]


def normalise(text: str, library: Library, seen: Seen | None = None) -> tuple[str, list[Citation]]:
    """Return text with valid citations rewritten to canonical form, plus the
    unique valid citations in order of first appearance.

    With `seen` (the pages the model was shown), a citation of a page the model
    never saw gets an UNVERIFIED marker. That catches answers given from memory
    and dressed up with a plausible page number; prompting alone doesn't
    reliably prevent it.

    The model cites PDF positions; this is where printed page numbers are
    added for people (see pagelabels.py)."""
    found: dict[str, Citation] = {}

    def fix(match: re.Match[str]) -> str:
        doc = _resolve(library, match["name"])
        page = int(match["page"])
        if doc is None or not 1 <= page <= len(doc.pages):
            return match[0]
        end = min(max(int(match["end"] or page), page), len(doc.pages))
        verified = seen is None or any((doc.id, n) in seen for n in range(page, end + 1))
        citation = Citation(doc, page, end, verified)
        found.setdefault(citation.label, citation)
        return f"[{citation.label}]" if verified else f"[{citation.label}] {UNVERIFIED}"

    return _CITATION.sub(fix, text), list(found.values())
