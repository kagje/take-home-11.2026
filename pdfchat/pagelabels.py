"""Printed page numbers ("page labels") vs. PDF page positions.

A PDF page's position in the file (what viewers and our tools use) often
differs from the number printed on it: front matter is unnumbered or roman,
chapters restart at 1, documents get merged. Users with a printed copy, or
anyone reading a page number off the page, mean the printed one.

Design: the model works exclusively with PDF positions (one unambiguous
numbering, shared with all tools). Printed labels are resolved here, at ingest,
and only added when citations are rendered for people.

Sources, in order of trust:
1. The PDF's own /PageLabels (the standard mechanism, e.g. "i, ii, ... 1, 2").
   Ignored when it merely restates positions (1..N), which many tools write.
2. Numbers printed in the header/footer zone, accepted only when they form a
   run of >= MIN_RUN consecutive pages with a constant offset. Using local runs
   (not one global offset) handles roman front matter, per-chapter restarts and
   merged documents; requiring a run filters stray numbers (table cells, years).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pdfchat import config

if TYPE_CHECKING:
    from pdfchat.ingest import Page

MIN_RUN = 3

_ARABIC = re.compile(r"^(?:page\s+)?(\d{1,4})(?:\s*(?:of|/)\s*\d{1,4})?$")  # 9 | page 9 | 9 of 24 | 9/24
_DASHED = re.compile(r"^[-\u2013\u2014]\s*(\d{1,4})\s*[-\u2013\u2014]$")  # - 9 -
_ROMAN = re.compile(r"^(?:page\s+)?(m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3}))$")
_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def roman_to_int(numeral: str) -> int:
    total = 0
    values = [_ROMAN_VALUES[c] for c in numeral.lower()]
    for value, nxt in zip(values, [*values[1:], 0], strict=True):
        total += -value if value < nxt else value
    return total


def parse_label(label: str) -> tuple[str, int] | None:
    """'12' -> ('arabic', 12), 'iv' -> ('roman', 4), anything else -> None."""
    low = " ".join(label.split()).lower()
    if m := _ARABIC.match(low) or _DASHED.match(low):
        return "arabic", int(m[1])
    if (m := _ROMAN.match(low)) and m[1]:
        return "roman", roman_to_int(m[1])
    return None


def _candidates(text: str, zone: int) -> dict[tuple[str, int], str]:
    """Page-number-looking lines in the header/footer zone, keyed by (kind, value)."""
    lines = [" ".join(ln.split()) for ln in text.splitlines() if ln.strip()]
    found: dict[tuple[str, int], str] = {}
    for line in lines[:zone] + lines[-zone:]:
        if parsed := parse_label(line):
            kind, value = parsed
            found.setdefault(parsed, str(value) if kind == "arabic" else line.split()[-1])
    return found


def detect_printed_labels(texts: Sequence[str], zone: int = config.RUNNING_LINE_ZONE) -> list[str | None]:
    """Printed page label per page (None where none was found reliably)."""
    # Re-key each candidate by (kind, offset): pages in one numbered run share it.
    by_page = [
        {(kind, index - value): label for (kind, value), label in _candidates(text, zone).items()}
        for index, text in enumerate(texts)
    ]
    labels: list[str | None] = [None] * len(texts)
    for key in sorted({k for page in by_page for k in page}):  # "arabic" sorts before "roman"
        start = 0
        while start < len(texts):
            if key not in by_page[start]:
                start += 1
                continue
            end = start
            while end < len(texts) and key in by_page[end]:
                end += 1
            if end - start >= MIN_RUN:
                for i in range(start, end):
                    labels[i] = labels[i] or by_page[i][key]
            start = end
    return labels


def resolve_labels(pdf_labels: Sequence[str], texts: Sequence[str]) -> tuple[list[str | None], str | None]:
    """Pick the label source and return (labels, source). A label equal to the
    PDF position carries no information and is dropped (None)."""
    if any(label and label != str(i) for i, label in enumerate(pdf_labels, start=1)):
        labels, source = [label or None for label in pdf_labels], "PDF page labels"
    else:
        labels, source = detect_printed_labels(texts), "printed page numbers"
    labels = [None if label == str(i) else label for i, label in enumerate(labels, start=1)]
    return labels, (source if any(labels) else None)


def describe_mapping(pages: Sequence[Page], max_runs: int = 12) -> str | None:
    """Compact mapping for prompts/UI, e.g. 'PDF p.1-4 = i-iv; PDF p.5-28 = 1-24'."""
    runs: list[list] = []  # [first_pdf, last_pdf, first_label, last_label, parsed_last]
    for page in pages:
        if not page.label:
            continue
        parsed = parse_label(page.label)
        prev = runs[-1] if runs else None
        if (
            prev
            and parsed
            and prev[4]
            and page.number == prev[1] + 1
            and parsed[0] == prev[4][0]
            and parsed[1] == prev[4][1] + 1
        ):
            prev[1], prev[3], prev[4] = page.number, page.label, parsed
        else:
            runs.append([page.number, page.number, page.label, page.label, parsed])
    if not runs:
        return None
    parts = [f"PDF p.{a} = {la}" if a == b else f"PDF p.{a}-{b} = {la}-{lb}" for a, b, la, lb, _ in runs[:max_runs]]
    return "; ".join(parts) + ("; ..." if len(runs) > max_runs else "")
