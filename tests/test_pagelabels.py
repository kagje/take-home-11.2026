import asyncio
from pathlib import Path

import pymupdf
import pytest
from fakes import make_doc, make_library

from pdfchat.citations import normalise
from pdfchat.ingest import ingest_pdf
from pdfchat.pagelabels import describe_mapping, detect_printed_labels, parse_label, resolve_labels, roman_to_int
from pdfchat.prompt import build_instructions


def page(body: str, top: str = "", bottom: str = "") -> str:
    return "\n".join(x for x in (top, body, bottom) if x)


BODY = "Some paragraph text.\nMore text here.\nAnd a closing line."

# --- parsing ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("9", ("arabic", 9)),
        ("Page 9", ("arabic", 9)),
        ("page 9 of 24", ("arabic", 9)),
        ("9/24", ("arabic", 9)),
        ("- 9 -", ("arabic", 9)),
        ("xiv", ("roman", 14)),
        ("IV", ("roman", 4)),
        ("I.", None),  # section numbering, not a page number
        ("2004 Annual Report", None),
        ("", None),
    ],
)
def test_parse_label(label: str, expected) -> None:
    assert parse_label(label) == expected


def test_roman_to_int() -> None:
    assert [roman_to_int(r) for r in ("i", "iv", "ix", "xl", "mcmxcix")] == [1, 4, 9, 40, 1999]


# --- detection --------------------------------------------------------------------


def test_front_matter_roman_then_arabic_body() -> None:
    # Like the sample handbook: unnumbered cover, roman ToC pages, body from 1.
    texts = [
        page(BODY),
        *(page(BODY, bottom=r) for r in ("ii", "iii", "iv")),
        *(page(BODY, top=str(n)) for n in range(1, 6)),
    ]
    assert detect_printed_labels(texts) == [None, "ii", "iii", "iv", "1", "2", "3", "4", "5"]


def test_numbering_that_restarts_per_chapter() -> None:
    texts = [page(BODY, bottom=str(n)) for n in (1, 2, 3, 1, 2, 3, 4)]
    assert detect_printed_labels(texts) == ["1", "2", "3", "1", "2", "3", "4"]


def test_stray_numbers_do_not_form_a_run() -> None:
    texts = [page(BODY, top="42"), page(BODY), page(BODY, top="7"), page(BODY, top="8"), page(BODY)]
    assert detect_printed_labels(texts) == [None] * 5  # isolated, and a run of only 2


def test_numbers_outside_the_header_footer_zone_are_ignored() -> None:
    middle = "\n".join(["line"] * 6 + ["{n}"] + ["line"] * 6)
    texts = [middle.format(n=n) for n in (1, 2, 3, 4)]
    assert detect_printed_labels(texts) == [None] * 4


def test_page_n_of_m_footer() -> None:
    texts = [page(BODY, bottom=f"Page {n} of 4") for n in range(1, 5)]
    assert detect_printed_labels(texts) == ["1", "2", "3", "4"]


# --- source selection -------------------------------------------------------------


def test_identity_pdf_labels_are_ignored_in_favour_of_printed_numbers() -> None:
    texts = [page(BODY), *(page(BODY, top=str(n)) for n in range(1, 4))]
    labels, source = resolve_labels(["1", "2", "3", "4"], texts)
    assert labels == [None, "1", "2", "3"]
    assert source == "printed page numbers"


def test_meaningful_pdf_labels_win() -> None:
    labels, source = resolve_labels(["i", "ii", "1", "2"], [BODY] * 4)
    assert labels == ["i", "ii", "1", "2"]
    assert source == "PDF page labels"


def test_labels_equal_to_positions_carry_no_information() -> None:
    labels, source = resolve_labels(["", "", ""], [page(BODY, bottom=str(n)) for n in (1, 2, 3)])
    assert labels == [None, None, None]
    assert source is None


# --- end to end -------------------------------------------------------------------


def test_ingest_reads_pdf_page_labels(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    for n in range(6):
        pdf.new_page().insert_text((72, 300), f"Content of page {n} " * 5)
    pdf.set_page_labels(
        [
            {"startpage": 0, "prefix": "", "style": "r", "firstpagenum": 1},
            {"startpage": 2, "prefix": "", "style": "D", "firstpagenum": 1},
        ]
    )
    path = tmp_path / "labelled.pdf"
    pdf.save(path)

    doc = ingest_pdf(path)
    assert [p.label for p in doc.pages] == ["i", "ii", "1", "2", "3", "4"]
    assert any("PDF page labels" in note for note in doc.notes)


def test_page_ref_and_mapping() -> None:
    doc = make_doc("h", ["x"] * 6, name="H.pdf")
    for p, label in zip(doc.pages[2:], ["1", "2", "3", "4"], strict=True):
        p.label = label
    assert doc.page_ref(5) == "p.3 (PDF p.5)"
    assert doc.page_ref(4, 5) == "p.2-3 (PDF p.4-5)"
    assert doc.page_ref(1) == "p.1"  # unlabeled: PDF position only
    assert doc.page_ref(2, 3) == "p.2-3"  # mixed range: PDF positions only
    assert describe_mapping(doc.pages) == "PDF p.3-6 = 1-4"


def test_citations_show_printed_number_and_link_to_pdf_page() -> None:
    doc = make_doc("h", ["x"] * 14, name="Handbook.pdf")
    for p in doc.pages[4:]:
        p.label = str(p.number - 4)
    library = asyncio.run(make_library(doc))

    text, citations = normalise("Retirement plan [Handbook.pdf p.13].", library)
    assert text == "Retirement plan [Handbook.pdf p.9 (PDF p.13)]."
    assert citations[0].label == "Handbook.pdf p.9 (PDF p.13)"  # the UI element name
    assert citations[0].page == 13  # what the link opens

    assert "Printed page numbers: PDF p.5-14 = 1-10" in build_instructions(library)
