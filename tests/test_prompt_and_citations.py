import asyncio
import re

from fakes import make_doc, make_library

from pdfchat.citations import UNVERIFIED, normalise
from pdfchat.ingest import PageKind, format_ranges
from pdfchat.prompt import build_instructions, select_in_context

# --- prompt ---------------------------------------------------------------------


def test_in_context_budget_is_shared_across_documents() -> None:
    docs = [make_doc(f"d{i}", ["word " * 400]) for i in range(3)]  # ~400 tokens each
    assert select_in_context(docs, budget=900) == {"d0", "d1"}  # third one doesn't fit


def test_small_docs_are_inlined_large_ones_are_search_only(monkeypatch) -> None:
    small = make_doc("small", ["The secret word is pineapple."])
    big = make_doc("big", ["filler " * 2000])
    library = asyncio.run(make_library(small, big))
    monkeypatch.setattr("pdfchat.config.FULL_CONTEXT_TOKEN_BUDGET", 1000)

    prompt = build_instructions(library)
    assert "small.pdf (id small): 1 pages" in prompt
    assert "IN CONTEXT" in prompt
    assert re.search(r"^- big\.pdf \(id big\): 1 pages, ~[\d,]+ tokens, SEARCH-ONLY$", prompt, re.M)
    assert "[small.pdf p.1]\nThe secret word is pineapple." in prompt
    assert "filler filler" not in prompt


def test_pages_without_text_are_flagged_for_view_page() -> None:
    doc = make_doc("scan", ["text", "", "", "", "text", ""])
    for n in (2, 3, 4, 6):
        doc.pages[n - 1].kind = PageKind.SCANNED
    prompt = build_instructions(asyncio.run(make_library(doc)))
    assert "No extracted text on pages 2-4, 6" in prompt


def test_prompt_is_deterministic_for_prefix_caching() -> None:
    library = asyncio.run(make_library(make_doc("a", ["alpha"]), make_doc("b", ["beta"])))
    assert build_instructions(library) == build_instructions(library)


def test_ranges() -> None:
    assert format_ranges([1, 2, 3, 7, 9, 10]) == "1-3, 7, 9-10"
    assert format_ranges([]) == ""


# --- citations ------------------------------------------------------------------


def _library():
    return asyncio.run(make_library(make_doc("abc", ["x"] * 20, name="Report 2026.pdf")))


def test_citation_variants_are_normalised() -> None:
    text = "A [Report 2026.pdf p. 3]. B [report 2026.pdf, page 12]. C [Report 2026.pdf p3]. D [abc p.4]."
    fixed, citations = normalise(text, _library())
    assert fixed == (
        "A [Report 2026.pdf p.3]. B [Report 2026.pdf p.12]. C [Report 2026.pdf p.3]. D [Report 2026.pdf p.4]."
    )
    assert [c.label for c in citations] == ["Report 2026.pdf p.3", "Report 2026.pdf p.12", "Report 2026.pdf p.4"]


def test_page_ranges_are_kept() -> None:
    fixed, citations = normalise("See [Report 2026.pdf pp. 3\u20135].", _library())
    assert fixed == "See [Report 2026.pdf p.3-5]."
    assert [c.page for c in citations] == [3]


def test_invalid_citations_are_left_alone() -> None:
    text = "[Other.pdf p.1] [Report 2026.pdf p.99] [just a bracket]"
    fixed, citations = normalise(text, _library())
    assert fixed == text
    assert citations == []


def test_citations_of_unseen_pages_are_flagged() -> None:
    library = _library()
    seen = {("abc", 3), ("abc", 7)}
    text = "A [Report 2026.pdf p.3]. B [Report 2026.pdf p.5]. C [Report 2026.pdf pp.6-8]."
    fixed, citations = normalise(text, library, seen)
    assert fixed == (
        f"A [Report 2026.pdf p.3]. B [Report 2026.pdf p.5] {UNVERIFIED}. C [Report 2026.pdf p.6-8]."  # 7 is in range
    )
    assert [(c.page, c.verified) for c in citations] == [(3, True), (5, False), (6, True)]
