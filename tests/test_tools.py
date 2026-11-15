import asyncio
from pathlib import Path

import pymupdf
import pytest
from fakes import make_doc, make_library

from pdfchat import config
from pdfchat.ingest import ingest_pdf
from pdfchat.tools import Toolbox


def call(toolbox: Toolbox, name: str, arguments: str):
    return asyncio.run(toolbox.call(name, arguments))


@pytest.fixture
def toolbox() -> Toolbox:
    pages = [f"Page {n} text about topic {chr(64 + n)}." for n in range(1, 11)]
    return Toolbox(asyncio.run(make_library(make_doc("abc123", pages, name="Report.pdf"))))


def test_search_results_carry_citation_markers(toolbox: Toolbox) -> None:
    result = call(toolbox, "search", '{"query": "topic C", "document": null}')
    assert "[Report.pdf p.3]" in result.output


def test_documents_resolve_by_id_or_name(toolbox: Toolbox) -> None:
    by_id = call(toolbox, "read_pages", '{"document": "abc123", "first_page": 2, "last_page": 2}')
    by_name = call(toolbox, "read_pages", '{"document": "report.pdf", "first_page": 2, "last_page": 2}')
    assert by_id.output == by_name.output
    assert by_id.output.startswith("[Report.pdf p.2]")


def test_read_pages_is_capped(toolbox: Toolbox) -> None:
    result = call(toolbox, "read_pages", '{"document": "abc123", "first_page": 1, "last_page": 10}')
    assert f"[Report.pdf p.{config.READ_PAGES_MAX}]" in result.output
    assert f"[Report.pdf p.{config.READ_PAGES_MAX + 1}]" not in result.output


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("read_pages", '{"document": "abc123", "first_page": 0, "last_page": 2}', "pages 1-10"),
        ("read_pages", '{"document": "abc123", "first_page": 5, "last_page": 2}', "last_page must be"),
        ("search", '{"query": "x", "document": "nope.pdf"}', "Known documents: Report.pdf"),
        ("delete_everything", "{}", "Unknown tool"),
    ],
)
def test_model_errors_come_back_as_text_not_exceptions(toolbox: Toolbox, name, arguments, message) -> None:
    result = call(toolbox, name, arguments)
    assert result.output.startswith("Error:")
    assert message in result.output


def test_unexpected_failures_are_contained(toolbox: Toolbox) -> None:
    result = call(toolbox, "search", "not json")
    assert result.output == "Error: the tool failed unexpectedly."


def test_view_page_returns_an_image(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 72), "A table would be here.")
    path = tmp_path / "t.pdf"
    pdf.save(path)
    toolbox = Toolbox(asyncio.run(make_library(ingest_pdf(path))))

    result = call(toolbox, "view_page", '{"document": "t.pdf", "page": 1}')
    assert result.image is not None
    assert result.image.startswith(b"\x89PNG")
    assert result.output[1]["type"] == "input_image"
    assert result.output[1]["image_url"].startswith("data:image/png;base64,")


def test_results_report_which_pages_the_model_saw(toolbox: Toolbox) -> None:
    search = call(toolbox, "search", '{"query": "topic C", "document": null}')
    assert ("abc123", 3) in search.pages
    read = call(toolbox, "read_pages", '{"document": "abc123", "first_page": 2, "last_page": 3}')
    assert read.pages == {("abc123", 2), ("abc123", 3)}
    assert call(toolbox, "read_pages", '{"document": "x", "first_page": 1, "last_page": 1}').pages == frozenset()


def test_pages_without_text_are_not_seen_by_reading() -> None:
    doc = make_doc("scan", ["text", ""])
    toolbox = Toolbox(asyncio.run(make_library(doc)))
    result = call(toolbox, "read_pages", '{"document": "scan", "first_page": 1, "last_page": 2}')
    assert "use view_page" in result.output
    assert result.pages == {("scan", 1)}
