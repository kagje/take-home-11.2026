"""OCR orchestration tests. `ocr_page` (the only function that calls the API)
is replaced by a fake; everything around it (cap, cache, failures, cleanup,
concurrency) runs for real on a real PDF."""

import asyncio
from pathlib import Path

import pymupdf
import pytest

from pdfchat import config, ocr
from pdfchat.ingest import PageKind, ingest_pdf


@pytest.fixture(autouse=True)
def empty_cache():
    ocr._cache.clear()
    yield
    ocr._cache.clear()


def scanned_pdf(tmp_path: Path, pages: int = 6) -> Path:
    """Image-only pages: no text layer at all."""
    pix = pymupdf.Pixmap(pymupdf.csGRAY, pymupdf.IRect(0, 0, 60, 80), False)
    pix.clear_with(200)
    png = pix.tobytes("png")
    pdf = pymupdf.open()
    for _ in range(pages):
        page = pdf.new_page()
        page.insert_image(page.rect, stream=png)
    path = tmp_path / "scan.pdf"
    pdf.save(path)
    return path


def fake_transcriber(calls: list[int], fail: set[int] = frozenset()):
    """Returns what a real scan of a small report would OCR to: a running
    header, body text, and a printed page number (offset by 1: the cover is
    unnumbered)."""

    async def fake(client, doc, page: int) -> str:
        calls.append(page)
        await asyncio.sleep(0)
        if page in fail:
            raise RuntimeError("rate limited")
        footer = str(page - 1) if page > 1 else ""
        return f"ACME QUARTERLY REPORT\nThis is the body of page {page}, about topic {chr(64 + page)}.\n{footer}"

    return fake


def run(coro):
    return asyncio.run(coro)


def test_ocr_text_gets_the_same_cleanup_as_digital_text(tmp_path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber(calls))
    doc = ingest_pdf(scanned_pdf(tmp_path))
    assert all(p.kind == PageKind.SCANNED for p in doc.pages)
    assert doc.token_count == 0

    assert run(ocr.ocr_document(doc, client=object())) == 6
    assert sorted(calls) == [1, 2, 3, 4, 5, 6]
    assert all(p.ocr and not p.needs_ocr for p in doc.pages)
    assert doc.pages[2].text == "This is the body of page 3, about topic C."  # header + page number stripped
    assert [p.label for p in doc.pages] == [None, "1", "2", "3", "4", "5"]  # printed numbers found
    assert doc.token_count > 0  # cached size was refreshed, so routing sees the new text


def test_page_cap_leaves_the_rest_image_only(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "MAX_OCR_PAGES", 2)
    calls: list[int] = []
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber(calls))
    doc = ingest_pdf(scanned_pdf(tmp_path))

    assert run(ocr.ocr_document(doc, client=object())) == 2
    assert sorted(calls) == [1, 2]
    assert [p.number for p in doc.pages if p.needs_ocr] == [3, 4, 5, 6]
    assert "pages 3-6 stay image-only" in doc.notes[0]


def test_failed_pages_are_reported_not_raised(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber([], fail={2, 3}))
    doc = ingest_pdf(scanned_pdf(tmp_path))

    assert run(ocr.ocr_document(doc, client=object())) == 4
    assert [p.number for p in doc.pages if p.needs_ocr] == [2, 3]
    assert "OCR failed for pages 2-3" in doc.notes[0]


def test_same_file_is_not_ocrd_twice(tmp_path, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber(calls))
    path = scanned_pdf(tmp_path)
    run(ocr.ocr_document(ingest_pdf(path), client=object()))
    run(ocr.ocr_document(ingest_pdf(path), client=object()))  # e.g. re-upload in a new chat
    assert len(calls) == 6


def test_concurrency_is_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(config, "OCR_CONCURRENCY", 2)
    active = peak = 0

    async def slow(client, doc, page):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return f"text of page {page} " * 10

    monkeypatch.setattr(ocr, "ocr_page", slow)
    run(ocr.ocr_document(ingest_pdf(scanned_pdf(tmp_path)), client=object()))
    assert peak == 2


def test_progress_is_reported_for_every_page(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber([], fail={4}))
    seen: list[tuple[int, int]] = []

    async def progress(done: int, total: int) -> None:
        seen.append((done, total))

    run(ocr.ocr_document(ingest_pdf(scanned_pdf(tmp_path)), client=object(), progress=progress))
    assert seen[-1] == (6, 6)  # failures count as finished too
