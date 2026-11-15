"""UI glue tests. Chainlit elements validate on construction and need a
Chainlit context, so build one instead of trusting the code by inspection
(a missing `path` once crashed every answer that had citations)."""

import asyncio
from types import SimpleNamespace

import chainlit as cl
from chainlit.context import init_http_context
from fakes import fake_embed, make_doc, make_library
from test_ocr import fake_transcriber, scanned_pdf

import app
from pdfchat import ocr
from pdfchat.citations import normalise
from pdfchat.index import Index
from pdfchat.library import Library


def in_chainlit_context(fn, *args):
    """Run fn inside a Chainlit HTTP context (which needs a running loop)."""

    async def runner():
        init_http_context()
        return fn(*args)

    return asyncio.run(runner())


def _citations():
    doc = make_doc("h", ["x"] * 14, name="Handbook.pdf")
    for p in doc.pages[4:]:
        p.label = str(p.number - 4)
    _, citations = normalise("A [Handbook.pdf p.13]. B [Handbook.pdf p.2].", asyncio.run(make_library(doc)))
    return citations


def test_citation_links_reuse_the_uploaded_file() -> None:
    elements = in_chainlit_context(app.citation_elements, _citations(), {"h": "uploaded-file-id"})
    assert [(e.name, e.page) for e in elements] == [("Handbook.pdf p.9 (PDF p.13)", 13), ("Handbook.pdf p.2", 2)]
    assert all(e.chainlit_key == "uploaded-file-id" for e in elements)  # no per-citation copy


def test_citation_links_without_a_known_upload_fall_back_to_the_path() -> None:
    elements = in_chainlit_context(app.citation_elements, _citations(), {})
    assert all(e.chainlit_key is None and e.path for e in elements)


def test_upload_of_a_scanned_pdf_reads_ocrs_and_indexes(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ocr, "ocr_page", fake_transcriber([]))
    ocr._cache.clear()
    library = Library(index=Index(embed=fake_embed))
    upload = SimpleNamespace(path=str(scanned_pdf(tmp_path)), name="scan.pdf", chainlit_key="file-1")
    sent: list[str] = []
    monkeypatch.setattr(cl.Message, "send", lambda self: _record(sent, self.content))

    async def flow():
        cl.user_session.set(app.FILE_KEYS, {})
        await app.ingest_element(upload, library)
        return cl.user_session.get(app.FILE_KEYS)

    file_keys = in_chainlit_context_async(flow)
    doc = next(iter(library.docs.values()))
    assert all(p.ocr for p in doc.pages)
    assert len(library.index.chunks) == 6
    assert file_keys == {doc.id: "file-1"}
    assert "OCR'd pages: 1-6" in sent[-1]
    assert "page numbering differs from PDF positions (from printed page numbers): PDF p.2-6 = 1-5" in sent[-1]


async def _record(sent: list[str], content: str) -> None:
    sent.append(content)


def in_chainlit_context_async(coro_fn):
    async def runner():
        init_http_context()
        return await coro_fn()

    return asyncio.run(runner())
