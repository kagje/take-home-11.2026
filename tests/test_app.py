"""UI glue tests. Chainlit elements validate on construction and need a
Chainlit context, so build one instead of trusting the code by inspection
(a missing `path` once crashed every answer that had citations)."""

import asyncio

from chainlit.context import init_http_context
from fakes import make_doc, make_library

import app
from pdfchat.citations import normalise


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
