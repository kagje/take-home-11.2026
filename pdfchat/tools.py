"""Tools the agent can call. Each returns a ToolResult with two views: what the
model sees (`output`) and a short summary for the UI step.

Design notes:
- Few, orthogonal tools. Listing documents isn't a tool: the catalogue is in
  the system prompt, so the model never wastes a round-trip on it.
- Tools never raise into the agent loop. Bad arguments come back as an error
  string the model can read and correct (e.g. a page number out of range).
- Every text result carries `[name p.N]` markers in exactly the format the
  model is asked to cite, so copying a citation is the path of least resistance.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pdfchat import config
from pdfchat.ingest import Document, count_tokens, encoder, render_page
from pdfchat.library import Library

log = logging.getLogger(__name__)


class ToolError(Exception):
    """An error the model caused and can fix; its message is shown to the model."""


@dataclass
class ToolResult:
    output: str | list[dict[str, Any]]  # function_call_output payload for the model
    summary: str  # for the UI
    image: bytes | None = None  # PNG to show in the UI (view_page)
    # (doc id, page) pairs whose content this result showed the model; used to
    # check that citations point at pages the model actually saw.
    pages: frozenset[tuple[str, int]] = frozenset()


def cite(doc: Document, page: int) -> str:
    return f"[{doc.name} p.{page}]"


def _truncate(text: str, max_tokens: int = config.TOOL_OUTPUT_MAX_TOKENS) -> str:
    tokens = encoder().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return encoder().decode(tokens[:max_tokens]) + "\n[... truncated; request fewer pages ...]"


_NULLABLE_DOC = {"type": ["string", "null"], "description": "Document id or file name; null searches all documents."}
_DOC = {"type": "string", "description": "Document id or file name."}

SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "search",
        "description": (
            "Hybrid semantic + keyword search over the uploaded documents. Returns the most relevant "
            "passages with their page numbers. Use specific terms: identifiers, names, section titles."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}, "document": _NULLABLE_DOC},
            "required": ["query", "document"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "read_pages",
        "description": (
            f"Read the full extracted text of a page range (max {config.READ_PAGES_MAX} pages). Use it to see "
            "the context around a search hit, or to read a section start to end."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document": _DOC,
                "first_page": {"type": "integer", "description": "1-based"},
                "last_page": {"type": "integer", "description": "1-based, inclusive"},
            },
            "required": ["document", "first_page", "last_page"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "view_page",
        "description": (
            "Look at an image of a page. Use it when extracted text is missing or not enough: scanned pages, "
            "tables, charts, figures, diagrams, handwriting."
        ),
        "parameters": {
            "type": "object",
            "properties": {"document": _DOC, "page": {"type": "integer", "description": "1-based"}},
            "required": ["document", "page"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


class Toolbox:
    def __init__(self, library: Library) -> None:
        self.library = library
        self._handlers: dict[str, Callable[..., Awaitable[ToolResult]]] = {
            "search": self.search,
            "read_pages": self.read_pages,
            "view_page": self.view_page,
        }

    async def call(self, name: str, arguments: str) -> ToolResult:
        try:
            handler = self._handlers.get(name)
            if handler is None:
                raise ToolError(f"Unknown tool {name!r}.")
            return await handler(**json.loads(arguments))
        except ToolError as exc:
            return ToolResult(f"Error: {exc}", f"⚠️ {exc}")
        except Exception:  # a bug on our side: log it, let the model carry on
            log.exception("tool %s failed with arguments %s", name, arguments)
            return ToolResult("Error: the tool failed unexpectedly.", "❌ tool crashed (see server log)")

    # --- helpers --------------------------------------------------------------

    def _doc(self, ref: str) -> Document:
        """Resolve by id or by file name: models often pass the name they see."""
        if doc := self.library.get(ref):
            return doc
        matches = [d for d in self.library.docs.values() if d.name.lower() == ref.strip().lower()]
        if len(matches) == 1:
            return matches[0]
        known = ", ".join(f"{d.name} (id {d.id})" for d in self.library.docs.values()) or "none"
        raise ToolError(f"No document {ref!r}. Known documents: {known}.")

    @staticmethod
    def _check_page(doc: Document, page: int) -> None:
        if not 1 <= page <= len(doc.pages):
            raise ToolError(f"{doc.name} has pages 1-{len(doc.pages)}; got {page}.")

    # --- tools ----------------------------------------------------------------

    async def search(self, query: str, document: str | None = None) -> ToolResult:
        scope = {self._doc(document).id} if document else None
        hits = await self.library.index.search(query, doc_ids=scope)
        if not hits:
            return ToolResult("No results.", "no results")
        parts = [f"{cite(self.library.docs[h.chunk.doc_id], h.chunk.page)}\n{h.chunk.text}" for h in hits]
        summary = ", ".join(f"{h.chunk.doc_name} p.{h.chunk.page}" for h in hits)
        seen = frozenset((h.chunk.doc_id, h.chunk.page) for h in hits)
        return ToolResult(_truncate("\n\n---\n\n".join(parts)), summary, pages=seen)

    async def read_pages(self, document: str, first_page: int, last_page: int) -> ToolResult:
        doc = self._doc(document)
        self._check_page(doc, first_page)
        self._check_page(doc, last_page)
        if last_page < first_page:
            raise ToolError("last_page must be >= first_page.")
        last_page = min(last_page, first_page + config.READ_PAGES_MAX - 1)

        pages = doc.pages[first_page - 1 : last_page]
        parts = []
        for page in pages:
            body = page.text or f"(no extracted text: page is {page.kind}; use view_page to look at it)"
            parts.append(f"{cite(doc, page.number)}\n{body}")
        text = _truncate("\n\n".join(parts))
        summary = f"{doc.name} p.{first_page}-{last_page} (~{count_tokens(text):,} tokens)"
        # Pages without text weren't really "seen": the model only got a placeholder.
        return ToolResult(text, summary, pages=frozenset((doc.id, p.number) for p in pages if p.text))

    async def view_page(self, document: str, page: int) -> ToolResult:
        doc = self._doc(document)
        self._check_page(doc, page)
        png = await asyncio.to_thread(render_page, doc, page, config.VIEW_PAGE_DPI)
        data_url = "data:image/png;base64," + base64.b64encode(png).decode()
        output = [
            {"type": "input_text", "text": f"Image of {cite(doc, page)}:"},
            {"type": "input_image", "image_url": data_url, "detail": "high"},
        ]
        return ToolResult(output, f"{doc.name} p.{page}", image=png, pages=frozenset({(doc.id, page)}))
