"""Chainlit entry point.  Run:  uv run chainlit run app.py -w"""

import asyncio
from collections import Counter

import chainlit as cl
from chainlit.element import Element

from pdfchat.ingest import Document, IngestError, PageKind, ingest_pdf
from pdfchat.library import Library


def describe(doc: Document) -> str:
    kinds = Counter(p.kind for p in doc.pages)
    lines = [
        f"**{doc.name}**: {len(doc.pages)} pages, ~{doc.token_count:,} tokens → strategy `{doc.strategy}`",
        "- " + ", ".join(f"{n} {k}" for k, n in kinds.items()),
    ]
    if ocr := [p.number for p in doc.pages if p.needs_ocr]:
        shown = ", ".join(map(str, ocr[:15])) + (" …" if len(ocr) > 15 else "")
        lines.append(f"- pages needing OCR: {shown}")
    if kinds[PageKind.TEXT] == 0:
        lines.append("- ⚠️ no usable text layer at all")
    lines += [f"- {n}" for n in doc.notes]
    return "\n".join(lines)


async def ingest_element(element: Element, library: Library) -> None:
    async with cl.Step(name=f"Reading {element.name}", type="tool") as step:
        try:
            # PDF parsing is CPU-bound; keep the event loop (and the UI) responsive.
            doc = await asyncio.to_thread(ingest_pdf, element.path, element.name)
        except IngestError as exc:
            step.output = f"❌ {exc}"
            await cl.Message(f"Couldn't use **{element.name}**: {exc}").send()
            return
        step.output = describe(doc)

    async with cl.Step(name=f"Indexing {doc.name}", type="embedding") as step:
        chunks = await library.add(doc)
        step.output = f"{chunks} chunks embedded and indexed" if chunks else "already indexed"

    await cl.Message(describe(doc)).send()


@cl.on_chat_start
async def on_chat_start() -> None:
    cl.user_session.set("library", Library())
    await cl.Message(
        "Hello there! Upload one or more PDFs (click upload or drag & drop), then ask questions about them."
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    library: Library = cl.user_session.get("library")
    for element in message.elements or []:
        # Chainlit turns uploads into cl.Pdf / cl.File depending on the browser's
        # MIME guess, so match on name/mime rather than class. ingest_pdf still
        # verifies the bytes really are a PDF.
        if element.path and (element.mime == "application/pdf" or element.name.lower().endswith(".pdf")):
            await ingest_element(element, library)

    if message.content.strip():
        # Placeholder until the agent lands (step 3): show raw retrieval results.
        hits = await library.index.search(message.content)
        lines = [
            f"- **{h.chunk.doc_name}** p{h.chunk.page} (dense #{h.dense_rank}, bm25 #{h.keyword_rank}): "
            f"{' '.join(h.chunk.text.split())[:160]}…"
            for h in hits
        ]
        body = "\n".join(lines) or "_nothing indexed_"
        await cl.Message(f"Top search hits (agent comes in step 3):\n{body}").send()
