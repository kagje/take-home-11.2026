"""Chainlit entry point.  Run:  uv run chainlit run app.py -w

This file is only UI glue: upload handling, and rendering the agent's event
stream (text deltas -> streamed message, tool calls -> steps, citations ->
links that open the PDF at the cited page).
"""

import asyncio
import json
import logging
from collections import Counter

import chainlit as cl
import openai
from chainlit.element import Element
from chainlit.utils import utc_now

from pdfchat import config
from pdfchat.agent import Agent, Completed, TextDelta, ToolFinished, ToolStarted
from pdfchat.citations import UNVERIFIED, Citation, normalise
from pdfchat.ingest import Document, IngestError, PageKind, ingest_pdf
from pdfchat.library import Library

log = logging.getLogger(__name__)

# Chainlit file key of each uploaded PDF, so citation links can reference the
# already-uploaded file instead of re-uploading a copy per citation.
FILE_KEYS = "file_keys"


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

    cl.user_session.get(FILE_KEYS)[doc.id] = element.chainlit_key
    await cl.Message(describe(doc)).send()


def citation_elements(citations: list[Citation], file_keys: dict[str, str | None]) -> list[cl.Pdf]:
    """One side-panel PDF link per citation.

    display="side": Chainlit turns every occurrence of the element's name in the
    message into a link that opens the PDF at `page` (longest names match
    first, so "x.pdf p.12" beats "x.pdf p.1").

    `path` is required by the constructor; `chainlit_key` (the id of the
    already-uploaded file) stops Chainlit from storing a fresh copy of the
    whole PDF for every citation."""
    return [
        cl.Pdf(
            name=c.label,
            page=c.page,
            display="side",
            mime="application/pdf",
            path=str(c.doc.path),
            chainlit_key=file_keys.get(c.doc.id),
        )
        for c in citations
    ]


def usage_footer(event: Completed, model: str) -> str:
    u = event.usage
    parts = [
        f"{u.tool_calls} tool call{'' if u.tool_calls == 1 else 's'}",
        f"{u.input_tokens + u.output_tokens:,} tokens ({u.cached_tokens:,} cached)",
    ]
    if (cost := u.cost(model)) is not None:
        parts.append(f"${cost:.4f}")
    return "\n\n*" + " · ".join(parts) + "*"


async def answer(agent: Agent, question: str) -> None:
    reply = cl.Message(content="")
    steps: dict[str, cl.Step] = {}

    async for event in agent.run(question):
        match event:
            case TextDelta(text=text):
                await reply.stream_token(text)
            case ToolStarted(call_id=call_id, name=name, arguments=arguments):
                step = cl.Step(name=name, type="tool", show_input="json")
                step.input = json.dumps(json.loads(arguments), indent=2)
                step.start = utc_now()
                await step.send()
                steps[call_id] = step
            case ToolFinished(call_id=call_id, result=result):
                step = steps.pop(call_id)
                step.output = result.summary
                if result.image:
                    step.elements = [cl.Image(name="page", content=result.image, display="inline", size="large")]
                step.end = utc_now()
                await step.update()
            case Completed() as done:
                # Replace the streamed text with the final answer: drops any
                # "let me search..." preamble and canonicalises citations.
                text, citations = normalise(done.text, agent.library, done.seen_pages)
                if any(not c.verified for c in citations):
                    text += f"\n\n⚠️ *{UNVERIFIED}: the assistant cited a page it never actually read. Double-check it.*"
                footer = usage_footer(done, agent.model) if config.SHOW_USAGE else ""
                reply.content = text + footer
                try:
                    reply.elements = citation_elements(citations, cl.user_session.get(FILE_KEYS))
                except Exception:  # links are a nicety; never lose the answer over them
                    log.exception("could not build citation links")
                log.info("turn usage: %s", done.usage)
    await reply.send()


@cl.on_chat_start
async def on_chat_start() -> None:
    library = Library()
    cl.user_session.set("library", library)
    cl.user_session.set("agent", Agent(library))
    cl.user_session.set(FILE_KEYS, {})
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

    if not message.content.strip():
        return
    try:
        await answer(cl.user_session.get("agent"), message.content)
    except openai.OpenAIError as exc:
        log.exception("model call failed")
        await cl.ErrorMessage(f"The model call failed: {exc}").send()
