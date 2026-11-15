"""Vision OCR for pages without a usable text layer (scanned or garbled).

Why a vision LLM rather than Tesseract: no system dependency, it reads tables
into Markdown and describes figures (both useful for retrieval), and on our
degraded scan it recovered ~95-99% of words. The trade-offs are cost (~$0.001
per page with gpt-6-luna) and latency (~10s per page), handled below with a
page cap and concurrency. A local engine (Tesseract, PaddleOCR, docTR) would be
the choice for very large scanned archives or data that mustn't leave the
building; it would slot in behind the same `ocr_page` function.

Policy:
- Eager, at upload, for up to MAX_OCR_PAGES pages: their text becomes
  searchable and citable like any other page.
- Pages beyond the cap (or whose OCR failed) stay image-only. The agent can
  still read them on demand with `view_page`, so nothing becomes unreachable.
- Results are cached by (content hash, page), so re-uploading the same file in
  a new chat costs nothing.
- OCR output keeps headers, footers and page numbers, then goes through the
  same cleanup as digital text (`clean_text`): running lines are stripped and
  printed page numbers detected, for scanned documents too.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from pdfchat import config
from pdfchat.ingest import Document, clean_text, format_ranges, render_page

log = logging.getLogger(__name__)

ProgressFn = Callable[[int, int], Awaitable[None]]

OCR_PROMPT = """\
Transcribe all text on this page exactly as written, in reading order, including headers, footers \
and page numbers. Use Markdown: headings as headings, tables as Markdown tables. For a figure, chart \
or photo, add one line: [Figure: <short description of what it shows>]. Do not summarise, translate, \
correct or comment. If the page has no text, return nothing."""

# (document content hash, page number) -> transcription. Process-wide: survives
# across chat sessions, so the same file is never OCR'd twice by this server.
_cache: dict[tuple[str, int], str] = {}


async def ocr_page(client: AsyncOpenAI, doc: Document, page: int) -> str:
    png = await asyncio.to_thread(render_page, doc, page, config.OCR_DPI)
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    response = await client.responses.create(
        model=config.VISION_MODEL,
        reasoning={"effort": config.OCR_REASONING_EFFORT},
        store=False,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": OCR_PROMPT},
                    {"type": "input_image", "image_url": image, "detail": "high"},
                ],
            }
        ],
    )
    return response.output_text.strip()


async def ocr_document(doc: Document, client: AsyncOpenAI | None = None, progress: ProgressFn | None = None) -> int:
    """OCR the pages that need it, in parallel, then re-run text cleanup.
    Returns the number of pages OCR'd. Failures are reported in doc.notes,
    never raised: a half-OCR'd document is still useful."""
    todo = [p for p in doc.pages if p.needs_ocr]
    if not todo:
        return 0
    if skipped := todo[config.MAX_OCR_PAGES :]:
        doc.notes.append(
            f"OCR is capped at {config.MAX_OCR_PAGES} pages per document; pages "
            f"{format_ranges([p.number for p in skipped])} stay image-only (the assistant can still look at them)"
        )
        todo = todo[: config.MAX_OCR_PAGES]

    client = client or AsyncOpenAI()
    limit = asyncio.Semaphore(config.OCR_CONCURRENCY)
    finished = 0

    async def run(page_number: int) -> str:
        nonlocal finished
        try:
            key = (doc.id, page_number)
            if key not in _cache:
                async with limit:
                    _cache[key] = await ocr_page(client, doc, page_number)
            return _cache[key]
        finally:
            finished += 1
            if progress:
                await progress(finished, len(todo))

    results = await asyncio.gather(*(run(p.number) for p in todo), return_exceptions=True)

    failed = []
    for page, result in zip(todo, results, strict=True):
        if isinstance(result, BaseException):
            log.warning("OCR failed for %s p.%d: %r", doc.name, page.number, result)
            failed.append(page.number)
        else:
            page.raw_text, page.ocr = result, True
    if failed:
        doc.notes.append(
            f"OCR failed for pages {format_ranges(failed)}; they stay image-only (the assistant can still look at them)"
        )

    await asyncio.to_thread(clean_text, doc)
    return len(todo) - len(failed)
