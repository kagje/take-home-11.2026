"""Quick head-to-head of candidate models on the two jobs this app gives them:

1. OCR of a degraded scanned page (vs. ground truth from the original PDF)
2. Tool calling through the Responses API

Not a rigorous eval: a few pages, one run each. Enough to pick a default.

Usage:  uv run python scripts/compare_models.py [model ...]
"""

import base64
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
SAMPLES = Path(__file__).resolve().parent.parent / "samples"
PRICES = {  # USD per 1M tokens (input, output), from OpenAI's pricing page 2026-11
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-6.1-terra": (1.00, 5.00),
}
PAGES = [1, 4, 9]  # title/abstract, dense prose + formulas, a results table
OCR_PROMPT = (
    "Transcribe all text on this scanned page exactly as written, in reading order. "
    "Use Markdown; render tables as Markdown tables. Do not summarise or add commentary."
)


def words(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.lower()))


def word_recall(truth: str, ocr: str) -> float:
    """Share of ground-truth words (with multiplicity) that appear in the OCR."""
    t, o = words(truth), words(ocr)
    return sum((t & o).values()) / max(sum(t.values()), 1)


def ocr_page(client: OpenAI, model: str, png: bytes):
    image = "data:image/png;base64," + base64.b64encode(png).decode()
    return client.responses.create(
        model=model,
        reasoning={"effort": "low"},
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


def tool_call_ok(client: OpenAI, model: str) -> bool:
    tool = {
        "type": "function",
        "name": "search",
        "description": "Search the uploaded documents.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    r = client.responses.create(
        model=model,
        reasoning={"effort": "low"},
        tools=[tool],
        input="What BLEU score did the Transformer get on WMT 2014 English-German? Use the search tool.",
    )
    return any(item.type == "function_call" and item.name == "search" for item in r.output)


def main(models: list[str]) -> None:
    client = OpenAI()
    truth_doc = pymupdf.open(SAMPLES / "attention.pdf")
    scan_doc = pymupdf.open(SAMPLES / "attention-scanned.pdf")

    for model in models:
        recalls, tin, tout, secs = [], 0, 0, 0.0
        for n in PAGES:
            png = scan_doc[n - 1].get_pixmap(dpi=150).tobytes("png")
            t0 = time.perf_counter()
            r = ocr_page(client, model, png)
            secs += time.perf_counter() - t0
            tin, tout = tin + r.usage.input_tokens, tout + r.usage.output_tokens
            recalls.append(word_recall(truth_doc[n - 1].get_text(), r.output_text))
        pin, pout = PRICES.get(model, (0, 0))
        cost = (tin * pin + tout * pout) / 1e6
        print(
            f"{model:15} OCR word recall per page {[f'{x:.1%}' for x in recalls]} | "
            f"{secs / len(PAGES):4.1f}s/page | ${cost / len(PAGES):.4f}/page | "
            f"tools via Responses: {'ok' if tool_call_ok(client, model) else 'FAILED'}"
        )


if __name__ == "__main__":
    main(sys.argv[1:] or ["gpt-6-luna", "gpt-5-mini"])
