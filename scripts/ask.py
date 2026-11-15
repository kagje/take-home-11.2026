"""Ask questions about PDFs from the terminal: same agent, no UI.
Handy for quick checks and for showing the agent is UI-independent.

Usage:  uv run python scripts/ask.py samples/attention.pdf [more.pdf ...] -q "question" [-q "follow-up"]
"""

import argparse
import asyncio
import json

from pdfchat.agent import Agent, Completed, TextDelta, ToolFinished, ToolStarted
from pdfchat.citations import normalise
from pdfchat.ingest import ingest_pdf
from pdfchat.library import Library


async def main(paths: list[str], questions: list[str]) -> None:
    library = Library()
    for path in paths:
        doc = await asyncio.to_thread(ingest_pdf, path)
        await library.add(doc)
        print(f"loaded {doc.name}: {len(doc.pages)} pages, ~{doc.token_count:,} tokens")
    agent = Agent(library)

    for q in questions:
        print(f"\n>>> {q}")
        async for event in agent.run(q):
            match event:
                case TextDelta():
                    pass  # print the normalised final answer instead
                case ToolStarted(name=name, arguments=args):
                    print(f"  -> {name}({', '.join(f'{k}={v!r}' for k, v in json.loads(args).items())})")
                case ToolFinished(result=result):
                    print(f"     {result.summary[:150]}")
                case Completed(text=text, usage=usage, seen_pages=seen):
                    answer, citations = normalise(text, library, seen)
                    print(f"\n{answer}\n")
                    cost = usage.cost(agent.model)
                    print(
                        f"  [{usage.model_calls} model calls, {usage.tool_calls} tool calls, "
                        f"in={usage.input_tokens:,} (cached {usage.cached_tokens:,}) out={usage.output_tokens:,}, "
                        f"${cost or 0:.4f}; citations: {[c.label for c in citations]}]"
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdfs", nargs="+")
    parser.add_argument("-q", "--question", action="append", required=True)
    args = parser.parse_args()
    asyncio.run(main(args.pdfs, args.question))
