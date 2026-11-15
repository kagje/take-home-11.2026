"""Index a PDF with real embeddings and show what each retriever contributes.

Usage:  uv run python scripts/search_demo.py samples/nist-sp800-53r5.pdf "query" ["query" ...]
"""

import asyncio
import sys
import time

from pdfchat.index import Index
from pdfchat.ingest import ingest_pdf


async def main(path: str, queries: list[str]) -> None:
    t0 = time.perf_counter()
    doc = await asyncio.to_thread(ingest_pdf, path)
    t1 = time.perf_counter()
    index = Index()
    n = await index.add(doc)
    t2 = time.perf_counter()
    print(f"{doc.name}: {len(doc.pages)} pages, ingest {t1 - t0:.1f}s, {n} chunks embedded+indexed {t2 - t1:.1f}s\n")

    for q in queries:
        print(f"Q: {q}")
        for hit in await index.search(q, k=5):
            first = " ".join(hit.chunk.text.split())[:90]
            print(f"   p{hit.chunk.page:<4} dense={hit.dense_rank!s:4} bm25={hit.keyword_rank!s:4} | {first}")
        print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2:]))
