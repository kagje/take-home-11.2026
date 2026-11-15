"""Chunking + hybrid (dense + BM25) search over all documents in a session.

Why hybrid: embeddings capture meaning ("staff leaving the company" ~ "account
termination"), BM25 nails exact tokens that embeddings blur (control IDs like
"AC-2", error codes, names, numbers). Reciprocal Rank Fusion merges the two
rankings without having to calibrate their incomparable scores.

Everything is in memory per chat session: numpy for vectors, rank_bm25 for
keywords. Fine up to tens of thousands of chunks; beyond that, swap for
pgvector / Qdrant / LanceDB behind the same `search` signature.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

import numpy as np
from openai import AsyncOpenAI
from rank_bm25 import BM25Okapi

from pdfchat import config
from pdfchat.ingest import Document, count_tokens, encoder

EmbedFn = Callable[[list[str]], Awaitable[np.ndarray]]


@dataclass(frozen=True)
class Chunk:
    doc_id: str
    doc_name: str
    page: int
    text: str

    @property
    def embed_text(self) -> str:
        # A tiny "contextual header": the embedding knows which document and
        # page a chunk comes from, which helps queries that name the document.
        return f"{self.doc_name}, page {self.page}\n{self.text}"


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    dense_rank: int | None  # 1-based rank per retriever, None if not in its top-N
    keyword_rank: int | None


# --- Chunking -----------------------------------------------------------------


def _split_long_line(line: str, max_tokens: int) -> list[str]:
    """Hard-split a single line that alone exceeds the budget (e.g. text
    extracted without newlines, or OCR output of a dense table)."""
    tokens = encoder().encode(line, disallowed_special=())
    return [encoder().decode(tokens[i : i + max_tokens]) for i in range(0, len(tokens), max_tokens)]


def chunk_text(
    text: str,
    max_tokens: int = config.CHUNK_TOKENS,
    overlap_tokens: int = config.CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """Greedily pack whole lines into chunks of <= max_tokens, carrying the
    last ~overlap_tokens worth of lines into the next chunk so that a sentence
    straddling a boundary is retrievable from either side."""
    lines: list[tuple[str, int]] = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        n = count_tokens(raw)
        pieces = _split_long_line(raw, max_tokens) if n > max_tokens else [raw]
        lines += [(p, count_tokens(p)) for p in pieces]

    chunks: list[str] = []
    current: list[tuple[str, int]] = []
    size = 0
    for line, n in lines:
        if current and size + n > max_tokens:
            chunks.append("\n".join(t for t, _ in current))
            overlap: list[tuple[str, int]] = []
            for item in reversed(current):
                if sum(k for _, k in overlap) + item[1] > overlap_tokens:
                    break
                overlap.insert(0, item)
            current, size = overlap, sum(k for _, k in overlap)
        current.append((line, n))
        size += n
    if current:
        chunks.append("\n".join(t for t, _ in current))
    return chunks


def chunk_document(doc: Document) -> list[Chunk]:
    return [Chunk(doc.id, doc.name, page.number, text) for page in doc.pages for text in chunk_text(page.text)]


# --- Embeddings ---------------------------------------------------------------


class OpenAIEmbedder:
    """Batched, concurrent embedding calls. Returns L2-normalised vectors so
    cosine similarity is a plain dot product."""

    def __init__(self, client: AsyncOpenAI | None = None, model: str = config.EMBEDDING_MODEL) -> None:
        self._client = client or AsyncOpenAI()
        self._model = model
        self._limit = asyncio.Semaphore(4)

    async def _batch(self, texts: list[str]) -> list[list[float]]:
        async with self._limit:
            resp = await self._client.embeddings.create(model=self._model, input=texts)
        return [d.embedding for d in resp.data]

    async def __call__(self, texts: list[str]) -> np.ndarray:
        batches = [texts[i : i + config.EMBED_BATCH] for i in range(0, len(texts), config.EMBED_BATCH)]
        results = await asyncio.gather(*(self._batch(b) for b in batches))
        vectors = np.array([v for batch in results for v in batch], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


# --- Search -------------------------------------------------------------------

_WORD = re.compile(r"\w+(?:-\w+)*")  # keeps "AC-2", "SP-800-53" as single tokens


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def reciprocal_rank_fusion(rankings: Iterable[list[int]], k: int = config.RRF_K) -> dict[int, float]:
    """score(d) = sum over rankings of 1 / (k + rank(d)). Rank-based, so it
    needs no score normalisation between retrievers."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return scores


class Index:
    def __init__(self, embed: EmbedFn | None = None) -> None:
        self._embed = embed or OpenAIEmbedder()
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._doc_ids: set[str] = set()

    def __contains__(self, doc_id: str) -> bool:
        return doc_id in self._doc_ids

    async def add(self, doc: Document) -> int:
        """Index a document; returns the number of chunks added. Idempotent."""
        if doc.id in self._doc_ids:
            return 0
        new = await asyncio.to_thread(chunk_document, doc)
        if new:
            vectors = await self._embed([c.embed_text for c in new])
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            self.chunks += new
            # BM25 statistics (IDF) are corpus-wide, so rebuild. It's a few ms
            # per thousand chunks; an incremental index would matter at scale.
            self._bm25 = await asyncio.to_thread(BM25Okapi, [tokenize(c.text) for c in self.chunks])
        self._doc_ids.add(doc.id)
        return len(new)

    async def search(self, query: str, k: int = config.SEARCH_TOP_K, doc_ids: set[str] | None = None) -> list[Hit]:
        if not self.chunks or not query.strip():
            return []
        assert self._vectors is not None and self._bm25 is not None

        allowed = np.ones(len(self.chunks), dtype=bool)
        if doc_ids:
            allowed = np.array([c.doc_id in doc_ids for c in self.chunks])

        def top(scores: np.ndarray) -> list[int]:
            scores = np.where(allowed, scores, -np.inf)
            order = np.argsort(-scores)[: config.SEARCH_CANDIDATES]
            return [int(i) for i in order if np.isfinite(scores[i]) and scores[i] > 0]

        query_vec = (await self._embed([query]))[0]
        dense = top(self._vectors @ query_vec)
        keyword = top(np.asarray(self._bm25.get_scores(tokenize(query))))

        fused = reciprocal_rank_fusion([dense, keyword])
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        dense_rank = {idx: r for r, idx in enumerate(dense, 1)}
        keyword_rank = {idx: r for r, idx in enumerate(keyword, 1)}
        return [Hit(self.chunks[i], fused[i], dense_rank.get(i), keyword_rank.get(i)) for i in best]
