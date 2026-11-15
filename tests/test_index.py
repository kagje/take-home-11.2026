import asyncio
import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from pdfchat.index import Index, chunk_document, chunk_text, reciprocal_rank_fusion, tokenize
from pdfchat.ingest import Document, Page, PageKind, count_tokens


async def fake_embed(texts: list[str]) -> np.ndarray:
    """Deterministic bag-of-words hashing embedder: texts sharing words are
    similar. Lets us test search mechanics without calling the API."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def make_doc(doc_id: str, pages: list[str]) -> Document:
    return Document(
        id=doc_id,
        name=f"{doc_id}.pdf",
        path=Path(f"{doc_id}.pdf"),
        pages=[Page(i + 1, t, PageKind.TEXT if t else PageKind.EMPTY, 0.0) for i, t in enumerate(pages)],
    )


def run(coro):
    return asyncio.run(coro)


# --- chunking -----------------------------------------------------------------


def test_chunks_respect_token_budget_and_overlap() -> None:
    text = "\n".join(f"Line {i} talks about topic number {i} in some detail." for i in range(200))
    chunks = chunk_text(text, max_tokens=100, overlap_tokens=20)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)
    for prev, nxt in itertools.pairwise(chunks):
        assert prev.splitlines()[-1] == nxt.splitlines()[0]  # last line carried over


def test_single_huge_line_is_split() -> None:
    chunks = chunk_text("word " * 1000, max_tokens=100, overlap_tokens=0)
    assert len(chunks) >= 10
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_chunks_never_cross_pages_and_skip_empty_pages() -> None:
    doc = make_doc("d", ["alpha text", "", "beta text"])
    assert [(c.page, c.text) for c in chunk_document(doc)] == [(1, "alpha text"), (3, "beta text")]


# --- fusion ---------------------------------------------------------------------


def test_rrf_rewards_agreement_between_retrievers() -> None:
    # 7 is first in only one list; 3 is second in both -> 3 wins.
    scores = reciprocal_rank_fusion([[7, 3, 1], [5, 3, 9]])
    assert max(scores, key=scores.__getitem__) == 3


def test_tokenize_keeps_hyphenated_identifiers() -> None:
    assert tokenize("See control AC-2 and SP-800-53.") == ["see", "control", "ac-2", "and", "sp-800-53"]


# --- search ---------------------------------------------------------------------


@pytest.fixture
def index() -> Index:
    idx = Index(embed=fake_embed)
    run(idx.add(make_doc("security", ["Account management AC-2 requires reviewing accounts.", "Audit logs."])))
    run(idx.add(make_doc("recipes", ["Bake the bread for forty minutes.", "Account for altitude when baking."])))
    return idx


def test_search_finds_exact_identifier(index: Index) -> None:
    top = run(index.search("AC-2"))[0]
    assert (top.chunk.doc_id, top.chunk.page) == ("security", 1)
    assert top.keyword_rank == 1


def test_search_can_be_scoped_to_documents(index: Index) -> None:
    hits = run(index.search("account", doc_ids={"recipes"}))
    assert hits
    assert {h.chunk.doc_id for h in hits} == {"recipes"}


def test_adding_same_document_twice_is_a_no_op(index: Index) -> None:
    before = len(index.chunks)
    assert run(index.add(make_doc("security", ["whatever"]))) == 0
    assert len(index.chunks) == before


def test_empty_query_or_index_returns_nothing(index: Index) -> None:
    assert run(index.search("   ")) == []
    assert run(Index(embed=fake_embed).search("anything")) == []
