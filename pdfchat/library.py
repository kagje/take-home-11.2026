"""A chat session's documents plus their search index: the state the agent's
tools operate on."""

from __future__ import annotations

from dataclasses import dataclass, field

from pdfchat.index import Index
from pdfchat.ingest import Document


@dataclass
class Library:
    index: Index = field(default_factory=Index)
    docs: dict[str, Document] = field(default_factory=dict)

    async def add(self, doc: Document) -> int:
        """Register and index a document; returns the number of new chunks.

        Every document is indexed, including ones small enough to sit in the
        prompt whole: embedding is nearly free, and search still helps the
        agent pinpoint the page to cite, or cope once several "small"
        documents together exceed the context budget."""
        self.docs[doc.id] = doc
        return await self.index.add(doc)

    def get(self, doc_id: str) -> Document | None:
        return self.docs.get(doc_id)
