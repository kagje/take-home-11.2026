"""Test doubles: a deterministic embedder and a scripted OpenAI Responses client."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pdfchat.index import Index, tokenize
from pdfchat.ingest import Document, Page, PageKind
from pdfchat.library import Library


async def fake_embed(texts: list[str]) -> np.ndarray:
    """Bag-of-words hashing embedder: texts sharing words are similar."""
    out = np.zeros((len(texts), 256), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in tokenize(text):
            out[row, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def make_doc(doc_id: str, pages: list[str], name: str | None = None, path: Path | None = None) -> Document:
    return Document(
        id=doc_id,
        name=name or f"{doc_id}.pdf",
        path=path or Path(f"{doc_id}.pdf"),
        pages=[Page(i + 1, t, PageKind.TEXT if t else PageKind.EMPTY, 0.0) for i, t in enumerate(pages)],
    )


async def make_library(*docs: Document) -> Library:
    library = Library(index=Index(embed=fake_embed))
    for doc in docs:
        await library.add(doc)
    return library


# --- Scripted Responses API -------------------------------------------------------


@dataclass
class FakeItem:
    type: str
    fields: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        try:
            return self.fields[name]
        except KeyError:
            raise AttributeError(name) from None

    def model_dump(self, exclude_none: bool = False) -> dict[str, Any]:
        return {"type": self.type, **self.fields}


def tool_call(name: str, call_id: str = "call_1", **arguments: Any) -> FakeItem:
    return FakeItem("function_call", {"call_id": call_id, "name": name, "arguments": json.dumps(arguments)})


def reply(text: str) -> FakeItem:
    return FakeItem("message", {"role": "assistant", "content": [{"type": "output_text", "text": text}]})


def response(*items: FakeItem) -> SimpleNamespace:
    text = "".join(i.content[0]["text"] for i in items if i.type == "message")
    usage = SimpleNamespace(input_tokens=100, output_tokens=10, input_tokens_details=SimpleNamespace(cached_tokens=40))
    return SimpleNamespace(output=list(items), output_text=text, usage=usage)


class _Stream:
    def __init__(self, resp: SimpleNamespace) -> None:
        self._resp = resp

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def __aiter__(self):
        if self._resp.output_text:
            for word in self._resp.output_text.split(" "):
                yield SimpleNamespace(type="response.output_text.delta", delta=word + " ")

    async def get_final_response(self) -> SimpleNamespace:
        return self._resp


class FakeClient:
    """Replays one scripted response per model call and records the requests."""

    def __init__(self, *script: SimpleNamespace) -> None:
        self._script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.responses = self

    def stream(self, **kwargs: Any) -> _Stream:
        self.requests.append({**kwargs, "input": list(kwargs["input"])})  # snapshot: the agent mutates it
        return _Stream(self._script.pop(0))
