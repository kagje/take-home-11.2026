"""Hand-rolled tool-calling loop on the OpenAI Responses API.

    user question
      -> model (streams text, may request tool calls)
      -> run requested tools concurrently, append their results
      -> model again ... until it answers without tools (or MAX_AGENT_STEPS)

Why hand-rolled: ~100 lines, no framework magic, and every step is visible.
The agent knows nothing about Chainlit: `run()` yields events (text deltas,
tool start/finish, completion) and the UI decides how to render them. That
keeps it testable with a fake client and reusable from a CLI (scripts/ask.py).

Why the Responses API: gpt-6-luna only supports function calling in Chat
Completions with reasoning disabled; Responses lets it reason between calls.

Stateless on purpose (store=False): OpenAI keeps nothing between requests, so
uploaded documents aren't retained server-side. We resend context each turn
and carry the model's encrypted reasoning items forward ourselves.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from pdfchat import config
from pdfchat.library import Library
from pdfchat.prompt import build_instructions, in_context_pages
from pdfchat.tools import SPECS, Toolbox, ToolResult

# --- Events -------------------------------------------------------------------


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolStarted:
    call_id: str
    name: str
    arguments: str  # raw JSON as produced by the model


@dataclass
class ToolFinished:
    call_id: str
    name: str
    result: ToolResult


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0

    def add(self, usage: Any) -> None:
        self.model_calls += 1
        if usage is None:
            return
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        details = getattr(usage, "input_tokens_details", None)
        self.cached_tokens += getattr(details, "cached_tokens", 0) or 0

    def cost(self, model: str) -> float | None:
        if model not in config.PRICES:
            return None
        p_in, p_cached, p_out = config.PRICES[model]
        uncached = self.input_tokens - self.cached_tokens
        return (uncached * p_in + self.cached_tokens * p_cached + self.output_tokens * p_out) / 1e6


@dataclass
class Completed:
    text: str
    usage: Usage = field(default_factory=Usage)
    seen_pages: frozenset[tuple[str, int]] = frozenset()  # for citation verification


Event = TextDelta | ToolStarted | ToolFinished | Completed


# --- Agent --------------------------------------------------------------------

# Fields the SDK's streaming helper adds to output items for client-side
# convenience; the API rejects them when the items are sent back as input.
_CLIENT_ONLY_FIELDS = {"parsed_arguments", "parsed"}


def as_input(item: Any) -> dict[str, Any]:
    """Convert a response output item into an input item for the next request."""

    def strip(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: strip(v) for k, v in value.items() if k not in _CLIENT_ONLY_FIELDS}
        if isinstance(value, list):
            return [strip(v) for v in value]
        return value

    return strip(item.model_dump(exclude_none=True))


class Agent:
    def __init__(self, library: Library, client: AsyncOpenAI | None = None, model: str = config.CHAT_MODEL) -> None:
        self.library = library
        self.tools = Toolbox(library)
        self.client = client or AsyncOpenAI()
        self.model = model
        # Conversation memory across turns: user questions and final answers
        # only. Tool calls/results are scratch work for one turn; dropping them
        # keeps later turns cheap, and the answer (with citations) carries the
        # facts forward. The model can always re-search if it needs more.
        self.history: list[dict[str, Any]] = []
        # Pages whose content the model has been shown this session (in-context
        # text or tool results). Citations of any other page are unverified.
        self.seen_pages: set[tuple[str, int]] = set()

    async def run(self, question: str) -> AsyncIterator[Event]:
        items: list[dict[str, Any]] = [*self.history, {"role": "user", "content": question}]
        instructions = build_instructions(self.library)
        usage = Usage()
        self.seen_pages |= in_context_pages(self.library)

        for step in range(config.MAX_AGENT_STEPS):
            final_step = step == config.MAX_AGENT_STEPS - 1
            async with self.client.responses.stream(
                model=self.model,
                instructions=instructions,
                input=items,
                tools=SPECS,
                # On the last step tools are disabled so we always get an answer.
                tool_choice="none" if final_step else "auto",
                parallel_tool_calls=True,
                reasoning={"effort": config.REASONING_EFFORT},
                store=False,
                include=["reasoning.encrypted_content"],
            ) as stream:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        yield TextDelta(event.delta)
                response = await stream.get_final_response()
            usage.add(response.usage)

            # Echo everything back (reasoning, messages, calls) for the next step.
            items += [as_input(item) for item in response.output]
            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                answer = response.output_text
                self.history += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
                yield Completed(answer, usage, frozenset(self.seen_pages))
                return

            for call in calls:
                yield ToolStarted(call.call_id, call.name, call.arguments)
            usage.tool_calls += len(calls)
            results = await asyncio.gather(*(self.tools.call(c.name, c.arguments) for c in calls))
            for call, result in zip(calls, results, strict=True):
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": result.output})
                self.seen_pages |= result.pages
                yield ToolFinished(call.call_id, call.name, result)

        raise RuntimeError("unreachable: the final step has tools disabled")  # pragma: no cover
