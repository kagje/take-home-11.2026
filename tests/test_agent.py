import asyncio

import pytest
from fakes import FakeClient, FakeItem, make_doc, make_library, reply, response, tool_call

from pdfchat import config
from pdfchat.agent import Agent, Completed, TextDelta, ToolFinished, ToolStarted, as_input


def run_turn(agent: Agent, question: str) -> list:
    async def collect():
        return [e async for e in agent.run(question)]

    return asyncio.run(collect())


@pytest.fixture
def library():
    doc = make_doc("nist", ["Account management AC-2 requires reviewing accounts.", "Audit logs AU-2."])
    return asyncio.run(make_library(doc))


def test_tool_call_then_answer(library) -> None:
    client = FakeClient(
        response(tool_call("search", query="AC-2", document=None)),
        response(reply("Accounts must be reviewed [nist.pdf p.1].")),
    )
    agent = Agent(library, client=client, model="gpt-6-luna")
    events = run_turn(agent, "What does AC-2 require?")

    kinds = [type(e) for e in events]
    assert kinds[:2] == [ToolStarted, ToolFinished]
    assert TextDelta in kinds
    assert kinds[-1] is Completed
    assert events[-1].text == "Accounts must be reviewed [nist.pdf p.1]."

    # The search result went back to the model, with citable page markers.
    second_input = client.requests[1]["input"]
    tool_output = next(i for i in second_input if i.get("type") == "function_call_output")
    assert tool_output["call_id"] == "call_1"
    assert "[nist.pdf p.1]" in tool_output["output"]

    usage = events[-1].usage
    assert (usage.model_calls, usage.tool_calls, usage.cached_tokens) == (2, 1, 80)
    assert usage.cost("gpt-6-luna") > 0


def test_history_keeps_only_questions_and_answers(library) -> None:
    client = FakeClient(
        response(tool_call("search", query="AC-2", document=None)),
        response(reply("First answer.")),
        response(reply("Second answer.")),
    )
    agent = Agent(library, client=client)
    run_turn(agent, "Q1")
    run_turn(agent, "Q2")

    assert client.requests[2]["input"] == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "First answer."},
        {"role": "user", "content": "Q2"},
    ]


def test_last_step_disables_tools(library, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "MAX_AGENT_STEPS", 2)
    client = FakeClient(
        response(tool_call("search", query="x", document=None)),
        response(reply("Best effort answer.")),
    )
    events = run_turn(Agent(library, client=client), "Q")
    assert [r["tool_choice"] for r in client.requests] == ["auto", "none"]
    assert events[-1].text == "Best effort answer."


def test_parallel_tool_calls_all_get_results(library) -> None:
    client = FakeClient(
        response(
            tool_call("read_pages", call_id="a", document="nist", first_page=1, last_page=1),
            tool_call("read_pages", call_id="b", document="nist.pdf", first_page=2, last_page=2),
        ),
        response(reply("Done.")),
    )
    run_turn(Agent(library, client=client), "Q")
    outputs = {
        i["call_id"]: i["output"] for i in client.requests[1]["input"] if i.get("type") == "function_call_output"
    }
    assert "AC-2" in outputs["a"]
    assert "AU-2" in outputs["b"]


def test_requests_are_stateless(library) -> None:
    client = FakeClient(response(reply("Hi.")))
    run_turn(Agent(library, client=client), "Hello")
    assert client.requests[0]["store"] is False
    assert "reasoning.encrypted_content" in client.requests[0]["include"]


def test_completion_reports_pages_seen_via_in_context_text_and_tools(library) -> None:
    client = FakeClient(
        response(tool_call("read_pages", document="nist", first_page=2, last_page=2)),
        response(reply("Answer.")),
    )
    events = run_turn(Agent(library, client=client), "Q")
    # nist is tiny, so it's in context: both pages count as seen anyway.
    assert events[-1].seen_pages == {("nist", 1), ("nist", 2)}


def test_search_only_document_pages_are_seen_only_when_returned(monkeypatch) -> None:
    monkeypatch.setattr(config, "FULL_CONTEXT_TOKEN_BUDGET", 0)  # force SEARCH-ONLY
    library = asyncio.run(make_library(make_doc("big", ["alpha", "beta", "gamma"])))
    client = FakeClient(
        response(tool_call("read_pages", document="big", first_page=2, last_page=2)),
        response(reply("Answer.")),
    )
    events = run_turn(Agent(library, client=client), "Q")
    assert events[-1].seen_pages == {("big", 2)}


def test_echoed_items_drop_sdk_only_fields() -> None:
    # Regression: the streaming helper adds `parsed_arguments` to function calls;
    # the API rejects it when the item is sent back ("Unknown parameter").
    item = FakeItem("function_call", {"call_id": "c", "name": "search", "arguments": "{}", "parsed_arguments": {}})
    assert as_input(item) == {"type": "function_call", "call_id": "c", "name": "search", "arguments": "{}"}
