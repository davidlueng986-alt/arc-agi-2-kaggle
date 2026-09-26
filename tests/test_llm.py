from __future__ import annotations

import json

import pytest

from harness.grids import to_csv
from harness.llm import (
    EFFORT_RANK,
    VALID_EFFORTS,
    ChatRequest,
    ChatResponse,
    EffortError,
    HTTPChatClient,
    LLMClient,
    MockLLM,
    RoundRobinClient,
    Usage,
    validate_effort,
)


def test_valid_efforts_are_exactly_three():
    assert set(VALID_EFFORTS) == {"xhigh", "medium", "low"}
    assert EFFORT_RANK == {"low": 0, "medium": 1, "xhigh": 2}


@pytest.mark.parametrize("effort", ["xhigh", "medium", "low"])
def test_validate_effort_accepts_valid(effort):
    assert validate_effort(effort) == effort


@pytest.mark.parametrize("effort", ["high", "x-high", "XHIGH", "", "none", "max"])
def test_validate_effort_rejects_invalid(effort):
    with pytest.raises(EffortError):
        validate_effort(effort)


def test_chat_request_rejects_high_at_construction():
    with pytest.raises(EffortError):
        ChatRequest(messages=[], reasoning_effort="high")


def test_chat_request_defaults_to_xhigh():
    assert ChatRequest(messages=[]).reasoning_effort == "xhigh"


def test_usage_totals():
    usage = Usage(prompt_tokens=10, completion_tokens=20, reasoning_tokens=15)
    assert usage.total == 30
    assert usage.as_dict()["reasoning_tokens"] == 15


def test_chat_response_flags():
    assert ChatResponse().ok
    assert not ChatResponse(error="boom").ok
    assert ChatResponse(finish_reason="length").truncated
    assert not ChatResponse(finish_reason="stop").truncated


def test_http_payload_chat_template_kwargs_mechanism():
    client = HTTPChatClient(effort_mechanism="chat_template_kwargs")
    payload = client._payload(ChatRequest(messages=[{"role": "user", "content": "hi"}], max_tokens=99, seed=7))
    assert payload["chat_template_kwargs"] == {"reasoning_effort": "xhigh"}
    assert "reasoning_effort" not in payload
    assert payload["max_tokens"] == 99
    assert payload["seed"] == 7
    assert payload["stream"] is False
    assert payload["model"] == "qwen3.8-27b"


def test_http_payload_top_level_mechanism():
    client = HTTPChatClient(effort_mechanism="top_level")
    payload = client._payload(ChatRequest(messages=[], reasoning_effort="low"))
    assert payload["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in payload


def test_http_payload_none_mechanism():
    client = HTTPChatClient(effort_mechanism="none")
    payload = client._payload(ChatRequest(messages=[]))
    assert "chat_template_kwargs" not in payload
    assert "reasoning_effort" not in payload


def test_http_payload_rejects_unknown_mechanism():
    with pytest.raises(ValueError):
        HTTPChatClient(effort_mechanism="magic")


def test_http_payload_extra_and_seed_omission():
    client = HTTPChatClient(extra_payload={"n": 1, "logprobs": False})
    payload = client._payload(ChatRequest(messages=[]))
    assert payload["n"] == 1
    assert "seed" not in payload


def test_http_parse_extracts_reasoning_and_usage():
    client = HTTPChatClient()
    raw = {
        "choices": [
            {
                "message": {"content": "ANSWER 1:\n1,2", "reasoning_content": "thoughts"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 5000,
            "completion_tokens_details": {"reasoning_tokens": 4900},
        },
    }
    resp = client._parse(raw, 1.5, 1, {"task_id": "x"})
    assert resp.content == "ANSWER 1:\n1,2"
    assert resp.reasoning == "thoughts"
    assert resp.usage.reasoning_tokens == 4900
    assert resp.usage.completion_tokens == 5000
    assert resp.finish_reason == "stop"
    assert not resp.truncated
    assert resp.meta == {"task_id": "x"}


def test_http_parse_handles_content_parts_and_missing_fields():
    client = HTTPChatClient()
    raw = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}}]}
    resp = client._parse(raw, 0.1, 1, {})
    assert resp.content == "ab"
    assert resp.usage.total == 0


def test_http_parse_handles_empty_choices():
    resp = HTTPChatClient()._parse({}, 0.1, 1, {})
    assert resp.content == ""
    assert resp.ok


def test_http_complete_retries_then_fails(monkeypatch):
    client = HTTPChatClient(max_retries=2, backoff=0.0)
    calls = {"n": 0}

    def boom(*_a, **_k):
        calls["n"] += 1
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    resp = client.complete(ChatRequest(messages=[]))
    assert not resp.ok
    assert "connection refused" in resp.error
    assert calls["n"] == 2


def test_http_complete_does_not_retry_client_errors(monkeypatch):
    import urllib.error

    client = HTTPChatClient(max_retries=3, backoff=0.0)
    calls = {"n": 0}

    def http400(*_a, **_k):
        calls["n"] += 1
        raise urllib.error.HTTPError("url", 400, "Bad Request", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", http400)
    resp = client.complete(ChatRequest(messages=[]))
    assert not resp.ok
    assert "HTTP 400" in resp.error
    assert calls["n"] == 1


def test_http_health_and_wait(monkeypatch):
    client = HTTPChatClient()
    monkeypatch.setattr(client, "health", lambda: False)
    assert client.wait_healthy(0.0, poll=0.0) is False
    monkeypatch.setattr(client, "health", lambda: True)
    assert client.wait_healthy(1e9, poll=0.0) is True


def test_mock_llm_returns_correct_grids_when_probability_one():
    truth = [[[1, 2], [3, 4]]]
    mock = MockLLM(answers={"t1": truth}, correct_prob=1.0, seed=1, tok_per_sec=0)
    resp = mock.complete(
        ChatRequest(messages=[{"role": "user", "content": "x"}], meta={"task_id": "t1"}, reasoning_effort="low")
    )
    assert resp.ok
    assert f"ANSWER 1:\n{to_csv(truth[0])}" in resp.content
    assert resp.usage.completion_tokens > 0


def test_mock_llm_reports_token_usage_by_effort():
    mock = MockLLM(answers={"t1": [[[0]]]}, correct_prob=1.0, tok_per_sec=0)
    low = mock.complete(ChatRequest(messages=[], meta={"task_id": "t1"}, reasoning_effort="low"))
    xhigh = mock.complete(ChatRequest(messages=[], meta={"task_id": "t1"}, reasoning_effort="xhigh"))
    assert low.usage.reasoning_tokens < xhigh.usage.reasoning_tokens
    assert mock.tokens_emitted == low.usage.completion_tokens + xhigh.usage.completion_tokens


def test_mock_llm_can_emit_malformed_output():
    mock = MockLLM(answers={"t1": [[[0]]]}, correct_prob=0.0, malformed_prob=1.0, tok_per_sec=0)
    resp = mock.complete(ChatRequest(messages=[], meta={"task_id": "t1"}))
    assert "ANSWER" not in resp.content


def test_mock_llm_unknown_task_yields_parseable_placeholder():
    mock = MockLLM(answers={}, tok_per_sec=0)
    resp = mock.complete(ChatRequest(messages=[], meta={"task_id": "zzz"}))
    assert "ANSWER 1:" in resp.content


def test_mock_llm_answer_fn_override():
    mock = MockLLM(answer_fn=lambda req: [[[req.meta["sample"]]]], tok_per_sec=0)
    resp = mock.complete(ChatRequest(messages=[], meta={"sample": 7}))
    assert "7" in resp.content


def test_mock_llm_records_calls():
    mock = MockLLM(answers={"t": [[[0]]]}, correct_prob=1.0, tok_per_sec=0)
    for i in range(3):
        mock.complete(ChatRequest(messages=[], meta={"task_id": "t"}, seed=i))
    assert len(mock.calls) == 3


def test_mock_llm_multi_test_answers():
    mock = MockLLM(answers={"t": [[[1]], [[2]]]}, correct_prob=1.0, tok_per_sec=0)
    resp = mock.complete(ChatRequest(messages=[], meta={"task_id": "t"}))
    assert "ANSWER 1:" in resp.content and "ANSWER 2:" in resp.content


def test_mock_llm_is_deterministic_for_same_seed():
    kwargs = dict(answers={"t": [[[1, 1]]]}, correct_prob=0.5, tok_per_sec=0)
    a = MockLLM(seed=5, **kwargs)
    b = MockLLM(seed=5, **kwargs)
    reqs = [ChatRequest(messages=[], meta={"task_id": "t"}, seed=i) for i in range(5)]
    assert [a.complete(r).content for r in reqs] == [b.complete(r).content for r in reqs]


def test_http_client_payload_is_valid_json():
    client = HTTPChatClient()
    payload = client._payload(ChatRequest(messages=[{"role": "user", "content": "é"}]))
    assert json.loads(json.dumps(payload))["messages"][0]["content"] == "é"


class _Tagged(LLMClient):
    name = "tagged"

    def __init__(self, tag, healthy=True):
        self.tag = tag
        self._healthy = healthy
        self.seen = []

    def complete(self, request):
        self.seen.append(request)
        return ChatResponse(content=self.tag, usage=Usage(1, 1, 0), meta=dict(request.meta))

    def health(self):
        return self._healthy


def test_round_robin_distributes_evenly():
    a, b = _Tagged("a"), _Tagged("b")
    rr = RoundRobinClient([a, b])
    out = [rr.complete(ChatRequest(messages=[])).content for _ in range(6)]
    assert out == ["a", "b", "a", "b", "a", "b"]
    assert len(a.seen) == len(b.seen) == 3


def test_round_robin_single_client():
    a = _Tagged("a")
    rr = RoundRobinClient([a])
    assert [rr.complete(ChatRequest(messages=[])).content for _ in range(3)] == ["a"] * 3


def test_round_robin_rejects_empty():
    with pytest.raises(ValueError):
        RoundRobinClient([])


def test_round_robin_health_is_any_and_healthy_filters():
    a, b = _Tagged("a", True), _Tagged("b", False)
    rr = RoundRobinClient([a, b])
    assert rr.health() is True
    assert rr.healthy() == [a]
    assert RoundRobinClient([_Tagged("x", False)]).health() is False


def test_round_robin_close_propagates():
    closed = []
    a = _Tagged("a")
    a.close = lambda: closed.append("a")
    RoundRobinClient([a]).close()
    assert closed == ["a"]
