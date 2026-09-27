"""LLM client: payload construction, response parsing, markup recovery, retries."""

from __future__ import annotations

import json

import pytest

from harness.llm import (
    VALID_REASONING_EFFORTS,
    ContextOverflowError,
    ConcurrencyLimitedClient,
    LLMError,
    LLMRequest,
    MockLLM,
    Usage,
    VLLMClient,
    parse_tool_arguments,
    recover_tool_calls,
    strip_code_fence,
    text_response,
    tool_response,
)


def _payload(**overrides):
    payload = {
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "content": "Hypothesis: flip.",
                    "reasoning_content": "thinking hard",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "python", "arguments": '{"code": "print(1)"}'},
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 140,
            "completion_tokens_details": {"reasoning_tokens": 25},
        },
    }
    payload.update(overrides)
    return payload


class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.sent = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.sent.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, Exception):
            raise item
        return item


class TestParsing:
    def test_parses_text_reasoning_tools_usage(self):
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=_payload())]))
        response = client.complete(LLMRequest(messages=[]))
        assert response.text == "Hypothesis: flip."
        assert response.reasoning == "thinking hard"
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "python"
        assert response.tool_calls[0].arguments == {"code": "print(1)"}
        assert response.usage.prompt_tokens == 100
        assert response.usage.reasoning_tokens == 25
        assert response.finish_reason == "tool_calls"

    def test_to_usage_maps_onto_budget_usage(self):
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=_payload())]))
        usage = client.complete(LLMRequest(messages=[])).to_usage()
        assert usage.calls == 1
        assert usage.total_tokens == 140
        assert usage.reasoning_tokens == 25

    def test_dict_arguments_accepted(self):
        payload = _payload()
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = {"code": "x"}
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        assert client.complete(LLMRequest(messages=[])).tool_calls[0].arguments == {"code": "x"}

    def test_tool_call_without_a_name_is_skipped(self):
        payload = _payload()
        payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"] = ""
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        assert client.complete(LLMRequest(messages=[])).tool_calls == ()

    def test_no_choices_raises(self):
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload={"choices": []})]))
        with pytest.raises(LLMError):
            client.complete(LLMRequest(messages=[]))

    def test_list_content_is_flattened(self):
        payload = _payload()
        payload["choices"][0]["message"]["content"] = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        assert client.complete(LLMRequest(messages=[])).text == "a\nb"


class TestMarkupRecovery:
    def _markup(self, name="python", code="print(1)"):
        return (
            "Let me inspect.\n"
            + chr(60)
            + "tool_call"
            + chr(62)
            + "\n"
            + chr(60)
            + f"function={name}"
            + chr(62)
            + "\n"
            + chr(60)
            + "parameter=code"
            + chr(62)
            + f"\n{code}\n"
            + chr(60)
            + "/parameter"
            + chr(62)
            + "\n"
            + chr(60)
            + "/function"
            + chr(62)
            + "\n"
            + chr(60)
            + "/tool_call"
            + chr(62)
        )

    def test_recovers_a_tool_call_from_plain_text(self):
        payload = _payload()
        message = payload["choices"][0]["message"]
        message["tool_calls"] = []
        message["content"] = self._markup(code="verify(transform)")
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        response = client.complete(LLMRequest(messages=[]))
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "python"
        assert response.tool_calls[0].arguments == {"code": "verify(transform)"}

    def test_markup_is_removed_from_the_visible_text(self):
        payload = _payload()
        message = payload["choices"][0]["message"]
        message["tool_calls"] = []
        message["content"] = self._markup()
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        assert "tool_call" not in client.complete(LLMRequest(messages=[])).text

    def test_recovery_from_reasoning_channel(self):
        payload = _payload()
        message = payload["choices"][0]["message"]
        message["tool_calls"] = []
        message["content"] = "no markup here"
        message["reasoning_content"] = self._markup()
        client = VLLMClient(session=_FakeSession([_FakeResponse(payload=payload)]))
        assert len(client.complete(LLMRequest(messages=[])).tool_calls) == 1

    def test_no_markup_means_no_calls(self):
        assert recover_tool_calls("just prose") == ()
        assert recover_tool_calls("") == ()

    def test_duplicate_calls_deduplicated(self):
        doubled = self._markup() + "\n" + self._markup()
        assert len(recover_tool_calls(doubled)) == 1


class TestArgumentParsing:
    def test_plain_json(self):
        assert parse_tool_arguments('{"code": "x"}') == {"code": "x"}

    def test_fenced_json(self):
        assert parse_tool_arguments('```json\n{"code": "x"}\n```') == {"code": "x"}

    def test_non_object_json_is_wrapped(self):
        assert parse_tool_arguments("[1,2]") == {"value": [1, 2]}

    def test_bare_code_falls_back_to_code_key(self):
        assert parse_tool_arguments("print(1)") == {"code": "print(1)"}

    def test_empty(self):
        assert parse_tool_arguments("") == {}
        assert parse_tool_arguments(None) == {}

    def test_strip_code_fence(self):
        assert strip_code_fence("```python\nx=1\n```") == "x=1"
        assert strip_code_fence("x=1") == "x=1"


class TestPayload:
    def test_vllm_extras(self):
        client = VLLMClient(model="qwen3.8-27b")
        payload = client.build_payload(
            LLMRequest(messages=[{"role": "user", "content": "hi"}], top_k=20, seed=7, max_tokens=99)
        )
        assert payload["model"] == "qwen3.8-27b"
        assert payload["stream"] is False
        assert payload["top_k"] == 20
        assert payload["seed"] == 7
        assert payload["max_tokens"] == 99
        assert payload["chat_template_kwargs"] == {"enable_thinking": True}

    def test_tools_and_choice(self):
        payload = VLLMClient().build_payload(
            LLMRequest(messages=[], tools=[{"type": "function"}], tool_choice="auto")
        )
        assert payload["tools"] and payload["tool_choice"] == "auto"

    @pytest.mark.parametrize("effort", VALID_REASONING_EFFORTS)
    def test_valid_efforts_are_sent(self, effort):
        payload = VLLMClient().build_payload(LLMRequest(messages=[], reasoning_effort=effort))
        assert payload["reasoning_effort"] == effort

    def test_high_is_rejected_locally(self):
        # QwenLM/Qwen3.8#217: `high` returns HTTP 500. Fail fast with a clear message.
        with pytest.raises(LLMError) as info:
            VLLMClient().build_payload(LLMRequest(messages=[], reasoning_effort="high"))
        assert "HTTP 500" in str(info.value)

    def test_extra_body_wins(self):
        client = VLLMClient(extra_body={"temperature": 0.1})
        assert client.build_payload(LLMRequest(messages=[]))["temperature"] == 0.1

    def test_negative_seed_omitted(self):
        assert "seed" not in VLLMClient().build_payload(LLMRequest(messages=[], seed=-1))


class TestErrorHandling:
    def test_context_overflow_is_typed(self):
        session = _FakeSession([_FakeResponse(400, text="This model's maximum context length is 32768 tokens")])
        with pytest.raises(ContextOverflowError):
            VLLMClient(session=session).complete(LLMRequest(messages=[]))

    def test_500_on_effort_drops_it_and_retries(self):
        session = _FakeSession(
            [_FakeResponse(500, text="boom"), _FakeResponse(payload=_payload())]
        )
        client = VLLMClient(session=session, drop_effort_on_500=True)
        response = client.complete(LLMRequest(messages=[], reasoning_effort="medium"))
        assert response.text
        assert client.stats["effort_dropped"] == 1
        assert "reasoning_effort" not in session.sent[-1]["json"]

    def test_effort_stays_dropped_for_the_rest_of_the_run(self):
        session = _FakeSession([_FakeResponse(500, text="boom"), _FakeResponse(payload=_payload())])
        client = VLLMClient(session=session)
        client.complete(LLMRequest(messages=[], reasoning_effort="medium"))
        client.complete(LLMRequest(messages=[], reasoning_effort="medium"))
        assert "reasoning_effort" not in session.sent[-1]["json"]

    def test_500_without_effort_is_retried_then_raises(self):
        session = _FakeSession([_FakeResponse(500, text="boom")])
        client = VLLMClient(session=session, max_retries=1, retry_backoff=0.0)
        with pytest.raises(LLMError):
            client.complete(LLMRequest(messages=[]))
        assert client.stats["retries"] >= 1

    def test_transport_error_is_retried(self):
        session = _FakeSession([ConnectionError("refused"), _FakeResponse(payload=_payload())])
        client = VLLMClient(session=session, max_retries=2, retry_backoff=0.0)
        assert client.complete(LLMRequest(messages=[])).text

    def test_4xx_other_than_context_is_fatal(self):
        session = _FakeSession([_FakeResponse(404, text="not found")])
        with pytest.raises(LLMError):
            VLLMClient(session=session, max_retries=0).complete(LLMRequest(messages=[]))

    def test_malformed_json_is_retried(self):
        session = _FakeSession([_FakeResponse(200, payload=None, text="not json"), _FakeResponse(payload=_payload())])
        client = VLLMClient(session=session, max_retries=2, retry_backoff=0.0)
        assert client.complete(LLMRequest(messages=[])).text

    def test_auth_header(self):
        session = _FakeSession([_FakeResponse(payload=_payload())])
        VLLMClient(base_url="http://x/v1/", api_key="KEY", session=session).complete(LLMRequest(messages=[]))
        assert session.sent[0]["headers"]["Authorization"] == "Bearer KEY"
        assert session.sent[0]["url"] == "http://x/v1/chat/completions"


class TestMockLLM:
    def test_script_cycles(self):
        mock = MockLLM([text_response("a"), text_response("b")])
        assert [mock.complete(LLMRequest(messages=[])).text for _ in range(3)] == ["a", "b", "a"]

    def test_responder_sees_the_request(self):
        seen = []

        def respond(request):
            seen.append(request)
            return text_response("ok")

        mock = MockLLM(responder=respond)
        mock.complete(LLMRequest(messages=[{"role": "user", "content": "hi"}]))
        assert seen[0].messages[0]["content"] == "hi"
        assert mock.last_messages[0]["content"] == "hi"

    def test_needs_script_or_responder(self):
        with pytest.raises(ValueError):
            MockLLM()

    def test_tool_response_helper(self):
        response = tool_response("print(1)", text="hi")
        assert response.tool_calls[0].arguments == {"code": "print(1)"}
        assert response.text == "hi"
        assert response.tool_calls[0].name == "python"

    def test_calls_counter(self):
        mock = MockLLM([text_response("a")])
        mock.complete(LLMRequest(messages=[]))
        mock.complete(LLMRequest(messages=[]))
        assert mock.calls == 2


class TestConcurrencyLimiter:
    def test_passes_through(self):
        inner = MockLLM([text_response("a")])
        client = ConcurrencyLimitedClient(inner, 2)
        assert client.complete(LLMRequest(messages=[])).text == "a"
