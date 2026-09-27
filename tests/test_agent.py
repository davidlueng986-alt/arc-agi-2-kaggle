"""The agent loop: turn accounting, ledger, early stop, forced final, recovery."""

from __future__ import annotations

import json

import pytest

from harness.agent import (
    AgentResult,
    DuckREPLAgent,
    Ledger,
    harvest_notes,
)
from harness.config import HarnessConfig
from harness.context import estimate_message_tokens
from harness.llm import ContextOverflowError, LLMRequest, LLMResponse, MockLLM, Usage, text_response, tool_response


def make_config(**budget_overrides) -> HarnessConfig:
    config = HarnessConfig(workers=1)
    config.budget.use_image = False
    config.budget.max_turns = 6
    config.budget.hard_max_turns = 8
    config.budget.max_task_seconds = 120
    config.budget.tool_timeout_s = 20
    config.budget.keep_recent_turns = 3
    config.min_turns = 1
    config.diversity_wait_turns = 2
    for key, value in budget_overrides.items():
        setattr(config.budget, key, value)
    return config


VERIFY_OK = "def transform(g):\n    return list(reversed(g))\nrep = verify(transform)\nprint(rep['summary'])"
VERIFY_BAD = "def transform(g):\n    return g\nrep = verify(transform)\nprint(rep['summary'])"
INSPECT = "print(shapes())"
FINISH = "finish('done')"


def scripted(*items):
    return MockLLM(list(items))


class TestLedger:
    def test_harvest_only_labelled_lines(self):
        text = "Hypothesis: the grid is flipped vertically.\nNext: verify.\nrandom prose line\nverify."
        notes = harvest_notes(text)
        assert notes == [
            "hypothesis: the grid is flipped vertically.",
            "next: verify.",
        ]

    def test_harvest_strips_markdown_bullets(self):
        assert harvest_notes("- Note: something") == ["note: something"]

    def test_harvest_ignores_empty_labels(self):
        assert harvest_notes("Hypothesis:") == []

    def test_harvest_caps_at_three(self):
        text = "\n".join(f"Note: {i}" for i in range(10))
        assert len(harvest_notes(text)) == 3

    def test_notes_are_truncated(self):
        notes = harvest_notes("Note: " + "x" * 500)
        assert len(notes[0]) < 200

    def test_ledger_dedupes_and_caps(self):
        ledger = Ledger(max_entries=3)
        for i in range(10):
            ledger.add_tried(f"entry {i}")
        assert len(ledger.tried) == 3
        ledger.add_note("a")
        ledger.add_note("a")
        assert ledger.notes == ["a"]

    def test_ledger_lines_respect_the_char_budget(self):
        from harness.submission import CandidatePool

        ledger = Ledger(max_entries=8, max_chars=300)
        for i in range(8):
            ledger.add_tried("x" * 200)
        assert len("\n".join(ledger.lines(CandidatePool()))) < 300

    def test_best_line(self):
        from harness.submission import CandidatePool

        ledger = Ledger()
        ledger.set_best("VERIFIED soft=1.000")
        assert ledger.lines(CandidatePool())[0].startswith("best:")


class TestAgentLoop:
    def test_happy_path_verifies_and_early_stops(self, flip_task):
        llm = scripted(
            tool_response(INSPECT, text="Hypothesis: flip.\nNext: verify."),
            tool_response(VERIFY_OK, text="Next: verify the flip."),
            tool_response(FINISH),
        )
        result = DuckREPLAgent(llm, make_config()).solve(flip_task)
        assert result.error == ""
        assert result.verified is True
        assert result.n_verified == 1
        # The verified candidate on turn 2 triggers the early stop, so the third
        # scripted response is never consumed -- that saving is the point.
        assert result.turns == 2
        assert "early-stop" in result.stop_reason
        assert result.python_calls == 2
        assert result.verify_calls == 1
        assert result.entries[0]["attempt_1"] == [[8, 8], [7, 7]]

    def test_explicit_finish_is_honoured(self, flip_task):
        config = make_config()
        config.stop_on_verified = False
        llm = scripted(tool_response(INSPECT), tool_response(VERIFY_OK), tool_response(FINISH))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert result.turns == 3
        assert "finish" in result.stop_reason
        assert result.verified is True

    def test_early_stop_on_verified_without_finish(self, flip_task):
        config = make_config()
        config.want_distinct_verified = False
        llm = scripted(tool_response(VERIFY_OK), tool_response(VERIFY_OK), tool_response(INSPECT))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert result.verified is True
        assert "early-stop" in result.stop_reason
        assert result.turns == 1

    def test_early_stop_waits_for_a_distinct_second_answer(self, flip_task):
        config = make_config()
        config.diversity_wait_turns = 10
        # identical outputs both times -> no distinct pair, so no early stop
        llm = scripted(tool_response(VERIFY_OK), tool_response(VERIFY_OK), tool_response(FINISH))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert result.turns == 3
        assert result.entries[0]["attempt_1"] == result.entries[0]["attempt_2"]

    def test_early_stop_fires_on_two_distinct_verified_outputs(self, ambiguous_task):
        config = make_config()
        config.diversity_wait_turns = 10
        rule_a = "def transform(g):\n    return [[1] * len(row) for row in g]\nverify(transform)"
        rule_b = "def transform(g):\n    return [[1, 1], [1, 1]]\nverify(transform)"
        llm = scripted(tool_response(rule_a), tool_response(rule_b), tool_response(FINISH))
        result = DuckREPLAgent(llm, config).solve(ambiguous_task)
        assert result.n_verified == 2
        assert "early-stop" in result.stop_reason
        assert result.turns == 2
        assert result.entries[0]["attempt_1"] == [[1, 1, 1], [1, 1, 1]]
        assert result.entries[0]["attempt_2"] == [[1, 1], [1, 1]]

    def test_min_turns_is_respected(self, flip_task):
        config = make_config()
        config.min_turns = 3
        config.want_distinct_verified = False
        llm = scripted(tool_response(VERIFY_OK), tool_response(INSPECT), tool_response(INSPECT), tool_response(FINISH))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert result.turns >= 3

    def test_turn_cap_forces_a_final_answer(self, flip_task):
        config = make_config(max_turns=2, hard_max_turns=4)
        llm = scripted(tool_response(VERIFY_BAD), tool_response(VERIFY_BAD), tool_response(VERIFY_OK))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert "turn cap" in result.stop_reason or result.verified
        assert result.entries

    def test_no_tool_call_produces_a_nudge(self, flip_task):
        llm = scripted(text_response("I think it is a flip."), tool_response(FINISH))
        result = DuckREPLAgent(llm, make_config()).solve(flip_task)
        nudges = [
            m
            for request in llm.requests
            for m in request.messages
            if m.get("role") == "user" and "did not call the `python` tool" in str(m.get("content"))
        ]
        assert nudges
        assert result.python_calls == 1

    def test_unknown_tool_is_answered_not_executed(self, flip_task):
        llm = MockLLM(
            [
                LLMResponse(
                    text="",
                    tool_calls=(
                        type(
                            "TC",
                            (),
                            {"id": "c1", "name": "shell", "arguments": {"cmd": "ls"}, "raw_arguments": "{}"},
                        )(),
                    ),
                    usage=Usage(prompt_tokens=1, completion_tokens=1),
                ),
                tool_response(FINISH),
            ]
        )
        result = DuckREPLAgent(llm, make_config()).solve(flip_task)
        tool_messages = [
            m
            for m in llm.requests[-1].messages
            if m.get("role") == "tool"
        ]
        assert any("unknown tool" in str(m.get("content")) for m in tool_messages)
        assert result.python_calls == 1  # the finish() call only

    def test_extra_tool_calls_beyond_the_cap_get_a_stub_result(self, flip_task):
        config = make_config(max_python_calls_per_turn=1)
        calls = tuple(
            type("TC", (), {"id": f"c{i}", "name": "python", "arguments": {"code": INSPECT}, "raw_arguments": "{}"})()
            for i in range(3)
        )
        llm = MockLLM(
            [
                LLMResponse(text="", tool_calls=calls, usage=Usage(prompt_tokens=1, completion_tokens=1)),
                tool_response(FINISH),
            ]
        )
        result = DuckREPLAgent(llm, config).solve(flip_task)
        contents = [str(m.get("content")) for m in llm.requests[-1].messages if m.get("role") == "tool"]
        assert any("only 1 tool calls are executed" in c for c in contents)
        assert result.python_calls == 2  # one executed in turn 1 + finish() in turn 2

    def test_tool_messages_always_follow_their_assistant(self, flip_task):
        llm = scripted(tool_response(INSPECT), tool_response(VERIFY_BAD), tool_response(FINISH))
        DuckREPLAgent(llm, make_config()).solve(flip_task)
        for request in llm.requests:
            messages = request.messages
            for index, message in enumerate(messages):
                if message.get("role") == "tool":
                    assert index > 0
                    previous = messages[index - 1]
                    assert previous.get("role") in {"assistant", "tool"}

    def test_status_message_carries_the_ledger(self, flip_task):
        llm = scripted(
            tool_response(VERIFY_BAD, text="Hypothesis: identity mapping."),
            tool_response(VERIFY_OK, text="Hypothesis: vertical flip."),
            tool_response(FINISH),
        )
        DuckREPLAgent(llm, make_config()).solve(flip_task)
        last = llm.requests[-1].messages
        status = [m for m in last if m.get("role") == "user" and "Ledger" in str(m.get("content"))]
        assert status
        assert "best:" in str(status[-1]["content"])

    def test_context_window_is_bounded_by_eviction(self, flip_task):
        # The invariant that matters: the rolling history never grows past
        # keep_recent_turns assistant blocks. The floor is the protected prefix
        # plus those turns, which eviction is forbidden to touch.
        config = make_config(context_budget=8000, keep_recent_turns=2)
        long_code = "print('x' * 2000)\n" * 3
        llm = scripted(*[tool_response(long_code) for _ in range(6)], tool_response(FINISH))
        DuckREPLAgent(llm, config).solve(flip_task)
        assert len(llm.requests) >= 5  # it really ran long enough to evict
        for request in llm.requests:
            assistants = [m for m in request.messages if m.get("role") == "assistant"]
            assert len(assistants) <= config.budget.keep_recent_turns + 1

    def test_a_negative_context_budget_is_clamped_not_fatal(self, flip_task):
        config = make_config(context_budget=100, completion_reserve=4096)
        llm = scripted(tool_response(INSPECT), tool_response(FINISH))
        result = DuckREPLAgent(llm, config).solve(flip_task)
        assert result.stop_reason != "context overflow"
        for request in llm.requests:
            assert request.messages[0]["role"] == "system"

    def test_system_and_task_prefix_are_never_evicted(self, flip_task):
        config = make_config(context_budget=1200, keep_recent_turns=1)
        llm = scripted(*[tool_response("print('y' * 3000)") for _ in range(5)], tool_response(FINISH))
        DuckREPLAgent(llm, config).solve(flip_task)
        for request in llm.requests:
            assert request.messages[0]["role"] == "system"
            assert "TASK flipud-synthetic" in str(request.messages[1]["content"])

    def test_llm_error_is_captured_not_raised(self, flip_task):
        class Boom:
            def complete(self, request):
                raise RuntimeError("server down")

        result = DuckREPLAgent(Boom(), make_config()).solve(flip_task)
        assert result.error.startswith("RuntimeError")
        assert result.stop_reason == "llm error"
        assert result.entries  # fallback still produced

    def test_context_overflow_recovers_by_dropping_history(self, flip_task):
        calls = {"n": 0}

        class OverflowOnce:
            def complete(self, request):
                calls["n"] += 1
                if calls["n"] == 1:
                    return tool_response(INSPECT)  # build some history first
                if calls["n"] == 2:
                    raise ContextOverflowError("maximum context length")
                return tool_response(FINISH)

        result = DuckREPLAgent(OverflowOnce(), make_config()).solve(flip_task)
        assert calls["n"] == 3
        assert result.stop_reason != "context overflow"
        assert "finish" in result.stop_reason

    def test_context_overflow_shrinks_the_task_block_as_a_last_resort(self, flip_task):
        calls = {"n": 0}

        class AlwaysOverflow:
            def complete(self, request):
                calls["n"] += 1
                if calls["n"] <= 3:
                    raise ContextOverflowError("maximum context length")
                return tool_response(FINISH)

        result = DuckREPLAgent(AlwaysOverflow(), make_config(use_image=False)).solve(flip_task)
        assert calls["n"] == 4
        assert result.stop_reason != "context overflow"

    def test_unrecoverable_context_overflow_gives_up_cleanly(self, flip_task):
        class AlwaysOverflow:
            def complete(self, request):
                raise ContextOverflowError("maximum context length")

        result = DuckREPLAgent(AlwaysOverflow(), make_config()).solve(flip_task)
        assert result.stop_reason == "context overflow"
        assert result.entries  # a schema-valid fallback still exists

    def test_usage_is_accumulated(self, flip_task):
        llm = scripted(
            tool_response(INSPECT, usage={"prompt_tokens": 1000, "completion_tokens": 100}),
            tool_response(FINISH, usage={"prompt_tokens": 1200, "completion_tokens": 50}),
        )
        result = DuckREPLAgent(llm, make_config()).solve(flip_task)
        assert result.usage.prompt_tokens == 2200
        assert result.usage.completion_tokens == 150
        assert result.usage.calls == 2

    def test_transcript_is_kept_when_asked(self, flip_task):
        llm = scripted(tool_response(INSPECT), tool_response(FINISH))
        result = DuckREPLAgent(llm, make_config(), keep_transcript=True).solve(flip_task)
        assert result.transcript
        assert result.transcript[0]["role"] == "assistant"

    def test_image_is_attached_on_the_first_turn_only(self, flip_task):
        config = make_config()
        config.budget.use_image = True
        llm = scripted(tool_response(INSPECT), tool_response(INSPECT), tool_response(FINISH))
        DuckREPLAgent(llm, config).solve(flip_task)
        for request in llm.requests:
            images = [
                part
                for message in request.messages
                if isinstance(message.get("content"), list)
                for part in message["content"]
                if part.get("type") == "image_url"
            ]
            assert len(images) <= 1
            if images:
                assert request.messages[1]["content"] == images[0] or images[0] in request.messages[1]["content"]

    def test_reasoning_effort_schedule(self, flip_task):
        llm = scripted(tool_response(VERIFY_BAD), tool_response(VERIFY_BAD), tool_response(FINISH))
        DuckREPLAgent(llm, make_config()).solve(flip_task)
        efforts = [request.reasoning_effort for request in llm.requests]
        assert efforts[0] == "medium"  # first turn
        assert all(e in ("low", "medium") for e in efforts)
        assert "xhigh" not in efforts and "high" not in efforts

    def test_shape_failure_raises_effort(self, crop_task):
        llm = scripted(tool_response(VERIFY_BAD), tool_response(VERIFY_BAD), tool_response(FINISH))
        DuckREPLAgent(llm, make_config()).solve(crop_task)
        # identity on a crop task is a shape mismatch -> effort escalates
        assert llm.requests[1].reasoning_effort == "medium"

    def test_reasoning_is_never_replayed_into_the_prompt(self, flip_task):
        # Thinking tokens are paid once as completion. If they were echoed back
        # into the message list, every later turn would re-prefill them -- at
        # ~1.5K reasoning tokens/turn that is the single biggest cost leak
        # available, and prefill is the dominant term in the budget.
        llm = MockLLM(
            [
                LLMResponse(
                    text="Hypothesis: flip.",
                    reasoning="R" * 4000,
                    tool_calls=tool_response(INSPECT).tool_calls,
                    usage=Usage(prompt_tokens=10, completion_tokens=1500, reasoning_tokens=1400),
                ),
                tool_response(FINISH),
            ]
        )
        result = DuckREPLAgent(llm, make_config()).solve(flip_task)
        for request in llm.requests:
            blob = json.dumps(request.messages, default=str)
            assert "RRRR" not in blob
            for message in request.messages:
                assert "reasoning_content" not in message
        assert result.usage.reasoning_tokens == 1400  # accounted, but not resent

    def test_status_message_is_omitted_when_nothing_changed(self, flip_task):
        config = make_config()
        config.stop_on_verified = False
        llm = scripted(tool_response(INSPECT), tool_response(INSPECT), tool_response(FINISH))
        DuckREPLAgent(llm, config).solve(flip_task)
        statuses = [
            m
            for m in llm.requests[-1].messages
            if m.get("role") == "user" and "Ledger" in str(m.get("content"))
        ]
        # inspection-only turns register nothing, so they get no status message
        assert len(statuses) == 0

        result = AgentResult(task_id="x")
        assert result.turns == 0
        assert result.entries == []
        assert result.verified is False

    def test_agent_result_defaults(self):
        result = AgentResult(task_id="x")
        assert result.turns == 0
        assert result.entries == []
        assert result.verified is False

    def test_sandbox_factory_injection(self, flip_task):
        from harness.sandbox import ReplSandbox

        made = []

        def factory(task, on_verify):
            sandbox = ReplSandbox(task, timeout_s=20, on_verify=on_verify)
            made.append(sandbox)
            return sandbox

        llm = scripted(tool_response(VERIFY_OK), tool_response(FINISH))
        result = DuckREPLAgent(llm, make_config(), sandbox_factory=factory).solve(flip_task)
        assert made and result.verified

    def test_global_deadline_stops_the_task(self, flip_task):
        from harness.budget import Deadline

        deadline = Deadline(0.01)
        import time

        time.sleep(0.05)
        llm = scripted(tool_response(INSPECT), tool_response(FINISH))
        result = DuckREPLAgent(llm, make_config(), deadline=deadline).solve(flip_task)
        assert "deadline" in result.stop_reason
        assert result.turns == 0
        assert result.entries
