"""Context estimation and eviction."""

from __future__ import annotations

import pytest

from harness.context import (
    PROTECTED_PREFIX,
    block_starts,
    count_assistant_turns,
    drop_oldest_block,
    estimate_message_tokens,
    estimate_tokens,
    strip_image,
    trim_to_budget,
)


def system(text="S" * 300):
    return {"role": "system", "content": text}


def task_block(text="T" * 300, with_image=False):
    if with_image:
        return {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    return {"role": "user", "content": text}


def turn(index, body="x" * 300):
    return [
        {
            "role": "assistant",
            "content": f"Hypothesis: {index}",
            "tool_calls": [
                {
                    "id": f"call_{index}",
                    "type": "function",
                    "function": {"name": "python", "arguments": '{"code": "%s"}' % body},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{index}", "content": f"result {index} {body}"},
    ]


def conversation(n_turns, **kwargs):
    messages = [system(), task_block(**kwargs)]
    for i in range(n_turns):
        messages.extend(turn(i))
    return messages


class TestEstimation:
    def test_chars_per_token(self):
        # json-encodes first, so the quotes are counted: ceil(32/3) = 11
        assert estimate_tokens("a" * 30) == 11
        assert estimate_tokens("") >= 1

    def test_tools_are_counted(self):
        messages = [system()]
        without = estimate_message_tokens(messages)
        with_tools = estimate_message_tokens(messages, tools=[{"type": "function", "function": {"name": "python"}}])
        assert with_tools > without

    def test_image_part_costs_the_supplied_estimate(self):
        messages = [system(), task_block(with_image=True)]
        assert estimate_message_tokens(messages, image_tokens=0) < estimate_message_tokens(
            messages, image_tokens=500
        )
        assert estimate_message_tokens(messages, image_tokens=500) - estimate_message_tokens(
            messages, image_tokens=0
        ) == 500

    def test_tool_calls_are_counted(self):
        plain = estimate_message_tokens([{"role": "assistant", "content": "hi"}])
        with_calls = estimate_message_tokens(
            [
                {
                    "role": "assistant",
                    "content": "hi",
                    "tool_calls": [{"id": "1", "function": {"name": "python", "arguments": "{}"}}],
                }
            ]
        )
        assert with_calls > plain


class TestBlockStructure:
    def test_block_starts_skip_the_prefix(self):
        messages = conversation(3)
        assert block_starts(messages) == [2, 4, 6]

    def test_count_turns(self):
        assert count_assistant_turns(conversation(4)) == 4

    def test_drop_oldest_removes_assistant_and_its_tool(self):
        messages = conversation(3)
        assert drop_oldest_block(messages, preserve_recent_turns=1) is True
        assert len(messages) == 2 + 4
        assert messages[2]["tool_calls"][0]["id"] == "call_1"

    def test_drop_refuses_below_preserve_recent(self):
        messages = conversation(2)
        assert drop_oldest_block(messages, preserve_recent_turns=2) is False
        assert len(messages) == 6

    def test_drop_with_nothing_droppable(self):
        messages = [system(), task_block()]
        assert drop_oldest_block(messages, preserve_recent_turns=0) is False

    def test_never_leaves_an_orphan_tool_message(self):
        messages = [system(), task_block(), {"role": "tool", "tool_call_id": "x", "content": "orphan"}]
        messages.extend(turn(0))
        assert drop_oldest_block(messages, preserve_recent_turns=0) is True
        # the dropped block's assistant is gone and the orphaned tool result that
        # preceded it was cleaned up too, leaving only the protected prefix
        assert len(messages) == PROTECTED_PREFIX
        assert [m["role"] for m in messages] == ["system", "user"]
        assert not any(m.get("role") == "tool" for m in messages)

    def test_orphan_cleanup_only_touches_the_prefix_boundary(self):
        messages = [system(), task_block()]
        messages.extend(turn(0))
        messages.append({"role": "user", "content": "status"})
        messages.extend(turn(1))
        drop_oldest_block(messages, preserve_recent_turns=1)
        assert messages[PROTECTED_PREFIX]["role"] == "assistant"
        assert messages[PROTECTED_PREFIX]["tool_calls"][0]["id"] == "call_1"

    def test_status_user_messages_are_dropped_with_their_block(self):
        messages = conversation(3)
        messages.insert(3, {"role": "user", "content": "status"})
        drop_oldest_block(messages, preserve_recent_turns=1)
        assert not any(m.get("content") == "status" for m in messages)


class TestImageStrip:
    def test_strip_replaces_with_text(self):
        messages = [system(), task_block(with_image=True)]
        assert strip_image(messages) is True
        assert isinstance(messages[1]["content"], str)
        assert "image dropped" in messages[1]["content"]
        assert "T" * 100 in messages[1]["content"]

    def test_strip_is_idempotent(self):
        messages = [system(), task_block(with_image=True)]
        assert strip_image(messages) is True
        assert strip_image(messages) is False

    def test_strip_without_image(self):
        messages = [system(), task_block()]
        assert strip_image(messages) is False


class TestTrimToBudget:
    def test_noop_when_under_budget(self):
        messages = conversation(3)
        out, stats = trim_to_budget(messages, context_budget=100_000)
        assert len(out) == len(messages)
        assert stats["blocks_dropped"] == 0

    def test_drops_oldest_first_and_keeps_recent(self):
        messages = conversation(10)
        out, stats = trim_to_budget(messages, context_budget=600, preserve_recent_turns=2)
        assert stats["blocks_dropped"] > 0
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"
        assert count_assistant_turns(out) <= 3
        # the most recent turn survived
        assert out[-1]["content"].startswith("result 9")

    def test_protected_prefix_never_dropped(self):
        messages = conversation(6)
        out, _ = trim_to_budget(messages, context_budget=1, preserve_recent_turns=0)
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

    def test_image_is_the_last_resort(self):
        messages = conversation(2, with_image=True)
        prefix_cost = estimate_message_tokens(messages[:2], image_tokens=10_000)
        out, stats = trim_to_budget(
            messages, context_budget=prefix_cost - 9_000, image_tokens=10_000, preserve_recent_turns=2
        )
        assert stats["image_stripped"] == 1
        assert count_assistant_turns(out) == 2

    def test_returns_a_copy(self):
        messages = conversation(3)
        out, _ = trim_to_budget(messages, context_budget=100_000)
        assert out is not messages
        assert len(messages) == 8

    def test_empty(self):
        out, stats = trim_to_budget([], context_budget=100)
        assert out == []
        assert stats["blocks_dropped"] == 0

    def test_converges_even_when_impossible(self):
        # A prefix bigger than the budget must terminate, not spin.
        messages = [system("S" * 30000), task_block("T" * 30000)]
        out, stats = trim_to_budget(messages, context_budget=10)
        assert len(out) == 2
