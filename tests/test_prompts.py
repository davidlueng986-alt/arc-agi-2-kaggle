"""Prompt surface: the encoded knowledge and the tool contract."""

from __future__ import annotations

import re

import pytest

from harness.grids import compute_stats, text_tokens
from harness.prompts import (
    ARC_KNOWLEDGE_ADDENDUM,
    DIFF_READING_ADDENDUM,
    FINAL_ANSWER_INSTRUCTION,
    ORACLE_ADDENDUM,
    PYTHON_TOOL_SPEC,
    TOOLS,
    build_status_message,
    build_system_prompt,
    build_task_block,
)


class TestSystemPrompt:
    def test_contains_every_addendum(self):
        prompt = build_system_prompt()
        for fragment in (
            "coding agent solving ARC-AGI-2",
            "verify(fn)",
            "finish(reason)",
            "TRAIN",
            "TEST",
            "segmentation",
            "kind: shape",
            "expected_crop",
            "aug",
            "numpy",
            "30 seconds" if False else "30s",
        ):
            assert fragment in prompt, fragment

    def test_encodes_arc_domain_knowledge(self):
        prompt = build_system_prompt()
        for family in ("crop", "symmetry", "gravity", "tile", "bounding box", "colour remapping"):
            assert family.lower() in prompt.lower(), family

    def test_encodes_the_anti_overfitting_rule(self):
        prompt = build_system_prompt()
        assert "Never key on absolute coordinates" in prompt
        assert "RULE, not a lookup" in prompt

    def test_encodes_the_compact_output_rule(self):
        prompt = build_system_prompt()
        assert "NEVER print or echo a full grid" in prompt

    def test_encodes_state_reset(self):
        assert "FRESH process" in build_system_prompt()

    def test_limits_are_interpolated(self):
        prompt = build_system_prompt(tool_timeout_s=7, tool_output_chars=99)
        assert "7s" in prompt
        assert "99 characters" in prompt

    def test_size_is_bounded(self):
        # ~2.9K tokens by chars/3. Deliberately not tiny: the Duck's authors note
        # they "had to encode a lot of specific knowledge into the prompts", and
        # this prefix is cacheable. The guard is against accidental growth.
        assert text_tokens(build_system_prompt()) < 3600

    def test_tool_guidance_can_be_omitted(self):
        assert len(build_system_prompt(with_tool_guidance=True)) > len(
            build_system_prompt(with_tool_guidance=False)
        )


class TestToolSpec:
    def test_single_tool_named_python(self):
        assert len(TOOLS) == 1
        assert TOOLS[0]["function"]["name"] == "python"

    def test_single_required_string_parameter(self):
        parameters = PYTHON_TOOL_SPEC["function"]["parameters"]
        assert list(parameters["properties"]) == ["code"]
        assert parameters["required"] == ["code"]
        assert parameters["properties"]["code"]["type"] == "string"

    def test_description_documents_the_oracle(self):
        description = PYTHON_TOOL_SPEC["function"]["description"]
        assert "verify(fn)" in description
        assert "exact oracle" in description
        assert "does NOT persist" in description


class TestStatusMessage:
    def test_minimal(self):
        text = build_status_message(
            turn_index=0, max_turns=10, ledger_lines=[], tokens_used=5, token_budget=100
        )
        assert "turn 1/10" in text
        assert "Call `python` now" in text

    def test_ledger_and_headline_and_nudge(self):
        text = build_status_message(
            turn_index=3,
            max_turns=10,
            ledger_lines=["best: VERIFIED", "tried: identity"],
            tokens_used=5,
            token_budget=100,
            seconds_left=42.0,
            headline="A candidate registered.",
            nudge="verify it",
        )
        assert "turn 4/10" in text
        assert "42s left" in text
        assert "- best: VERIFIED" in text
        assert "A candidate registered." in text
        assert "verify it" in text

    def test_final_instruction_is_urgent_and_bounded(self):
        assert "TURN BUDGET REACHED" in FINAL_ANSWER_INSTRUCTION
        assert "finish(" in FINAL_ANSWER_INSTRUCTION
        assert text_tokens(FINAL_ANSWER_INSTRUCTION) < 120


class TestTaskBlock:
    def test_structural_summary_first(self, flip_task):
        block = build_task_block(flip_task, compute_stats(flip_task))
        first_lines = block.splitlines()[:3]
        assert any("TASK flipud-synthetic" in line for line in first_lines)
        assert any("3 train pairs" in line for line in first_lines)

    def test_no_csv_by_default(self, flip_task):
        # ASCII is ~3x cheaper than CSV for the same grid and the model never has
        # to emit a grid, so Land's "CSV is safest to generate" does not bind.
        block = build_task_block(flip_task, token_budget=4000)
        # COLOR_CHARS = ".brgysmocd": 1->b 2->r 3->g 4->y
        assert "br\ngy" in block  # train[0].input as colour symbols
        assert "gy\nbr" in block  # train[0].output
        for line in block.splitlines():
            assert not re.fullmatch(r"[0-9](,[0-9])+\s*", line), line

    def test_stats_can_be_omitted(self, flip_task):
        assert build_task_block(flip_task).startswith("TASK")
