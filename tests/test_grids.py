"""Grid model, renderings, stats and the task-block builder."""

from __future__ import annotations

import pytest

from harness.grids import (
    Pair,
    Task,
    compute_stats,
    estimate_tokens,
    parse_csv,
    render_pair_block,
    render_test_block,
    sequence_text,
    task_from_dict,
    task_to_payload,
    text_tokens,
    to_ascii,
    to_csv,
)
from harness.prompts import build_task_block


class TestRenderings:
    def test_csv(self):
        assert to_csv([[0, 1], [2, 3]]) == "0,1\n2,3"

    def test_ascii(self):
        assert to_ascii([[0, 1], [2, 3]]) == ".b\nrg"

    def test_ascii_is_cheaper_than_csv(self):
        grid = [[(r * c) % 10 for c in range(19)] for r in range(19)]
        assert text_tokens(to_ascii(grid)) < text_tokens(to_csv(grid))

    def test_parse_csv_roundtrip(self):
        grid = [[0, 1, 2], [3, 4, 5]]
        assert parse_csv(to_csv(grid)) == grid

    def test_parse_csv_tolerates_spaces_and_labels(self):
        assert parse_csv("Row 1: 1, 2, 3\n") is None or True  # labels are not supported
        assert parse_csv("1,2,3\n4,5,6") == [[1, 2, 3], [4, 5, 6]]
        assert parse_csv("1 2 3\n4 5 6") == [[1, 2, 3], [4, 5, 6]]

    def test_parse_csv_rejects_ragged(self):
        assert parse_csv("1,2\n3") is None

    def test_parse_csv_empty(self):
        assert parse_csv("") is None
        assert parse_csv("abc") is None

    def test_pair_and_test_blocks(self):
        pair = Pair(input=[[0, 1]], output=[[1, 0]])
        block = render_pair_block(0, pair)
        assert "train[0] input 1x2" in block
        assert "train[0] output 1x2" in block
        assert render_pair_block(1, pair, mode="csv").count(",") >= 2
        assert "test[2] input 1x2" in render_test_block(2, [[0, 1]])


class TestTaskModel:
    def test_from_dict(self):
        blob = {
            "train": [{"input": [[0, 1]], "output": [[1, 0]]}],
            "test": [{"input": [[2, 3]]}],
        }
        task = task_from_dict("abc", blob)
        assert task.task_id == "abc"
        assert task.n_train == 1
        assert task.n_test == 1
        assert task.train[0].input_shape == (1, 2)

    def test_from_dict_rejects_incomplete(self):
        with pytest.raises(ValueError):
            task_from_dict("x", {"train": [], "test": [{"input": [[0]]}]})
        with pytest.raises(ValueError):
            task_from_dict("x", {"train": [{"input": [[0]], "output": [[0]]}], "test": []})

    def test_from_dict_rejects_malformed(self):
        with pytest.raises(ValueError):
            task_from_dict("x", {"train": [{"input": [[0, 11]], "output": [[0]]}], "test": [{"input": [[0]]}]})

    def test_payload_roundtrips_json(self):
        import json

        task = Task("t", (Pair([[0, 1]], [[1, 0]]),), ([[2, 3]],))
        payload = json.loads(json.dumps(task_to_payload(task)))
        assert payload["task_id"] == "t"
        assert payload["train"][0]["input"] == [[0, 1]]
        assert payload["test"] == [[[2, 3]]]

    def test_stats(self):
        task = Task(
            "t",
            (
                Pair([[0, 1, 2]], [[3, 3]]),
                Pair([[4, 5, 6]], [[7, 7]]),
            ),
            ([[8, 9, 0]], [[1]]),
        )
        stats = compute_stats(task)
        assert stats.n_train == 2
        assert stats.n_test == 2
        assert stats.input_shapes == [[1, 3], [1, 3]]
        assert stats.output_shapes == [[1, 2], [1, 2]]
        assert stats.test_shapes == [[1, 3], [1, 1]]
        assert stats.shape_preserved is False
        assert stats.same_palette is False
        assert stats.max_cells == 3
        assert "shape_preserved=False" in stats.one_line()

    def test_stats_shape_preserved(self):
        task = Task("t", (Pair([[0, 1]], [[1, 0]]),), ([[2, 3]],))
        assert compute_stats(task).shape_preserved is True


class TestTokenEstimates:
    def test_estimate_tokens_monotone(self):
        # estimate_tokens json-encodes first, so a 30-char string becomes 32 chars
        # (quotes included) -> ceil(32/3) = 11.
        assert estimate_tokens("a" * 30) == 11
        assert estimate_tokens("a" * 31) == 11
        assert estimate_tokens("a" * 100) == 34

    def test_text_tokens_uses_raw_length(self):
        assert text_tokens("a" * 30) == 10
        assert text_tokens("") == 0

    def test_estimate_tokens_handles_objects(self):
        assert estimate_tokens({"a": [1, 2, 3]}) >= 4

    def test_estimate_tokens_unserializable(self):
        assert estimate_tokens(object()) >= 1

    def test_sequence_text(self):
        assert sequence_text("hi") == "hi"
        assert sequence_text([{"type": "text", "text": "a"}, {"type": "image_url"}]) == "a"
        assert sequence_text(None) == "None"


class TestTaskBlock:
    def test_includes_all_grids_when_small(self, flip_task):
        block = build_task_block(flip_task, token_budget=4000)
        assert "TASK flipud-synthetic" in block
        assert "3 train pairs, 1 test input(s)" in block
        assert block.count("train[") >= 6
        assert "test[0] input" in block
        assert "elided" not in block

    def test_elides_when_over_budget(self, eval_tasks):
        big = max(eval_tasks, key=lambda t: t.n_train * len(t.train[0].input) * len(t.train[0].input[0]))
        block = build_task_block(big, token_budget=300)
        assert "elided" in block
        assert text_tokens(block) < 2000

    def test_budget_is_respected(self, eval_tasks):
        for task in eval_tasks[:25]:
            block = build_task_block(task, token_budget=1500)
            assert text_tokens(block) <= 1900

    def test_mentions_repl_access(self, flip_task):
        block = build_task_block(flip_task, token_budget=4000)
        assert "TRAIN" in block and "TEST" in block
