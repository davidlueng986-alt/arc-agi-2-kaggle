"""Candidate pool, 2-attempt selection, schema validation, atomic writes, scoring."""

from __future__ import annotations

import json
import os

import pytest

from harness.grids import Pair, Task
from harness.oracle import normalize_grid
from harness.submission import (
    Candidate,
    CandidatePool,
    build_submission,
    score_submission,
    score_task_entries,
    select_attempts,
    validate_submission,
    write_submission,
)


def make_pool_entry(pool, *, exact, soft, outputs, source="def transform(g):\n    return g\n", turn=0):
    candidate, ack = pool.add(
        source=source,
        exact=exact,
        soft_score=soft,
        pass_rate=1.0 if exact else soft,
        n_pairs=3,
        summary=("VERIFIED 3/3" if exact else f"FAIL soft={soft}"),
        test_outputs=outputs,
        turn_index=turn,
    )
    return candidate, ack


class TestPool:
    def test_add_and_rank(self):
        pool = CandidatePool()
        make_pool_entry(pool, exact=False, soft=0.4, outputs=[[[0]]], turn=0)
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]], turn=1)
        ranked = pool.ranked()
        assert ranked[0].exact is True
        assert ranked[1].exact is False

    def test_verified_first_then_soft_then_simpler_then_earlier(self):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]], source="a\nb\nc\n", turn=5)
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[2]]], source="a\n", turn=9)
        assert len(pool.ranked()[0].source.splitlines()) == 1

    def test_rejects_entries_with_no_test_output(self):
        pool = CandidatePool()
        candidate, ack = pool.add(
            source="x", exact=True, soft_score=1.0, pass_rate=1.0, n_pairs=1,
            summary="", test_outputs=[],
        )
        assert candidate is None
        assert ack["registered"] is False
        assert pool.rejected == 1

    def test_normalizes_outputs(self):
        pool = CandidatePool()
        np = pytest.importorskip("numpy")
        candidate, _ = pool.add(
            source="x", exact=True, soft_score=1.0, pass_rate=1.0, n_pairs=1,
            summary="", test_outputs=[np.array([[1, 2], [3, 4]])],
        )
        assert candidate.test_outputs == [[[1, 2], [3, 4]]]

    def test_drops_invalid_outputs(self):
        pool = CandidatePool()
        candidate, ack = pool.add(
            source="x", exact=True, soft_score=1.0, pass_rate=1.0, n_pairs=1,
            summary="", test_outputs=[[[0, 12]]],
        )
        assert candidate is None
        assert ack["registered"] is False

    def test_pool_is_bounded_and_keeps_the_best(self):
        pool = CandidatePool(max_size=3)
        for i in range(10):
            make_pool_entry(pool, exact=(i == 7), soft=i / 10.0, outputs=[[[i % 10]]], turn=i)
        assert len(pool.candidates) == 3
        assert pool.ranked()[0].exact is True
        # every later entry was strictly better than the running worst, so the
        # pool evicted the worst rather than rejecting the newcomer
        assert pool.rejected == 0
        assert {c.distinct_id for c in pool.candidates} == {8, 9, 10}

    def test_pool_rejects_when_full_of_better_candidates(self):
        pool = CandidatePool(max_size=2)
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]], turn=0)
        make_pool_entry(pool, exact=True, soft=0.9, outputs=[[[2]]], turn=1)
        candidate, ack = make_pool_entry(pool, exact=False, soft=0.1, outputs=[[[3]]], turn=2)
        assert candidate is None
        assert ack["registered"] is False
        assert pool.rejected == 1
        assert len(pool.candidates) == 2

    def test_distinct_verified_outputs(self):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]], turn=0)
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]], turn=1)
        assert pool.distinct_verified_outputs() == 1
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[2]]], turn=2)
        assert pool.distinct_verified_outputs() == 2

    def test_summary_lines(self):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1]]])
        lines = pool.best_summary_lines(limit=2)
        assert lines and "VERIFIED" in lines[0]


class TestAttemptSelection:
    def test_single_verified_candidate_duplicates_for_attempt_2(self, flip_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[8, 8], [7, 7]]])
        entries = select_attempts(pool, flip_task)
        assert len(entries) == 1
        assert entries[0]["attempt_1"] == [[8, 8], [7, 7]]
        assert entries[0]["attempt_2"] == entries[0]["attempt_1"]

    def test_second_distinct_candidate_becomes_attempt_2(self, flip_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[8, 8], [7, 7]]], turn=0)
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[1, 1], [2, 2]]], turn=1)
        entries = select_attempts(pool, flip_task)
        assert entries[0]["attempt_1"] != entries[0]["attempt_2"]

    def test_unverified_candidate_can_be_attempt_2(self, flip_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[8, 8], [7, 7]]], turn=0)
        make_pool_entry(pool, exact=False, soft=0.6, outputs=[[[0, 0], [0, 0]]], turn=1)
        entries = select_attempts(pool, flip_task)
        assert entries[0]["attempt_2"] == [[0, 0], [0, 0]]

    def test_when_verified_attempt_2_follows_likelihood_not_shape(self, flip_task):
        # Land's ~97% verification precision means the leader is probably right,
        # so the runner-up is chosen by soft_score, not by shape diversity.
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[8, 8], [7, 7]]], turn=0)
        make_pool_entry(pool, exact=False, soft=0.9, outputs=[[[8, 7], [8, 7]]], turn=1)
        make_pool_entry(pool, exact=False, soft=0.5, outputs=[[[1, 2, 3]]], turn=2)
        entries = select_attempts(pool, flip_task)
        assert entries[0]["attempt_2"] == [[8, 7], [8, 7]]

    def test_when_unverified_attempt_2_prefers_shape_diversity(self, flip_task):
        # With no verification signal the dominant risk is having the whole
        # transformation family wrong, so cover a different output shape.
        pool = CandidatePool()
        make_pool_entry(pool, exact=False, soft=0.9, outputs=[[[8, 8], [7, 7]]], turn=0)
        make_pool_entry(pool, exact=False, soft=0.8, outputs=[[[8, 7], [8, 7]]], turn=1)
        make_pool_entry(pool, exact=False, soft=0.5, outputs=[[[1, 2, 3]]], turn=2)
        entries = select_attempts(pool, flip_task)
        assert entries[0]["attempt_1"] == [[8, 8], [7, 7]]
        assert entries[0]["attempt_2"] == [[1, 2, 3]]

    def test_shape_diversity_falls_back_when_all_shapes_agree(self, flip_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=False, soft=0.9, outputs=[[[8, 8], [7, 7]]], turn=0)
        make_pool_entry(pool, exact=False, soft=0.4, outputs=[[[1, 1], [2, 2]]], turn=1)
        entries = select_attempts(pool, flip_task)
        assert entries[0]["attempt_2"] == [[1, 1], [2, 2]]

    def test_multiple_test_inputs_are_aligned(self, two_test_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[2, 2], [1, 1]], [[4, 4, 4], [3, 3, 3]]])
        entries = select_attempts(pool, two_test_task)
        assert len(entries) == 2
        assert entries[0]["attempt_1"] == [[2, 2], [1, 1]]
        assert entries[1]["attempt_1"] == [[4, 4, 4], [3, 3, 3]]

    def test_missing_test_output_falls_back_to_input_copy(self, two_test_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[2, 2], [1, 1]]])  # only one of two
        entries = select_attempts(pool, two_test_task)
        assert entries[1]["attempt_1"] == two_test_task.test_inputs[1]

    def test_empty_pool_falls_back_to_input_copy(self, flip_task):
        entries = select_attempts(CandidatePool(), flip_task)
        assert entries[0]["attempt_1"] == flip_task.test_inputs[0]
        assert entries[0]["attempt_2"] == flip_task.test_inputs[0]

    def test_all_entries_are_valid_grids(self, crop_task):
        pool = CandidatePool()
        make_pool_entry(pool, exact=True, soft=1.0, outputs=[[[3, 3, 3]], [[4, 4]]])
        entries = select_attempts(pool, crop_task)
        assert validate_submission({"crop-synthetic": entries}, {"crop-synthetic": 2}) == []


class TestSchema:
    def test_build_fills_every_task(self):
        shapes = {"a": 1, "b": 2}
        submission = build_submission({"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}, shapes)
        assert set(submission) == {"a", "b"}
        assert len(submission["b"]) == 2
        assert submission["b"][0] == {"attempt_1": [], "attempt_2": []}

    def test_build_drops_extra_entries(self):
        submission = build_submission({"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}] * 3}, {"a": 1})
        assert len(submission["a"]) == 1

    def test_validate_accepts_good(self):
        submission = {"a": [{"attempt_1": [[0, 1], [2, 3]], "attempt_2": [[1, 0], [3, 2]]}]}
        assert validate_submission(submission, {"a": 1}) == []

    @pytest.mark.parametrize(
        "submission,shapes,needle",
        [
            ({}, {"a": 1}, "missing task_id"),
            ({"a": []}, {"a": 1}, "expected 1 entries"),
            ({"a": [{}]}, {"a": 1}, "missing attempt_1"),
            ({"a": [{"attempt_1": [[0]]}]}, {"a": 1}, "missing attempt_2"),
            ({"a": [{"attempt_1": [], "attempt_2": [[0]]}]}, {"a": 1}, "empty or not a list"),
            ({"a": [{"attempt_1": [[]], "attempt_2": [[0]]}]}, {"a": 1}, "malformed row"),
            ({"a": [{"attempt_1": [[0, 1], [2]], "attempt_2": [[0]]}]}, {"a": 1}, "ragged"),
            ({"a": [{"attempt_1": [[0, 12]], "attempt_2": [[0]]}]}, {"a": 1}, "not an int 0-9"),
            ({"a": [{"attempt_1": [[0, -1]], "attempt_2": [[0]]}]}, {"a": 1}, "not an int 0-9"),
            ({"a": [{"attempt_1": [["x"]], "attempt_2": [[0]]}]}, {"a": 1}, "not an int 0-9"),
            ({"a": [{"attempt_1": [[True]], "attempt_2": [[0]]}]}, {"a": 1}, "not an int 0-9"),
            ({"a": "nope"}, {"a": 1}, "not a list"),
            ("nope", {"a": 1}, "not a JSON object"),
        ],
    )
    def test_validate_rejects(self, submission, shapes, needle):
        problems = validate_submission(submission, shapes)
        assert problems, f"expected a problem for {submission}"
        assert any(needle in p for p in problems), problems


class TestWriteAndScore:
    def test_write_is_atomic_and_readable(self, tmp_path):
        path = tmp_path / "sub" / "submission.json"
        write_submission(str(path), {"a": [{"attempt_1": [[0]], "attempt_2": [[0]]}]})
        assert path.exists()
        assert json.loads(path.read_text())["a"][0]["attempt_1"] == [[0]]
        assert not list(path.parent.glob("*.tmp.*"))

    def test_write_overwrites(self, tmp_path):
        path = tmp_path / "submission.json"
        write_submission(str(path), {"a": []})
        write_submission(str(path), {"b": []})
        assert set(json.loads(path.read_text())) == {"b"}

    def test_score_exact_match_only(self):
        solutions = {"a": [[[1, 2], [3, 4]]]}
        good = {"a": [{"attempt_1": [[1, 2], [3, 4]], "attempt_2": [[0]]}]}
        assert score_submission(good, solutions)["score"] == 1.0
        near = {"a": [{"attempt_1": [[1, 2], [3, 5]], "attempt_2": [[0]]}]}
        assert score_submission(near, solutions)["score"] == 0.0

    def test_score_second_attempt_counts(self):
        solutions = {"a": [[[1]]]}
        submission = {"a": [{"attempt_1": [[0]], "attempt_2": [[1]]}]}
        assert score_submission(submission, solutions)["score"] == 1.0

    def test_score_averages_over_outputs_not_tasks(self):
        solutions = {"a": [[[1]], [[2]]]}
        submission = {"a": [{"attempt_1": [[1]], "attempt_2": [[1]]}, {"attempt_1": [[9]], "attempt_2": [[9]]}]}
        result = score_submission(submission, solutions)
        assert result["n_outputs"] == 2
        assert result["n_correct"] == 1
        assert result["score"] == 0.5
        assert result["per_task"]["a"] == 0.5

    def test_score_handles_missing_task(self):
        assert score_submission({}, {"a": [[[1]]]})["score"] == 0.0

    def test_score_task_entries(self):
        assert score_task_entries([{"attempt_1": [[1]], "attempt_2": [[0]]}], [[[1]]]) == 1.0
        assert score_task_entries([], [[[1]]]) == 0.0
        assert score_task_entries([{"attempt_1": [[1]], "attempt_2": [[0]]}], []) == 0.0
