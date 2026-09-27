"""Data loading, Kaggle path detection, challenge-file selection."""

from __future__ import annotations

import json
import os

import pytest

from harness import dataio
from harness.dataio import (
    KAGGLE_COMPETITION_DIR,
    default_submission_path,
    find_data_dir,
    is_competition_rerun,
    iter_tasks,
    load_challenges,
    load_sample_submission,
    load_solutions,
    on_kaggle,
    pick_challenge_file,
    task_shapes,
)


def test_competition_dir_constant():
    assert KAGGLE_COMPETITION_DIR == "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"


def test_rerun_flag(monkeypatch):
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    assert is_competition_rerun() is False
    for value in ("1", "true", "TRUE", " true "):
        monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", value)
        assert is_competition_rerun() is True
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "0")
    assert is_competition_rerun() is False


def test_find_data_dir_explicit(real_data_dir):
    assert find_data_dir(real_data_dir) == real_data_dir


def test_find_data_dir_via_env(monkeypatch, real_data_dir):
    monkeypatch.setenv("ARC_DATA_DIR", real_data_dir)
    assert find_data_dir(None) == real_data_dir


def test_find_data_dir_rejects_a_dir_without_challenges(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_data_dir(str(tmp_path))


def test_load_challenges(real_data_dir):
    challenges = load_challenges(real_data_dir, "arc-agi_evaluation_challenges.json")
    assert len(challenges) == 120
    first = next(iter(challenges.values()))
    assert "train" in first and "test" in first


def test_load_challenges_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_challenges(str(tmp_path), "nope.json")


def test_load_solutions(real_data_dir):
    solutions = load_solutions(real_data_dir, "arc-agi_evaluation_challenges.json")
    assert solutions is not None
    assert len(solutions) == 120
    for task_id, grids in solutions.items():
        assert grids
        for grid in grids:
            assert grid and all(0 <= cell <= 9 for row in grid for cell in row)


def test_load_solutions_absent(tmp_path):
    assert load_solutions(str(tmp_path), "arc-agi_evaluation_challenges.json") is None
    assert load_solutions(str(tmp_path), "unknown.json") is None


def test_solutions_align_with_challenges(real_data_dir):
    challenges = load_challenges(real_data_dir, "arc-agi_evaluation_challenges.json")
    solutions = load_solutions(real_data_dir, "arc-agi_evaluation_challenges.json")
    assert set(challenges) == set(solutions)
    for task_id, blob in challenges.items():
        assert len(solutions[task_id]) == len(blob["test"])


def test_pick_challenge_file_prefers_test_at_rerun(monkeypatch, real_data_dir):
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "1")
    chosen = pick_challenge_file(real_data_dir)
    assert chosen in ("arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json")
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN")
    assert pick_challenge_file(real_data_dir) == "arc-agi_evaluation_challenges.json"
    assert pick_challenge_file(real_data_dir, prefer_test=True) == "arc-agi_test_challenges.json"


def test_pick_challenge_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        pick_challenge_file(str(tmp_path))


def test_iter_tasks_limit_and_filter(eval_challenges):
    assert len(iter_tasks(eval_challenges, limit=5)) == 5
    ids = list(eval_challenges)[:3]
    tasks = iter_tasks(eval_challenges, task_ids=ids)
    assert [t.task_id for t in tasks] == ids


def test_iter_tasks_skips_invalid():
    blob = {
        "good": {"train": [{"input": [[0]], "output": [[0]]}], "test": [{"input": [[1]]}]},
        "bad": {"train": [], "test": []},
    }
    tasks = iter_tasks(blob, skip_invalid=True)
    assert [t.task_id for t in tasks] == ["good"]
    with pytest.raises(ValueError):
        iter_tasks(blob, skip_invalid=False)


def test_task_shapes(eval_challenges):
    shapes = task_shapes(eval_challenges)
    assert len(shapes) == 120
    assert sum(shapes.values()) == 172  # eval split has 172 scored test outputs
    assert set(shapes.values()) == {1, 2, 3}


def test_sample_submission_matches_the_test_file(real_data_dir):
    sample = load_sample_submission(real_data_dir)
    challenges = load_challenges(real_data_dir, "arc-agi_test_challenges.json")
    assert sample is not None
    assert set(sample) == set(challenges)
    for task_id, entries in sample.items():
        assert len(entries) == len(challenges[task_id]["test"])
        assert set(entries[0]) == {"attempt_1", "attempt_2"}


def test_default_submission_path_local(monkeypatch):
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    path = default_submission_path()
    assert path.endswith("submission.json")


def test_on_kaggle_is_false_locally():
    assert on_kaggle() is False or os.path.isdir("/kaggle/input")
