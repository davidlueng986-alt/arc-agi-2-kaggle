from __future__ import annotations

import json
import os

import pytest

from harness.tasks import (
    ARCTask,
    SPLITS,
    load_challenges,
    resolve_challenges_path,
    resolve_data_dir,
    resolve_output_dir,
    resolve_solutions_path,
    task_ids,
    total_test_outputs,
)


def test_load_real_eval_split_with_solutions(data_dir):
    path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    sols = os.path.join(data_dir, "arc-agi_evaluation_solutions.json")
    tasks = load_challenges(path, sols)
    assert len(tasks) == 120
    assert all(t.solutions is not None for t in tasks)
    assert all(len(t.solutions) == t.n_test for t in tasks)
    assert total_test_outputs(tasks) == 172


def test_load_real_test_split_has_no_solutions(data_dir):
    path = os.path.join(data_dir, "arc-agi_test_challenges.json")
    tasks = load_challenges(path)
    assert len(tasks) == 240
    assert all(t.solutions is None for t in tasks)
    assert total_test_outputs(tasks) == 259


def test_test_split_test_input_distribution(data_dir):
    tasks = load_challenges(os.path.join(data_dir, "arc-agi_test_challenges.json"))
    dist = {}
    for t in tasks:
        dist[t.n_test] = dist.get(t.n_test, 0) + 1
    assert dist == {1: 223, 2: 15, 3: 2}


def test_task_ids_are_sorted_and_unique(data_dir):
    path = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    ids = task_ids(path)
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)


def test_all_grids_are_valid(data_dir):
    tasks = load_challenges(os.path.join(data_dir, "arc-agi_evaluation_challenges.json"))
    for t in tasks:
        assert t.n_test >= 1
        for gi, go in t.train:
            assert 1 <= len(gi) <= 30 and 1 <= len(gi[0]) <= 30
            assert all(0 <= c <= 9 for row in gi for c in row)
            assert all(0 <= c <= 9 for row in go for c in row)
        for gi in t.test:
            assert 1 <= len(gi) <= 30


def test_prompt_size_statistics_on_test_split(data_dir):
    tasks = load_challenges(os.path.join(data_dir, "arc-agi_test_challenges.json"))
    tokens = sorted(t.approx_prompt_tokens() for t in tasks)
    mean = sum(tokens) / len(tokens)
    assert 700 < mean < 1600, f"mean prompt tokens {mean} outside the budget model range"
    assert tokens[-1] < 6000, f"max prompt tokens {tokens[-1]} exceeds max-model-len plan"


def test_prompt_chars_positive(tiny_task):
    assert tiny_task.prompt_chars() > 0
    assert tiny_task.approx_prompt_tokens() > tiny_task.prompt_chars() / 4


def test_rejects_malformed_task(tmp_path):
    path = str(tmp_path / "bad.json")
    json.dump({"x1": {"train": [{"input": [[0]], "output": [[1]]}], "test": []}}, open(path, "w"))
    with pytest.raises(ValueError, match="no test inputs"):
        load_challenges(path)


def test_rejects_out_of_range_cells(tmp_path):
    path = str(tmp_path / "bad2.json")
    json.dump(
        {"x1": {"train": [{"input": [[10]], "output": [[1]]}], "test": [{"input": [[0]]}]}},
        open(path, "w"),
    )
    with pytest.raises(ValueError, match="bad train pair"):
        load_challenges(path)


def test_solutions_length_mismatch_yields_none(tmp_path):
    ch = str(tmp_path / "c.json")
    so = str(tmp_path / "s.json")
    json.dump({"x1": {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[2]]}]}}, open(ch, "w"))
    json.dump({"x1": [[[3]], [[4]]]}, open(so, "w"))
    tasks = load_challenges(ch, so)
    assert tasks[0].solutions is None


def test_resolve_data_dir_finds_shared_data(data_dir):
    assert resolve_data_dir(data_dir) == data_dir
    assert resolve_data_dir(None)


def test_resolve_data_dir_missing_raises(tmp_path, monkeypatch):
    import harness.tasks as tasks_mod

    monkeypatch.setattr(tasks_mod, "DEV_DATA_DIR", str(tmp_path / "nope"))
    monkeypatch.delenv("ARC_DATA_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        resolve_data_dir(str(tmp_path / "also-nope"))


def test_resolve_challenges_path_rejects_bad_split(data_dir):
    with pytest.raises(ValueError):
        resolve_challenges_path(data_dir, "nope")
    for split in SPLITS:
        assert resolve_challenges_path(data_dir, split).endswith(f"arc-agi_{split}_challenges.json")


def test_resolve_solutions_path_absent_returns_none(data_dir):
    assert resolve_solutions_path(data_dir, "test") is None or os.path.exists(
        resolve_solutions_path(data_dir, "test")
    )


def test_resolve_output_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARC_OUTPUT_DIR", str(tmp_path))
    assert resolve_output_dir(None) == str(tmp_path)
    assert resolve_output_dir(str(tmp_path / "explicit")) == str(tmp_path / "explicit")


def test_arc_task_is_frozen(tiny_task):
    with pytest.raises(Exception):
        tiny_task.task_id = "other"


def test_n_test_property(two_test_task):
    assert two_test_task.n_test == 2
    assert isinstance(two_test_task, ARCTask)
