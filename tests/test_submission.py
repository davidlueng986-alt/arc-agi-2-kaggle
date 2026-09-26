from __future__ import annotations

import json
import os

import pytest

from harness.submission import (
    PLACEHOLDER,
    SubmissionWriter,
    build_entry,
    empty_submission,
    validate_submission_obj,
)
from harness.tasks import ARCTask


TRAIN = (([[0]], [[1]]),)


def make_tasks():
    return [
        ARCTask("aaa", TRAIN, ([[2]],)),
        ARCTask("bbb", TRAIN, ([[2]], [[3]])),
        ARCTask("ccc", TRAIN, ([[4]],)),
    ]


@pytest.fixture
def tasks():
    return make_tasks()


def test_empty_submission_covers_every_task_and_test_index(tasks):
    sub = empty_submission(tasks)
    assert set(sub) == {"aaa", "bbb", "ccc"}
    assert len(sub["bbb"]) == 2
    assert set(sub["bbb"][0]) == {"attempt_1", "attempt_2"}
    assert sub["aaa"][0]["attempt_1"] == PLACEHOLDER


def test_build_entry_duplicates_when_only_one_grid():
    entry = build_entry([[[1, 2]]])
    assert entry["attempt_1"] == [[1, 2]]
    assert entry["attempt_2"] == [[1, 2]]


def test_build_entry_placeholder_when_empty():
    entry = build_entry([])
    assert entry == {"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER}


def test_writer_starts_schema_valid(tmp_path, tasks):
    path = str(tmp_path / "submission.json")
    writer = SubmissionWriter(path, tasks)
    assert writer.validate() == []
    writer.write()
    assert os.path.exists(path)
    loaded = json.load(open(path, encoding="utf-8"))
    assert set(loaded) == {"aaa", "bbb", "ccc"}


def test_update_replaces_both_attempts(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.update("bbb", [{"attempt_1": [[1]], "attempt_2": [[2]]}, {"attempt_1": [[3]], "attempt_2": [[4]]}])
    assert writer.data["bbb"][1]["attempt_2"] == [[4]]
    assert writer.validate() == []


def test_update_rejects_unknown_task(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    with pytest.raises(KeyError):
        writer.update("zzz", [{"attempt_1": [[1]], "attempt_2": [[1]]}])


def test_update_rejects_wrong_prediction_count(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    with pytest.raises(ValueError):
        writer.update("bbb", [{"attempt_1": [[1]], "attempt_2": [[1]]}])


def test_update_fills_missing_attempt_from_first(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.update("aaa", [{"attempt_1": [[7]]}])
    assert writer.data["aaa"][0]["attempt_2"] == [[7]]


def test_validate_catches_bad_cells(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.data["aaa"][0]["attempt_1"] = [[12]]
    assert any("outside 0..9" in e for e in writer.validate())


def test_validate_catches_non_rectangular(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.data["aaa"][0]["attempt_1"] = [[1, 2], [3]]
    assert any("non-rectangular" in e for e in writer.validate())


def test_validate_catches_empty_grid(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.data["aaa"][0]["attempt_1"] = []
    assert any("empty" in e for e in writer.validate())


def test_validate_catches_wrong_entry_count(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    writer.data["bbb"] = writer.data["bbb"][:1]
    assert any("expected 2" in e for e in writer.validate())


def test_validate_catches_missing_task(tmp_path, tasks):
    writer = SubmissionWriter(str(tmp_path / "s.json"), tasks)
    del writer.data["ccc"]
    assert any("missing" in e for e in writer.validate())


def test_write_is_atomic_and_replaces(tmp_path, tasks):
    path = str(tmp_path / "s.json")
    writer = SubmissionWriter(path, tasks)
    writer.write()
    first = json.load(open(path, encoding="utf-8"))
    writer.update("aaa", [{"attempt_1": [[9]], "attempt_2": [[9]]}])
    writer.write()
    second = json.load(open(path, encoding="utf-8"))
    assert writer.writes == 2
    assert first["aaa"][0]["attempt_1"] == PLACEHOLDER
    assert second["aaa"][0]["attempt_1"] == [[9]]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_writer_creates_parent_dirs(tmp_path, tasks):
    path = str(tmp_path / "nested" / "deep" / "s.json")
    writer = SubmissionWriter(path, tasks)
    writer.write()
    assert os.path.exists(path)


def test_load_roundtrip(tmp_path, tasks):
    path = str(tmp_path / "s.json")
    writer = SubmissionWriter(path, tasks)
    writer.update("ccc", [{"attempt_1": [[5, 5]], "attempt_2": [[6, 6]]}])
    writer.write()
    loaded = SubmissionWriter.load(path)
    assert loaded["ccc"][0]["attempt_2"] == [[6, 6]]


def test_validate_submission_obj_helper(tasks):
    sub = empty_submission(tasks)
    assert validate_submission_obj(sub, tasks) == []
    sub["aaa"] = []
    assert validate_submission_obj(sub, tasks)


def test_schema_matches_sample_submission_shape(tmp_path, data_dir, test_tasks):
    sample_path = os.path.join(data_dir, "sample_submission.json")
    if not os.path.exists(sample_path):
        pytest.skip("sample_submission.json unavailable")
    sample = json.load(open(sample_path, encoding="utf-8"))
    writer = SubmissionWriter(str(tmp_path / "submission.json"), test_tasks)
    assert set(writer.data) == set(sample)
    for task_id, entries in sample.items():
        assert len(writer.data[task_id]) == len(entries)
        assert set(writer.data[task_id][0]) == set(entries[0])


def test_concurrent_watchdog_and_scheduler_writes_do_not_race(tmp_path):
    """Regression: the watchdog thread and the scheduler both called write().

    Concurrent `os.replace` raised PermissionError on Windows, and concurrent
    `json.dump` over a dict being mutated raises on Linux. Both would have fired
    at the generation deadline and cost the entire run.
    """
    import threading

    tasks = make_tasks()
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)
    errors: list = []

    def hammer_updates():
        try:
            for i in range(300):
                writer.update("aaa", [{"attempt_1": [[i % 10]], "attempt_2": [[i % 10]]}])
                writer.update("bbb", [{"attempt_1": [[1]], "attempt_2": [[2]]},
                                      {"attempt_1": [[3]], "attempt_2": [[4]]}])
        except Exception as exc:
            errors.append(("update", repr(exc)))

    def hammer_writes():
        try:
            for _ in range(200):
                writer.write()
        except Exception as exc:
            errors.append(("write", repr(exc)))

    threads = [threading.Thread(target=hammer_updates)]
    threads += [threading.Thread(target=hammer_writes) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)

    assert not errors, f"concurrent submission writes raised: {errors[:3]}"
    assert not any(t.is_alive() for t in threads), "a writer thread deadlocked"
    assert writer.writes == 600
    assert writer.validate() == []
    assert json.load(open(writer.path, encoding="utf-8"))["aaa"][0]["attempt_1"] == [[299 % 10]]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_write_retries_through_a_transient_permission_error(tmp_path, monkeypatch):
    tasks = make_tasks()
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)
    real_replace = os.replace
    state = {"calls": 0}

    def flaky_replace(src, dst):
        state["calls"] += 1
        if state["calls"] <= 2:
            raise PermissionError("[WinError 5] Access is denied")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    monkeypatch.setattr("harness.submission.WRITE_RETRY_DELAY", 0.0)
    writer.write()
    assert state["calls"] == 3
    assert writer.writes == 1
    assert json.load(open(writer.path, encoding="utf-8"))["aaa"][0]["attempt_1"] == PLACEHOLDER
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_write_raises_after_exhausting_retries(tmp_path, monkeypatch):
    tasks = make_tasks()
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)

    def always_denied(src, dst):
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(os, "replace", always_denied)
    monkeypatch.setattr("harness.submission.WRITE_RETRY_DELAY", 0.0)
    with pytest.raises(PermissionError):
        writer.write()
    assert writer.writes == 0
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_write_propagates_non_permission_errors_without_retry(tmp_path, monkeypatch):
    tasks = make_tasks()
    writer = SubmissionWriter(str(tmp_path / "submission.json"), tasks)
    calls = {"n": 0}

    def boom(src, dst):
        calls["n"] += 1
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        writer.write()
    assert calls["n"] == 1
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
