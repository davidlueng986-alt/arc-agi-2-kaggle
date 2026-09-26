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


def test_schema_matches_sample_submission_shape(data_dir, test_tasks):
    sample_path = os.path.join(data_dir, "sample_submission.json")
    if not os.path.exists(sample_path):
        pytest.skip("sample_submission.json unavailable")
    sample = json.load(open(sample_path, encoding="utf-8"))
    writer = SubmissionWriter(os.devnull, test_tasks)
    assert set(writer.data) == set(sample)
    for task_id, entries in sample.items():
        assert len(writer.data[task_id]) == len(entries)
        assert set(writer.data[task_id][0]) == set(entries[0])
