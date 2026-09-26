import json
import time
from pathlib import Path

import numpy as np
import pytest

from harness.budget import Budget, Watchdog
from harness.grid import to_grid
from harness.submission import (
    SubmissionWriter,
    fallback_attempts,
    score_submission,
    validate_submission,
)
from harness.types import Task


def _task(tid="t1", n_test=1, shape=(2, 2)):
    x = to_grid([[1, 2], [3, 4]])[: shape[0], : shape[1]]
    y = x.copy()
    tests = [np.zeros(shape, dtype=np.int64) for _ in range(n_test)]
    return Task(tid, [(x, y)] * 2, tests)


class TestBudget:
    def test_charge_and_caps(self):
        b = Budget(deadline_ts=time.time() + 3600, per_task_tokens=1000, total_tokens=5000)
        assert b.can_call("t1", 500)
        b.charge("t1", 400, 200)
        assert b.task_spent("t1") == 600
        assert not b.can_call("t1", 500)  # exceeds per-task cap
        assert b.can_call("t2", 500)
        b.charge("t2", 2000, 2500)
        assert not b.can_call("t3", 1)  # total cap exceeded

    def test_deadline_expiry(self):
        b = Budget(deadline_ts=time.time() + 10, reserve_seconds=20)
        assert b.expired()  # reserve pushes effective deadline into the past
        assert not b.can_call("t", 0)

    def test_snapshot_and_task_done(self):
        b = Budget(deadline_ts=time.time() + 600)
        b.charge("t", 10, 5)
        b.mark_task_done()
        snap = b.snapshot()
        assert snap["prompt_tokens"] == 10 and snap["completion_tokens"] == 5
        assert snap["tasks_done"] == 1 and snap["llm_calls"] == 1

    def test_per_task_time_cap_floored(self):
        b = Budget(deadline_ts=time.time() + 600)
        assert b.per_task_time_cap(1000) >= 60.0


class TestWatchdog:
    def test_fires_callback(self):
        hits = []
        w = Watchdog(time.time() + 0.2, lambda: hits.append(1), hard_kill_grace_s=60)
        w.start()
        time.sleep(0.6)
        w.cancel()
        assert hits == [1]
        assert w.fired

    def test_cancel_prevents(self):
        hits = []
        w = Watchdog(time.time() + 0.3, lambda: hits.append(1))
        w.start()
        w.cancel()
        time.sleep(0.5)
        assert hits == []


class TestSubmissionWriter:
    def test_initial_file_has_all_tasks_and_schema(self, tmp_path):
        tasks = [_task("a", 1), _task("b", 2)]
        w = SubmissionWriter(tmp_path / "submission.json", tasks)
        data = json.loads((tmp_path / "submission.json").read_text())
        assert validate_submission(data, tasks) == []
        assert len(data["b"]) == 2

    def test_update_persists_atomically(self, tmp_path):
        tasks = [_task("a", 1)]
        w = SubmissionWriter(tmp_path / "submission.json", tasks)
        g = to_grid([[5, 6], [7, 8]])
        w.update("a", [(g, g)])
        data = json.loads((tmp_path / "submission.json").read_text())
        assert data["a"][0]["attempt_1"] == [[5, 6], [7, 8]]
        assert not (tmp_path / "submission.tmp").exists()

    def test_update_rejects_unknown_task(self, tmp_path):
        w = SubmissionWriter(tmp_path / "s.json", [_task("a", 1)])
        with pytest.raises(KeyError):
            w.update("nope", [(None, None)])

    def test_none_attempts_become_zero_grids(self, tmp_path):
        tasks = [_task("a", 1, shape=(3, 3))]
        w = SubmissionWriter(tmp_path / "s.json", tasks)
        w.update("a", [(None, None)])
        data = w.data()
        assert validate_submission(data, tasks) == []
        assert np.array(data["a"][0]["attempt_1"]).shape == (3, 3)

    def test_fallback_attempts_shapes(self):
        x = to_grid([[1, 1], [1, 0]])
        packs = fallback_attempts(x)
        assert packs[0]["attempt_1"] == x.tolist()
        assert packs[0]["attempt_2"] == [[1, 1], [1, 1]]  # background fill


class TestValidationAndScoring:
    def test_validate_catches_missing_and_ragged(self):
        tasks = [_task("a", 1), _task("b", 2)]
        bad = {"a": [{"attempt_1": [[1]], "attempt_2": [[1, 2], [3]]}]}  # b missing, ragged a2
        problems = validate_submission(bad, tasks)
        assert any("b" in p for p in problems)
        assert any("attempt_2" in p for p in problems)

    def test_validate_catches_bad_values(self):
        tasks = [_task("a", 1)]
        bad = {"a": [{"attempt_1": [[10, 0]], "attempt_2": [[0, 0]]}]}
        assert validate_submission(bad, tasks)

    def test_score_pass_at_2(self):
        t = Task("a", [], [np.zeros((2, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64)])
        truth = [[[1, 1], [1, 1]], [[2, 2], [2, 2]]]
        data = {
            "a": [
                {"attempt_1": [[9, 9], [9, 9]], "attempt_2": [[1, 1], [1, 1]]},  # hit via attempt_2
                {"attempt_1": [[3, 3], [3, 3]], "attempt_2": [[4, 4], [4, 4]]},  # miss
            ]
        }
        report = score_submission(data, [t], {"a": truth})
        assert report["outputs_total"] == 2
        assert report["outputs_hit"] == 1
        assert report["score"] == 0.5
