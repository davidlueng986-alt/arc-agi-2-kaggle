import json
import os

from harness.submission import (
    SubmissionWriter,
    fallback_attempts,
    validate_submission,
)

CHALLENGES = {
    "task_a": {"train": [{"input": [[0]], "output": [[1]]}], "test": [{"input": [[2]]}]},
    "task_b": {
        "train": [{"input": [[0, 0]], "output": [[1, 1]]}],
        "test": [{"input": [[3, 3]]}, {"input": [[4, 4]]}],
    },
}


def _good():
    return {
        "task_a": [{"attempt_1": [[5]], "attempt_2": [[6]]}],
        "task_b": [
            {"attempt_1": [[1, 2]], "attempt_2": [[3, 4]]},
            {"attempt_1": [[5, 6]], "attempt_2": [[7, 8]]},
        ],
    }


class TestValidation:
    def test_valid(self):
        assert validate_submission(_good(), CHALLENGES) == []

    def test_missing_task(self):
        sub = _good()
        del sub["task_b"]
        errs = validate_submission(sub, CHALLENGES)
        assert any("task_b" in e and "missing" in e for e in errs)

    def test_wrong_entry_count(self):
        sub = _good()
        sub["task_b"] = sub["task_b"][:1]
        errs = validate_submission(sub, CHALLENGES)
        assert any("expected 2 entries" in e for e in errs)

    def test_missing_attempt(self):
        sub = _good()
        del sub["task_a"][0]["attempt_2"]
        errs = validate_submission(sub, CHALLENGES)
        assert any("attempt_2" in e for e in errs)

    def test_bad_grid_types(self):
        sub = _good()
        sub["task_a"][0]["attempt_1"] = [["x"]]
        assert validate_submission(sub, CHALLENGES)

    def test_ragged_grid(self):
        sub = _good()
        sub["task_a"][0]["attempt_1"] = [[1, 2], [3]]
        assert validate_submission(sub, CHALLENGES)

    def test_empty_grid_rejected(self):
        sub = _good()
        sub["task_a"][0]["attempt_1"] = []
        assert validate_submission(sub, CHALLENGES)

    def test_bool_rejected(self):
        sub = _good()
        sub["task_a"][0]["attempt_1"] = [[True]]
        assert validate_submission(sub, CHALLENGES)

    def test_extra_task_rejected(self):
        sub = _good()
        sub["ghost"] = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
        errs = validate_submission(sub, CHALLENGES)
        assert any("ghost" in e for e in errs)


class TestFallbacks:
    def test_fallback_attempts_schema(self):
        fb = fallback_attempts([[[2, 2]], [[3, 3]]], [[[1, 1]]])
        assert len(fb) == 2
        for entry in fb:
            assert entry["attempt_1"] and entry["attempt_2"]
            assert entry["attempt_1"] != entry["attempt_2"]
        assert fb[0]["attempt_1"] == [[2, 2]]  # identity

    def test_fallback_all_zero_train(self):
        fb = fallback_attempts([[[0]]], [[[0, 0]]])
        assert fb[0]["attempt_1"] != fb[0]["attempt_2"]


class TestWriter:
    def test_cumulative_writes_and_finalize(self, tmp_path):
        path = str(tmp_path / "submission.json")
        w = SubmissionWriter(path, CHALLENGES)
        w.update("task_a", [{"attempt_1": [[5]], "attempt_2": [[6]]}])
        # after each update the file exists, parses, and holds what we wrote
        with open(path, encoding="utf-8") as f:
            mid = json.load(f)
        assert set(mid) == {"task_a"}

        res = w.finalize()
        assert res["errors"] == []
        with open(path, encoding="utf-8") as f:
            full = json.load(f)
        assert set(full) == {"task_a", "task_b"}
        assert len(full["task_b"]) == 2
        for entry in full["task_b"]:
            assert entry["attempt_1"] and entry["attempt_2"]

    def test_finalize_custom_solver(self, tmp_path):
        path = str(tmp_path / "sub.json")
        w = SubmissionWriter(path, CHALLENGES)

        def solver(task_id):
            if task_id == "task_a":
                return [{"attempt_1": [[9]], "attempt_2": [[8]]}]
            return None

        res = w.finalize(solver_fn=solver)
        assert res["errors"] == []
        assert w.submission["task_a"][0]["attempt_1"] == [[9]]
        assert w.submission["task_b"]  # fallback-filled

    def test_atomic_replace_no_tmp_left(self, tmp_path):
        path = str(tmp_path / "sub.json")
        w = SubmissionWriter(path, CHALLENGES)
        w.update("task_a", [{"attempt_1": [[1]], "attempt_2": [[2]]}])
        w.finalize()
        leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
        assert leftovers == []
