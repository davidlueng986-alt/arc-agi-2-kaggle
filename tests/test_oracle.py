import asyncio

from harness.oracle import array_diff, build_feedback, evaluate_code, soft_score
from harness.types import RunResult


class TestSoftScore:
    def test_exact(self):
        assert soft_score([[1, 2], [3, 4]], [[1, 2], [3, 4]]) == 1.0

    def test_partial(self):
        assert soft_score([[1, 0], [3, 4]], [[1, 2], [3, 4]]) == 0.75

    def test_shape_mismatch_zero(self):
        assert soft_score([[1, 2, 3]], [[1, 2]]) == 0.0

    def test_none_zero(self):
        assert soft_score(None, [[1]]) == 0.0

    def test_all_wrong(self):
        assert soft_score([[0, 0]], [[1, 1]]) == 0.0


class TestArrayDiff:
    def test_marks_wrong_cells(self):
        out = array_diff([[1, 5]], [[1, 2]])
        assert out == "1 5/2"

    def test_multiline(self):
        out = array_diff([[1, 2], [9, 4]], [[1, 2], [3, 4]])
        assert out.splitlines() == ["1 2", "9/3 4"]


class TestBuildFeedback:
    def test_mixed(self):
        rr_ok = RunResult(success=True, grid=[[1]], soft_score=1.0)
        rr_bad = RunResult(success=False, grid=[[0]], soft_score=0.0)
        text, score = build_feedback([rr_ok, rr_bad], [[[1]], [[5]]])
        assert score == 0.5
        assert "correctly" in text
        assert "incorrectly" in text
        assert "0/5" in text  # diff visualisation present

    def test_shape_mismatch_message(self):
        rr = RunResult(success=False, grid=[[1, 2, 3]], soft_score=0.0)
        text, score = build_feedback([rr], [[[1]]])
        assert score == 0.0
        assert "Shape mismatch" in text
        assert "(1, 3)" in text and "(1, 1)" in text

    def test_error_message_included(self):
        rr = RunResult(success=False, grid=None, error="timeout")
        text, score = build_feedback([rr], [[[0]]])
        assert score == 0.0
        assert "timeout" in text
        assert "rectangular" in text

    def test_empty(self):
        text, score = build_feedback([], [])
        assert score == 0.0


class TestEvaluateCode:
    def test_full_pipeline(self, tiny_task):
        train_in = [ex["input"] for ex in tiny_task["train"]]
        train_out = [ex["output"] for ex in tiny_task["train"]]
        test_in = [ex["input"] for ex in tiny_task["test"]]
        code = (
            "import numpy as np\n"
            "def transform(grid):\n"
            "    return (grid + 1) % 10\n"
        )
        train_res, test_res = asyncio.run(
            evaluate_code(code, train_in, train_out, test_in, timeout_s=30)
        )
        assert all(r.success for r in train_res)
        assert all(r.soft_score == 1.0 for r in train_res)
        assert test_res[0].grid == [[6, 6], [1, 2]]

    def test_broken_code_reports_per_example(self, tiny_task):
        train_in = [ex["input"] for ex in tiny_task["train"]]
        train_out = [ex["output"] for ex in tiny_task["train"]]
        code = "def transform(grid):\n    return grid\n"
        train_res, _ = asyncio.run(
            evaluate_code(code, train_in, train_out, [], timeout_s=30)
        )
        assert not any(r.success for r in train_res)
        # identity matches nothing here; soft scores strictly between 0 and 1
        assert all(0.0 <= r.soft_score < 1.0 for r in train_res)
