"""Oracle: exact verification, diff diagnostics, partial credit, augmentation."""

from __future__ import annotations

import pytest

from harness.oracle import (
    AUGMENTATIONS,
    ascii_grid,
    augment,
    cell_accuracy,
    crop_around,
    grid_diff,
    grid_equal,
    normalize_grid,
    pair_credit,
    shape_of,
    strip_private,
    summarize,
    verify_candidate,
)


class TestNormalize:
    def test_plain_lists(self):
        assert normalize_grid([[0, 1], [2, 3]]) == [[0, 1], [2, 3]]

    def test_tuples_and_floats(self):
        assert normalize_grid(((0, 1.0), (2.0, 3))) == [[0, 1], [2, 3]]

    def test_numpy(self):
        np = pytest.importorskip("numpy")
        assert normalize_grid(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]

    @pytest.mark.parametrize(
        "bad",
        [None, 5, "abc", [], [[]], [[0], [0, 1]], [[0, 10]], [[0, -1]], [[0, "x"]], object()],
    )
    def test_rejects_bad(self, bad):
        assert normalize_grid(bad) is None

    def test_rejects_oversized(self):
        big = [[0] * 65 for _ in range(65)]
        assert normalize_grid(big) is None

    def test_rejects_ragged(self):
        assert normalize_grid([[0, 1], [2]]) is None


class TestDiff:
    def test_exact(self):
        assert grid_diff([[1]], [[1]])["kind"] == "exact"

    def test_shape_mismatch_reports_delta(self):
        diff = grid_diff([[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0]])
        assert diff["kind"] == "shape"
        assert diff["expected_shape"] == [2, 2]
        assert diff["got_shape"] == [2, 3]
        assert diff["delta_cols"] == 1
        assert diff["delta_rows"] == 0

    def test_cell_diff_carries_coordinates_and_values(self):
        expected = [[1, 1, 1], [1, 2, 1], [1, 1, 1]]
        got = [[1, 1, 1], [1, 3, 1], [1, 1, 1]]
        diff = grid_diff(expected, got)
        assert diff["kind"] == "cells"
        assert diff["n_diff"] == 1
        assert diff["n_cells"] == 9
        assert diff["bbox"] == [1, 1, 1, 1]
        assert diff["samples"] == [[1, 1, 2, 3]]
        assert diff["expected_hist_at_diffs"] == {"2": 1}
        assert diff["got_hist_at_diffs"] == {"3": 1}
        assert diff["accuracy"] == round(8 / 9, 4)

    def test_samples_are_capped(self):
        expected = [[0] * 10 for _ in range(10)]
        got = [[1] * 10 for _ in range(10)]
        diff = grid_diff(expected, got, max_samples=4)
        assert len(diff["samples"]) == 4
        assert diff["samples_truncated"] is True
        assert diff["n_diff"] == 100
        assert diff["diff_frac"] == 1.0

    def test_crop_marks_the_mismatching_cell(self):
        expected = [[0, 0, 0, 0, 0]] * 5
        got = [list(row) for row in expected]
        got[2][2] = 1
        diff = grid_diff(expected, got)
        assert "b" in diff["got_crop"].upper().replace("B", "b") or "B" in diff["got_crop"]
        assert diff["expected_crop"] != diff["got_crop"]

    def test_crop_marks_centre_uppercase(self):
        grid = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        crop = crop_around(grid, 1, 1, radius=1)
        assert crop.splitlines()[1] == ".B."


class TestCredit:
    def test_exact_is_one(self):
        assert pair_credit([[1, 2]], [[1, 2]]) == 1.0

    def test_none_is_zero(self):
        assert pair_credit([[1]], None) == 0.0

    def test_same_shape_partial(self):
        credit = pair_credit([[0, 0], [0, 0]], [[0, 0], [0, 1]])
        assert 0.4 < credit < 0.95

    def test_shape_mismatch_is_heavily_discounted(self):
        assert pair_credit([[0] * 10 for _ in range(10)], [[0] * 5 for _ in range(5)]) == pytest.approx(0.075)

    def test_exact_iff_credit_one(self):
        assert pair_credit([[3, 3], [3, 3]], [[3, 3], [3, 4]]) < 1.0


class TestAugment:
    def test_names(self):
        assert AUGMENTATIONS == ("transpose", "rot180", "flipud")

    def test_transpose(self):
        assert augment([[1, 2, 3], [4, 5, 6]], "transpose") == [[1, 4], [2, 5], [3, 6]]

    def test_rot180(self):
        assert augment([[1, 2], [3, 4]], "rot180") == [[4, 3], [2, 1]]

    def test_flipud(self):
        assert augment([[1, 2], [3, 4]], "flipud") == [[3, 4], [1, 2]]

    def test_fliplr(self):
        assert augment([[1, 2], [3, 4]], "fliplr") == [[2, 1], [4, 3]]

    def test_identity_and_double_roundtrip(self):
        grid = [[1, 2], [3, 4]]
        assert augment(grid, "identity") == grid
        assert augment(augment(grid, "rot180"), "rot180") == grid
        assert augment(augment(grid, "transpose"), "transpose") == grid

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            augment([[0]], "spin")


class TestVerify:
    PAIRS = [
        ([[1, 2], [3, 4]], [[3, 4], [1, 2]]),
        ([[0, 5], [5, 0]], [[5, 0], [0, 5]]),
    ]
    TEST = [[[9, 9], [1, 1]]]

    def test_correct_program_verifies_exactly(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: list(reversed(g)))
        assert report["exact"] is True
        assert report["passed"] == 2
        assert report["pass_rate"] == 1.0
        assert report["soft_score"] == 1.0
        assert report["failures"] == []
        assert report["errors"] == []
        assert report["test"][0]["shape"] == [2, 2]
        assert report["test"][0]["_grid"] == [[1, 1], [9, 9]]

    def test_flipud_program_is_symmetry_equivariant(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: list(reversed(g)))
        assert report["aug"]["flipud"] == 1.0

    def test_wrong_program_reports_shape_and_cell_failures(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: g)
        assert report["exact"] is False
        assert report["passed"] == 0
        assert len(report["failures"]) == 2
        assert all(f["kind"] == "cells" for f in report["failures"])

    def test_shape_failure_is_labelled(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: [g[0]])
        assert report["failures"][0]["kind"] == "shape"

    def test_exception_is_reported_not_raised(self):
        def boom(grid):
            raise IndexError("nope")

        report = verify_candidate(self.PAIRS, self.TEST, boom)
        assert report["exact"] is False
        assert report["passed"] == 0
        assert report["errors"][0]["error"].startswith("IndexError")
        assert report["soft_score"] == 0.0

    def test_non_grid_return_is_reported(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: "not a grid")
        assert report["errors"]
        assert "non-grid" in report["errors"][0]["error"]

    def test_out_of_range_values_rejected(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: [[12] * len(r) for r in g])
        assert report["errors"]

    def test_test_errors_recorded_separately(self):
        calls = {"n": 0}

        def flaky(grid):
            calls["n"] += 1
            if calls["n"] > len(self.PAIRS):
                raise ValueError("test only")
            return list(reversed(grid))

        report = verify_candidate(self.PAIRS, self.TEST, flaky)
        assert report["exact"] is True
        assert report["test"][0]["error"].startswith("ValueError")

    def test_label_dependence_probe_runs_only_when_exact(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: g)
        assert "runs_without_labels" not in report
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: list(reversed(g)))
        assert report["runs_without_labels"] == 1.0

    def test_label_dependent_program_is_flagged(self):
        table = {tuple(map(tuple, p[0])): p[1] for p in self.PAIRS}

        def lookup(grid):
            return table[tuple(map(tuple, grid))]

        report = verify_candidate(self.PAIRS, self.TEST, lookup, use_augmentations=False)
        assert report["exact"] is True
        # It never reads the (blanked) labels, so it still "runs" -- but it cannot
        # answer the test input, which shows up as a test error.
        assert report["test"][0].get("error", "").startswith("KeyError")

    def test_augmentation_can_be_disabled(self):
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: g, use_augmentations=False)
        assert "aug" not in report

    def test_no_test_inputs(self):
        report = verify_candidate(self.PAIRS, [], lambda g: list(reversed(g)))
        assert report["test"] == []
        assert report["exact"] is True

    def test_empty_train(self):
        report = verify_candidate([], self.TEST, lambda g: g)
        assert report["exact"] is False
        assert report["pairs"] == 0

    def test_numpy_output_accepted(self):
        np = pytest.importorskip("numpy")
        report = verify_candidate(self.PAIRS, self.TEST, lambda g: np.array(g)[::-1])
        assert report["exact"] is True


class TestSummarizeAndStrip:
    def test_summary_verified(self):
        report = verify_candidate(TestVerify.PAIRS, TestVerify.TEST, lambda g: list(reversed(g)))
        assert report["summary"].startswith("VERIFIED 2/2")

    def test_summary_error(self):
        def boom(grid):
            raise RuntimeError("bad")

        report = verify_candidate(TestVerify.PAIRS, TestVerify.TEST, boom)
        assert report["summary"].startswith("ERROR RuntimeError")

    def test_summary_failure_counts_shape_mismatches(self):
        report = verify_candidate(TestVerify.PAIRS, TestVerify.TEST, lambda g: [g[0]])
        assert "shape mismatches" in report["summary"]

    def test_strip_private_removes_grids(self):
        report = verify_candidate(TestVerify.PAIRS, TestVerify.TEST, lambda g: list(reversed(g)))
        assert "_grid" in report["test"][0]
        stripped = strip_private(report)
        assert "_grid" not in stripped["test"][0]
        assert stripped["test"][0]["shape"] == [2, 2]
        assert stripped["exact"] is True

    def test_ascii_and_shapes(self):
        assert ascii_grid([[0, 1], [2, 3]]) == ".b\nrg"
        assert shape_of([]) == [0, 0]
        assert shape_of([[1, 2, 3]]) == [1, 3]
        assert grid_equal([[1]], [[1]]) is True
        assert grid_equal([[1]], [[2]]) is False
        assert cell_accuracy([[0, 1]], [[0, 2]]) == 0.5
