import numpy as np
import pytest

from harness.grids import (
    distinct_grids,
    extract_all_grids,
    grid_key,
    grid_to_csv,
    grid_to_diagram,
    grids_equal,
    modal_color,
    normalize_grid,
    normalized_distance,
    parse_grid_from_text,
    shape_of,
    suggest_output_shape,
)


class TestFormatting:
    def test_csv(self):
        assert grid_to_csv([[1, 2], [3, 4]]) == "1,2\n3,4"

    def test_diagram(self):
        assert grid_to_diagram([[1, 2], [3, 4]]) == "1 2\n3 4"

    def test_shape_and_key(self):
        g = [[1, 2], [3, 4]]
        assert shape_of(g) == (2, 2)
        assert grid_key(g) == ((1, 2), (3, 4))
        assert grids_equal(g, [[1, 2], [3, 4]])
        assert not grids_equal(g, [[1, 2], [3, 5]])
        assert shape_of([]) is None


class TestNormalize:
    def test_numpy(self):
        assert normalize_grid(np.array([[1, 2], [3, 4]])) == [[1, 2], [3, 4]]

    def test_ragged_rejected(self):
        assert normalize_grid([[1, 2], [3]]) is None

    def test_empty_rejected(self):
        assert normalize_grid([]) is None
        assert normalize_grid([[]]) is None
        assert normalize_grid(None) is None

    def test_non_numeric_rejected(self):
        assert normalize_grid([["a", "b"]]) is None

    def test_floats_truncated_to_int(self):
        assert normalize_grid([[1.0, 2.0]]) == [[1, 2]]


class TestTolerantCSVParsing:
    """Land's tolerant parser must survive noisy local-model output."""

    def test_clean(self):
        assert parse_grid_from_text("1,2,3\n4,5,6") == [[1, 2, 3], [4, 5, 6]]

    def test_prose_around(self):
        text = "The answer is:\n\n1,2\n3,4\n\nHope this helps!"
        # 'Hope this helps!' has no digits -> not a row; block intact
        assert parse_grid_from_text(text) == [[1, 2], [3, 4]]

    def test_row_labels(self):
        text = "Row 1:\n1,2\nRow 2:\n3,4"
        assert parse_grid_from_text(text) == [[1, 2], [3, 4]]

    def test_numbered_list_rows(self):
        text = "1. 8,8,8\n2. 7,7,7"
        assert parse_grid_from_text(text) == [[8, 8, 8], [7, 7, 7]]

    def test_fences_are_hard_separators_last_block_wins(self):
        text = "```\n1,1\n1,1\n```\nsome words\n```\n2,2\n2,2\n```"
        assert parse_grid_from_text(text) == [[2, 2], [2, 2]]

    def test_big_gap_splits_blocks(self):
        text = "1,1\n\n\n\n\n\n9,9\n9,9"
        blocks = extract_all_grids(text)
        assert blocks[-1] == [[9, 9], [9, 9]]

    def test_small_gap_tolerated(self):
        text = "1,1\n\nsome noise\n2,2"
        # gap of 2 lines (blank + noise) within MAX_GAP=2 and width matches
        assert parse_grid_from_text(text) == [[1, 1], [2, 2]]

    def test_width_jump_splits(self):
        text = "1,2,3,4,5,6,7,8\n9,1"
        blocks = extract_all_grids(text)
        assert len(blocks) == 2

    def test_bracketed_python_style_rows(self):
        text = "[1, 2],\n[3, 4],"
        assert parse_grid_from_text(text) == [[1, 2], [3, 4]]

    def test_no_grid_returns_none(self):
        assert parse_grid_from_text("just prose, no numbers at all") is None
        assert parse_grid_from_text("") is None

    def test_extract_all_multiple(self):
        text = "```\n1,1\n```\ntext\n```\n2,2\n2,2\n```"
        blocks = extract_all_grids(text)
        assert [b for b in blocks] == [[[1, 1]], [[2, 2], [2, 2]]]

    def test_bullet_points_ignored(self):
        text = "- move the object\n1,2\n- recolor\n3,4"
        assert parse_grid_from_text(text) == [[1, 2], [3, 4]]

    def test_distinct_grids_order_preserved(self):
        a, b = [[1]], [[2]]
        assert distinct_grids([a, b, a, [ [1] ]]) == [a, b]


class TestDistance:
    def test_identical(self):
        assert normalized_distance([[1, 2]], [[1, 2]]) == 0.0

    def test_half_different(self):
        assert normalized_distance([[1, 2]], [[1, 9]]) == 0.5

    def test_shape_mismatch_is_max(self):
        assert normalized_distance([[1, 2]], [[1, 2, 3]]) == 1.0

    def test_none(self):
        assert normalized_distance(None, [[1]]) == 1.0


class TestModalColor:
    def test_modal_excluding_zero_preference(self):
        outs = [[[0, 0], [2, 2]], [[0, 2], [2, 2]]]
        # counts: 0x4, 2x6 -> modal is 2 (nonzero pool wins when present)
        assert modal_color(outs) == 2

    def test_all_zero(self):
        assert modal_color([[[0, 0]]]) == 0

    def test_empty(self):
        assert modal_color([]) == 0


class TestShapePrior:
    def test_identity_rule(self):
        tin = [[[0, 0, 0], [0, 0, 0]], [[1, 1], [1, 1]]]
        tout = [[[0, 0, 0], [0, 0, 0]], [[1, 1], [1, 1]]]
        test = [[5] * 7 for _ in range(4)]
        assert suggest_output_shape(tin, tout, test) == (4, 7)

    def test_constant_output(self):
        tin = [[[0] * 5 for _ in range(5)], [[0] * 3 for _ in range(7)]]
        tout = [[[1] * 3 for _ in range(3)], [[1] * 3 for _ in range(3)]]
        test = [[0] * 9 for _ in range(2)]
        assert suggest_output_shape(tin, tout, test) == (3, 3)

    def test_double_scale(self):
        tin = [[[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        tout = [
            [[1] * 4 for _ in range(4)],
            [[1] * 6 for _ in range(6)],
        ]
        test = [[0] * 5 for _ in range(3)]
        assert suggest_output_shape(tin, tout, test) == (6, 10)

    def test_ambiguous_single_pair(self):
        # one train pair 3->5: many (a,b) fits; they disagree on a 7-row test
        tin = [[[0] * 3 for _ in range(3)]]
        tout = [[[1] * 5 for _ in range(5)]]
        test = [[0] * 7 for _ in range(7)]
        assert suggest_output_shape(tin, tout, test) is None

    def test_constant_output_single_pair_is_ambiguous(self):
        # with ONE pair, "constant 5x5" cannot be distinguished from scaling
        tin = [[[0] * 3 for _ in range(3)]]
        tout = [[[1] * 5 for _ in range(5)]]
        assert suggest_output_shape(tin, tout, [[0] * 4 for _ in range(4)]) is None

    def test_constant_output_two_identical_input_pairs(self):
        tin = [[[0] * 3 for _ in range(3)]] * 2
        tout = [[[1] * 5 for _ in range(5)]] * 2
        test = [[0] * 7 for _ in range(7)]
        assert suggest_output_shape(tin, tout, test) == (5, 5)

    def test_mixed_relation_falls_through_to_fit_or_none(self):
        tin = [[[0, 0], [0, 0]], [[0, 0, 0], [0, 0, 0]], [[0] * 4 for _ in range(4)]]
        tout = [[[0, 0], [0, 0]], [[1] * 3 for _ in range(3)], [[1] * 8 for _ in range(8)]]
        test = [[0] * 5 for _ in range(5)]
        # rows: (2->2, 3->3, 4->8) — no single linear fit (a<=4) -> None
        assert suggest_output_shape(tin, tout, test) is None

    def test_empty_inputs(self):
        assert suggest_output_shape([], [], [[0]]) is None
