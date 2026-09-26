from __future__ import annotations

import pytest

from harness.grids import (
    fallback_grid,
    from_key,
    is_grid,
    parse_csv_strict,
    parse_grid_blocks,
    parse_last_grid,
    regularize,
    shape,
    to_csv,
    to_csv_indented,
    to_key,
)


def test_is_grid_accepts_rectangular_ints():
    assert is_grid([[0, 1], [2, 3]])
    assert is_grid([[5]])
    assert is_grid([list(range(10))])


def test_is_grid_rejects_bad_inputs():
    assert not is_grid([])
    assert not is_grid([[]])
    assert not is_grid([[0, 1], [2]])
    assert not is_grid([[0, 10]])
    assert not is_grid([[0, -1]])
    assert not is_grid([[0, "1"]])
    assert not is_grid([[True, False]])
    assert not is_grid("0,1")
    assert not is_grid(None)


def test_csv_roundtrip():
    grid = [[0, 1, 2], [3, 4, 5]]
    assert to_csv(grid) == "0,1,2\n3,4,5"
    assert parse_csv_strict(to_csv(grid)) == grid


def test_key_roundtrip_and_shape():
    grid = [[1, 2, 3], [4, 5, 6]]
    assert from_key(to_key(grid)) == grid
    assert shape(grid) == (2, 3)


def test_indented_csv():
    assert to_csv_indented([[1, 2]], "  ") == "  1,2"


def test_strict_rejects_ragged_and_noise():
    assert parse_csv_strict("1,2\n3") is None
    assert parse_csv_strict("1,2\nhello") is None
    assert parse_csv_strict("") is None


def test_parse_blocks_from_clean_answer():
    text = "ANSWER 1:\n0,1,2\n3,4,5\n"
    assert parse_grid_blocks(text) == [[[0, 1, 2], [3, 4, 5]]]


def test_parse_blocks_from_markdown_fence():
    text = "Here is the grid:\n```\n1,0\n0,1\n```\nDone."
    assert parse_grid_blocks(text) == [[[1, 0], [0, 1]]]


def test_parse_blocks_from_python_style_rows():
    text = "Output: [1, 2, 3]\n        [4, 5, 6]"
    blocks = parse_grid_blocks(text)
    assert blocks == [[[1, 2, 3], [4, 5, 6]]]


def test_parse_blocks_ignores_bullets_and_prose():
    text = "- the shape moves right\n- colour 2 becomes 8\n\n7,7,7\n7,0,7\n7,7,7\n"
    assert parse_grid_blocks(text) == [[[7, 7, 7], [7, 0, 7], [7, 7, 7]]]


def test_parse_blocks_splits_on_width_jump():
    text = "1,2,3\n4,5,6\n\n1,2,3,4,5,6,7,8,9\n1,2,3,4,5,6,7,8,9"
    blocks = parse_grid_blocks(text)
    assert len(blocks) == 2
    assert shape(blocks[0]) == (2, 3)
    assert shape(blocks[1]) == (2, 9)


def test_parse_blocks_splits_on_fence_between_grids():
    text = "1,1\n1,1\n```\n2,2\n2,2"
    blocks = parse_grid_blocks(text)
    assert blocks == [[[1, 1], [1, 1]], [[2, 2], [2, 2]]]


def test_parse_last_grid_returns_final_answer():
    text = "Draft:\n0,0\n0,0\n\nFinal:\n9,9\n9,9\n"
    assert parse_last_grid(text) == [[9, 9], [9, 9]]


def test_parse_blocks_handles_row_labels():
    text = "Row 1: 4,5\nRow 2: 6,7"
    assert parse_grid_blocks(text) == [[[4, 5], [6, 7]]]


def test_parse_blocks_numbered_list():
    text = "1. 8,8\n2. 0,0"
    assert parse_grid_blocks(text) == [[[8, 8], [0, 0]]]


def test_parse_blocks_returns_empty_on_no_grid():
    assert parse_grid_blocks("I cannot solve this.") == []
    assert parse_grid_blocks("") == []
    assert parse_last_grid("no digits here") is None


def test_parse_blocks_drops_out_of_range_cells():
    assert parse_grid_blocks("1,12\n3,4") == [[[3, 4]]]
    assert parse_grid_blocks("12,12\n99,99") == []


def test_parse_blocks_single_cell_needs_explicit_opt_in():
    assert parse_grid_blocks("5") == []
    assert parse_grid_blocks("5", allow_single_cell=True) == [[[5]]]


def test_regularize_drops_minority_width_rows():
    rows = [[1, 2, 3], [4, 5, 6], [7, 8], [9, 10, 11]]
    assert regularize(rows) == [[1, 2, 3], [4, 5, 6], [9, 10, 11]]


def test_regularize_rejects_when_too_ragged():
    rows = [[1, 2], [3, 4, 5], [6, 7, 8, 9]]
    assert regularize(rows) is None


def test_regularize_empty():
    assert regularize([]) is None


def test_fallback_grid_uses_modal_output_shape_and_colour():
    train = [([[0]], [[1, 1], [1, 1]]), ([[2]], [[1, 1], [1, 1]])]
    out = fallback_grid([[5, 5, 5]], train)
    assert out == [[1, 1], [1, 1]]


def test_fallback_grid_identity_without_train():
    out = fallback_grid([[1, 2], [3, 4]], [])
    assert out == [[1, 2], [3, 4]]


@pytest.mark.parametrize(
    "text",
    [
        "1,2\n3,4",
        "  1,2  \n  3,4  ",
        "`1,2`\n`3,4`",
    ],
)
def test_parse_blocks_whitespace_tolerant(text):
    assert parse_grid_blocks(text) == [[[1, 2], [3, 4]]]
