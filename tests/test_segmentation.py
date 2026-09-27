"""Segmentation: 4-connected components, adjacency, nesting, invariant hashes."""

from __future__ import annotations

import pytest

from harness.segmentation import (
    MAX_CELLS_PER_NODE,
    MAX_NODES,
    segment_grid,
    segment_summary,
)


def test_single_component():
    grid = [[1, 1], [1, 1]]
    seg = segment_grid(grid)
    assert seg["n_components"] == 1
    node = seg["nodes"][0]
    assert node["pixels"] == 4
    assert node["color"] == 1
    assert node["char"] == "b"
    assert node["bbox"] == [0, 0, 1, 1]
    assert seg["adjacency_list"] == []
    assert seg["containment_computed"] is True


def test_four_connectivity_not_eight():
    # diagonal-only contact must NOT merge
    grid = [
        [1, 0],
        [0, 1],
    ]
    seg = segment_grid(grid)
    ones = [n for n in seg["nodes"] if n["color"] == 1]
    assert len(ones) == 2
    zeros = [n for n in seg["nodes"] if n["color"] == 0]
    assert len(zeros) == 2


def test_reading_order_ids():
    grid = [
        [0, 2, 0],
        [1, 0, 0],
    ]
    seg = segment_grid(grid)
    # ids follow the top-most-left-most cell of each component, in reading order:
    # (0,0)=0 -> 0, (0,1)=2 -> 1, (0,2)=0 -> 2 (joins (1,1),(1,2)), (1,0)=1 -> 3
    # `cells` is emitted sorted, so cells[0] is the reading-order minimum.
    starts = {n["id"]: tuple(n["cells"][0]) for n in seg["nodes"]}
    assert starts[0] == (0, 0)
    assert starts[1] == (0, 1)
    assert starts[2] == (0, 2)
    assert starts[3] == (1, 0)
    assert seg["n_components"] == 4


def test_adjacency_list():
    grid = [
        [1, 2],
        [1, 2],
    ]
    seg = segment_grid(grid)
    assert seg["adjacency_list"] == [[0, 1]]


def test_containment_parent_child():
    grid = [
        [1, 1, 1],
        [1, 2, 1],
        [1, 1, 1],
    ]
    seg = segment_grid(grid)
    outer = next(n for n in seg["nodes"] if n["color"] == 1)
    inner = next(n for n in seg["nodes"] if n["color"] == 2)
    assert inner["id"] in outer["children"]
    assert inner["children"] == []


def test_nested_three_levels():
    grid = [
        [1, 1, 1, 1, 1],
        [1, 2, 2, 2, 1],
        [1, 2, 3, 2, 1],
        [1, 2, 2, 2, 1],
        [1, 1, 1, 1, 1],
    ]
    seg = segment_grid(grid)
    by_color = {n["color"]: n for n in seg["nodes"]}
    assert by_color[2]["id"] in by_color[1]["children"]
    assert by_color[3]["id"] in by_color[2]["children"]
    # innermost encloser only -- 3 is NOT a direct child of 1
    assert by_color[3]["id"] not in by_color[1]["children"]


def test_hash_is_translation_invariant():
    a = segment_grid([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    b = segment_grid([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    ha = next(n["hash"] for n in a["nodes"] if n["color"] == 1)
    hb = next(n["hash"] for n in b["nodes"] if n["color"] == 1)
    assert ha == hb


def test_hash_distinguishes_shape_and_color():
    a = segment_grid([[1, 1, 0], [0, 0, 0]])
    b = segment_grid([[1, 0, 0], [1, 0, 0]])
    c = segment_grid([[2, 2, 0], [0, 0, 0]])
    ha = next(n["hash"] for n in a["nodes"] if n["pixels"] == 2)
    hb = next(n["hash"] for n in b["nodes"] if n["pixels"] == 2)
    hc = next(n["hash"] for n in c["nodes"] if n["pixels"] == 2)
    assert ha != hb
    assert ha != hc


def test_boundary_is_corner_points_of_a_rectangle():
    grid = [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]
    seg = segment_grid(grid)
    node = next(n for n in seg["nodes"] if n["color"] == 1)
    assert len(node["boundary"]) == 4
    assert sorted(map(tuple, node["boundary"])) == sorted([(1, 1), (1, 2), (2, 2), (2, 1)])


def test_single_cell_boundary():
    seg = segment_grid([[0, 5], [0, 0]])
    node = next(n for n in seg["nodes"] if n["color"] == 5)
    assert node["boundary"] == [[0, 1]]
    assert node["pixels"] == 1


def test_cells_are_capped():
    grid = [[1] * 30 for _ in range(30)]
    seg = segment_grid(grid)
    assert seg["nodes"][0]["pixels"] == 900
    assert len(seg["nodes"][0]["cells"]) == MAX_CELLS_PER_NODE
    assert seg["nodes"][0]["cells_truncated"] is True


def test_too_many_components_skips_containment():
    # checkerboard: every cell its own component
    grid = [[(r + c) % 2 for c in range(30)] for r in range(30)]
    seg = segment_grid(grid)
    assert seg["n_components"] == 900
    assert seg["truncated"] is True
    assert len(seg["nodes"]) == MAX_NODES
    assert seg["containment_computed"] is False
    assert all(n["children"] == [] for n in seg["nodes"])


def test_empty_and_degenerate():
    assert segment_grid([])["nodes"] == []
    assert segment_grid([[]])["shape"] == [0, 0]
    seg = segment_grid([[7]])
    assert seg["shape"] == [1, 1]
    assert seg["nodes"][0]["char"] == "o"


def test_summary_is_compact_and_readable():
    grid = [
        [0, 0, 2, 0],
        [0, 0, 2, 0],
        [1, 1, 0, 0],
    ]
    seg = segment_grid(grid)
    text = segment_summary(grid, max_nodes=10)
    assert "shape=3x4" in text
    assert f"components={seg['n_components']}" in text
    assert seg["n_components"] == 4  # two 0-regions, one 2-region, one 1-region
    assert text.count("\n") == seg["n_components"]
    assert "#0" in text


def test_summary_truncates_node_list():
    grid = [[(r + c) % 3 for c in range(20)] for r in range(20)]
    text = segment_summary(grid, max_nodes=3)
    assert "more components" in text


def test_matches_oracle_color_chars():
    from harness.oracle import COLOR_CHARS
    from harness.segmentation import DEFAULT_COLOR_CHARS

    assert DEFAULT_COLOR_CHARS == COLOR_CHARS
    seg = segment_grid([[9]])
    assert seg["nodes"][0]["char"] == COLOR_CHARS[9]
