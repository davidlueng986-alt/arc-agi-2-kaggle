from __future__ import annotations

from harness.parse import merge_reasoning_and_content, parse_answers


def test_single_answer_marker():
    text = "Some reasoning.\n\nANSWER 1:\n1,2\n3,4\n"
    parsed = parse_answers(text, 1)
    assert parsed.method == "marker"
    assert parsed.grids == {1: [[1, 2], [3, 4]]}


def test_markdown_bold_marker():
    text = "blah\n**ANSWER 1:**\n```\n5,5\n6,6\n```\n"
    parsed = parse_answers(text, 1)
    assert parsed.grids == {1: [[5, 5], [6, 6]]}


def test_multiple_answers_in_order():
    text = (
        "ANSWER 1:\n1,1\n2,2\n\nANSWER 2:\n3,3,3\n4,4,4\n\nANSWER 3:\n9\n"
    )
    parsed = parse_answers(text, 3)
    assert parsed.indices() == [1, 2, 3]
    assert parsed.grids[1] == [[1, 1], [2, 2]]
    assert parsed.grids[2] == [[3, 3, 3], [4, 4, 4]]
    assert parsed.grids[3] == [[9]]


def test_alias_markers():
    text = "OUTPUT 1:\n7,7\n\nRESULT 2:\n8,8\n"
    parsed = parse_answers(text, 2)
    assert parsed.grids == {1: [[7, 7]], 2: [[8, 8]]}


def test_lower_case_marker():
    parsed = parse_answers("answer 1:\n0,1\n1,0\n", 1)
    assert parsed.grids == {1: [[0, 1], [1, 0]]}


def test_marker_mentioned_in_reasoning_then_real_answer():
    text = (
        "If ANSWER 1: were the identity we would be done, but it is not.\n"
        "The rule flips vertically.\n\nANSWER 1:\n4,3\n2,1\n"
    )
    parsed = parse_answers(text, 1)
    assert parsed.grids == {1: [[4, 3], [2, 1]]}


def test_labels_out_of_range_are_ignored():
    parsed = parse_answers("ANSWER 1:\n1,1\nANSWER 4:\n2,2\n", 1)
    assert parsed.indices() == [1]


def test_no_markers_ordered_blocks():
    text = "1,1\n2,2\n\n3,3\n4,4\n"
    parsed = parse_answers(text, 2)
    assert parsed.method == "ordered-blocks"
    assert parsed.grids == {1: [[1, 1], [2, 2]], 2: [[3, 3], [4, 4]]}


def test_no_markers_tail_blocks_used():
    text = "scratch work:\n0,0\n\nreal:\n5,5\n\n6,6\n"
    parsed = parse_answers(text, 2)
    assert parsed.grids[1] == [[5, 5]]
    assert parsed.grids[2] == [[6, 6]]


def test_no_markers_single_falls_back_to_last_block():
    text = "draft:\n1,1\n\nfinal:\n9,9\n"
    parsed = parse_answers(text, 1)
    assert parsed.method == "last-block"
    assert parsed.grids == {1: [[9, 9]]}


def test_empty_and_unparseable():
    assert not parse_answers("", 1).ok
    assert parse_answers("", 1).method == "empty"
    assert not parse_answers("I do not know.", 1).ok
    assert parse_answers("I do not know.", 1).method == "none"


def test_as_list_pads_missing_indices():
    parsed = parse_answers("ANSWER 1:\n1,2\n", 3)
    assert parsed.as_list(3) == [[[1, 2]], None, None]


def test_partial_marker_coverage_still_yields_parsed_index():
    text = "ANSWER 2:\n3,3\n"
    parsed = parse_answers(text, 2)
    assert parsed.grids == {2: [[3, 3]]}
    assert parsed.as_list(2) == [None, [[3, 3]]]


def test_merge_prefers_content_and_includes_reasoning():
    merged = merge_reasoning_and_content("REASONING TAIL\n1,1", "ANSWER 1:\n2,2")
    assert merged.index("ANSWER 1:") < merged.index("REASONING TAIL")


def test_merge_recovers_answer_from_truncated_content():
    parsed = parse_answers(merge_reasoning_and_content("...\nANSWER 1:\n7,8\n", ""), 1)
    assert parsed.grids == {1: [[7, 8]]}


def test_marker_slice_stops_at_next_marker():
    text = "ANSWER 1:\n1,2\n3,4\nANSWER 2:\n5,6\n7,8\n"
    parsed = parse_answers(text, 2)
    assert parsed.grids[1] == [[1, 2], [3, 4]]
    assert parsed.grids[2] == [[5, 6], [7, 8]]
