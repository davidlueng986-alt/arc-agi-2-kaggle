from __future__ import annotations

import pytest

from harness.select import (
    Candidate,
    CandidatePool,
    consensus_reached,
    task_consensus,
)


def test_candidate_weight_by_effort():
    grid = [[1]]
    assert Candidate(grid, effort="low").weight == pytest.approx(1.0)
    assert Candidate(grid, effort="medium").weight == pytest.approx(1.5)
    assert Candidate(grid, effort="xhigh").weight == pytest.approx(2.0)
    assert Candidate(grid, effort="xhigh", truncated=True).weight == pytest.approx(0.6)


def test_candidate_unknown_effort_is_neutral():
    assert Candidate([[1]], effort="banana").weight == pytest.approx(1.0)


def test_pool_requires_positive_n_test():
    with pytest.raises(ValueError):
        CandidatePool(0)


def test_pool_index_bounds():
    pool = CandidatePool(1)
    with pytest.raises(IndexError):
        pool.add(1, [[0]])


def test_plurality_wins_attempt_one():
    pool = CandidatePool(1)
    pool.add(0, [[1, 1]], effort="xhigh")
    pool.add(0, [[1, 1]], effort="xhigh")
    pool.add(0, [[2, 2]], effort="xhigh")
    result = pool.select([[[0, 0]]])[0]
    assert result.attempt_1 == [[1, 1]]
    assert result.n_candidates == 3
    assert result.n_distinct == 2
    assert result.consensus == pytest.approx(4.0 / 6.0)


def test_attempt_two_is_shape_diverse_when_available():
    pool = CandidatePool(1)
    for _ in range(3):
        pool.add(0, [[1, 1], [1, 1]], effort="xhigh")
    pool.add(0, [[5, 5, 5]], effort="xhigh")
    pool.add(0, [[9, 9], [9, 9]], effort="xhigh")
    result = pool.select([[[0]]])[0]
    assert result.attempt_1 == [[1, 1], [1, 1]]
    assert result.attempt_2 == [[5, 5, 5]]
    assert result.shape_diverse is True


def test_attempt_two_falls_back_to_second_plurality_when_no_shape_diversity():
    pool = CandidatePool(1)
    pool.add(0, [[1, 1]], effort="xhigh")
    pool.add(0, [[1, 1]], effort="xhigh")
    pool.add(0, [[2, 2]], effort="xhigh")
    result = pool.select([[[0, 0]]])[0]
    assert result.attempt_1 == [[1, 1]]
    assert result.attempt_2 == [[2, 2]]
    assert result.shape_diverse is False


def test_single_candidate_duplicated_into_attempt_two():
    pool = CandidatePool(1)
    pool.add(0, [[4, 4]], effort="xhigh")
    result = pool.select([[[0, 0]]])[0]
    assert result.attempt_1 == [[4, 4]]
    assert result.attempt_2 == [[4, 4]]


def test_empty_pool_uses_fallback():
    pool = CandidatePool(2)
    results = pool.select([[[1]], [[2, 2]]])
    assert results[0].attempt_1 == [[1]]
    assert results[0].used_fallback is True
    assert results[1].attempt_2 == [[2, 2]]


def test_higher_effort_breaks_a_tie_in_count():
    pool = CandidatePool(1)
    pool.add(0, [[1]], effort="low")
    pool.add(0, [[2]], effort="xhigh")
    result = pool.select([[[0]]])[0]
    assert result.attempt_1 == [[2]]


def test_deterministic_order_breaks_a_full_tie():
    pool = CandidatePool(1)
    pool.add(0, [[3]], effort="xhigh")
    pool.add(0, [[7]], effort="xhigh")
    first = pool.select([[[0]]])[0]
    pool2 = CandidatePool(1)
    pool2.add(0, [[3]], effort="xhigh")
    pool2.add(0, [[7]], effort="xhigh")
    second = pool2.select([[[0]]])[0]
    assert first.attempt_1 == second.attempt_1 == [[3]]


def test_truncation_penalty_demotes_a_candidate():
    pool = CandidatePool(1)
    pool.add(0, [[1]], effort="xhigh", truncated=True)
    pool.add(0, [[2]], effort="low", truncated=False)
    result = pool.select([[[0]]])[0]
    assert result.attempt_1 == [[2]]


def test_per_test_index_votes_are_independent():
    pool = CandidatePool(2)
    pool.add_sample([[[1]], [[2]]], effort="xhigh")
    pool.add_sample([[[1]], None], effort="xhigh")
    pool.add_sample([None, [[8]]], effort="xhigh")
    results = pool.select([[[0]], [[0]]])
    assert results[0].attempt_1 == [[1]]
    assert results[0].n_candidates == 2
    assert results[1].n_candidates == 2


def test_select_rejects_wrong_fallback_count():
    pool = CandidatePool(2)
    with pytest.raises(ValueError):
        pool.select([[[0]]])


def test_consensus_thresholds():
    pool = CandidatePool(1)
    for _ in range(4):
        pool.add(0, [[1]], effort="xhigh")
    result = pool.select([[[0]]])[0]
    assert result.consensus == pytest.approx(1.0)
    assert consensus_reached(result, min_candidates=4, threshold=0.7)
    assert not consensus_reached(result, min_candidates=5, threshold=0.7)
    assert consensus_reached(result, min_candidates=4, threshold=1.0)


def test_consensus_false_when_split():
    pool = CandidatePool(1)
    pool.add(0, [[1]], effort="xhigh")
    pool.add(0, [[2]], effort="xhigh")
    pool.add(0, [[3]], effort="xhigh")
    pool.add(0, [[4]], effort="xhigh")
    result = pool.select([[[0]]])[0]
    assert not consensus_reached(result, min_candidates=4, threshold=0.7)


def test_task_consensus_requires_all_indices():
    pool = CandidatePool(2)
    for _ in range(4):
        pool.add(0, [[1]], effort="xhigh")
    pool.add(1, [[2]], effort="xhigh")
    pool.add(1, [[3]], effort="xhigh")
    results = pool.select([[[0]], [[0]]])
    assert not task_consensus(results, 4, 0.7)
    assert task_consensus(results[:1], 4, 0.7)


def test_task_consensus_empty_is_false():
    assert task_consensus([], 4, 0.7) is False


def test_pool_counters():
    pool = CandidatePool(2)
    assert pool.add_sample([[[1]], [[2]]], effort="xhigh") == 2
    assert pool.add_sample([None, [[3]]], effort="low") == 1
    assert pool.total() == 3
    assert pool.min_index_count() == 1
    assert pool.candidates(0)[0].effort == "xhigh"
