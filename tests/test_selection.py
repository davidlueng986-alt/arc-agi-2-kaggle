from harness.grids import grid_key
from harness.selection import (
    VERIFIED_BONUS_JUDGE,
    build_grid_stats,
    early_accept,
    score_stat,
    select_attempts,
)
from harness.types import Candidate, GridStat

GA = [[1, 1], [1, 1]]           # 2x2
GB = [[2, 2], [2, 2]]           # 2x2 (same shape as GA)
GC = [[3, 3, 3], [3, 3, 3], [3, 3, 3]]  # 3x3 (different shape)
TEST_IN = [[0, 0], [0, 0]]
TRAIN_OUT = [[[1, 1], [1, 1]]]


def _stat(grid, jp=0, vv=0, votes=0, soft=0.0, sources=("expert_a",), synth=False):
    return GridStat(grid=grid, judge_points=jp, verified_votes=vv, votes=votes,
                    best_soft=soft, sources=set(sources), synthesized=synth)


class TestGridStats:
    def test_votes_and_sources(self):
        cands = [
            Candidate(source="expert_a", grids=[GA], verified=True, soft=1.0),
            Candidate(source="expert_b", grids=[GA], verified=True, soft=1.0),
            Candidate(source="text[0]", grids=[GB], soft=0.0),
        ]
        stats = build_grid_stats(cands, 0)
        by_key = {grid_key(s.grid): s for s in stats}
        sa = by_key[grid_key(GA)]
        assert sa.votes == 2 and sa.verified_votes == 2
        assert sa.sources == {"expert_a", "expert_b"}
        sb = by_key[grid_key(GB)]
        assert sb.votes == 1 and sb.verified_votes == 0

    def test_judge_points_merged_and_synth_added(self):
        from harness.types import ScoreEntry

        cands = [Candidate(source="expert_a", grids=[GA], soft=0.5)]
        board = [
            ScoreEntry(grid=GA, points=5),
            ScoreEntry(grid=GC, points=2, origin="synthesized", matched=False),
        ]
        stats = build_grid_stats(cands, 0, board)
        by_key = {grid_key(s.grid): s for s in stats}
        assert by_key[grid_key(GA)].judge_points == 5
        assert by_key[grid_key(GC)].synthesized is True
        assert by_key[grid_key(GC)].votes == 0


class TestScoring:
    def test_judge_mode_formula(self):
        st = _stat(GA, jp=4, vv=1, votes=3)
        import math

        expected = 4 + VERIFIED_BONUS_JUDGE * 1 + 0.25 * math.log1p(3)
        assert abs(score_stat(st, judge_on=True) - expected) < 1e-9

    def test_single_judge_first_choice_cannot_beat_verified(self):
        synth = _stat(GC, jp=2, vv=0, votes=0)          # one judge first choice
        verified = _stat(GA, jp=0, vv=1, votes=2)        # verified, no judge support
        assert score_stat(verified, True) > score_stat(synth, True)

    def test_two_of_three_majority_can_beat_verified(self):
        synth = _stat(GC, jp=4, vv=0, votes=0)           # 2/3 first choices
        verified = _stat(GA, jp=0, vv=1, votes=2)
        assert score_stat(synth, True) > score_stat(verified, True)


class TestEarlyAccept:
    def test_two_verified_votes_unique(self):
        stats = [_stat(GA, vv=2, votes=2), _stat(GB, votes=1)]
        acc = early_accept(stats)
        assert acc is not None and acc.grid == GA

    def test_conflicting_verified_grids_need_judge(self):
        stats = [_stat(GA, vv=1, votes=1), _stat(GB, vv=1, votes=1)]
        assert early_accept(stats) is None

    def test_single_verified_vote_needs_judge(self):
        stats = [_stat(GA, vv=1, votes=1)]
        assert early_accept(stats) is None


class TestAttemptSelection:
    def test_naive_top2_when_diversity_off(self):
        stats = [_stat(GA, jp=6), _stat(GB, jp=3), _stat(GC, jp=1)]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, (3, 3),
                                 judge_on=True, diversity_on=False)
        assert a1 == GA and a2 == GB  # naive: second-highest score, wrong shape

    def test_shape_repair_hedge(self):
        # judge loves the 2x2 cluster but the train-implied shape is 3x3
        stats = [_stat(GA, jp=6), _stat(GB, jp=3), _stat(GC, jp=1)]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, (3, 3),
                                 judge_on=True, diversity_on=True)
        assert a1 == GA
        assert a2 == GC  # best S*-shaped candidate, not the same-shape GB

    def test_no_hedge_when_attempt1_shape_matches_prior(self):
        stats = [_stat(GC, jp=6), _stat(GA, jp=3), _stat(GB, jp=1)]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, (3, 3),
                                 judge_on=True, diversity_on=True)
        assert a1 == GC
        assert a2 == GA  # plain second best (different source? both expert_a -> score decides)

    def test_source_decorrelation_tiebreak(self):
        # exact score tie; attempt_2 must prefer the different-source candidate
        s_same = _stat(GB, jp=3, votes=1, sources=("expert_a",))
        s_diff = _stat(GC, jp=3, votes=1, sources=("text[0]",))
        stats = [_stat(GA, jp=6, sources=("expert_a",)), s_same, s_diff]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, None,
                                 judge_on=True, diversity_on=True)
        assert a1 == GA
        assert a2 == GC

    def test_supported_alternative_swap(self):
        # top-2 same shape AND same source; a supported different-shape exists
        stats = [
            _stat(GA, jp=5, votes=2, sources=("expert_a",)),
            _stat(GB, jp=2, votes=1, sources=("expert_a",)),
            _stat(GC, jp=0, votes=2, sources=("expert_b",)),
        ]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, None,
                                 judge_on=True, diversity_on=True)
        assert a1 == GA
        assert a2 == GC  # swapped in: votes>=2 support

    def test_unsupported_singleton_not_swapped(self):
        stats = [
            _stat(GA, jp=5, votes=2, sources=("expert_a",)),
            _stat(GB, jp=2, votes=1, sources=("expert_a",)),
            _stat(GC, jp=0, votes=1, sources=("expert_b",)),  # no support
        ]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, None,
                                 judge_on=True, diversity_on=True)
        assert a2 == GB  # rule 5 requires votes>=2 / judge point / verified

    def test_single_candidate_gets_distinct_fallback(self):
        stats = [_stat(GA, votes=1)]
        a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, None,
                                 judge_on=False, diversity_on=True)
        assert a1 == GA
        assert a2 != GA
        assert a2  # non-empty

    def test_no_candidates_fallbacks(self):
        a1, a2 = select_attempts([], TEST_IN, TRAIN_OUT, (2, 2),
                                 judge_on=False, diversity_on=True)
        assert a1 == TEST_IN            # identity
        assert a2 == [[1, 1], [1, 1]]   # modal train-out colour fill (colour 1)
        assert a1 != a2

    def test_attempt2_never_equals_attempt1(self):
        stats = [_stat(GA, jp=9, votes=5)]
        for suggested in (None, (2, 2), (3, 3)):
            a1, a2 = select_attempts(stats, TEST_IN, TRAIN_OUT, suggested,
                                     judge_on=True, diversity_on=True)
            assert grid_key(a1) != grid_key(a2)

    def test_no_judge_mode_prefers_verified(self):
        stats = [
            _stat(GA, jp=0, vv=1, votes=1, soft=1.0),
            _stat(GB, jp=0, vv=0, votes=4, soft=0.4),
        ]
        a1, _ = select_attempts(stats, TEST_IN, TRAIN_OUT, None,
                                judge_on=False, diversity_on=True)
        assert a1 == GA
