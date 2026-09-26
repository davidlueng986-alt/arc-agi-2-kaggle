"""pass@2-aware attempt selection (HYPOTHESIS.md §5).

pass@2 = P(A1 correct) + P(A1 wrong) * P(A2 correct | A1 wrong), so attempt_2
must maximise CONDITIONAL COVERAGE, not raw rank. The decorrelation axis is the
output shape, checked against the train-implied shape prior S* (measured: the
train shape relation generalises to the test output in >=98% of eval tasks, so
shape errors are systematic and correlate across candidates).

Rules, in order:
  1. score every distinct grid (judge points + verified bonus + vote mass);
  2. attempt_1 = argmax score;
  3. shape-repair hedge: if S* known and shape(attempt_1) != S*, attempt_2 =
     best-scoring grid WITH shape S*;
  4. else attempt_2 = best-scoring different grid, tie-broken toward a different
     source channel and larger content distance (source decorrelation);
  5. supported-alternative swap: if the top-2 share shape AND source, and a
     different-shape grid with support (votes >= 2 or judge points >= 1) exists,
     swap it into attempt_2 (ambiguous-shape hedge, support-gated);
  6. hard guarantees: attempt_2 != attempt_1 as grids; missing slots fall back
     to identity(test_input), then the modal-train-output-colour grid.
"""

from __future__ import annotations

import math
from typing import Optional

from harness.grids import (
    grid_key,
    modal_color,
    normalized_distance,
    shape_of,
)
from harness.types import Candidate, Grid, GridStat, ScoreEntry

VERIFIED_BONUS_JUDGE = 2.0  # equal to one judge first-choice: a single noisy
# judge sample cannot displace a verified grid, a 2/3 majority (4+ pts) can.
VERIFIED_BONUS_NOJUDGE = 3.0
VOTE_WEIGHT_JUDGE = 0.25
VOTE_WEIGHT_NOJUDGE = 1.0
SOFT_WEIGHT_NOJUDGE = 0.5


def build_grid_stats(
    candidates: list[Candidate],
    test_idx: int,
    scoreboard: Optional[list[ScoreEntry]] = None,
) -> list[GridStat]:
    stats: dict[tuple, GridStat] = {}

    def get(grid: Grid) -> GridStat:
        k = grid_key(grid)
        if k not in stats:
            stats[k] = GridStat(grid=grid)
        return stats[k]

    for c in candidates:
        if test_idx < len(c.grids) and c.grids[test_idx] is not None:
            st = get(c.grids[test_idx])
            st.votes += 1
            st.verified_votes += int(c.verified)
            st.best_soft = max(st.best_soft, c.soft)
            st.sources.add(c.source)

    for e in scoreboard or []:
        st = get(e.grid)
        st.judge_points = max(st.judge_points, e.points)
        if e.origin == "synthesized":
            st.synthesized = True

    return list(stats.values())


def score_stat(st: GridStat, judge_on: bool) -> float:
    if judge_on:
        return st.judge_points + VERIFIED_BONUS_JUDGE * st.verified_votes + \
            VOTE_WEIGHT_JUDGE * math.log1p(st.votes)
    return VERIFIED_BONUS_NOJUDGE * st.verified_votes + VOTE_WEIGHT_NOJUDGE * math.log1p(st.votes) + \
        SOFT_WEIGHT_NOJUDGE * st.best_soft


def _rank(st: GridStat, judge_on: bool):
    """Deterministic ordering key: score, then verified, then votes, then soft,
    then grid key for total determinism."""
    return (
        -score_stat(st, judge_on),
        -st.verified_votes,
        -st.votes,
        -st.best_soft,
        str(grid_key(st.grid)),
    )


def early_accept(stats: list[GridStat]) -> Optional[GridStat]:
    """Skip the judge when >= 2 independent verified votes land on ONE grid and
    no other verified grid competes (verification precision ~97%, Land)."""
    verified = [st for st in stats if st.verified_votes > 0]
    if len(verified) != 1:
        return None
    top = verified[0]
    if top.verified_votes >= 2:
        return top
    return None


def select_attempts(
    stats: list[GridStat],
    test_input: Grid,
    train_out: list[Grid],
    suggested_shape: Optional[tuple[int, int]],
    judge_on: bool,
    diversity_on: bool = True,
) -> tuple[Grid, Grid]:
    """Return (attempt_1, attempt_2) — never equal, never empty."""
    fallbacks = _fallback_grids(test_input, train_out, suggested_shape)

    if not stats:
        return fallbacks[0], fallbacks[1]

    ranked = sorted(stats, key=lambda st: _rank(st, judge_on))
    a1 = ranked[0].grid

    if len(ranked) == 1:
        alt = next((f for f in fallbacks if grid_key(f) != grid_key(a1)), fallbacks[0])
        return a1, alt

    if not diversity_on:
        # naive top-2 (ablation baseline)
        return a1, ranked[1].grid

    a1_shape = shape_of(a1)
    rest = ranked[1:]

    # Rule 3: shape-repair hedge.
    if suggested_shape is not None and a1_shape != suggested_shape:
        matching = [st for st in rest if shape_of(st.grid) == suggested_shape]
        if matching:
            return a1, matching[0].grid

    # Rule 4: best remaining, tie-broken toward a different source and larger
    # content distance from attempt_1.
    def decorrelation_key(st: GridStat):
        base = _rank(st, judge_on)
        same_source = 0 if st.sources - (set(ranked[0].sources)) else 1
        dist = normalized_distance(st.grid, a1)
        # within near-ties (same integer score bucket) prefer different source
        # and larger distance; _rank already dominates.
        return (base[0], same_source, -dist) + base[1:]

    a2_stat = min(rest, key=decorrelation_key)

    # Rule 5: supported-alternative swap when top-2 are fully correlated
    # (same shape AND same source pool) and a corroborated different-shape
    # hypothesis exists.
    if (
        suggested_shape is None
        and shape_of(a2_stat.grid) == a1_shape
        and a2_stat.sources
        and a2_stat.sources <= set(ranked[0].sources)
    ):
        alternatives = [
            st for st in rest
            if shape_of(st.grid) != a1_shape
            and (st.votes >= 2 or st.judge_points >= 1 or st.verified_votes >= 1)
        ]
        if alternatives:
            a2_stat = min(alternatives, key=lambda st: _rank(st, judge_on))

    return a1, a2_stat.grid


def _fallback_grids(
    test_input: Grid, train_out: list[Grid], suggested_shape: Optional[tuple[int, int]]
) -> list[Grid]:
    identity = [list(row) for row in test_input]
    color = modal_color(train_out)
    shape = suggested_shape or shape_of(test_input) or (1, 1)
    filled = [[color] * shape[1] for _ in range(shape[0])]
    if grid_key(filled) == grid_key(identity):
        filled = [[(color + 1) % 10] * shape[1] for _ in range(shape[0])]
    return [identity, filled]
