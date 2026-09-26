"""Synthetic ARC fixtures with KNOWN multiple-passing-programs.

Every fixture here is constructed so that >= 2 programs pass ALL train pairs
exactly, yet only one behavior is truly correct on the test input. The
`expect` field names the label whose output must be ranked first / submitted
as attempt_1. These fixtures are the offline proof that the disambiguation
layer works (no GPU, no LLM needed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from harness.grid import to_grid
from harness.types import Task


@dataclass
class Fixture:
    name: str
    task: Task
    programs: list[tuple[str, str]]  # (label, code); label "correct" is truth
    true_outputs: list[np.ndarray]
    expect: str = "correct"
    notes: str = ""


def _g(rows: list[str]) -> np.ndarray:
    return to_grid([[int(ch) for ch in row] for row in rows])


# ---------------------------------------------------------------------------
# F1 — position overfit vs. geometric rule (D4 equivariance discriminates)
# ---------------------------------------------------------------------------
# True rule: recolor every 2 as 8 (pixelwise, fully D4-equivariant).
# Train pairs keep all 2s inside the top-left 3x3 window, so a program that
# only recolors inside that window ALSO passes every train pair. The test
# input puts 2s outside the window.

F1_TRAIN = [
    (
        _g(["02000", "22000", "00000", "00000", "00000"]),
        _g(["08000", "88000", "00000", "00000", "00000"]),
    ),
    (
        _g(["20000", "00000", "00200", "00000", "00000"]),
        _g(["80000", "00000", "00800", "00000", "00000"]),
    ),
    (
        _g(["00000", "02200", "20000", "00000", "00000"]),
        _g(["00000", "08800", "80000", "00000", "00000"]),
    ),
]
# NOTE pair2 has a 2 at (2,2) - inside window rows 0..2 cols 0..2 ✓
F1_TEST = [_g(["00000", "00000", "00000", "00020", "02000"])]

F1_CORRECT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[grid == 2] = 8
    return out
"""

F1_OVERFIT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    sub = out[:3, :3]
    sub[grid[:3, :3] == 2] = 8
    return out
"""

# ---------------------------------------------------------------------------
# F2 — color-identity overfit vs role-free rule (color-perm equivariance)
# ---------------------------------------------------------------------------
# True rule: swap colors 2 and 3 (pixelwise). Train pairs place every 3 in
# the top half, so "2->3 everywhere; 3->2 only in the top half" passes all
# train pairs. The test input has 3s in the bottom half. Colors 2 and 3 have
# IDENTICAL usage profiles across train pairs, so the transposition (2 3) is
# a data-plausible color symmetry; only the true rule survives it.

F2_TRAIN = [
    (
        _g(["22003", "20003", "02000", "00002"]),
        _g(["33002", "30002", "03000", "00003"]),
    ),
    (
        # every 3 sits in the top half; counts: 3s=5, 2s=2 (mirrors pair 1)
        _g(["30330", "03003", "20000", "02000"]),
        _g(["20220", "02002", "30000", "03000"]),
    ),
]
F2_TEST = [_g(["20000", "02000", "00000", "00033"])]

F2_CORRECT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[grid == 2] = 3
    out[grid == 3] = 2
    return out
"""

F2_OVERFIT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    out[grid == 2] = 3
    h = grid.shape[0] // 2
    top = out[:h]
    top[grid[:h] == 3] = 2
    return out
"""

# ---------------------------------------------------------------------------
# F3 — shape-constant memorizer vs affine shape law
# ---------------------------------------------------------------------------
# True rule: keep the top half of the rows. Train shapes (4,4)->(2,4) and
# (6,4)->(3,4) pin the scale law out_h = in_h/2. The memorizer embeds the
# train outputs as literals keyed by input shape and falls back to identity:
# it passes train, violates the unanimous shape law on the (8,6) test input,
# and trips the literal-memorization screen.

_F3_OUT1 = ["1234", "5678"]
_F3_IN1 = ["1234", "5678", "9012", "3456"]
_F3_IN2 = ["1112", "2223", "3334", "4445", "5556", "6667"]
_F3_OUT2 = ["1112", "2223", "3334"]
F3_TRAIN = [
    (_g(_F3_IN1), _g(_F3_OUT1)),
    (_g(_F3_IN2), _g(_F3_OUT2)),
]
_F3_TEST_IN = ["121212", "343434", "565656", "787878", "909090", "121212", "343434", "565656"]
F3_TEST = [_g(_F3_TEST_IN)]

F3_CORRECT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid[: grid.shape[0] // 2, :].copy()
"""

F3_OVERFIT = (
    """import numpy as np

LOOKUP = {
    (4, 4): [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ],
    (6, 4): [
        [1, 1, 1, 2],
        [2, 2, 2, 3],
        [3, 3, 3, 4],
    ],
}

def transform(grid: np.ndarray) -> np.ndarray:
    key = (grid.shape[0], grid.shape[1])
    if key in LOOKUP:
        return np.array(LOOKUP[key])
    return grid.copy()
"""
)

# ---------------------------------------------------------------------------
# F4 — behavioral consensus: 3 independent correct implementations vs 1 deviant
# ---------------------------------------------------------------------------
# True rule: flip vertically. Train inputs contain no 7s; the deviant flips
# then recolors 7->8, passing train, differing on the test input (which has
# 7s). Cluster support must elect the majority behavior.

F4_TRAIN = [
    (
        _g(["123", "456", "708"]),
        _g(["708", "456", "123"]),
    ),
    (
        _g(["1100", "0220", "0030", "4444"]),
        _g(["4444", "0030", "0220", "1100"]),
    ),
]
# NOTE train pair 1 contains a 7 -> move it out of train so the deviant passes:
F4_TRAIN = [
    (
        _g(["123", "456", "908"]),
        _g(["908", "456", "123"]),
    ),
    (
        _g(["1100", "0220", "0030", "4444"]),
        _g(["4444", "0030", "0220", "1100"]),
    ),
]
F4_TEST = [_g(["712", "345", "608", "999"])]

F4_CORRECT_A = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid).copy()
"""

F4_CORRECT_B = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, copy=True)
    h = grid.shape[0]
    for r in range(h):
        out[r, :] = grid[h - 1 - r, :]
    return out
"""

F4_CORRECT_C = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid[::-1, :])
"""

F4_DEVIANT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid[::-1, :], copy=True)
    out[out == 7] = 8
    return out
"""

# ---------------------------------------------------------------------------
# F6 — no verified program: soft-score fallback + nondeterminism penalty
# ---------------------------------------------------------------------------
# No pool program passes all pairs. A: deterministic, soft ~0.78 (recolors bg
# only in row 0). B: higher soft (~0.89) but nondeterministic — it corrupts
# the bottom-right cell with a clock-driven value that is GUARANTEED different
# from the truth (so it never verifies by luck) and differs between runs.
# C: identity, soft ~0.56. Expected order: A > C > B (the nondeterminism
# penalty must demote B below both), attempts = (A output, C output).

F6_TRAIN = [
    (
        _g(["112", "112", "333"]),
        _g(["002", "002", "333"]),
    ),
    (
        _g(["441", "441", "222"]),
        _g(["001", "001", "222"]),
    ),
]
F6_TEST = [_g(["553", "553", "111"])]

F6_A = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    row0 = out[0]
    row0[grid[0] == bg] = 0
    return out
"""

F6_B = """import numpy as np
import time

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out[out == bg] = 0
    r, c = grid.shape[0] - 1, grid.shape[1] - 1
    v = (time.perf_counter_ns() // 1000) % 9
    out[r, c] = (int(grid[r, c]) + 1 + v) % 10
    return out
"""

F6_C = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid.copy()
"""

# ---------------------------------------------------------------------------
# F5 — DOCUMENTED LIMITATION: Bayes-irreducible ambiguity + pass@2 hedge
# ---------------------------------------------------------------------------
# True rule: flip left-right. BUT every train input is horizontally symmetric,
# so y_i == x_i and the identity program passes all train pairs too. No
# offline signal can identify the true rule from this data (the train set
# literally does not distinguish them); the MDL/equivariance priors prefer
# identity. What the layer MUST still guarantee: attempt_2 carries the other
# behavior (pass@2 diversity hedge), so the task is still winnable.

F5_TRAIN = [
    (
        _g(["12321", "04040", "00000"]),
        _g(["12321", "04040", "00000"]),
    ),
    (
        _g(["51115", "00200", "77077"]),
        _g(["51115", "00200", "77077"]),
    ),
]
F5_TEST = [_g(["12345", "00000", "98760"])]

F5_CORRECT = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid[:, ::-1])
"""

F5_IDENTITY = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid.copy()
"""

# ---------------------------------------------------------------------------
# F7 — pass@2 diversity selection (synthetic FeatureBundle, no sandbox)
# ---------------------------------------------------------------------------
# Built directly in test_disambiguate.py: three verified clusters with scores
# c1 > c2 > c3 where c3 has a DIFFERENT output shape. naive_top2 must pick c2
# for attempt_2; the full policy must pick c3 (shape-diversity bonus 0.15
# exceeds the c2-c3 score gap).


# ---------------------------------------------------------------------------
# F8 — three-way disagreement separated by TEST-side self-consistency
# ---------------------------------------------------------------------------
# True rule: recolor every 2 as 8 (pixelwise). Train contains no 7s, so two
# deviants that additionally recolor 7s (conditionally!) also pass all train
# pairs. Train-side pseudo-labels cannot see the difference (the 7-clause is
# inert on train); the TEST input contains 7s, and only the true rule stays
# self-consistent under D4 transforms of the test input (equiv_test), and it
# is also the shortest program (MDL). Three distinct behaviors -> attempt
# selection must actively choose, unlike 2-cluster fixtures.

F8_TRAIN = F1_TRAIN
F8_TEST = [_g(["77000", "00000", "00200", "00000", "00072"])]

F8_CORRECT = F1_CORRECT

F8_DEVIANT_TOPROW = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[grid == 2] = 8
    row0 = out[0]
    row0[grid[0] == 7] = 3
    return out
"""

F8_DEVIANT_LEFTCOL = """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    out[grid == 2] = 8
    col0 = out[:, 0]
    col0[grid[:, 0] == 7] = 4
    return out
"""

# ---------------------------------------------------------------------------
# assemble
# ---------------------------------------------------------------------------


def _true_output(code: str, test_inputs) -> list[np.ndarray]:
    ns: dict = {}
    exec(compile(code, "<truth>", "exec"), ns)
    return [np.asarray(ns["transform"](np.asarray(x, dtype=np.int64).copy()), dtype=np.int64) for x in test_inputs]


def build_fixtures() -> list[Fixture]:
    fixtures = []

    t1 = Task("fixture_f1_equivariance", F1_TRAIN, F1_TEST)
    fixtures.append(
        Fixture(
            name="f1_position_overfit",
            task=t1,
            programs=[("overfit", F1_OVERFIT), ("correct", F1_CORRECT)],
            true_outputs=_true_output(F1_CORRECT, F1_TEST),
            notes="D4 pseudo-labels expose the hardcoded top-left window",
        )
    )

    t2 = Task("fixture_f2_color_overfit", F2_TRAIN, F2_TEST)
    fixtures.append(
        Fixture(
            name="f2_color_identity_overfit",
            task=t2,
            programs=[("overfit", F2_OVERFIT), ("correct", F2_CORRECT)],
            true_outputs=_true_output(F2_CORRECT, F2_TEST),
            notes="data-plausible color transposition (2 3) exposes color-identity overfit",
        )
    )

    t3 = Task("fixture_f3_shape_law", F3_TRAIN, F3_TEST)
    fixtures.append(
        Fixture(
            name="f3_shape_memorizer",
            task=t3,
            programs=[("overfit", F3_OVERFIT), ("correct", F3_CORRECT)],
            true_outputs=_true_output(F3_CORRECT, F3_TEST),
            notes="literal-grid memorization screen + unanimous affine shape law",
        )
    )

    t4 = Task("fixture_f4_consensus", F4_TRAIN, F4_TEST)
    fixtures.append(
        Fixture(
            name="f4_behavioral_consensus",
            task=t4,
            programs=[
                ("deviant", F4_DEVIANT),
                ("correct", F4_CORRECT_A),
                ("correct_b", F4_CORRECT_B),
                ("correct_c", F4_CORRECT_C),
            ],
            true_outputs=_true_output(F4_CORRECT_A, F4_TEST),
            notes="cluster support elects the 3-program majority over 1 deviant",
        )
    )

    t6 = Task("fixture_f6_soft_fallback", F6_TRAIN, F6_TEST)
    fixtures.append(
        Fixture(
            name="f6_soft_fallback_nondeterminism",
            task=t6,
            programs=[("weak", F6_C), ("nondet", F6_B), ("best_partial", F6_A)],
            true_outputs=_true_output(F6_A, F6_TEST),  # not achievable; fallback test
            expect="best_partial",
            notes="no program verifies: rank by soft score, penalize hidden RNG",
        )
    )

    t5 = Task("fixture_f5_irreducible", F5_TRAIN, F5_TEST)
    fixtures.append(
        Fixture(
            name="f5_symmetric_train_ambiguity",
            task=t5,
            programs=[("identity_deviant", F5_IDENTITY), ("correct", F5_CORRECT)],
            true_outputs=_true_output(F5_CORRECT, F5_TEST),
            expect="correct",
            notes=(
                "DOCUMENTED LIMITATION: train pairs cannot distinguish identity "
                "from fliplr; offline priors prefer identity, but attempt_2 must "
                "carry the other behavior (pass@2 hedge)."
            ),
        )
    )

    t8 = Task("fixture_f8_three_way", F8_TRAIN, F8_TEST)
    fixtures.append(
        Fixture(
            name="f8_three_way_test_consistency",
            task=t8,
            programs=[
                ("deviant_toprow", F8_DEVIANT_TOPROW),
                ("deviant_leftcol", F8_DEVIANT_LEFTCOL),
                ("correct", F8_CORRECT),
            ],
            true_outputs=_true_output(F8_CORRECT, F8_TEST),
            notes="3 distinct verified behaviors; test-side equivariance + MDL separate them",
        )
    )
    return fixtures


FIXTURES = build_fixtures()
FIXTURES_BY_NAME = {f.name: f for f in FIXTURES}


def make_pool(fixture: Fixture, executor):
    """Evaluate every fixture program through the oracle and build a pool,
    exactly like the synthesis loop would (minus the LLM)."""
    from harness.oracle import evaluate_program, run_on_inputs
    from harness.pool import ProgramPool

    pool = ProgramPool()
    for label, code in fixture.programs:
        ev = evaluate_program(executor, code, fixture.task.train)
        outs = run_on_inputs(executor, code, fixture.task.test) if ev.verified else []
        pool.add(code, label, ev, outs)
    return pool


def output_of(code: str, grids) -> list[np.ndarray]:
    return _true_output(code, grids)
