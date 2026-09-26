"""The disambiguation test suite.

Fixtures are synthetic ARC tasks where MULTIPLE programs pass all train pairs
but only one behavior is truly correct on the test input. These tests are the
offline proof (no GPU, no LLM) that the ranking layer prefers the truth.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness.disambiguate import (
    DisambiguationConfig,
    FeatureBundle,
    ast_size,
    compute_features,
    memorization_score,
    rank_and_select,
    disambiguate,
    score_bundle,
)
from harness.grid import grids_equal, to_grid
from harness.sandbox import LocalExecutor, Sandbox
from harness.types import Task
from tests.fixtures.synthetic import (
    F1_CORRECT,
    F1_OVERFIT,
    F1_TEST,
    F1_TRAIN,
    F3_CORRECT,
    F3_OVERFIT,
    F5_CORRECT,
    F5_IDENTITY,
    F5_TEST,
    F6_A,
    F6_B,
    F6_C,
    F6_TEST,
    FIXTURES_BY_NAME,
    make_pool,
    output_of,
)

EX = LocalExecutor()


def _rng():
    return np.random.default_rng(1234)


def _dis(fixture_name, executor=None, cfg=None):
    fx = FIXTURES_BY_NAME[fixture_name]
    pool = make_pool(fx, executor or EX)
    cfg = cfg or DisambiguationConfig(seed=7)
    result, attempts, bundle = disambiguate(
        fx.task, pool, executor or EX, cfg=cfg, rng=_rng(), fallback_outputs=list(fx.task.test)
    )
    return fx, pool, result, attempts, bundle


# ---------------------------------------------------------------------------
# fixture-driven end-to-end disambiguation
# ---------------------------------------------------------------------------


def test_f1_position_overfit_loses_to_geometric_rule():
    fx, pool, result, attempts, bundle = _dis("f1_position_overfit")
    a1, _ = attempts[0]
    assert grids_equal(a1, fx.true_outputs[0]), "equivariance must expose the hardcoded window"
    # the overfit program is ranked strictly below the correct one
    corr = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "correct")
    over = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "overfit")
    assert corr["score"] > over["score"]
    assert corr["equiv_train"] > over["equiv_train"]
    assert over["memorization"] == 0.0  # window indices are not memorized grids


def test_f2_color_identity_overfit_loses():
    fx, pool, result, attempts, bundle = _dis("f2_color_identity_overfit")
    a1, _ = attempts[0]
    assert grids_equal(a1, fx.true_outputs[0]), "color-perm pseudo-labels must expose identity overfit"
    corr = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "correct")
    over = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "overfit")
    assert corr["equiv_train"] > over["equiv_train"]


def test_f3_shape_memorizer_loses():
    fx, pool, result, attempts, bundle = _dis("f3_shape_memorizer")
    a1, _ = attempts[0]
    assert grids_equal(a1, fx.true_outputs[0])
    over = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "overfit")
    assert over["memorization"] == 1.0
    assert over["shape_violation"] == 1.0
    corr = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "correct")
    assert corr["memorization"] == 0.0
    assert corr["shape_violation"] == 0.0


def test_f4_consensus_majority_wins():
    fx, pool, result, attempts, bundle = _dis("f4_behavioral_consensus")
    a1, _ = attempts[0]
    assert grids_equal(a1, fx.true_outputs[0]), "cluster support must elect the 3-program majority"
    dev = next(f for f in result.features if pool.candidates[f["pool_index"]].source == "deviant")
    corrs = [f for f in result.features if pool.candidates[f["pool_index"]].source.startswith("correct")]
    assert len(corrs) == 3
    assert all(c["score"] > dev["score"] for c in corrs)
    # the three correct implementations share one cluster
    assert len({c["cluster"] for c in corrs}) == 1
    assert dev["cluster"] not in {c["cluster"] for c in corrs}


def test_f5_irreducible_ambiguity_still_hedges_pass2():
    """DOCUMENTED LIMITATION: symmetric train pairs cannot distinguish
    identity from fliplr; offline priors (MDL+equivariance) prefer identity.
    The hard requirement is that attempt_2 carries the OTHER behavior."""
    fx, pool, result, attempts, bundle = _dis("f5_symmetric_train_ambiguity")
    a1, a2 = attempts[0]
    flip_out = output_of(F5_CORRECT, F5_TEST)[0]
    id_out = F5_TEST[0]
    assert not grids_equal(a1, a2), "pass@2 hedge lost: attempts must differ"
    assert {a1.tobytes(), a2.tobytes()} == {flip_out.tobytes(), id_out.tobytes()}


def test_f6_soft_fallback_and_nondeterminism_penalty():
    fx, pool, result, attempts, bundle = _dis("f6_soft_fallback_nondeterminism")
    a1, a2 = attempts[0]
    exp_A = output_of(F6_A, F6_TEST)[0]
    exp_C = output_of(F6_C, F6_TEST)[0]
    assert grids_equal(a1, exp_A), "highest-soft deterministic program must lead"
    assert grids_equal(a2, exp_C), "attempt_2 = next-best distinct deterministic output"
    feat = {pool.candidates[f["pool_index"]].source: f for f in result.features}
    assert feat["nondet"]["determinism_ok"] is False
    assert feat["best_partial"]["determinism_ok"] is True
    assert feat["nondet"]["score"] < feat["weak"]["score"] < feat["best_partial"]["score"]


def test_disambiguation_is_deterministic():
    _, _, _, attempts1, _ = _dis("f1_position_overfit")
    _, _, _, attempts2, _ = _dis("f1_position_overfit")
    assert grids_equal(attempts1[0][0], attempts2[0][0])
    assert grids_equal(attempts1[0][1], attempts2[0][1])


def test_f8_three_way_separated_by_test_consistency():
    fx, pool, result, attempts, bundle = _dis("f8_three_way_test_consistency")
    a1, a2 = attempts[0]
    assert grids_equal(a1, fx.true_outputs[0]), (
        "test-side self-consistency + MDL must elect the pixelwise rule over "
        "two conditionally-recoloring deviants"
    )
    assert not grids_equal(a1, a2), "3 distinct behaviors -> attempts must differ"
    feat = {pool.candidates[f["pool_index"]].source: f for f in result.features}
    assert len({f["cluster"] for f in result.features}) == 3, "three distinct clusters expected"
    # train-side pseudo-labels are blind here (the 7-clause is inert on train)
    assert feat["correct"]["equiv_train"] == feat["deviant_toprow"]["equiv_train"] == 1.0
    # test-side consistency separates them
    assert feat["correct"]["equiv_test"] > feat["deviant_toprow"]["equiv_test"]
    assert feat["correct"]["equiv_test"] > feat["deviant_leftcol"]["equiv_test"]
    assert feat["correct"]["score"] > feat["deviant_toprow"]["score"]
    assert feat["correct"]["score"] > feat["deviant_leftcol"]["score"]


def test_f1_survives_real_subprocess_sandbox():
    """The whole battery must also work through the isolated subprocess sandbox."""
    with Sandbox(per_job_timeout=2.0) as sb:
        fx, pool, result, attempts, bundle = _dis("f1_position_overfit", executor=sb)
    assert grids_equal(attempts[0][0], fx.true_outputs[0])


def test_empty_pool_falls_back():
    task = Task("empty", F1_TRAIN, F1_TEST)
    from harness.pool import ProgramPool

    pool = ProgramPool()
    cfg = DisambiguationConfig()
    result, attempts, bundle = disambiguate(
        task, pool, EX, cfg=cfg, rng=_rng(), fallback_outputs=list(task.test)
    )
    assert grids_equal(attempts[0][0], F1_TEST[0])
    assert grids_equal(attempts[0][1], F1_TEST[0])


# ---------------------------------------------------------------------------
# injected-signal tests (LOO / adjudication are LLM-in-the-loop; the scoring
# path is pure and must respond to them correctly)
# ---------------------------------------------------------------------------


def _two_cluster_bundle(loo=(0.0, 0.0), adj=(0.0, 0.0), equiv=(1.0, 1.0)):
    g1 = to_grid([[1, 1], [1, 1]])
    g2 = to_grid([[2, 2], [2, 2]])
    return FeatureBundle(
        task_id="synthetic",
        candidate_indices=[0, 1],
        test_outputs=[[g1], [g2]],
        soft_mean=[1.0, 1.0],
        verified=[True, True],
        equiv_train=list(equiv),
        equiv_test=list(equiv),
        shape_violation=[0.0, 0.0],
        memorization=[0.0, 0.0],
        determinism_ok=[True, True],
        mdl_nodes=[20, 20],
        robustness=[1.0, 1.0],
        agreement=np.array([[1.0, 0.0], [0.0, 1.0]]),
        kept_syms=[],
        loo=list(loo),
        adj=list(adj),
        generation_order=[0, 1],
    )


def _task_1pair():
    x = to_grid([[1, 0], [0, 1]])
    return Task("synthetic", [(x, x)], [to_grid([[0, 1], [1, 0]])])


def test_loo_corroboration_flips_ranking():
    task = _task_1pair()
    cfg = DisambiguationConfig()
    b_plain = _two_cluster_bundle()
    res_plain, att_plain = rank_and_select(b_plain, cfg, task)
    assert grids_equal(att_plain[0][0], to_grid([[1, 1], [1, 1]]))  # tie -> first

    b_loo = _two_cluster_bundle(loo=(0.0, 1.0))
    res, att = rank_and_select(b_loo, cfg, task)
    assert grids_equal(att[0][0], to_grid([[2, 2], [2, 2]])), "LOO-corroborated behavior must win"

    b_loo_off = _two_cluster_bundle(loo=(0.0, 1.0))
    cfg_off = DisambiguationConfig()
    cfg_off.enable["loo"] = False
    _, att_off = rank_and_select(b_loo_off, cfg_off, task)
    assert grids_equal(att_off[0][0], to_grid([[1, 1], [1, 1]])), "ablation must neutralize LOO"


def test_adjudication_vote_flips_ranking():
    task = _task_1pair()
    b = _two_cluster_bundle(adj=(0.0, 1.0))
    _, att = rank_and_select(b, DisambiguationConfig(), task)
    assert grids_equal(att[0][0], to_grid([[2, 2], [2, 2]]))


def test_f7_pass2_diversity_vs_naive_top2():
    task = Task("synthetic", [(to_grid([[0]]), to_grid([[0]]))], [to_grid([[0]])])
    g_44a = np.zeros((4, 4), dtype=np.int64)
    g_44b = np.ones((4, 4), dtype=np.int64)
    g_22 = np.full((2, 2), 2, dtype=np.int64)
    bundle = FeatureBundle(
        task_id="f7",
        candidate_indices=[0, 1, 2],
        test_outputs=[[g_44a], [g_44b], [g_22]],
        soft_mean=[1.0, 1.0, 1.0],
        verified=[True, True, True],
        equiv_train=[1.0, 0.9, 0.8],  # score gaps of 0.08 < diversity bonus 0.15
        equiv_test=[1.0, 0.9, 0.8],
        shape_violation=[0.0, 0.0, 0.0],
        memorization=[0.0, 0.0, 0.0],
        determinism_ok=[True, True, True],
        mdl_nodes=[20, 20, 20],
        robustness=[1.0, 1.0, 1.0],
        agreement=np.eye(3),
        kept_syms=[],
        generation_order=[0, 1, 2],
    )
    cfg_full = DisambiguationConfig()
    _, att_full = rank_and_select(bundle, cfg_full, task)
    assert grids_equal(att_full[0][0], g_44a)
    assert grids_equal(att_full[0][1], g_22), "shape-diverse cluster must win attempt_2"

    import dataclasses

    bundle2 = dataclasses.replace(bundle)
    cfg_naive = DisambiguationConfig(policy="naive_top2")
    _, att_naive = rank_and_select(bundle2, cfg_naive, task)
    assert grids_equal(att_naive[0][1], g_44b), "naive top-2 ignores shape diversity"


def test_naive_vs_diverse_expected_gain_documented():
    """Quantify: under correlated errors the diverse pair strictly dominates.

    Model: P(top1 wrong) = 0.5. Naive top-2 shares the failure mode
    (rho = 0.9): P(win) = p1 + (1-p1)*p2*(1-rho). Diverse top-2 (rho = 0.3).
    """
    p1 = 0.5
    p2 = 0.4
    rho_corr, rho_div = 0.9, 0.3
    naive = p1 + (1 - p1) * p2 * (1 - rho_corr)
    diverse = p1 + (1 - p1) * p2 * (1 - rho_div)
    assert diverse - naive == pytest.approx((1 - p1) * p2 * (rho_corr - rho_div))
    assert diverse > naive


# ---------------------------------------------------------------------------
# static-analysis units
# ---------------------------------------------------------------------------


def test_ast_size_prefers_shorter():
    assert ast_size("def transform(g):\n    return g\n") < ast_size(
        "def transform(g):\n    out = g.copy()\n    for i in range(g.shape[0]):\n        for j in range(g.shape[1]):\n            out[i, j] = g[i, j]\n    return out\n"
    )


def test_ast_size_ignores_comments_and_docstrings():
    a = 'def transform(g):\n    """long docstring here"""  \n    return g\n'
    b = "def transform(g):\n    return g\n"
    assert ast_size(a) == pytest.approx(ast_size(b), abs=2)


def test_memorization_full_grid_literal():
    train = [(to_grid([[1, 2], [3, 4]]), to_grid([[9, 9], [9, 9]]))]
    code = """
import numpy as np
TABLE = {
    (2, 2): [[9, 9], [9, 9]],
}
def transform(grid):
    return np.array(TABLE[grid.shape])
"""
    assert memorization_score(code, train) == 1.0


def test_memorization_csv_string():
    train = [(to_grid([[1, 2, 3], [4, 5, 6], [7, 8, 0]]), to_grid([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))]
    code = '''
import numpy as np
MEM = "0,0,0,0,0,0,0,0,0"
def transform(grid):
    return np.array([[int(v) for v in MEM.split(",")]]).reshape(3, 3)
'''
    assert memorization_score(code, train) == 1.0


def test_memorization_row_pairs():
    x = to_grid([[1, 2, 3, 4, 5, 6], [7, 8, 9, 0, 1, 2]])
    y = to_grid([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    code = """
import numpy as np
R1 = [1, 2, 3, 4, 5, 6]
R2 = [7, 8, 9, 0, 1, 2]
def transform(grid):
    return np.stack([R1, R2])
"""
    assert memorization_score(code, [(x, y)]) == 0.75


def test_memorization_clean_program():
    train = [(to_grid([[1, 2], [3, 4]]), to_grid([[9, 9], [9, 9]]))]
    assert memorization_score(F1_CORRECT, train) == 0.0
    assert memorization_score(F1_OVERFIT, train) == 0.0  # window constants are not grids


def test_f3_overfit_flagged_f3_correct_clean():
    fx = FIXTURES_BY_NAME["f3_shape_memorizer"]
    train = fx.task.train
    assert memorization_score(F3_OVERFIT, train) == 1.0
    assert memorization_score(F3_CORRECT, train) == 0.0


# ---------------------------------------------------------------------------
# scoring-level ablation units
# ---------------------------------------------------------------------------


def test_equiv_ablation_neutralizes_symmetry_signal():
    task = _task_1pair()
    b = _two_cluster_bundle(equiv=(1.0, 0.0))
    cfg_on = DisambiguationConfig()
    s_on = score_bundle(b, cfg_on, task.n_pairs)
    assert s_on[0] > s_on[1]
    b2 = _two_cluster_bundle(equiv=(1.0, 0.0))
    cfg_off = DisambiguationConfig()
    cfg_off.enable["equiv"] = False
    s_off = score_bundle(b2, cfg_off, task.n_pairs)
    assert s_off[0] == pytest.approx(s_off[1])


def test_two_pair_tasks_boost_equivariance_weight():
    task2 = Task("t", [(to_grid([[0]]), to_grid([[0]]))] * 2, [to_grid([[0]])])
    task4 = Task("t", [(to_grid([[0]]), to_grid([[0]]))] * 4, [to_grid([[0]])])
    b1 = _two_cluster_bundle(equiv=(1.0, 0.0))
    b2 = _two_cluster_bundle(equiv=(1.0, 0.0))
    s2 = score_bundle(b1, DisambiguationConfig(), 2)
    s4 = score_bundle(b2, DisambiguationConfig(), 4)
    gap2 = s2[0] - s2[1]
    gap4 = s4[0] - s4[1]
    assert gap2 == pytest.approx(gap4 * 1.25)
