"""The disambiguation layer — the intellectual core of branch H3.

Given a pool of programs that all passed (or partially passed) the train-pair
oracle but DISAGREE on the test input, decide what to submit as attempt_1 and
attempt_2 — without any further LLM calls (everything here is numpy + sandbox
re-execution, i.e. effectively free in the token budget).

Signals (each justified in HYPOTHESIS.md):

* S1 equivariance     — a verified program should commute with the symmetries
                        the task plausibly exhibits (D4 + data-plausible color
                        permutations). Free pseudo-labeled held-out data:
                        (s·x_i -> s·y_i). Two-stage gating keeps this sound:
                        (a) data-level plausibility (pseudo-pairs must not
                        contradict train pairs), (b) pool-majority resolvability
                        (a symmetry only counts when a support-weighted majority
                        of distinct verified behaviors respects it — otherwise
                        the true rule itself breaks it and everyone would be
                        penalized equally, e.g. trivially-equivariant identity
                        programs must not outrank the true rule).
* S1c shape law       — affine/scale/const/identity shape laws fitted on train
                        pairs; a program whose test output shape contradicts
                        EVERY consistent law is penalized.
* S2 LOO corroboration— programs whose test behavior is reproduced by
                        held-out-fold resyntheses that passed their held-out
                        pair (LLM-in-the-loop, injected as `loo_map`; the
                        scoring logic itself is pure).
* S3 MDL prior        — shorter AST (comments/docstrings stripped) = higher
                        prior (CompressARC insight).
* S3b memorization    — literal train grids (or >=2 full rows) embedded in the
                        source = near-definitive overfit; heavy penalty.
* S4 behavioral
   clustering         — pairwise output agreement over a probe battery
                        (test inputs, their D4/color-perm variants, single-cell
                        mutations, border padding); cluster support = number of
                        independent programs agreeing.
* S5 soft fallback    — verification failure never discards: unverified
                        programs stay ranked by soft score (Land: precision
                        ~97%, recall ~43% => strong positive, weak negative).
* S6 determinism      — nondeterministic programs (hidden RNG) are penalized.
* S7 adjudication vote— an LLM adjudicator program (gated, budgeted) that
                        passes verification adds a vote to the matching
                        behavior (injected as `adj_votes`).

pass@2-aware selection: attempt_2 must be a DIFFERENT grid than attempt_1
(duplicating attempt_1 is strictly dominated under exact-match scoring), with a
shape-diversity bonus so the two attempts hedge different shape hypotheses.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from .grid import (
    D4_ELEMENTS,
    apply_color_perm,
    apply_d4,
    consistent_shape_laws,
    d4_name,
    grid_key,
    grids_equal,
    pad_border,
    perturb_cells,
    plausible_color_perms,
    plausible_d4,
    shape_law_violation,
)
from .types import Candidate, ClusterScore, DisambiguationResult, Grid, Task

# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "verified": 2.0,      # gate-like bonus: verification is a strong positive
    "soft": 0.5,          # mean train cell accuracy (dominant when unverified)
    "support": 1.0,       # cluster support (behavioral consensus)
    "equiv": 0.8,         # symmetry equivariance (train pseudo-labels + test consistency)
    "shape": 1.5,         # penalty: contradicts every consistent shape law
    "memorization": 1.5,  # penalty: literal train grids in source
    "nondet": 1.0,        # penalty: nondeterministic outputs
    "mdl": 0.25,          # MDL prior (shorter = better)
    "robust": 0.2,        # fraction of probe battery that executes cleanly
    "loo": 0.6,           # LOO corroboration (when available)
    "adj": 0.6,           # adjudication vote (when available)
}

DEFAULT_ENABLE: dict[str, bool] = {
    "equiv": True,
    "shape": True,
    "memorization": True,
    "determinism": True,
    "mdl": True,
    "support": True,
    "robust": True,
    "loo": True,
    "adj": True,
}


@dataclass
class DisambiguationConfig:
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    enable: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_ENABLE))
    max_battery_programs: int = 8
    cluster_theta: float = 0.75          # probe agreement needed to share a cluster
    diversity_bonus: float = 0.15        # attempt_2 bonus for a different SHAPE
    two_pair_equiv_boost: float = 1.25   # 2-pair tasks overfit more -> lean on symmetry
    fallback_soft_boost: float = 3.0     # no verified program: likelihood must dominate priors
    n_mutations: int = 2
    det_runs: int = 3                    # extra repeats of the test input for the RNG check
    max_color_perms: int = 4
    policy: str = "full"                 # full | naive_top2 | first_verified | random
    seed: int = 0

    def weight(self, name: str) -> float:
        if name in self.enable and not self.enable[name]:
            return 0.0
        return float(self.weights.get(name, 0.0))


# ---------------------------------------------------------------------------
# static program analysis (pure, no execution)
# ---------------------------------------------------------------------------


def ast_size(code: str) -> int:
    """AST node count with docstrings stripped — an MDL proxy robust to
    whitespace/comment formatting."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return len(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                node.body = body[1:] or [ast.Pass()]
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            n += max(1, len(node.value) // 32)  # long strings cost a little
        else:
            n += 1
    return n


def memorization_score(code: str, train: Sequence[tuple[Grid, Grid]]) -> float:
    """1.0 when a full train grid appears as a literal; 0.75 when >=2 distinct
    full rows (len>=6) of one train grid appear; else 0.0."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 0.0

    def flatten_const(node) -> Optional[list[int]]:
        """Flatten nested list/tuple integer constants."""
        if isinstance(node, (ast.List, ast.Tuple)):
            out: list[int] = []
            for e in node.elts:
                sub = flatten_const(e)
                if sub is None:
                    return None
                out.extend(sub)
            return out
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, bool)) :
            return [int(node.value)]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            sub = flatten_const(node.operand)
            return [-v for v in sub] if sub is not None else None
        return None

    literal_seqs: list[list[int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.List, ast.Tuple)):
            flat = flatten_const(node)
            if flat and len(flat) >= 4:
                literal_seqs.append(flat)
    code_norm = re.sub(r"\s+", "", code)
    full_hit = 0.0
    row_hits: dict[int, set[tuple[int, ...]]] = {}
    for gi, (x, y) in enumerate(train):
        for g in (x, y):
            a = np.asarray(g)
            flat = a.reshape(-1).tolist()
            if len(flat) >= 4:
                # exact flattened grid as a literal
                if any(seq == flat for seq in literal_seqs):
                    full_hit = 1.0
                # CSV serialization inside a string constant
                csv = ",".join(str(v) for v in flat)
                if len(flat) >= 8 and csv in code_norm:
                    full_hit = 1.0
            if a.shape[1] >= 6:
                for row in a.tolist():
                    key = tuple(row)
                    if any(seq == list(key) for seq in literal_seqs):
                        row_hits.setdefault(gi, set()).add(key)
    if full_hit:
        return 1.0
    if any(len(v) >= 2 for v in row_hits.values()):
        return 0.75
    return 0.0


# ---------------------------------------------------------------------------
# feature computation (sandbox battery — still zero LLM tokens)
# ---------------------------------------------------------------------------


@dataclass
class FeatureBundle:
    """Everything the ranking needs; picklable so eval_local can cache it and
    run ablations offline without touching the sandbox again."""

    task_id: str
    candidate_indices: list[int]          # indices into pool.candidates (battery set)
    test_outputs: list[list[Optional[Grid]]]  # per battery candidate
    soft_mean: list[float]
    verified: list[bool]
    equiv_train: list[float]
    equiv_test: list[float]
    shape_violation: list[float]
    memorization: list[float]
    determinism_ok: list[bool]
    mdl_nodes: list[int]
    robustness: list[float]
    agreement: Optional[np.ndarray]       # pairwise probe agreement, battery set
    kept_syms: list[str]                  # names of gated symmetries (diagnostics)
    loo: list[float] = field(default_factory=list)
    adj: list[float] = field(default_factory=list)
    generation_order: list[int] = field(default_factory=list)  # battery idx -> pool order


def _probe_battery(
    task: Task, cfg: DisambiguationConfig, rng: np.random.Generator
) -> tuple[list[tuple[str, Grid, float]], list[str]]:
    """(label, grid, weight) probes for behavioral clustering, plus symmetry names."""
    probes: list[tuple[str, Grid, float]] = []
    sym_names: list[str] = []
    d4 = [e for e in D4_ELEMENTS]
    perms = plausible_color_perms(task.train, max_perms=cfg.max_color_perms)
    for t, x in enumerate(task.test):
        probes.append((f"test{t}", x, 1.0))
        for e in d4[1:]:
            probes.append((f"test{t}:{d4_name(e)}", apply_d4(x, e), 0.25))
        for pi, perm in enumerate(perms):
            probes.append((f"test{t}:perm{pi}", apply_color_perm(x, perm), 0.25))
        for m in range(cfg.n_mutations):
            probes.append((f"test{t}:mut{m}", perturb_cells(x, rng, 1), 0.25))
        probes.append((f"test{t}:pad", pad_border(x), 0.25))
    return probes, sym_names


def _battery_candidates(pool, task: Task, cfg: DisambiguationConfig) -> list[int]:
    """Pick up to max_battery_programs candidates: one representative (shortest
    code) per verified behavior group first, then extra members, then top
    partials. Preserves pool (generation) order in the returned list."""
    groups = pool.verified_behaviors()
    reps: list[int] = []
    extras: list[int] = []
    for key, members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if key == ("incomplete",):
            extras.extend(members)
            continue
        ordered = sorted(members, key=lambda i: ast_size(pool.candidates[i].code))
        reps.append(ordered[0])
        extras.extend(ordered[1:])
    partials = pool.best_partial(n=cfg.max_battery_programs)
    chosen = reps[: cfg.max_battery_programs]
    for i in extras + partials:
        if len(chosen) >= cfg.max_battery_programs:
            break
        if i not in chosen:
            chosen.append(i)
    return sorted(chosen)


def compute_features(
    task: Task,
    pool,
    executor,
    cfg: DisambiguationConfig,
    rng: Optional[np.random.Generator] = None,
    loo_map: Optional[dict] = None,
    adj_votes: Optional[dict] = None,
) -> FeatureBundle:
    """Run the full offline battery. `executor` is a Sandbox (or LocalExecutor).

    loo_map:  {behavior_key_tuple: corroboration in [0,1]} keyed by the group
              keys of ProgramPool.verified_behaviors().
    adj_votes: {behavior_key_tuple: vote count}.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    battery = _battery_candidates(pool, task, cfg)
    n = len(battery)
    empty = FeatureBundle(
        task_id=task.task_id, candidate_indices=battery, test_outputs=[], soft_mean=[],
        verified=[], equiv_train=[], equiv_test=[], shape_violation=[], memorization=[],
        determinism_ok=[], mdl_nodes=[], robustness=[], agreement=None, kept_syms=[],
        generation_order=battery,
    )
    if n == 0:
        return empty

    cands = [pool.candidates[i] for i in battery]

    # ---- symmetries: data-level plausibility ------------------------------
    d4_plausible = plausible_d4(task.train)
    perm_plausible = plausible_color_perms(task.train, max_perms=cfg.max_color_perms)

    # ---- phase 1 jobs: train pseudo-labels for ALL plausible syms ---------
    # job layout helper
    jobs: list[tuple[str, Grid]] = []
    job_tags: list[tuple] = []  # (kind, cand_i, sym_i, pair_k)

    syms = [("d4", e) for e in d4_plausible] + [("perm", p) for p in perm_plausible]

    def apply_sym(g: Grid, sym) -> Grid:
        kind, s = sym
        return apply_d4(g, s) if kind == "d4" else apply_color_perm(g, s)

    verified_idx = [i for i in range(n) if cands[i].verified]
    gate_set = verified_idx if verified_idx else list(range(n))
    if not verified_idx:
        # fallback mode: equivariance inactive (see per-candidate features) -
        # skip all pseudo-label jobs to save sandbox time
        syms = []
    for ci in gate_set:
        for si, sym in enumerate(syms):
            for k, (x, y) in enumerate(task.train):
                jobs.append((cands[ci].code, apply_sym(x, sym)))
                job_tags.append(("pseudo", ci, si, k))
    results = executor.run_batch(jobs) if jobs else []

    # pseudo-pass[c][s] = all pairs exact
    pseudo_pass: dict[tuple[int, int], bool] = {}
    pseudo_cell: dict[tuple[int, int], list[bool]] = {}
    for res, tag in zip(results, job_tags):
        _, ci, si, k = tag
        x, y = task.train[k]
        expected = apply_sym(y, syms[si])
        ok = res.ok and grids_equal(res.output, expected)
        pseudo_cell.setdefault((ci, si), []).append(ok)
    for key, cells in pseudo_cell.items():
        pseudo_pass[key] = all(cells)

    # ---- pool-majority resolvability gate ---------------------------------
    # Group gated candidates by test-output behavior; weight = min(3, size).
    beh_groups: dict[tuple, list[int]] = {}
    for ci in gate_set:
        key = tuple(
            grid_key(o) if o is not None else None for o in cands[ci].test_outputs
        )
        beh_groups.setdefault(key, []).append(ci)
    kept_syms_idx: list[int] = []
    if verified_idx:
        total_w = 0.0
        pass_w: dict[int, float] = {si: 0.0 for si in range(len(syms))}
        for key, members in beh_groups.items():
            w = float(min(3, len(members)))
            total_w += w
            for si in range(len(syms)):
                n_pass = sum(1 for ci in members if pseudo_pass.get((ci, si), False))
                if n_pass * 2 >= len(members):
                    pass_w[si] += w
        for si in range(len(syms)):
            if total_w == 0 or pass_w[si] / total_w >= 0.5:
                kept_syms_idx.append(si)
    else:
        kept_syms_idx = list(range(len(syms)))

    kept_names = [
        (d4_name(syms[si][1]) if syms[si][0] == "d4" else f"perm{syms[si][1]}")
        for si in kept_syms_idx
    ]

    # ---- phase 2 jobs: equiv on kept syms (all battery cands), probes,
    #      determinism, plain test outputs -----------------------------------
    jobs2: list[tuple[str, Grid]] = []
    tags2: list[tuple] = []
    probes, _ = _probe_battery(task, cfg, rng)
    for ci in range(n):
        code = cands[ci].code
        for si in kept_syms_idx:
            sym = syms[si]
            if ci not in gate_set or pseudo_pass.get((ci, si)) is None:
                for k, (x, y) in enumerate(task.train):
                    jobs2.append((code, apply_sym(x, sym)))
                    tags2.append(("pseudo", ci, si, k))
            for t, x in enumerate(task.test):
                jobs2.append((code, apply_sym(x, sym)))
                tags2.append(("symtest", ci, si, t))
        for pi, (label, g, w) in enumerate(probes):
            jobs2.append((code, g))
            tags2.append(("probe", ci, pi, 0))
        for t, x in enumerate(task.test):
            for run_k in range(cfg.det_runs):
                jobs2.append((code, x))
                tags2.append(("det", ci, run_k, t))
    results2 = executor.run_batch(jobs2) if jobs2 else []

    # collect all phase-2 outputs in one pass
    symtest_out: dict[tuple[int, int, int], Optional[Grid]] = {}
    probe_out: dict[tuple[int, int], Optional[Grid]] = {}
    det_out: dict[tuple[int, int, int], Optional[Grid]] = {}
    for res, tag in zip(results2, tags2):
        kind, ci, si, k = tag
        out = res.output if res.ok else None
        if kind == "pseudo":
            # deferred pseudo-label runs for non-gate candidates on kept syms
            expected = apply_sym(task.train[k][1], syms[si])
            ok = res.ok and grids_equal(res.output, expected)
            pseudo_cell.setdefault((ci, si), []).append(ok)
            pseudo_pass[(ci, si)] = all(pseudo_cell[(ci, si)])
        elif kind == "symtest":
            symtest_out[(ci, si, k)] = out
        elif kind == "probe":
            probe_out[(ci, si)] = out
        elif kind == "det":
            det_out[(ci, si, k)] = out

    # plain test outputs come from probe index of label test{t}
    test_outputs: list[list[Optional[Grid]]] = []
    for ci in range(n):
        outs = []
        for t in range(len(task.test)):
            pi = next(i for i, (label, _, _) in enumerate(probes) if label == f"test{t}")
            outs.append(probe_out.get((ci, pi)))
        test_outputs.append(outs)

    # ---- per-candidate features -------------------------------------------
    # Equivariance is a generalization REFINER among train-exact programs. In
    # fallback mode (nothing verified) it is disabled: trivially-equivariant
    # programs (identity-like) would otherwise outrank high-soft-score ones,
    # and pool-majority gating is meaningless without verified behaviors.
    equiv_active = bool(verified_idx)
    equiv_train: list[float] = []
    equiv_test: list[float] = []
    determinism_ok: list[bool] = []
    robustness: list[float] = []
    shape_viol: list[float] = []
    memor: list[float] = []
    mdl: list[int] = []
    laws = consistent_shape_laws(task.train)
    for ci in range(n):
        cand = cands[ci]
        # equiv train over kept syms
        if equiv_active:
            vals = [1.0 if pseudo_pass.get((ci, si), False) else 0.0 for si in kept_syms_idx]
            equiv_train.append(float(np.mean(vals)) if vals else 0.0)
        else:
            equiv_train.append(0.0)
        # equiv test: P(s·x_t) == s·P(x_t)
        tt = []
        if equiv_active:
            for si in kept_syms_idx:
                sym = syms[si]
                for t in range(len(task.test)):
                    po = symtest_out.get((ci, si, t))
                    base = test_outputs[ci][t] if t < len(test_outputs[ci]) else None
                    if po is None or base is None:
                        tt.append(0.0)
                        continue
                    expected = apply_sym(base, sym)
                    tt.append(1.0 if grids_equal(po, expected) else 0.0)
        equiv_test.append(float(np.mean(tt)) if tt else 0.0)
        # determinism: repeated test runs identical to the probe-run output
        det_ok = True
        for t in range(len(task.test)):
            base = test_outputs[ci][t] if t < len(test_outputs[ci]) else None
            for run_k in range(cfg.det_runs):
                a = det_out.get((ci, run_k, t))
                if a is not None and base is not None and not grids_equal(a, base):
                    det_ok = False
        determinism_ok.append(det_ok)
        # robustness: fraction of probe runs that produced output
        pruns = [probe_out.get((ci, pi)) is not None for pi in range(len(probes))]
        robustness.append(float(np.mean(pruns)) if pruns else 0.0)
        # shape-law violation fraction over test inputs
        if laws:
            viol = []
            for t in range(len(task.test)):
                o = test_outputs[ci][t] if t < len(test_outputs[ci]) else None
                viol.append(
                    1.0 if shape_law_violation(laws, task.test[t].shape, None if o is None else o.shape) else 0.0
                )
            shape_viol.append(float(np.mean(viol)))
        else:
            shape_viol.append(0.0)
        memor.append(memorization_score(cand.code, task.train))
        mdl.append(ast_size(cand.code))

    # ---- agreement matrix over probes --------------------------------------
    agreement = np.eye(n)
    weights = np.array([w for (_, _, w) in probes])
    for a in range(n):
        for b in range(a + 1, n):
            num = 0.0
            den = 0.0
            for pi in range(len(probes)):
                oa = probe_out.get((a, pi))
                ob = probe_out.get((b, pi))
                if oa is None or ob is None:
                    continue
                den += weights[pi]
                if grids_equal(oa, ob):
                    num += weights[pi]
            agree = num / den if den > 0 else 0.0
            agreement[a, b] = agreement[b, a] = agree

    # ---- LOO / adjudication injection ---------------------------------------
    groups = pool.verified_behaviors()
    loo_vals = [0.0] * n
    adj_vals = [0.0] * n
    if loo_map or adj_votes:
        for ci in range(n):
            idx = battery[ci]
            cand = pool.candidates[idx]
            if not cand.verified or any(o is None for o in cand.test_outputs):
                continue
            key = tuple(grid_key(o) for o in cand.test_outputs)
            if loo_map:
                loo_vals[ci] = float(loo_map.get(key, 0.0))
            if adj_votes:
                adj_vals[ci] = float(adj_votes.get(key, 0.0))

    return FeatureBundle(
        task_id=task.task_id,
        candidate_indices=battery,
        test_outputs=test_outputs,
        soft_mean=[c.soft_mean for c in cands],
        verified=[c.verified for c in cands],
        equiv_train=equiv_train,
        equiv_test=equiv_test,
        shape_violation=shape_viol,
        memorization=memor,
        determinism_ok=determinism_ok,
        mdl_nodes=mdl,
        robustness=robustness,
        agreement=agreement,
        kept_syms=kept_names,
        loo=loo_vals,
        adj=adj_vals,
        generation_order=list(range(n)),
    )


# ---------------------------------------------------------------------------
# ranking + selection (pure — used by ablations)
# ---------------------------------------------------------------------------


def _mdl_norm(nodes: Sequence[int]) -> list[float]:
    if not nodes:
        return []
    lo, hi = min(nodes), max(nodes)
    if hi == lo:
        return [1.0] * len(nodes)
    return [1.0 - (v - lo) / (hi - lo) for v in nodes]


def _cluster(bundle: FeatureBundle, cfg: DisambiguationConfig) -> list[list[int]]:
    """Greedy clustering by probe agreement (threshold cluster_theta)."""
    n = len(bundle.candidate_indices)
    if n == 0:
        return []
    clusters: list[list[int]] = []
    order = sorted(
        range(n),
        key=lambda i: (not bundle.verified[i], -bundle.soft_mean[i], bundle.mdl_nodes[i]),
    )
    A = bundle.agreement if bundle.agreement is not None else np.eye(n)
    for i in order:
        placed = False
        for cl in clusters:
            rep = cl[0]
            if A[i, rep] >= cfg.cluster_theta:
                cl.append(i)
                placed = True
                break
        if not placed:
            clusters.append([i])
    return clusters


def clusters_of(bundle: FeatureBundle, cfg: DisambiguationConfig) -> tuple[list[list[int]], dict[int, int]]:
    clusters = _cluster(bundle, cfg)
    cid_of = {i: c for c, cl in enumerate(clusters) for i in cl}
    return clusters, cid_of


def score_bundle(bundle: FeatureBundle, cfg: DisambiguationConfig, n_pairs: int) -> list[float]:
    n = len(bundle.candidate_indices)
    if n == 0:
        return []
    W = cfg.weight
    mdl_n = _mdl_norm(bundle.mdl_nodes)
    clusters, _ = clusters_of(bundle, cfg)
    support = [0.0] * n
    for cl in clusters:
        for i in cl:
            support[i] = float(len(cl))
    support_norm = [s / n for s in support]

    equiv_boost = cfg.two_pair_equiv_boost if n_pairs <= 2 else 1.0
    any_verified = any(bundle.verified)
    soft_w = W("soft") * (cfg.fallback_soft_boost if not any_verified else 1.0)
    scores: list[float] = []
    for i in range(n):
        s = 0.0
        s += W("verified") * (1.0 if bundle.verified[i] else 0.0)
        s += soft_w * bundle.soft_mean[i]
        if cfg.enable.get("support", True):
            s += W("support") * support_norm[i]
        if cfg.enable.get("equiv", True):
            equiv = 0.6 * bundle.equiv_train[i] + 0.4 * bundle.equiv_test[i]
            s += W("equiv") * equiv * equiv_boost
        if cfg.enable.get("shape", True):
            s -= W("shape") * bundle.shape_violation[i]
        if cfg.enable.get("memorization", True):
            s -= W("memorization") * bundle.memorization[i]
        if cfg.enable.get("determinism", True) and not bundle.determinism_ok[i]:
            s -= W("nondet")
        if cfg.enable.get("mdl", True):
            s += W("mdl") * mdl_n[i]
        if cfg.enable.get("robust", True):
            s += W("robust") * bundle.robustness[i]
        if cfg.enable.get("loo", True):
            s += W("loo") * (bundle.loo[i] if bundle.loo else 0.0)
        if cfg.enable.get("adj", True):
            s += W("adj") * (bundle.adj[i] if bundle.adj else 0.0)
        scores.append(s)
    return scores


def rank_and_select(
    bundle: FeatureBundle,
    cfg: DisambiguationConfig,
    task: Task,
    fallback_outputs: Optional[Sequence[Optional[Grid]]] = None,
) -> tuple[DisambiguationResult, list[tuple[Optional[Grid], Optional[Grid]]]]:
    """Pure ranking + pass@2-aware attempt selection from a FeatureBundle.

    fallback_outputs: per-test-input grids used only when NO candidate produced
    any usable output (schema must always be satisfiable).
    """
    n = len(bundle.candidate_indices)
    notes: list[str] = []
    scores = score_bundle(bundle, cfg, task.n_pairs)
    clusters, cid_of = clusters_of(bundle, cfg)

    cluster_scores: list[ClusterScore] = []
    for cl in clusters:
        best = max(cl, key=lambda i: scores[i])
        outs = bundle.test_outputs[best]
        cluster_scores.append(
            ClusterScore(
                rep_index=best,
                test_output=outs[0] if outs else None,
                score=scores[best],
                support=float(len(cl)),
                members=list(cl),
                features={
                    "soft": bundle.soft_mean[best],
                    "verified": float(bundle.verified[best]),
                    "equiv_train": bundle.equiv_train[best],
                    "equiv_test": bundle.equiv_test[best],
                    "shape_violation": bundle.shape_violation[best],
                    "memorization": bundle.memorization[best],
                    "mdl_nodes": float(bundle.mdl_nodes[best]),
                    "robust": bundle.robustness[best],
                    "loo": bundle.loo[best] if bundle.loo else 0.0,
                    "adj": bundle.adj[best] if bundle.adj else 0.0,
                },
            )
        )
    cluster_scores.sort(key=lambda c: -c.score)

    n_test = task.n_test
    attempts: list[tuple[Optional[Grid], Optional[Grid]]] = []

    if cfg.policy == "random" and n:
        rng = np.random.default_rng(cfg.seed)
        perm = list(rng.permutation(n))
        for t in range(n_test):
            outs = [bundle.test_outputs[i][t] for i in perm if t < len(bundle.test_outputs[i])]
            outs = [o for o in outs if o is not None]
            a1 = outs[0] if outs else None
            a2 = next((o for o in outs[1:] if not grids_equal(o, a1)), a1)
            attempts.append((a1, a2))
    elif cfg.policy == "first_verified":
        # Poetiq-style baseline: first verified program in generation order wins.
        order = sorted(range(n), key=lambda i: (not bundle.verified[i], bundle.generation_order[i]))
        for t in range(n_test):
            outs = [
                bundle.test_outputs[i][t]
                for i in order
                if t < len(bundle.test_outputs[i]) and bundle.test_outputs[i][t] is not None
            ]
            a1 = outs[0] if outs else None
            a2 = next((o for o in outs[1:] if not grids_equal(o, a1)), a1)
            attempts.append((a1, a2))
    else:
        for t in range(n_test):
            ranked_outs: list[tuple[float, Optional[Grid], tuple]] = []
            for cs in cluster_scores:
                # best member of this cluster that actually produced output for test t
                o = None
                sc = cs.score
                for m in sorted(cs.members, key=lambda i: -scores[i]):
                    cand_o = bundle.test_outputs[m][t] if t < len(bundle.test_outputs[m]) else None
                    if cand_o is not None:
                        o = cand_o
                        sc = scores[m]
                        break
                if o is not None:
                    ranked_outs.append((sc, o, o.shape))
            if not ranked_outs:
                for i in sorted(range(n), key=lambda i: -scores[i]):
                    o = bundle.test_outputs[i][t] if t < len(bundle.test_outputs[i]) else None
                    if o is not None:
                        ranked_outs.append((scores[i], o, o.shape))
            if not ranked_outs:
                fb = fallback_outputs[t] if fallback_outputs and t < len(fallback_outputs) else None
                attempts.append((fb, fb))
                notes.append(f"test{t}: no candidate output; fallback grid used")
                continue
            a1_score, a1, a1_shape = ranked_outs[0]
            # attempt_2: best DIFFERENT grid; +diversity bonus when shape differs
            a2 = None
            if cfg.policy == "naive_top2":
                for sc, o, shp in ranked_outs[1:]:
                    if not grids_equal(o, a1):
                        a2 = o
                        break
            else:
                best2 = None
                best2_val = None
                for sc, o, shp in ranked_outs[1:]:
                    if grids_equal(o, a1):
                        continue
                    val = sc + (cfg.diversity_bonus if shp != a1_shape else 0.0)
                    if best2_val is None or val > best2_val:
                        best2_val, best2 = val, o
                a2 = best2
            if a2 is None:
                a2 = a1
                notes.append(f"test{t}: only one distinct output; attempt_2 duplicates attempt_1")
            attempts.append((a1, a2))

    features_out = []
    for i in range(n):
        features_out.append(
            {
                "pool_index": bundle.candidate_indices[i],
                "score": scores[i] if i < len(scores) else 0.0,
                "cluster": cid_of.get(i, -1),
                "verified": bundle.verified[i],
                "soft_mean": bundle.soft_mean[i],
                "equiv_train": bundle.equiv_train[i],
                "equiv_test": bundle.equiv_test[i],
                "shape_violation": bundle.shape_violation[i],
                "memorization": bundle.memorization[i],
                "determinism_ok": bundle.determinism_ok[i],
                "mdl_nodes": bundle.mdl_nodes[i],
                "robust": bundle.robustness[i],
                "loo": bundle.loo[i] if bundle.loo else 0.0,
                "adj": bundle.adj[i] if bundle.adj else 0.0,
            }
        )

    return DisambiguationResult(
        attempt_1=attempts[0][0] if attempts else None,
        attempt_2=attempts[0][1] if attempts else None,
        clusters=cluster_scores,
        features=features_out,
        agreement=bundle.agreement,
        notes=notes,
    ), attempts


def disambiguate(
    task: Task,
    pool,
    executor,
    cfg: Optional[DisambiguationConfig] = None,
    rng: Optional[np.random.Generator] = None,
    loo_map: Optional[dict] = None,
    adj_votes: Optional[dict] = None,
    fallback_outputs: Optional[Sequence[Optional[Grid]]] = None,
) -> tuple[DisambiguationResult, list[tuple[Optional[Grid], Optional[Grid]]], FeatureBundle]:
    """Full pipeline: features (sandbox battery) -> rank -> select."""
    cfg = cfg or DisambiguationConfig()
    bundle = compute_features(task, pool, executor, cfg, rng=rng, loo_map=loo_map, adj_votes=adj_votes)
    result, attempts = rank_and_select(bundle, cfg, task, fallback_outputs=fallback_outputs)
    return result, attempts, bundle
