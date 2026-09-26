#!/usr/bin/env python
"""Local evaluation runner for the H3 branch.

Runs the full synthesis + oracle + disambiguation pipeline over the ARC-AGI-2
evaluation split (ground truth available) with either MockLLM (offline,
plumbing/schema validation + ablation mechanics) or a live vLLM server, and
reports ABLATION numbers: the pass@2 score with each disambiguation signal
disabled, recomputed from cached feature bundles (no re-synthesis needed).

Examples
--------
  python scripts/eval_local.py --mock --limit 10
  python scripts/eval_local.py --split fixtures
  python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b
  python scripts/eval_local.py --ablate-only out/cache
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from harness.budget import Budget
from harness.data import load_solutions_for, load_tasks
from harness.disambiguate import DisambiguationConfig, FeatureBundle, rank_and_select
from harness.llm import MockLLM, VLLMClient, library_responder
from harness.sandbox import Sandbox
from harness.solver import SolverConfig, solve_all
from harness.submission import SubmissionWriter, score_submission, validate_submission
from harness.types import Grid, Task

DEFAULT_DATA = r"C:\Users\daive\Downloads\arc-agi-2\data"


# ---------------------------------------------------------------------------
# ablation configurations
# ---------------------------------------------------------------------------

def ablation_configs() -> dict[str, DisambiguationConfig]:
    cfgs: dict[str, DisambiguationConfig] = {"full": DisambiguationConfig()}
    for sig in ("equiv", "support", "mdl", "shape", "memorization", "determinism", "robust", "loo", "adj"):
        c = DisambiguationConfig()
        c.enable[sig] = False
        cfgs[f"no_{sig}"] = c
    # combined removals: single-signal ablations are masked by deliberate
    # signal overlap (each fixture is catchable by >=2 independent signals)
    c = DisambiguationConfig()
    c.enable["equiv"] = False
    c.enable["mdl"] = False
    cfgs["no_equiv_no_mdl"] = c
    c = DisambiguationConfig()
    for sig in ("equiv", "mdl", "shape", "memorization", "determinism", "robust", "loo", "adj"):
        c.enable[sig] = False
    cfgs["all_offline_priors_off"] = c
    naive = DisambiguationConfig()
    naive.policy = "naive_top2"
    cfgs["naive_top2_no_diversity"] = naive
    fv = DisambiguationConfig()
    fv.policy = "first_verified"
    cfgs["first_verified_poetiq"] = fv
    rnd = DisambiguationConfig()
    rnd.policy = "random"
    cfgs["random_baseline"] = rnd
    return cfgs


def attempts_to_data(tasks: list[Task], attempts_by_task: dict[str, list]) -> dict:
    data = {}
    for t in tasks:
        packs = []
        atts = attempts_by_task.get(t.task_id, [])
        for i, x in enumerate(t.test):
            if i < len(atts) and atts[i][0] is not None:
                a1, a2 = atts[i]
                g1 = np.asarray(a1).tolist()
                g2 = np.asarray(a2 if a2 is not None else a1).tolist()
            else:
                g1 = np.asarray(x).tolist()
                g2 = np.zeros_like(np.asarray(x)).tolist()
            packs.append({"attempt_1": g1, "attempt_2": g2})
        data[t.task_id] = packs
    return data


# ---------------------------------------------------------------------------
# fixture split (synthetic, meaningful ablation numbers offline)
# ---------------------------------------------------------------------------


def load_fixture_tasks() -> tuple[list[Task], dict, dict[str, list[str]]]:
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures.synthetic import FIXTURES  # type: ignore

    tasks, solutions, scripts = [], {}, {}
    for fx in FIXTURES:
        tasks.append(fx.task)
        solutions[fx.task.task_id] = [g.tolist() for g in fx.true_outputs]
        scripts[fx.task.task_id] = [code for _, code in fx.programs]
    return tasks, solutions, scripts


def make_fixture_llm(scripts_by_task: dict[str, list[str]], task_ids_in_order: list[str]) -> MockLLM:
    """MockLLM whose responses depend on which task is in the prompt.

    Task membership is detected by a per-task signature (the CSV of its first
    train input) embedded in the rendered prompt."""
    from harness.grid import format_csv

    sigs = {}
    for tid, codes in scripts_by_task.items():
        sigs[tid] = codes
    # signature = first train input CSV requires the task; recompute from FIXTURES
    sys.path.insert(0, str(ROOT / "tests"))
    from fixtures.synthetic import FIXTURES  # type: ignore

    task_sig: list[tuple[str, str, str, list[str], str]] = []
    for fx in FIXTURES:
        # solver/nudge/adjudication prompts contain the test input; LOO prompts
        # do not, so also key on the train input. F1/F8 share train grids AND
        # share the same correct code, so a train-only match is unambiguous
        # enough for the mock.
        test_sig = format_csv(fx.task.test[0]) if fx.task.test else ""
        train_sig = format_csv(fx.task.train[0][0])
        codes = [c for _, c in fx.programs]
        correct = next((c for lab, c in fx.programs if lab == "correct"), codes[-1])
        task_sig.append((fx.task.task_id, test_sig, train_sig, codes, correct))
    state = {"counts": {}}

    def responder(messages, call_index):
        content = messages[0]["content"]
        is_adjudication = "<program_1>" in content
        is_loo = "withheld" in content
        # pass 1: unique test-input signature (solver/nudge/adjudication prompts)
        entry = next((e for e in task_sig if e[1] and e[1] in content), None)
        if entry is None and is_loo:
            # pass 2: LOO prompts omit the test input; match on train grids
            entry = next((e for e in task_sig if e[2] in content), None)
        if entry is None:
            return "I cannot solve this.\n"
        tid, test_sig, train_sig, codes, correct = entry
        if is_adjudication or is_loo:
            # emulate a COMPETENT judge/synthesizer: it identifies the
            # truly correct rule (exercises the vote-injection mechanics)
            return f"Final answer.\n\n```python\n{correct}\n```\n"
        k = state["counts"].get(tid, 0)
        state["counts"][tid] = k + 1
        code = codes[k % len(codes)]
        return f"Hypothesis {k}.\n\n```python\n{code}\n```\n"

    return MockLLM(responder)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--split", default="evaluation", choices=["evaluation", "training", "fixtures"])
    ap.add_argument("--mock", action="store_true", help="use MockLLM (offline plumbing + ablation mechanics)")
    ap.add_argument("--base-url", default=None, help="vLLM OpenAI-compatible base URL")
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--task-ids", default=None, help="comma-separated subset")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=600.0, help="wall-clock budget")
    ap.add_argument("--per-task-tokens", type=int, default=75_000)
    ap.add_argument("--experts", type=int, default=2)
    ap.add_argument("--iterations", type=int, default=5)
    ap.add_argument("--no-adjudication", action="store_true")
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--sandbox-timeout", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="out")
    ap.add_argument("--cache-name", default="cache.pkl")
    ap.add_argument("--ablate-only", default=None, help="recompute ablations from a cache file, no solving")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / args.cache_name

    # ---------------- ablation-only mode ----------------
    if args.ablate_only:
        with open(args.ablate_only, "rb") as f:
            cache = pickle.load(f)
        tasks = cache["tasks"]
        bundles = cache["bundles"]
        solutions = cache.get("solutions")
        print_ablations(tasks, bundles, solutions)
        return 0

    # ---------------- load tasks ----------------
    if args.split == "fixtures":
        tasks, solutions, scripts = load_fixture_tasks()
        if args.limit:
            tasks = tasks[: args.limit]
    else:
        ids = args.task_ids.split(",") if args.task_ids else None
        tasks = load_tasks(args.data_dir, split=args.split, limit=args.limit, task_ids=ids)
        solutions = load_solutions_for(args.data_dir, args.split)

    # ---------------- LLM ----------------
    if args.mock:
        if args.split == "fixtures":
            llm = make_fixture_llm({t.task_id: scripts[t.task_id] for t in tasks}, [t.task_id for t in tasks])
        else:
            llm = MockLLM(library_responder())
    elif args.base_url:
        llm = VLLMClient(base_url=args.base_url, model=args.model)
    else:
        print("ERROR: pass --mock or --base-url", file=sys.stderr)
        return 2

    # ---------------- run ----------------
    budget = Budget(
        deadline_ts=time.time() + args.minutes * 60,
        per_task_tokens=args.per_task_tokens,
        total_tokens=24_000_000,
    )
    cfg = SolverConfig(
        num_experts=args.experts,
        max_iterations=args.iterations,
        sandbox_timeout=args.sandbox_timeout,
        use_adjudication=not args.no_adjudication,
        use_loo=not args.no_loo,
        seed=args.seed,
        disambiguation=DisambiguationConfig(seed=args.seed),
    )
    writer = SubmissionWriter(out_dir / "submission.json", tasks)
    bundles: dict[str, FeatureBundle] = {}

    def on_bundle(task: Task, pool, bundle: FeatureBundle) -> None:
        bundles[task.task_id] = bundle

    t0 = time.time()

    def executor_factory():
        return Sandbox(per_job_timeout=args.sandbox_timeout)

    def on_task(res):
        print(
            f"  [{res.task_id}] cands={res.n_candidates} verified={res.n_verified} "
            f"behaviors={res.n_distinct_verified_behaviors} calls={res.llm_calls} "
            f"tok={res.prompt_tokens + res.completion_tokens} wall={res.wall_s:.1f}s"
        )

    summary = solve_all(
        tasks, llm, budget, cfg,
        submission_writer=writer, workers=args.workers,
        executor_factory=executor_factory, on_task=on_task, on_bundle=on_bundle,
    )
    wall = time.time() - t0

    data = json.loads((out_dir / "submission.json").read_text())
    problems = validate_submission(data, tasks)
    print(f"\nsubmission valid: {not problems} ({len(problems)} problems)")
    if problems:
        print(problems[:10])

    report = None
    if solutions:
        report = score_submission(data, tasks, solutions)
        print(f"pass@2 score: {report['score']:.4f} ({report['outputs_hit']}/{report['outputs_total']})")
        n_ver = sum(1 for r in summary["results"] if r.n_verified > 0)
        ver_and_hit = 0
        ver_and_miss = 0
        for r in summary["results"]:
            if r.n_verified > 0:
                tid = r.task_id
                per = report["per_task"].get(tid, 0.0)
                if per >= 1.0:
                    ver_and_hit += 1
                elif per == 0.0:
                    ver_and_miss += 1
        print(f"tasks with >=1 verified program: {n_ver}/{len(tasks)}")
        if n_ver:
            print(f"  of those, solved (pass@2): {ver_and_hit}  verified-but-wrong: {ver_and_miss}")
            print(f"  disambiguation precision on verified set: {ver_and_hit / max(1, ver_and_hit + ver_and_miss):.3f}")

    print(
        f"llm_calls={summary['llm_calls']} prompt_tokens={summary['prompt_tokens']} "
        f"completion_tokens={summary['completion_tokens']} wall={wall:.1f}s"
    )

    with open(cache_path, "wb") as f:
        pickle.dump({"tasks": tasks, "bundles": bundles, "solutions": solutions, "summary": {k: v for k, v in summary.items() if k != "results"}}, f)
    print(f"feature cache written: {cache_path}")

    if solutions and bundles:
        print("\n=== ABLATIONS (recomputed from cached feature bundles) ===")
        print_ablations(tasks, bundles, solutions)
    return 0


def print_ablations(tasks: list[Task], bundles: dict[str, FeatureBundle], solutions: dict) -> None:
    rows = []
    cfgs = ablation_configs()
    scored_tasks = [t for t in tasks if t.task_id in bundles and t.task_id in (solutions or {})]
    for name, cfg in cfgs.items():
        attempts_by_task: dict[str, list] = {}
        for t in scored_tasks:
            bundle = bundles[t.task_id]
            _, attempts = rank_and_select(
                dataclasses.replace(bundle), cfg, t, fallback_outputs=list(t.test)
            )
            attempts_by_task[t.task_id] = attempts
        data = attempts_to_data(scored_tasks, attempts_by_task)
        rep = score_submission(data, scored_tasks, solutions)
        # attempt_1-only accuracy (shows which signals earn the FIRST slot)
        a1_hits = 0
        a1_total = 0
        for t in scored_tasks:
            gt = solutions.get(t.task_id) or []
            packs = data.get(t.task_id, [])
            for i, truth in enumerate(gt):
                a1_total += 1
                if i < len(packs) and packs[i]["attempt_1"] == truth:
                    a1_hits += 1
        rows.append((name, rep["score"], rep["outputs_hit"], rep["outputs_total"], a1_hits / max(1, a1_total)))
    w = max(len(r[0]) for r in rows)
    base = next((r for r in rows if r[0] == "full"), rows[0])
    print(f"{'config'.ljust(w)}  {'pass@2':>7}  {'hits':>9}  {'a1_acc':>7}  delta_vs_full")
    for name, score, hits, total, a1 in rows:
        delta = score - base[1]
        print(f"{name.ljust(w)}  {score:7.4f}  {hits:4d}/{total:<4d}  {a1:7.4f}  {delta:+.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
