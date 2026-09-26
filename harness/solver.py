"""Per-task synthesis loop + full-run orchestration.

Loop shape (Poetiq-derived, but pool-preserving): Poetiq RETURNS at the first
train-verified program; we keep synthesizing to build a *verified set*, then
disambiguate offline. Rationale: many distinct programs pass all train pairs
yet disagree on the test input, and pass@2 rewards picking the right two —
that decision, not generation, is this branch's contribution.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from .budget import Budget
from .disambiguate import DisambiguationConfig, FeatureBundle, disambiguate
from .extraction import pick_program
from .grid import grid_key, grids_equal
from .llm import LLMClient
from .oracle import build_feedback, evaluate_program, run_on_inputs
from .pool import ProgramPool
from .prompts import (
    build_adjudication_messages,
    build_loo_messages,
    build_solver_messages,
    build_verified_nudge_messages,
)
from .sandbox import Sandbox
from .types import Grid, Task, TaskResult


@dataclass
class SolverConfig:
    num_experts: int = 2
    max_iterations: int = 5          # per expert
    max_solutions_feedback: int = 5
    temperature: float = 1.0
    reasoning_effort: str = "low"    # NEVER 'high' (HTTP 500 on Qwen3.8)
    max_completion_tokens: int = 3072
    stop_distinct_behaviors: int = 4 # early stop: enough verified diversity
    patience_no_new: int = 3         # stop after N consecutive fruitless calls
    min_calls: int = 4               # explore at least this many calls
    sandbox_timeout: float = 2.0
    use_adjudication: bool = True
    adjudication_max_calls: int = 1
    use_loo: bool = True
    loo_max_folds: int = 3
    loo_min_pairs: int = 3           # 2-pair tasks: a fold leaves 1 pair (useless)
    seed: int = 0
    disambiguation: DisambiguationConfig = field(default_factory=DisambiguationConfig)


def _estimate_tokens(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages) // 4


def solve_task(
    task: Task,
    llm: LLMClient,
    executor: Sandbox,
    budget: Optional[Budget],
    cfg: SolverConfig,
    time_cap_s: Optional[float] = None,
    on_bundle: Optional[Callable[[Task, ProgramPool, FeatureBundle], None]] = None,
) -> TaskResult:
    t_start = time.time()
    import zlib

    rng = np.random.default_rng(cfg.seed + zlib.crc32(task.task_id.encode()) % 10_000)
    pool = ProgramPool()
    calls = 0
    prompt_tokens = 0
    completion_tokens = 0
    no_new = 0
    last_verified_behavior_count = 0

    def _budget_ok(messages) -> bool:
        if budget is None:
            return True
        return budget.can_call(task.task_id, _estimate_tokens(messages) + cfg.max_completion_tokens)

    def _charge(resp) -> None:
        nonlocal prompt_tokens, completion_tokens
        prompt_tokens += resp.prompt_tokens
        completion_tokens += resp.completion_tokens
        if budget is not None:
            budget.charge(task.task_id, resp.prompt_tokens, resp.completion_tokens)

    def _add(code: str, source: str) -> bool:
        """Evaluate + add to pool. Returns True when behavior is new."""
        ev = evaluate_program(executor, code, task.train)
        test_outs: list[Optional[Grid]] = []
        if ev.verified:
            test_outs = run_on_inputs(executor, code, task.test)
        idx, is_new = pool.add(code, source, ev, test_outs)
        return is_new

    # ---- synthesis loop -----------------------------------------------------
    for expert in range(cfg.num_experts):
        for it in range(cfg.max_iterations):
            if budget is not None and budget.expired():
                break
            if time_cap_s is not None and time.time() - t_start > time_cap_s:
                break
            verified_groups = pool.verified_behaviors()
            n_beh = len([k for k in verified_groups if k != ("incomplete",)])
            if n_beh >= cfg.stop_distinct_behaviors and calls >= cfg.min_calls:
                break
            if calls >= cfg.min_calls and no_new >= cfg.patience_no_new:
                break

            verified = pool.verified()
            if verified:
                rep = max(verified, key=lambda i: len(pool.candidates[i].code) * -1)
                messages = build_verified_nudge_messages(
                    task, pool.candidates[rep].code, iteration=it, seed=cfg.seed + expert * 97
                )
            else:
                partials = []
                for i in pool.best_partial(n=cfg.max_solutions_feedback):
                    c = pool.candidates[i]
                    fb, _ = build_feedback(c.eval, task.train)
                    partials.append((c.code, fb, c.soft_mean))
                messages = build_solver_messages(
                    task,
                    iteration=it,
                    variant="A" if (expert + it) % 2 == 0 else "B",
                    partials=partials,
                    max_partials=cfg.max_solutions_feedback,
                    shuffle_seed=cfg.seed + expert * 97,
                )
            if not _budget_ok(messages):
                break
            try:
                resp = llm.chat(
                    messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_completion_tokens,
                    reasoning_effort=cfg.reasoning_effort,
                    seed=cfg.seed + expert * 1000 + it,
                )
            except Exception:
                no_new += 1
                calls += 1
                continue
            _charge(resp)
            calls += 1
            code = pick_program(resp.text)
            if code is None:
                no_new += 1
                continue
            is_new = _add(code, f"e{expert}i{it}")
            groups_now = len([k for k in pool.verified_behaviors() if k != ("incomplete",)])
            if is_new and groups_now > last_verified_behavior_count:
                no_new = 0
            elif is_new:
                no_new = max(0, no_new - 1)
            else:
                no_new += 1
            last_verified_behavior_count = groups_now
        if budget is not None and budget.expired():
            break

    # ---- no verified program: fall back to best partials (never discard) ----
    if not pool.verified():
        for i in pool.best_partial(n=5):
            c = pool.candidates[i]
            if not c.test_outputs:
                c.test_outputs = run_on_inputs(executor, c.code, task.test)

    verified_groups = {k: v for k, v in pool.verified_behaviors().items() if k != ("incomplete",)}
    disagreement = len(verified_groups) >= 2

    # ---- adjudication (gated LLM call: only on persistent disagreement) ----
    adj_votes: dict[tuple, float] = {}
    if (
        disagreement
        and cfg.use_adjudication
        and (budget is None or not budget.expired())
        and (time_cap_s is None or time.time() - t_start < time_cap_s * 0.8)
    ):
        reps = []
        for key, members in sorted(verified_groups.items(), key=lambda kv: -len(kv[1]))[:3]:
            rep = min(members, key=lambda i: len(pool.candidates[i].code))
            outs = pool.candidates[rep].test_outputs
            reps.append((pool.candidates[rep].code, outs[0] if outs else None))
        if len(reps) >= 2:
            messages = build_adjudication_messages(task, reps)
            if _budget_ok(messages):
                try:
                    resp = llm.chat(
                        messages,
                        temperature=0.4,
                        max_tokens=cfg.max_completion_tokens,
                        reasoning_effort="medium",
                        seed=cfg.seed + 7777,
                    )
                    _charge(resp)
                    code = pick_program(resp.text)
                    if code:
                        ev = evaluate_program(executor, code, task.train)
                        if ev.verified:
                            outs = run_on_inputs(executor, code, task.test)
                            pool.add(code, "adj", ev, outs)
                            if all(o is not None for o in outs):
                                key = tuple(grid_key(o) for o in outs)
                                adj_votes[key] = adj_votes.get(key, 0.0) + 1.0
                except Exception:
                    pass

    # ---- LOO corroboration (gated: disagreement + >=3 pairs + budget) -------
    loo_map: dict[tuple, float] = {}
    if (
        disagreement
        and cfg.use_loo
        and task.n_pairs >= cfg.loo_min_pairs
        and (budget is None or not budget.expired())
        and (time_cap_s is None or time.time() - t_start < time_cap_s * 0.9)
    ):
        folds = list(range(task.n_pairs))
        if len(folds) > cfg.loo_max_folds:
            folds = [int(i) for i in rng.choice(folds, size=cfg.loo_max_folds, replace=False)]
        n_pass = 0
        for fold in folds:
            messages = build_loo_messages(task, omit_pair=fold, seed=cfg.seed)
            if not _budget_ok(messages):
                break
            try:
                resp = llm.chat(
                    messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_completion_tokens,
                    reasoning_effort=cfg.reasoning_effort,
                    seed=cfg.seed + 31 * (fold + 1),
                )
            except Exception:
                continue
            _charge(resp)
            code = pick_program(resp.text)
            if not code:
                continue
            # held-out check: run on the withheld pair ONLY
            x_h, y_h = task.train[fold]
            res = executor.run_batch([(code, x_h)])
            if not (res[0].ok and res[0].output is not None):
                continue
            if not grids_equal(res[0].output, y_h):
                continue
            n_pass += 1
            outs = run_on_inputs(executor, code, task.test)
            pool.add(code, f"loo{fold}", evaluate_program(executor, code, task.train), outs)
            if all(o is not None for o in outs):
                key = tuple(grid_key(o) for o in outs)
                # credit the matching verified behavior group
                if key in verified_groups:
                    loo_map[key] = loo_map.get(key, 0.0) + 1.0
        if n_pass:
            for k in loo_map:
                loo_map[k] /= n_pass

    # ---- disambiguation (pure numpy + sandbox: zero LLM tokens) -------------
    fallback_outputs = [x for x in task.test]  # identity guess (writer re-fallbacks if None)
    result, attempts, bundle = disambiguate(
        task,
        pool,
        executor,
        cfg=cfg.disambiguation,
        rng=np.random.default_rng(cfg.seed + 1234),
        loo_map=loo_map or None,
        adj_votes=adj_votes or None,
        fallback_outputs=fallback_outputs,
    )
    if on_bundle is not None:
        on_bundle(task, pool, bundle)

    if budget is not None:
        budget.mark_task_done()
    return TaskResult(
        task_id=task.task_id,
        attempts=attempts,
        n_candidates=len(pool),
        n_verified=len(pool.verified()),
        n_distinct_verified_behaviors=len(verified_groups),
        llm_calls=calls,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        wall_s=round(time.time() - t_start, 2),
        disambiguation=result,
        extras={"adj_votes": {str(k): v for k, v in adj_votes.items()}, "loo_map": {str(k): v for k, v in loo_map.items()}},
    )


def solve_all(
    tasks: Sequence[Task],
    llm: LLMClient,
    budget: Optional[Budget],
    cfg: SolverConfig,
    submission_writer=None,
    workers: int = 1,
    executor_factory: Optional[Callable[[], Sandbox]] = None,
    on_task: Optional[Callable[[TaskResult], None]] = None,
    on_bundle: Optional[Callable[[Task, ProgramPool, FeatureBundle], None]] = None,
) -> dict:
    """Run all tasks (thread-parallel; each thread gets its own sandbox worker)."""
    results: list[TaskResult] = []
    import threading

    results_lock = threading.Lock()
    n_done = 0

    def _worker(task: Task) -> TaskResult:
        nonlocal n_done
        executor = (executor_factory() if executor_factory else Sandbox(per_job_timeout=cfg.sandbox_timeout))
        try:
            remaining = max(1, len(tasks) - n_done)
            cap = budget.per_task_time_cap(remaining) if budget is not None else None
            res = solve_task(task, llm, executor, budget, cfg, time_cap_s=cap, on_bundle=on_bundle)
        finally:
            executor.close()
        if submission_writer is not None:
            submission_writer.update(task.task_id, res.attempts)
        if on_task is not None:
            on_task(res)
        with results_lock:
            results.append(res)
            n_done += 1
        return res

    if workers <= 1:
        for task in tasks:
            if budget is not None and budget.expired():
                break
            _worker(task)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = []
            for task in tasks:
                if budget is not None and budget.expired():
                    break
                futures.append(ex.submit(_worker, task))
            for f in futures:
                try:
                    f.result()
                except Exception as e:  # keep going; watchdog guarantees a file
                    print(f"[solve_all] task failed: {e}")

    total_prompt = sum(r.prompt_tokens for r in results)
    total_completion = sum(r.completion_tokens for r in results)
    n_verified_tasks = sum(1 for r in results if r.n_verified > 0)
    return {
        "tasks_run": len(results),
        "tasks_with_verified": n_verified_tasks,
        "verified_rate": n_verified_tasks / max(1, len(results)),
        "llm_calls": sum(r.llm_calls for r in results),
        "prompt_tokens": total_prompt,
        "completion_tokens": total_completion,
        "wall_s": round(sum(r.wall_s for r in results), 1),
        "results": results,
    }
