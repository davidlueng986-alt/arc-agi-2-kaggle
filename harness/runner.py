"""Task + run orchestration.

solve_task: experts and text channel concurrently -> per-test-input candidate
stats -> early-accept or judge council -> pass@2-aware attempt selection.

TaskRunner: deadline-aware loop over a challenge set with cumulative submission
writes, per-task scoring against ground truth (when available), a watchdog
finalization hook, and the degradation ladder.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional

from harness.budget import Deadline, TokenLedger
from harness.config import PipelineConfig, degraded
from harness.experts import CodeExpert, TaskData, TextChannel
from harness.grids import grid_key, suggest_output_shape
from harness.judge import aggregate_picks, build_judge_entries, mark_synthesized, run_council
from harness.llm import LLMClient
from harness.selection import build_grid_stats, early_accept, score_stat, select_attempts
from harness.submission import SubmissionWriter
from harness.types import Candidate, Grid


async def solve_task(
    task_id: str,
    task: dict,
    client: LLMClient,
    cfg: PipelineConfig,
    ledger: TokenLedger,
    deadline: Optional[Deadline] = None,
    task_deadline_s: Optional[float] = None,
) -> tuple[list[dict], dict]:
    """Solve one task. Returns (attempts, meta) where attempts is the
    submission-schema list of {"attempt_1", "attempt_2"} per test input."""
    train_in: list[Grid] = [ex["input"] for ex in task["train"]]
    train_out: list[Grid] = [ex["output"] for ex in task["train"]]
    test_in: list[Grid] = [ex["input"] for ex in task["test"]]
    td = TaskData(task_id=task_id, train_in=train_in, train_out=train_out, test_in=test_in)

    level = deadline.level() if deadline is not None else 0
    ecfg = degraded(cfg, level)

    ledger.begin_task(task_id)
    meta: dict = {"task_id": task_id, "level": level, "judged": [], "verified": False}

    # --- candidate generation (all channels concurrently) -------------------
    jobs = []
    for expert_cfg in ecfg.expert_configs():
        if expert_cfg.max_iterations <= 0:
            continue
        jobs.append(CodeExpert(expert_cfg).run(td, client, ledger, deadline, task_deadline_s))
    if ecfg.text_candidates > 0 and ecfg.experts >= 0:
        jobs.append(
            TextChannel(k=ecfg.text_candidates, reasoning_effort=ecfg.effort_text).run(
                td, client, ledger, deadline
            )
        )
    results = await asyncio.gather(*jobs, return_exceptions=True)

    candidates: list[Candidate] = []
    channel_errors = 0
    for r in results:
        if isinstance(r, BaseException):
            channel_errors += 1
            continue
        candidates.extend(r)
    meta["candidates"] = len(candidates)
    meta["channel_errors"] = channel_errors
    meta["verified"] = any(c.verified for c in candidates)

    # --- per-test-input selection -------------------------------------------
    attempts: list[dict] = []
    for j, tin in enumerate(test_in):
        suggested = suggest_output_shape(train_in, train_out, tin)
        meta.setdefault("suggested_shapes", []).append(suggested)

        scoreboard = None
        judge_on = False
        stats_pre = build_grid_stats(candidates, j, None)
        accepted = early_accept(stats_pre)

        if accepted is None and ecfg.use_judge and ecfg.judges > 0:
            entries = build_judge_entries(candidates, j, max_entries=ecfg.judge_max_entries)
            if entries:  # even one grid is worth a judge pass (synthesis can rescue it)
                picks = await run_council(
                    task_id, train_in, train_out, tin, entries, client,
                    n_judges=ecfg.judges,
                    reasoning_effort=ecfg.effort_judge,
                    max_tokens=ecfg.max_tokens_judge,
                    ledger=ledger,
                    deadline=deadline,
                    trace_cap_chars=ecfg.judge_trace_cap_chars,
                )
                ok_picks = [p for p in picks if p.grids]
                if ok_picks:
                    scoreboard = aggregate_picks(picks)
                    cand_keys = {grid_key(e["grid"]) for e in entries}
                    mark_synthesized(scoreboard, cand_keys)
                    judge_on = True
                    meta["judged"].append(j)

        stats = build_grid_stats(candidates, j, scoreboard)
        a1, a2 = select_attempts(
            stats, tin, train_out, suggested,
            judge_on=judge_on, diversity_on=ecfg.diversity,
        )
        attempts.append({"attempt_1": a1, "attempt_2": a2})

    totals = ledger.end_task()
    meta["tokens"] = totals
    return attempts, meta


def score_attempts(attempts: list[dict], gt_outputs: list[Grid]) -> tuple[int, int, int]:
    """(pass@2 hits, pass@1 hits, n_outputs) for one task; a hit requires an
    exact grid match (dims + every cell) by either / the first attempt."""
    hits2 = hits1 = 0
    for j, gt in enumerate(gt_outputs):
        if j >= len(attempts):
            continue
        a1 = attempts[j].get("attempt_1")
        a2 = attempts[j].get("attempt_2")
        if a1 == gt:
            hits1 += 1
        if a1 == gt or a2 == gt:
            hits2 += 1
    return hits2, hits1, len(gt_outputs)


class TaskRunner:
    def __init__(
        self,
        challenges: dict,
        client: LLMClient,
        cfg: PipelineConfig,
        out_path: str,
        solutions: Optional[dict] = None,
        deadline: Optional[Deadline] = None,
    ):
        self.challenges = challenges
        self.client = client
        self.cfg = cfg
        self.solutions = solutions
        self.deadline = deadline or Deadline(cfg.total_time_s)
        self.ledger = TokenLedger(per_task_effective_cap=cfg.token_budget_per_task)
        # make the client account into this runner's ledger unless it has one
        if getattr(client, "ledger", None) is None:
            try:
                client.ledger = self.ledger
            except AttributeError:
                pass
        self.writer = SubmissionWriter(out_path, challenges)
        self.metas: dict[str, dict] = {}
        self.stop_requested = False
        self._lock = asyncio.Lock()

    def request_stop(self) -> None:
        """Watchdog hook: stop scheduling new tasks and finalize."""
        self.stop_requested = True

    def start_watchdog(self, finalize_before_s: float = 60.0) -> threading.Thread:
        """Daemon thread that force-finalizes the submission at
        deadline - finalize_before_s, whatever the solver is doing (lb3389's
        600s-buffer pattern). Safe to call multiple times."""
        def _watch():
            wait = max(0.0, self.deadline.remaining() - finalize_before_s)
            time.sleep(wait)
            self.request_stop()
            try:
                self.writer.finalize()
            except Exception:  # noqa: BLE001 — last resort: never raise in a watchdog
                pass

        t = threading.Thread(target=_watch, daemon=True, name="submission-watchdog")
        t.start()
        self._watchdog = t
        return t

    async def run(self, limit: Optional[int] = None, task_ids: Optional[list[str]] = None) -> dict:
        ids = task_ids if task_ids is not None else sorted(self.challenges.keys())
        if limit is not None:
            ids = ids[:limit]
        sem = asyncio.Semaphore(max(1, self.cfg.task_concurrency))
        start = time.monotonic()
        results: dict[str, dict] = {}
        done = {"n": 0}

        async def one(task_id: str) -> None:
            async with sem:
                if self.stop_requested or self.deadline.level() >= 4:
                    done["n"] += 1
                    return
                remaining = max(1, len(ids) - done["n"])
                t_soft = self.deadline.task_deadline(remaining, self.cfg.max_task_seconds)
                t0 = time.monotonic()
                try:
                    attempts, meta = await solve_task(
                        task_id, self.challenges[task_id], self.client, self.cfg,
                        self.ledger, self.deadline, task_deadline_s=t_soft,
                    )
                except Exception as e:  # noqa: BLE001 — never lose the run
                    attempts, meta = [], {"task_id": task_id, "error": f"{type(e).__name__}: {e}"}
                meta["wall_s"] = round(time.monotonic() - t0, 1)
                done["n"] += 1
                async with self._lock:
                    self.metas[task_id] = meta
                    if attempts:
                        self.writer.update(task_id, attempts)
                    if self.solutions is not None and task_id in self.solutions:
                        gt = self.solutions[task_id]
                        h2, h1, n = score_attempts(attempts, gt) if attempts else (0, 0, len(gt))
                        results[task_id] = {"hits2": h2, "hits1": h1, "n": n}
                        meta["hits2"], meta["hits1"], meta["n_outputs"] = h2, h1, n

        await asyncio.gather(*(one(tid) for tid in ids))

        # cumulative writer holds solved tasks; fill the rest with fallbacks
        fin = self.writer.finalize()
        summary = self._summarize(ids, results, start)
        summary["validation_errors"] = fin["errors"]
        return summary

    def _summarize(self, ids: list[str], results: dict[str, dict], start: float) -> dict:
        usage = self.ledger.snapshot()
        n_out = sum(r["n"] for r in results.values())
        h2 = sum(r["hits2"] for r in results.values())
        h1 = sum(r["hits1"] for r in results.values())
        metas = list(self.metas.values())
        return {
            "tasks_run": len(ids),
            "tasks_scored": len(results),
            "outputs_scored": n_out,
            "pass2": (h2 / n_out) if n_out else None,  # competition metric
            "pass1": (h1 / n_out) if n_out else None,
            "tasks_fully_solved_pass2": sum(
                1 for r in results.values() if r["n"] and r["hits2"] == r["n"]
            ),
            "verified_tasks": sum(1 for m in metas if m.get("verified")),
            "judged_test_inputs": sum(len(m.get("judged", [])) for m in metas),
            "channel_errors": sum(m.get("channel_errors", 0) for m in metas),
            "wall_s": round(time.monotonic() - start, 1),
            "tokens": usage,
        }
