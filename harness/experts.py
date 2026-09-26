"""Candidate-generation channels.

CodeExpert: the Poetiq refinement loop (solve_coding.py port) — shuffled train
order per iteration, pool-5 scored feedback, sandbox after every sample,
early-exit on full train verification, per-iteration budget/deadline checks.

TextChannel: Land's minimal free-form text candidates (no refinement, no
prescriptive template) for raw hypothesis diversity.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from harness.budget import Deadline, TokenLedger
from harness.grids import normalize_grid, parse_grid_from_text
from harness.llm import LLMClient
from harness.oracle import build_feedback, evaluate_code
from harness.parsing import looks_like_refusal, pick_transform_code
from harness.pool import SolutionPool
from harness.prompts import (
    build_feedback_prompt,
    build_solver_a_prompt,
    build_solver_b_prompt,
    build_text_prompt,
)
from harness.types import Candidate, Grid


@dataclass
class ExpertConfig:
    name: str = "expert_a"
    variant: str = "A"  # "A" = Poetiq prescriptive, "B" = Land minimal
    max_iterations: int = 8
    seed: int = 0
    temperature: float = 1.0
    reasoning_effort: str = "low"
    max_tokens: int = 2048
    request_timeout: float = 300.0
    pool_size: int = 5
    shuffle_examples: bool = True
    sandbox_timeout: float = 5.0


@dataclass
class TaskData:
    task_id: str
    train_in: list[Grid]
    train_out: list[Grid]
    test_in: list[Grid]


class CodeExpert:
    def __init__(self, cfg: ExpertConfig):
        self.cfg = cfg

    def _solver_prompt(self, task: TaskData, iteration: int) -> str:
        seed = self.cfg.seed + iteration  # Poetiq: seed + it
        if self.cfg.variant == "A":
            return build_solver_a_prompt(
                task.train_in, task.train_out, task.test_in,
                shuffle=self.cfg.shuffle_examples, seed=seed,
            )
        return build_solver_b_prompt(
            task.train_in, task.train_out, task.test_in,
            shuffle=self.cfg.shuffle_examples, seed=seed,
        )

    async def run(
        self,
        task: TaskData,
        client: LLMClient,
        ledger: Optional[TokenLedger] = None,
        deadline: Optional[Deadline] = None,
        task_deadline_s: Optional[float] = None,
    ) -> list[Candidate]:
        cfg = self.cfg
        pool = SolutionPool(max_solutions=cfg.pool_size)
        candidates: list[Candidate] = []
        start = asyncio.get_event_loop().time()

        for it in range(cfg.max_iterations):
            if deadline is not None and (deadline.level() >= 4 or deadline.expired()):
                break
            if task_deadline_s is not None and asyncio.get_event_loop().time() - start > task_deadline_s:
                break
            if ledger is not None and ledger.task_over_budget(task.task_id):
                break

            prompt = self._solver_prompt(task, it)
            if len(pool):
                prompt += "\n\n" + build_feedback_prompt(pool.entries, max_examples=cfg.pool_size)

            resp = await client.complete(
                prompt,
                tag=f"solver_{cfg.name}",
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
                reasoning_effort=cfg.reasoning_effort,
                timeout=cfg.request_timeout,
                # no cache_key: the prompt changes every iteration (shuffle +
                # pool feedback), so prefill is genuinely re-paid; only the
                # judge council shares an identical prompt.
            )
            if not resp.ok or looks_like_refusal(resp.text):
                continue

            picked = pick_transform_code(resp.text)
            if picked is None:
                continue  # malformed output: skip iteration (Poetiq semantics)
            code, entry = picked

            train_res, test_res = await evaluate_code(
                code, task.train_in, task.train_out, task.test_in,
                timeout_s=cfg.sandbox_timeout, entry=entry,
            )
            verified = bool(train_res) and all(r.success for r in train_res)
            mean_soft = (
                sum(r.soft_score for r in train_res) / len(train_res) if train_res else 0.0
            )
            feedback = ""
            if not verified:
                feedback, _score = build_feedback(train_res, task.train_out)

            candidates.append(
                Candidate(
                    source=cfg.name,
                    iteration=it + 1,
                    trace=resp.text,
                    code=code,
                    grids=[r.grid for r in test_res],
                    verified=verified,
                    soft=mean_soft,
                    feedback=feedback,
                )
            )
            if verified:
                # Poetiq early-exit: a fully verified program is ~97% precise
                # (Land's measurement) — stop spending tokens on this expert.
                return candidates

            if feedback:
                pool.add(code, feedback, mean_soft)

        return candidates


class TextChannel:
    """k free-form text samples per test input, minimal prompt (Land)."""

    def __init__(
        self,
        k: int = 3,
        temperature: float = 1.0,
        reasoning_effort: str = "low",
        max_tokens: int = 2500,
        request_timeout: float = 300.0,
    ):
        self.k = k
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout

    async def run(
        self,
        task: TaskData,
        client: LLMClient,
        ledger: Optional[TokenLedger] = None,
        deadline: Optional[Deadline] = None,
    ) -> list[Candidate]:
        candidates: list[Candidate] = []

        async def one(test_idx: int, sample: int) -> Optional[Candidate]:
            prompt = build_text_prompt(task.train_in, task.train_out, task.test_in[test_idx])
            resp = await client.complete(
                prompt,
                tag="text",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
                timeout=self.request_timeout,
            )
            if not resp.ok or looks_like_refusal(resp.text):
                return None
            grid = parse_grid_from_text(resp.text)
            grid = normalize_grid(grid)
            if grid is None:
                return None
            grids: list[Optional[Grid]] = [None] * len(task.test_in)
            grids[test_idx] = grid
            return Candidate(
                source=f"text[{test_idx}]",
                iteration=sample + 1,
                trace=resp.text,
                grids=grids,
                verified=False,
                soft=0.0,
            )

        jobs = [
            one(ti, s)
            for ti in range(len(task.test_in))
            for s in range(self.k)
        ]
        for fut in asyncio.as_completed([asyncio.ensure_future(j) for j in jobs]):
            try:
                cand = await fut
            except Exception:  # noqa: BLE001 — a dead channel never kills the task
                cand = None
            if cand is not None:
                candidates.append(cand)
        return candidates
