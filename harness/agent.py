"""The Duck-REPL agent: one task, one interactive coding session.

This is the ARC-AGI-2 analogue of the Duck's act-observe loop.  The mapping is
one-to-one:

===========================  ==================================================
The Duck (ARC-AGI-3)         This harness (ARC-AGI-2)
===========================  ==================================================
game state preloaded as      task preloaded as ``TRAIN`` / ``TEST``
Python variables             (train pairs WITH ground-truth outputs)
``action(actions)`` host     ``verify(fn)`` host round-trip: exact replay
round-trip                   against every train pair
new frame + ``last_action_   pass/fail + diff diagnostics (shape delta, bbox,
result`` observed            sample cells, value histograms, ASCII crop) +
                             test-output preview
``current_frame.ascii``      ``Grid.ascii`` / ``Grid.csv``
``current_frame.             ``Grid.segmentation`` / ``segment(grid)`` --
segmentation``               4-connected components, hashes, adjacency, nesting
4x-upscaled board image      one token-budgeted panel image, first turn only
pop oldest messages          ``context.trim_to_budget``
30s / 4096 chars / fresh     identical limits, fresh process per call
state per tool call
===========================  ==================================================

The thing that has no ARC-AGI-3 counterpart is the oracle.  ARC-AGI-3 has no
ground truth at inference time, so the Duck must learn a world model by probing;
hundreds of turns follow.  ARC-AGI-2 ships the train outputs, so the same loop
closes against an exact verifier and needs one to two orders of magnitude fewer
turns -- which is the whole reason this fits a 12h budget at all.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .budget import Deadline, TaskBudget, Usage
from .config import BudgetConfig, HarnessConfig
from .context import (
    estimate_message_tokens,
    drop_oldest_block,
    strip_image,
    trim_to_budget,
)
from .grids import Task, compute_stats, sequence_text, text_tokens
from .llm import ContextOverflowError, LLMClient, LLMRequest, LLMResponse, ToolCall
from .prompts import (
    FINAL_ANSWER_INSTRUCTION,
    TOOLS,
    build_status_message,
    build_system_prompt,
    build_task_block,
)
from .render import image_part, image_to_data_url, render_task_panel
from .sandbox import ReplSandbox, ToolResult, Verification
from .submission import CandidatePool, select_attempts

NOTE_LABELS = ("hypothesis:", "next:", "note:", "finding:", "plan:", "rule:", "open question:")
MAX_NOTE_CHARS = 160


@dataclass
class AgentResult:
    task_id: str
    entries: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    turns: int = 0
    python_calls: int = 0
    verify_calls: int = 0
    verified: bool = False
    n_verified: int = 0
    pool_size: int = 0
    best_soft_score: float = 0.0
    stop_reason: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    ledger: list[str] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


class Ledger:
    """Compact durable memory that survives context eviction.

    The Duck carries a "summarized knowledge" / world-model block forward for the
    same reason: once the oldest turns are popped, anything not in the ledger is
    gone.  Ours is deliberately tiny -- the arXiv ablation found simplification
    pressure helped, and every line here is re-sent every turn it appears in.
    """

    def __init__(self, max_entries: int = 8, max_chars: int = 2400) -> None:
        self.max_entries = max_entries
        self.max_chars = max_chars
        self.notes: list[str] = []
        self.tried: list[str] = []
        self.best: str = ""

    def add_note(self, text: str) -> None:
        clean = " ".join(str(text).split())[:MAX_NOTE_CHARS]
        if clean and clean not in self.notes:
            self.notes.append(clean)
            self._cap(self.notes, 4)

    def add_tried(self, text: str) -> None:
        clean = " ".join(str(text).split())[:MAX_NOTE_CHARS]
        if clean:
            self.tried.append(clean)
            self._cap(self.tried, self.max_entries)

    def set_best(self, text: str) -> None:
        self.best = " ".join(str(text).split())[:MAX_NOTE_CHARS]

    @staticmethod
    def _cap(items: list[str], limit: int) -> None:
        while len(items) > limit:
            items.pop(0)

    def lines(self, pool: CandidatePool) -> list[str]:
        out: list[str] = []
        if self.best:
            out.append(f"best: {self.best}")
        out.extend(pool.best_summary_lines(limit=2))
        out.extend(self.tried[-self.max_entries :])
        out.extend(self.notes[-3:])
        rendered: list[str] = []
        seen: set[str] = set()
        total = 0
        limit = self.max_chars // 3
        for line in out:
            if line in seen:
                continue
            cost = text_tokens(line)
            if total + cost > limit:
                break
            seen.add(line)
            rendered.append(line)
            total += cost
        return rendered


def harvest_notes(text: str) -> list[str]:
    """Pull the short *labelled* lines out of assistant text into the ledger.

    Only labelled lines are harvested.  An earlier version also kept any short
    unlabelled line, which filled the ledger with fragments like "verify." and
    cost tokens on every subsequent turn for zero information.
    """
    found: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*#> ").strip()
        if not line:
            continue
        lowered = line.lower()
        for label in NOTE_LABELS:
            if lowered.startswith(label):
                body = line[len(label) :].strip()
                if body:
                    found.append(f"{label[:-1]}: {body[:MAX_NOTE_CHARS]}")
                break
    return found[:3]


class DuckREPLAgent:
    def __init__(
        self,
        llm: LLMClient,
        config: HarnessConfig,
        *,
        deadline: Deadline | None = None,
        sandbox_factory: Callable[[Task, Callable[[dict[str, Any]], dict[str, Any]]], ReplSandbox]
        | None = None,
        keep_transcript: bool = False,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.llm = llm
        self.config = config
        self.deadline = deadline
        self._sandbox_factory = sandbox_factory
        self.keep_transcript = keep_transcript
        self._on_event = on_event
        self._system_prompt = build_system_prompt(
            tool_timeout_s=config.budget.tool_timeout_s,
            tool_output_chars=config.budget.tool_output_chars,
        )

    # ------------------------------------------------------------------ public

    def solve(self, task: Task) -> AgentResult:
        started = time.perf_counter()
        budget_cfg = self.config.budget
        pool = CandidatePool(max_size=budget_cfg.max_pool)
        ledger = Ledger(max_entries=budget_cfg.ledger_tried_entries)
        usage = Usage()
        result = AgentResult(task_id=task.task_id)

        task_budget = TaskBudget(
            max_turns=budget_cfg.max_turns,
            max_tokens=budget_cfg.max_task_tokens,
            max_seconds=budget_cfg.max_task_seconds,
            deadline=self.deadline,
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": self._system_prompt}]
        task_block = build_task_block(
            task, compute_stats(task), token_budget=budget_cfg.task_block_token_budget
        )
        image_tokens = 0
        if budget_cfg.use_image:
            try:
                image, info = render_task_panel(
                    task,
                    max_vision_tokens=budget_cfg.image_max_vision_tokens,
                    upscale=budget_cfg.image_upscale,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": task_block
                                + "\n\nComposite panel image (train pairs as input | output per "
                                "row, then the test inputs). It is the same data as the ASCII "
                                "above, at gestalt resolution: use it for shape, symmetry and "
                                "layout intuition, never to read a cell value.",
                            },
                            image_part(image_to_data_url(image)),
                        ],
                    }
                )
                image_tokens = int(info.get("vision_tokens", 0))
            except Exception as exc:
                messages.append({"role": "user", "content": task_block + f"\n[image unavailable: {exc}]"})
        else:
            messages.append({"role": "user", "content": task_block})

        turn_box = {"index": 0}
        sandbox = self._make_sandbox(task, pool, ledger, result, turn_box)

        last_shape_failure = False
        new_best_since_status = True
        turn = 0
        stop_reason = ""
        forced_final = False
        # A mis-set context_budget smaller than completion_reserve must not
        # produce a negative trimming target, which would evict everything.
        effective_context_budget = max(2048, budget_cfg.context_budget - budget_cfg.completion_reserve)
        task_block_budget = budget_cfg.task_block_token_budget
        overflow_retries = 0

        while turn < budget_cfg.hard_max_turns:
            turn_box["index"] = turn
            reason = task_budget.exceeded(usage, turn)
            if reason:
                stop_reason = reason
                break
            if turn >= budget_cfg.max_turns and not forced_final:
                forced_final = True
                messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})
                if self.keep_transcript:
                    result.transcript.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})

            effort = self.config.sampling.effort_for_turn(turn, last_shape_failure)
            if forced_final:
                effort = self.config.sampling.effort_final

            trimmed, trim_stats = trim_to_budget(
                messages,
                context_budget=effective_context_budget,
                tools=TOOLS,
                image_tokens=image_tokens,
                preserve_recent_turns=budget_cfg.keep_recent_turns,
            )

            try:
                response = self.llm.complete(
                    LLMRequest(
                        messages=trimmed,
                        tools=TOOLS,
                        tool_choice="auto",
                        temperature=self.config.sampling.temperature,
                        top_p=self.config.sampling.top_p,
                        top_k=self.config.sampling.top_k,
                        max_tokens=self.config.sampling.max_tokens,
                        reasoning_effort=effort,
                        seed=self.config.sampling.seed,
                    )
                )
            except ContextOverflowError:
                # Escalating recovery: drop the oldest block, then the image, then
                # halve the task block (the grids are all still in the REPL, so
                # this costs information the model can re-fetch, not correctness).
                overflow_retries += 1
                if drop_oldest_block(messages, preserve_recent_turns=0):
                    continue
                if strip_image(messages):
                    image_tokens = 0
                    continue
                if task_block_budget > 400 and overflow_retries <= 4:
                    task_block_budget //= 2
                    messages[1] = {
                        "role": "user",
                        "content": build_task_block(
                            task, compute_stats(task), token_budget=task_block_budget
                        ),
                    }
                    continue
                stop_reason = "context overflow"
                break
            except Exception as exc:
                result.error = f"{type(exc).__name__}: {exc}"[:400]
                stop_reason = "llm error"
                break

            usage.add(response.to_usage())
            usage.estimated_prompt_tokens += estimate_message_tokens(trimmed, tools=TOOLS, image_tokens=image_tokens)
            usage.estimated_completion_tokens += max(1, len(response.text) // 3)

            for note in harvest_notes(response.text):
                ledger.add_note(note)

            assistant_message: dict[str, Any] = {"role": "assistant", "content": response.text or ""}
            if response.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": call.raw_arguments or "{}"},
                    }
                    for call in response.tool_calls
                ]
            messages.append(assistant_message)
            if self.keep_transcript:
                result.transcript.append(dict(assistant_message))

            if not response.tool_calls:
                if pool.verified() and self.config.stop_on_verified:
                    stop_reason = "model stopped with a verified candidate"
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": build_status_message(
                            turn_index=turn,
                            max_turns=budget_cfg.max_turns,
                            ledger_lines=ledger.lines(pool),
                            tokens_used=usage.total_tokens,
                            token_budget=budget_cfg.max_task_tokens,
                            seconds_left=max(0.0, budget_cfg.max_task_seconds - task_budget.elapsed()),
                            nudge=(
                                "You did not call the `python` tool. Call it now: define "
                                "`transform` and `verify(transform)` in the same snippet."
                            ),
                        ),
                    }
                )
                turn += 1
                continue

            calls = response.tool_calls[: max(1, budget_cfg.max_python_calls_per_turn)]
            skipped = response.tool_calls[len(calls) :]
            turn_finished = False
            saw_shape_failure = False
            produced_verification = False

            for call in calls:
                if call.name != "python":
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": json.dumps({"error": f"unknown tool {call.name!r}; only `python` exists"}),
                        }
                    )
                    continue
                code = str(call.arguments.get("code", ""))
                tool_result = sandbox.run(code)
                usage.python_calls += 1
                usage.verify_calls += len(tool_result.verifications)
                usage.seconds += tool_result.elapsed_s
                produced_verification = produced_verification or bool(tool_result.verifications)
                for verification in tool_result.verifications:
                    if _has_shape_failure(verification.report):
                        saw_shape_failure = True
                if tool_result.finished:
                    turn_finished = True
                content = tool_result.render(max_chars=budget_cfg.tool_output_chars)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
                if self.keep_transcript:
                    result.transcript.append({"role": "tool", "content": content[:4000]})
                if self._on_event is not None:
                    self._on_event(
                        "python",
                        {
                            "task_id": task.task_id,
                            "turn": turn,
                            "error": tool_result.error[:200],
                            "verifications": [v.summary for v in tool_result.verifications],
                            "elapsed_s": round(tool_result.elapsed_s, 3),
                        },
                    )
                if turn_finished:
                    break

            for call in skipped:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(
                            {
                                "error": (
                                    f"only {budget_cfg.max_python_calls_per_turn} tool calls are executed per "
                                    "turn; batch the work into one snippet"
                                )
                            }
                        ),
                    }
                )

            last_shape_failure = saw_shape_failure
            _refresh_best(ledger, pool)
            if pool.verified():
                new_best_since_status = True

            if turn_finished:
                stop_reason = f"model called finish(): {result.stop_reason or 'done'}"
                break

            if self._should_stop(pool, turn, produced_verification):
                stop_reason = "verified early-stop"
                break

            if produced_verification or new_best_since_status or forced_final:
                new_best_since_status = False
                messages.append(
                    {
                        "role": "user",
                        "content": build_status_message(
                            turn_index=turn,
                            max_turns=budget_cfg.max_turns,
                            ledger_lines=ledger.lines(pool),
                            tokens_used=usage.total_tokens,
                            token_budget=budget_cfg.max_task_tokens,
                            seconds_left=max(0.0, budget_cfg.max_task_seconds - task_budget.elapsed()),
                            headline=(
                                "A candidate registered. Best so far is listed in the ledger."
                                if produced_verification
                                else None
                            ),
                            nudge=(
                                None
                                if produced_verification
                                else "That snippet called no `verify(...)`. Verify your candidate so the harness can score and register it."
                            ),
                        ),
                    }
                )
            turn += 1
        else:
            stop_reason = stop_reason or "hard turn cap"

        if not stop_reason:
            stop_reason = "turn loop ended"

        result.turns = usage.calls or turn
        result.python_calls = usage.python_calls
        result.verify_calls = usage.verify_calls
        result.usage = usage
        result.pool_size = len(pool.candidates)
        result.n_verified = len(pool.verified())
        result.verified = result.n_verified > 0
        result.best_soft_score = max((c.soft_score for c in pool.candidates), default=0.0)
        result.stop_reason = stop_reason
        result.ledger = ledger.lines(pool)
        try:
            result.entries = select_attempts(pool, task)
        except Exception as exc:
            result.error = (result.error + f" | select_attempts: {exc}")[:400]
            result.entries = [
                {
                    "attempt_1": [list(r) for r in grid],
                    "attempt_2": [list(r) for r in grid],
                }
                for grid in task.test_inputs
            ]
        result.elapsed_s = time.perf_counter() - started
        return result

    # ----------------------------------------------------------------- helpers

    def _make_sandbox(
        self,
        task: Task,
        pool: CandidatePool,
        ledger: Ledger,
        result: AgentResult,
        turn_box: dict[str, int],
    ) -> ReplSandbox:
        cfg = self.config.budget

        def on_verify(payload: dict[str, Any]) -> dict[str, Any]:
            candidate, ack = pool.add(
                source=payload.get("source"),
                exact=bool(payload.get("exact")),
                soft_score=float(payload.get("soft_score") or 0.0),
                pass_rate=float(payload.get("pass_rate") or 0.0),
                n_pairs=int(payload.get("n_pairs") or 0),
                summary=str(payload.get("summary") or ""),
                test_outputs=payload.get("test_outputs") or [],
                test_errors=payload.get("test_errors") or [],
                turn_index=turn_box.get("index", 0),
            )
            if candidate is not None:
                ledger.add_tried(candidate.summary)
            return ack

        if self._sandbox_factory is not None:
            return self._sandbox_factory(task, on_verify)
        return ReplSandbox(
            task,
            timeout_s=cfg.tool_timeout_s,
            max_output_chars=cfg.tool_output_chars,
            memory_mb=cfg.sandbox_memory_mb,
            verify_call_seconds=cfg.verify_call_seconds,
            verify_total_seconds=cfg.verify_total_seconds,
            on_verify=on_verify,
        )

    def _should_stop(self, pool: CandidatePool, turn: int, produced_verification: bool) -> bool:
        if not self.config.stop_on_verified:
            return False
        if turn + 1 < self.config.min_turns:
            return False
        verified = pool.verified()
        if not verified:
            return False
        if not self.config.want_distinct_verified:
            return True
        if pool.distinct_verified_outputs() >= 2:
            return True
        return turn + 1 >= self.config.diversity_wait_turns


def _has_shape_failure(report: dict[str, Any]) -> bool:
    for failure in report.get("failures") or []:
        if isinstance(failure, dict) and failure.get("kind") == "shape":
            return True
    return False


def _refresh_best(ledger: Ledger, pool: CandidatePool) -> None:
    ranked = pool.ranked()
    if not ranked:
        return
    best = ranked[0]
    ledger.set_best(
        "%s soft=%.3f %d/%d pairs exact=%s (%d lines)"
        % (
            "VERIFIED" if best.exact else "best",
            best.soft_score,
            int(round(best.pass_rate * best.n_pairs)),
            best.n_pairs,
            best.exact,
            best.source_lines,
        )
    )
