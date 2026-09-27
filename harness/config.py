"""Knobs for the Duck-REPL harness, with the budget arithmetic behind them.

Every default here is derived in ``HYPOTHESIS.md`` section 5.  The short version:

* prefill dominates (a ~9.5K-token prompt costs ~4.5s of GPU at a conservative
  2,000 tok/s aggregate prefill rate, versus ~1.4s for a 700-token completion at
  500 tok/s aggregate decode), so **context length is the primary cost lever**,
  not completion length;
* ``reasoning_effort=xhigh`` (the model default) would emit 8-16K thinking
  tokens per call and blow the 12h budget on its own, so effort is pinned to an
  explicit schedule of ``medium``/``low``;
* the environment is **static**, so unlike the Duck we inject the image once and
  only add a status/ledger user message when something actually changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BudgetConfig:
    # --- wall clock -------------------------------------------------------
    total_seconds: float = 12 * 3600.0
    safety_buffer_seconds: float = 600.0
    startup_reserve_seconds: float = 1200.0  # vLLM load + CUDA graph capture
    # --- per task ---------------------------------------------------------
    max_turns: int = 16
    hard_max_turns: int = 24
    max_python_calls_per_turn: int = 4
    max_task_tokens: int = 260_000
    max_task_seconds: float = 900.0
    # --- context ----------------------------------------------------------
    context_window: int = 32_768
    context_budget: int = 24_000
    completion_reserve: int = 4_096
    keep_recent_turns: int = 6
    task_block_token_budget: int = 4_000
    ledger_token_budget: int = 700
    # --- tool -------------------------------------------------------------
    tool_timeout_s: float = 30.0
    tool_output_chars: int = 4_096
    sandbox_memory_mb: int = 3_072
    verify_call_seconds: float = 1.5
    verify_total_seconds: float = 12.0
    # --- multimodal -------------------------------------------------------
    use_image: bool = True
    image_max_vision_tokens: int = 160
    image_upscale: int | None = None  # set to 4 to reproduce the Duck literally
    # --- pool / attempts --------------------------------------------------
    max_pool: int = 24
    ledger_tried_entries: int = 8

    @property
    def usable_seconds(self) -> float:
        return max(
            60.0,
            self.total_seconds - self.safety_buffer_seconds - self.startup_reserve_seconds,
        )


@dataclass
class SamplingConfig:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 3_072
    seed: int | None = None
    # Effort schedule.  `high` is NOT valid for Qwen3.8 (HTTP 500); xhigh is the
    # server default and unaffordable here.
    effort_first: str = "medium"
    effort_default: str = "low"
    effort_on_shape_fail: str = "medium"
    effort_final: str = "medium"

    def effort_for_turn(self, turn_index: int, last_had_shape_failure: bool) -> str:
        if turn_index == 0:
            return self.effort_first
        if last_had_shape_failure:
            return self.effort_on_shape_fail
        return self.effort_default


@dataclass
class HarnessConfig:
    model: str = "qwen3.8-27b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    request_timeout_s: float = 600.0
    max_retries: int = 2
    workers: int = 8
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    stop_on_verified: bool = True
    min_turns: int = 2
    # Stop early only once we have two *distinct* verified test outputs (so
    # attempt_2 is a real second chance), unless we are burning turns.
    want_distinct_verified: bool = True
    diversity_wait_turns: int = 6
    max_output_chars: int = 4_096
