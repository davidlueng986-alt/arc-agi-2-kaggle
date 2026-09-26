"""H1 "Pure Reasoning Ceiling" harness for ARC Prize 2026 (ARC-AGI-2).

Minimal, clean, no judge model, no code refinement, no test-time training. The only
levers are `reasoning_effort`, sample count, CSV grid encoding, tolerant parsing and
shape-diverse two-attempt selection.
"""

from __future__ import annotations

from .budget import BudgetConfig, BudgetGovernor, Clock, ManualClock, SystemClock, TokenLedger
from .config import HarnessConfig, SamplingConfig, ServingConfig
from .diagnostics import CoverageStats, CoverageTracker
from .grids import (
    Grid,
    fallback_grid,
    from_key,
    is_grid,
    parse_csv_strict,
    parse_grid_blocks,
    parse_last_grid,
    shape,
    to_csv,
    to_csv_indented,
    to_key,
)
from .llm import (
    EFFORT_RANK,
    VALID_EFFORTS,
    ChatRequest,
    ChatResponse,
    EffortError,
    HTTPChatClient,
    LLMClient,
    MockLLM,
    RoundRobinClient,
    Usage,
    validate_effort,
)
from .parse import AnswerParse, merge_reasoning_and_content, parse_answers
from .prompts import Prompt, build_messages, build_prompt, prompt_token_estimate
from .runner import RunOutcome, budget_table, load_tasks, mock_answers, run_harness
from .scoring import ScoreReport, TaskScore, score_predictions, score_submission
from .select import Candidate, CandidatePool, IndexResult, consensus_reached, task_consensus
from .submission import SubmissionWriter, build_entry, empty_submission, validate_submission_obj
from .tasks import ARCTask, load_challenges, resolve_data_dir, total_test_outputs
from .watchdog import Watchdog
from .waves import RunReport, Scheduler, TaskState, WaveRecord

__all__ = [
    "AnswerParse",
    "ARCTask",
    "BudgetConfig",
    "BudgetGovernor",
    "Candidate",
    "CandidatePool",
    "ChatRequest",
    "ChatResponse",
    "Clock",
    "CoverageStats",
    "CoverageTracker",
    "EffortError",
    "EFFORT_RANK",
    "Grid",
    "HTTPChatClient",
    "HarnessConfig",
    "IndexResult",
    "LLMClient",
    "ManualClock",
    "MockLLM",
    "RoundRobinClient",
    "Prompt",
    "RunOutcome",
    "RunReport",
    "SamplingConfig",
    "ScoreReport",
    "Scheduler",
    "ServingConfig",
    "SubmissionWriter",
    "SystemClock",
    "TaskScore",
    "TaskState",
    "TokenLedger",
    "Usage",
    "VALID_EFFORTS",
    "Watchdog",
    "WaveRecord",
    "budget_table",
    "build_entry",
    "build_messages",
    "build_prompt",
    "consensus_reached",
    "empty_submission",
    "fallback_grid",
    "from_key",
    "is_grid",
    "load_challenges",
    "load_tasks",
    "merge_reasoning_and_content",
    "mock_answers",
    "parse_answers",
    "parse_csv_strict",
    "parse_grid_blocks",
    "parse_last_grid",
    "prompt_token_estimate",
    "resolve_data_dir",
    "run_harness",
    "score_predictions",
    "score_submission",
    "shape",
    "task_consensus",
    "to_csv",
    "to_csv_indented",
    "to_key",
    "total_test_outputs",
    "validate_effort",
    "validate_submission_obj",
]
