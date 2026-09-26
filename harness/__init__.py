"""H3 branch harness: program synthesis over an exact oracle + verified-set
disambiguation for ARC-AGI-2 (Kaggle ARC Prize 2026)."""

from .budget import Budget, Watchdog
from .data import load_challenges, load_solutions, load_solutions_for, load_tasks, tasks_from_challenges
from .disambiguate import (
    DEFAULT_ENABLE,
    DEFAULT_WEIGHTS,
    DisambiguationConfig,
    FeatureBundle,
    compute_features,
    disambiguate,
    rank_and_select,
)
from .extraction import extract_code_blocks, pick_program
from .grid import format_csv, parse_grid_lenient, to_grid
from .llm import MockLLM, VLLMClient, library_responder, scripted_responder
from .oracle import evaluate_program, run_on_inputs, score_output
from .pool import ProgramPool
from .sandbox import LocalExecutor, Sandbox
from .solver import SolverConfig, solve_all, solve_task
from .submission import SubmissionWriter, score_submission, validate_submission
from .types import Candidate, Grid, LLMResponse, Task, TaskResult

__all__ = [
    "Budget",
    "Watchdog",
    "Candidate",
    "DEFAULT_ENABLE",
    "DEFAULT_WEIGHTS",
    "DisambiguationConfig",
    "FeatureBundle",
    "Grid",
    "LLMResponse",
    "LocalExecutor",
    "MockLLM",
    "ProgramPool",
    "Sandbox",
    "SolverConfig",
    "SubmissionWriter",
    "Task",
    "TaskResult",
    "VLLMClient",
    "compute_features",
    "disambiguate",
    "evaluate_program",
    "extract_code_blocks",
    "format_csv",
    "library_responder",
    "load_challenges",
    "load_solutions",
    "load_solutions_for",
    "load_tasks",
    "parse_grid_lenient",
    "pick_program",
    "rank_and_select",
    "run_on_inputs",
    "score_output",
    "score_submission",
    "scripted_responder",
    "solve_all",
    "solve_task",
    "tasks_from_challenges",
    "to_grid",
    "validate_submission",
]
