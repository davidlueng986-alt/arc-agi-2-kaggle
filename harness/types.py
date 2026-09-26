"""Core data types for the H4 harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

Grid = list[list[int]]


@dataclass
class LLMResponse:
    text: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: Optional[str] = None
    duration: float = 0.0

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class RunResult:
    """Result of executing one candidate program on one grid (train or test)."""

    success: bool = False
    grid: Optional[Grid] = None
    raw_output: str = ""
    soft_score: float = 0.0
    error: Optional[str] = None


@dataclass
class Candidate:
    """One hypothesis produced by any channel (code expert iteration or text sample)."""

    source: str  # "expert_a" | "expert_b" | "text" | ...
    iteration: int = 0
    trace: str = ""  # model-visible reasoning/code, fed to the judge (truncated later)
    code: Optional[str] = None
    grids: list[Optional[Grid]] = field(default_factory=list)  # per test input
    verified: bool = False  # all train pairs matched exactly
    soft: float = 0.0  # mean per-train-pair soft score
    feedback: str = ""


@dataclass
class JudgePick:
    """One judge's top-2 grids (Land: last two distinct grids in the response)."""

    grids: list[Grid] = field(default_factory=list)
    error: Optional[str] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ScoreEntry:
    """Aggregated judge scoreboard entry (2 pts first choice, 1 pt second choice)."""

    grid: Grid
    points: int = 0
    origin: str = "candidate"  # "candidate" | "synthesized"
    matched: bool = True
    voters: list[int] = field(default_factory=list)


@dataclass
class GridStat:
    """Per-distinct-grid statistics for one test input, used by attempt selection."""

    grid: Grid
    judge_points: int = 0
    verified_votes: int = 0
    votes: int = 0
    best_soft: float = 0.0
    sources: set[str] = field(default_factory=set)
    synthesized: bool = False

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.grid), len(self.grid[0]) if self.grid else 0)
