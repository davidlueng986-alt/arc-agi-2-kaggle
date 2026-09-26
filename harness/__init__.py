"""H4 harness: Poetiq code-refinement + Land holistic judge, pass@2-optimised.

Portable, dependency-light (numpy only at runtime; scipy/cv2 optional inside the
sandbox). Runs against any OpenAI-compatible endpoint (vLLM) or MockLLM.
"""

from harness.types import (  # noqa: F401
    Candidate,
    Grid,
    GridStat,
    JudgePick,
    LLMResponse,
    RunResult,
)
from harness.config import PipelineConfig  # noqa: F401

__all__ = [
    "Candidate",
    "Grid",
    "GridStat",
    "JudgePick",
    "LLMResponse",
    "RunResult",
    "PipelineConfig",
]
