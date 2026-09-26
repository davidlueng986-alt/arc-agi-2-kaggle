from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, Set, List

from src.grid import Grid

@dataclass
class Example:
    input: Grid
    output: Optional[Grid]

@dataclass
class Task:
    train: List[Example]
    test: List[Example]

# Constants
GPT_5_1_BASE = "gpt-5.1"
GPT_5_2_BASE = "gpt-5.2"
CLAUDE_SONNET_BASE = "claude-sonnet-4-5-20250929"
CLAUDE_OPUS_BASE = "claude-opus-4-5-20251101"
GEMINI_3_BASE = "gemini-3-pro-preview"

PRICING_PER_1M_TOKENS = {
    GPT_5_1_BASE: {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    GPT_5_2_BASE: {"input": 1.75, "cached_input": 0.175, "output": 14.00},
    "gpt-5.1-codex-max": {"input": 2.50, "cached_input": 0.25, "output": 20.00},
    "gpt-5.2-codex": {"input": 3.50, "cached_input": 0.35, "output": 28.00},
    CLAUDE_SONNET_BASE: {
        "input": 3.00,
        "cached_input": 0.30,
        "output": 15.00,
    },
    CLAUDE_OPUS_BASE: {
        "input": 5.00,
        "cached_input": 0.50,
        "output": 25.00,
    },
    GEMINI_3_BASE: {
        "input": 2.00,
        "cached_input": 0.20,
        "output": 12.00,
    },
}

ORDERED_MODELS = [
    "gpt-5.1-none",
    "gpt-5.1-low",
    "gpt-5.1-medium",
    "gpt-5.1-high",
    "gpt-5.1-codex-max-high",
    "gpt-5.1-codex-max-xhigh",
    "gpt-5.2-none",
    "gpt-5.2-low",
    "gpt-5.2-medium",
    "gpt-5.2-high",
    "gpt-5.2-xhigh",
    "gpt-5.2-codex-none",
    "gpt-5.2-codex-low",
    "gpt-5.2-codex-medium",
    "gpt-5.2-codex-high",
    "gpt-5.2-codex-xhigh",
    "claude-sonnet-4.5-no-thinking",
    "claude-sonnet-4.5-thinking-1024",
    "claude-sonnet-4.5-thinking-4000",
    "claude-sonnet-4.5-thinking-16000",
    "claude-sonnet-4.5-thinking-60000",
    "claude-opus-4.5-no-thinking",
    "claude-opus-4.5-thinking-1024",
    "claude-opus-4.5-thinking-4000",
    "claude-opus-4.5-thinking-16000",
    "claude-opus-4.5-thinking-60000",
    "gemini-3-low",
    "gemini-3-high",
]
SUPPORTED_MODELS: Set[str] = set(ORDERED_MODELS)

@dataclass
class TaskResult:
    task_path: Path
    test_index: int
    success: bool
    model_arg: str
    duration: float
    cost: float
    strategy: Optional[str] = None
    verified: Optional[bool] = None

@dataclass
class ModelResponse:
    text: str
    prompt_tokens: int
    cached_tokens: int
    completion_tokens: int
    thought_tokens: int = 0
    strategy: Optional[str] = None
    model_name: Optional[str] = None
    timing_breakdown: Optional[list[dict]] = None
    detailed_logs: Optional[List[dict]] = None

@dataclass
class ModelConfig:
    provider: str
    base_model: str
    config: object  # can be int (budget), str (effort), or something else
