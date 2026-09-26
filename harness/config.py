"""All tunable knobs in one place, with defaults derived from the budget model."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

from .budget import BudgetConfig


@dataclass
class SamplingConfig:
    phase_a_effort: str = "low"
    phase_a_samples: int = 2
    phase_a_max_tokens: int = 2560
    phase_a_temperature: float = 0.7

    phase_b_effort: str = "xhigh"
    wave_samples: Tuple[int, ...] = (3, 3, 3, 3)
    max_xhigh_samples: int = 12
    xhigh_max_tokens: int = 16384
    xhigh_temperature: float = 0.9
    top_p: float = 0.95

    consensus_threshold: float = 0.7
    consensus_min_candidates: int = 4

    concurrency: int = 48
    shuffle_train: bool = False
    request_timeout: float = 2400.0

    def max_tokens_for(self, effort: str) -> int:
        if effort == self.phase_b_effort:
            return self.xhigh_max_tokens
        if effort == "medium":
            return 6144
        return self.phase_a_max_tokens

    def temperature_for(self, effort: str) -> float:
        if effort == self.phase_a_effort:
            return self.phase_a_temperature
        return self.xhigh_temperature


@dataclass
class ServingConfig:
    model: str = "qwen3.8-27b"
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "EMPTY"
    effort_mechanism: str = "chat_template_kwargs"
    max_retries: int = 3
    health_wait_seconds: float = 900.0
    extra_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessConfig:
    data_dir: Optional[str] = None
    split: str = "test"
    output_dir: Optional[str] = None
    submission_name: str = "submission.json"
    limit: Optional[int] = None
    task_ids: Optional[Tuple[str, ...]] = None
    seed: int = 20260927

    serving: ServingConfig = field(default_factory=ServingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)

    log_every: int = 10
    write_after_each_task: bool = False

    @property
    def submission_path(self) -> str:
        from .tasks import resolve_output_dir

        return os.path.join(resolve_output_dir(self.output_dir), self.submission_name)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
