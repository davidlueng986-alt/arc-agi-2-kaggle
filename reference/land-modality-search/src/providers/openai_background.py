import time
from typing import Optional, TYPE_CHECKING

from src.types import ModelResponse
from src.logging import get_logger
from src.providers.openai_bg.job_manager import submit_job, poll_job

if TYPE_CHECKING:
    from src.providers.openai_runner import OpenAIRequestRunner

logger = get_logger("providers.openai")

class OpenAIBackgroundSolver:
    """Handles the lifecycle of OpenAI Background (Batch/Async) jobs."""

    def __init__(self, runner: 'OpenAIRequestRunner'):
        self.runner = runner
        self.client = runner.client
        self.verbose = runner.verbose

    def solve(self, prompt: str, image_path: Optional[str] = None, enable_code_execution: bool = False) -> ModelResponse:
        start_attempt_ts = time.perf_counter()

        try:
            # 1. Submit Job
            job_id = submit_job(self.runner, prompt, image_path, enable_code_execution)
            
            if self.verbose:
                print(f"[BACKGROUND] [{self.runner.model}] Job submitted. ID: {job_id}")

            # 2. Poll for Completion
            return poll_job(self.runner, job_id, prompt, image_path, start_attempt_ts)

        except Exception as e:
            raise e