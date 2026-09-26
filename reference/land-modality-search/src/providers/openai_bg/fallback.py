import time
from typing import Optional, TYPE_CHECKING
from src.types import ModelConfig, ModelResponse, CLAUDE_OPUS_BASE
from src.llm_utils import get_retries_enabled
from src.logging import get_logger, log_failure
from src.errors import RetryableProviderError, NonRetryableProviderError
from src.providers.anthropic import call_anthropic

if TYPE_CHECKING:
    from src.providers.openai_runner import OpenAIRequestRunner

logger = get_logger("providers.openai")

def fallback_to_claude(
    runner: 'OpenAIRequestRunner',
    prompt: str, 
    image_path: Optional[str], 
    reason: str, 
    start_ts: float,
    thinking: bool
) -> ModelResponse:
    """Handles fallback to Claude Opus when OpenAI jobs fail/timeout."""
    if not get_retries_enabled():
        raise RetryableProviderError(f"OpenAI Job failed: {reason}. Fallback disabled by --disable-retries.")

    if not runner.anthropic_client:
        raise NonRetryableProviderError("Fallback to Claude Opus required but anthropic_client is missing.")

    model_suffix = "thinking-60000" if thinking else "no-thinking"
    fallback_config = ModelConfig("anthropic", CLAUDE_OPUS_BASE, 60000 if thinking else 0)
    
    context_str = f"[{runner.task_id}:{runner.test_index}] ({runner.step_name})" if runner.task_id and runner.step_name else ""
    log_msg = f"{context_str} OpenAI Job failed: {reason}. Falling back to Claude Opus ({model_suffix})..."
    logger.warning(f"[BACKGROUND] {log_msg}")

    if runner.run_timestamp:
        log_failure(
            run_timestamp=runner.run_timestamp,
            task_id=runner.task_id if runner.task_id else "UNKNOWN",
            run_id="OPENAI_BG_FAILURE",
            error=RetryableProviderError(f"OpenAI Job failed: {reason}. Falling back..."),
            model=runner.full_model_name,
            step=runner.step_name if runner.step_name else (runner.task_id if runner.task_id else "UNKNOWN"),
            test_index=runner.test_index,
            is_retryable=True
        )

    duration = time.perf_counter() - start_ts
    if runner.timing_tracker is not None:
        runner.timing_tracker.append({
            "type": "attempt",
            "model": runner.full_model_name,
            "duration": duration,
            "status": "failed",
            "error": f"Failed: {reason}. Falling back."
        })

    response = call_anthropic(
        runner.anthropic_client,
        prompt,
        fallback_config,
        image_path=image_path,
        return_strategy=False,
        verbose=runner.verbose,
        task_id=runner.task_id,
        test_index=runner.test_index,
        run_timestamp=runner.run_timestamp,
        model_alias=f"claude-opus-4.5-{model_suffix}",
        timing_tracker=runner.timing_tracker
    )
    response.model_name = f"claude-opus-4.5-{model_suffix}"
    return response
