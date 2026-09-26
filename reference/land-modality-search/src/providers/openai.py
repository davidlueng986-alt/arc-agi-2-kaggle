from openai import OpenAI
from anthropic import Anthropic

from src.types import ModelConfig, ModelResponse
from src.providers.openai_runner import OpenAIRequestRunner

def call_openai_internal(
    client: OpenAI,
    prompt: str,
    config: ModelConfig,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
    task_id: str = None,
    test_index: int = None,
    step_name: str = None,
    use_background: bool = False,
    run_timestamp: str = None,
    anthropic_client: Anthropic = None,
    model_alias: str = None,
    timing_tracker: list[dict] = None,
    enable_code_execution: bool = False,
) -> ModelResponse:
    """
    Main entry point for OpenAI provider.
    Delegates actual execution to OpenAIRequestRunner.
    """
    runner = OpenAIRequestRunner(
        client=client,
        config=config,
        anthropic_client=anthropic_client,
        task_id=task_id,
        test_index=test_index,
        step_name=step_name,
        run_timestamp=run_timestamp,
        model_alias=model_alias,
        timing_tracker=timing_tracker,
        verbose=verbose
    )
    return runner.run(prompt, image_path=image_path, return_strategy=return_strategy, use_background=use_background, enable_code_execution=enable_code_execution)