import time
from typing import List, Dict, Any, Optional

from src.models import call_model, parse_model_arg, calculate_cost
from src.parallel.worker_utils.tokens import acquire_rate_limit_token

class ExecutionContext:
    def __init__(self):
        self.cost = 0.0
        self.duration = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.thought_tokens = 0
        self.cached_tokens = 0
        self.timings = []
        self.full_response = ""

    def update_from_response(self, response, model_name: str):
        self.full_response = response.text
        self.input_tokens += response.prompt_tokens
        self.output_tokens += response.completion_tokens
        self.thought_tokens += response.thought_tokens
        self.cached_tokens += response.cached_tokens
        
        try:
            model_config = parse_model_arg(model_name)
            self.cost += calculate_cost(model_config, response)
        except Exception:
            pass

def execute_model_call(
    client_config: Dict[str, Any],
    prompt: str,
    model_name: str,
    context: ExecutionContext,
    verbose: bool = False,
    prefix: str = "",
    image_path: str = None,
    task_id: str = None,
    test_index: int = None,
    step_name: str = None,
    use_background: bool = False,
    run_timestamp: str = None,
    execution_mode: str = "grid"
):
    # Acquire token
    acquire_rate_limit_token(model_name, verbose, prefix)

    start_ts = time.perf_counter()
    
    # Debug ID for tracking hangs
    llm_exec_id = f"LLM:{task_id}:{test_index}:{model_name}:{time.time():.6f}"
    import sys
    # print(f"DEBUG_LLM: START {llm_exec_id}", file=sys.stderr)
    
    try:
        response = call_model(
            openai_client=client_config['openai_client'],
            anthropic_client=client_config['anthropic_client'],
            google_keys=client_config['google_keys'],
            prompt=prompt,
            model_arg=model_name,
            image_path=image_path,
            return_strategy=False,
            verbose=verbose,
            task_id=task_id,
            test_index=test_index,
            step_name=step_name,
            use_background=use_background,
            run_timestamp=run_timestamp,
            timing_tracker=context.timings,
            enable_code_execution=(execution_mode == "v4")
        )
    finally:
        # print(f"DEBUG_LLM: FINISH {llm_exec_id}", file=sys.stderr)
        pass

    context.duration += time.perf_counter() - start_ts
    
    # Update context metrics
    context.update_from_response(response, model_name)
    
    return response
