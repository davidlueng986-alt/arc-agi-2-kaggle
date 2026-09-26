from typing import Dict, Any, Optional, List
from src.types import Example
from src.tasks import build_prompt_codegen_v3_stage2
from src.parallel.worker_utils.model_execution import execute_model_call, ExecutionContext

def run_v3_pipeline(
    hypothesis_plan: str,
    train_examples: List[Example],
    all_test_examples: List[Example],
    client_config: Dict[str, Any],
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
    execution_mode: str = "v3"
) -> Dict[str, Any]:
    
    # Store Stage 1 Details
    v3_details = {
        "stage_1": {
            "prompt": "NA (Handled in Caller)",
            "response": hypothesis_plan,
            "cost": context.cost,
            "duration": context.duration,
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens,
            "thought_tokens": context.thought_tokens,
            "cached_tokens": context.cached_tokens
        },
        "stage_2": {"status": "NOT_STARTED"}
    }

    prompt_stage2 = build_prompt_codegen_v3_stage2(train_examples, all_test_examples, hypothesis_plan)
    
    if verbose:
        print(f"{prefix} Initiating V3 Stage 2 (Engineer)...")

    try:
        response_s2 = execute_model_call(
            client_config=client_config,
            prompt=prompt_stage2,
            model_name=model_name,
            context=context, # Accumulates cost/tokens
            verbose=verbose,
            prefix=prefix,
            image_path=image_path,
            task_id=task_id,
            test_index=test_index,
            step_name=f"{step_name}_s2",
            use_background=use_background,
            run_timestamp=run_timestamp,
            execution_mode=execution_mode
        )

        grid_text = response_s2.text
        
        # We need specific metrics for Stage 2 for the logs
        # Note: 'context' has aggregated values, so we'd need to capture diffs if we want precise S2 stats
        # For simplicity, we can use the response object directly for tokens, but cost is tricky without recalculating.
        # But `context.update_from_response` was already called inside `execute_model_call`.
        
        # Let's approximate S2 stats from response_s2 directly for the details log
        from src.models import parse_model_arg, calculate_cost
        try:
            cost_s2 = calculate_cost(parse_model_arg(model_name), response_s2)
        except:
            cost_s2 = 0.0

        v3_details["stage_2"] = {
            "status": "SUCCESS",
            "prompt": prompt_stage2,
            "response": grid_text,
            "cost": cost_s2,
            "duration": 0.0, # Not easily available without diffing context.duration or tracking inside execute
            "input_tokens": response_s2.prompt_tokens,
            "output_tokens": response_s2.completion_tokens,
            "thought_tokens": response_s2.thought_tokens,
            "cached_tokens": response_s2.cached_tokens
        }
        
        return grid_text, v3_details

    except Exception as e:
        if verbose:
            print(f"{prefix} V3 Stage 2 Failed: {e}")
            
        v3_details["stage_2"] = {
            "status": "FAILED",
            "error": str(e),
            "prompt": prompt_stage2
        }
        
        # Return error message as text so it fails gracefully later
        fail_text = hypothesis_plan + "\n\n[STAGE 2 FAILED: " + str(e) + "]"
        return fail_text, v3_details