from typing import Dict, Any, List

def format_worker_result(
    model_name: str,
    requested_model: str,
    run_id: str,
    grid: List[List[int]] = None,
    is_correct: bool = False,
    context = None, # ExecutionContext
    prompt: str = "",
    verification_details: Dict = None,
    v3_details: Dict = None,
    detailed_logs: Dict = None,
    error_message: str = None
) -> Dict[str, Any]:
    
    full_response = context.full_response if context else error_message
    if error_message and context:
         full_response = error_message

    result = {
        "model": model_name,
        "requested_model": requested_model,
        "run_id": run_id,
        "grid": grid,
        "is_correct": is_correct,
        "prompt": prompt,
        "full_response": full_response,
        "verification_details": verification_details,
        "v3_details": v3_details,
        "detailed_logs": detailed_logs
    }
    
    if context:
        result.update({
            "cost": context.cost,
            "duration": context.duration,
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens + context.thought_tokens,
            "reasoning_tokens": context.thought_tokens,
            "cached_tokens": context.cached_tokens,
            "timing_breakdown": context.timings,
        })
    else:
        result.update({
            "cost": 0.0,
            "duration": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cached_tokens": 0,
            "timing_breakdown": [],
        })

    return result
