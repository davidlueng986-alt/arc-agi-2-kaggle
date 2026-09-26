from datetime import datetime
from typing import Dict, Any, List, Optional

def format_timestamp(ts_str: str) -> str:
    try:
        start_dt = datetime.strptime(ts_str, "%Y-%m-%d_%H-%M-%S")
        return start_dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    except ValueError:
        return ts_str

def get_iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S+00:00")

def halve_usage(value, is_int=False):
    if value is None:
        return None
    return value // 2 if is_int else value / 2

def build_usage_data(usage_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not usage_stats:
        return None
        
    return {
        "prompt_tokens": halve_usage(usage_stats.get("prompt_tokens"), True),
        "completion_tokens": halve_usage(usage_stats.get("completion_tokens"), True),
        "total_tokens": halve_usage(usage_stats.get("total_tokens"), True),
        "completion_tokens_details": {
            "reasoning_tokens": halve_usage(usage_stats.get("reasoning_tokens", 0), True),
            "accepted_prediction_tokens": halve_usage(usage_stats.get("accepted_prediction_tokens"), True),
            "rejected_prediction_tokens": halve_usage(usage_stats.get("rejected_prediction_tokens", 0), True)
        }
    }

def build_cost_data(usage_stats: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not usage_stats:
        return None

    return {
        "prompt_cost": halve_usage(usage_stats.get("prompt_cost")),
        "completion_cost": halve_usage(usage_stats.get("completion_cost")),
        "reasoning_cost": halve_usage(usage_stats.get("reasoning_cost")),
        "total_cost": halve_usage(usage_stats.get("total_cost"))
    }

def create_metadata(
    run_timestamp_iso: str,
    end_timestamp_iso: str,
    reasoning_summary: Optional[str],
    usage_data: Optional[Dict],
    cost_data: Optional[Dict],
    task_id: str,
    pair_index: int
) -> Dict[str, Any]:
    return {
        "model": "Johan_Land_Solver_V6",
        "provider": "Johan_Land",
        "start_timestamp": run_timestamp_iso,
        "end_timestamp": end_timestamp_iso,
        "choices": [
            {"index": 0, "message": {"role": "user", "content": "NA"}},
            {"index": 1, "message": {"role": "assistant", "content": "NA"}}
        ],
        "reasoning_summary": reasoning_summary,
        "kwargs": {
            "background": "mixed",
            "stream": "mixed",
            "reasoning": {
                "effort": "max with fallbacks for latency, cost and stability"
            },
            "max_output_tokens": "max less delta for stability"
        },
        "usage": usage_data,
        "cost": cost_data,
        "task_id": task_id,
        "pair_index": pair_index,
        "test_id": "Johan_Land_Solver_V6_Eval_2_Full_Run"
    }

def extract_solution_candidates(preds_raw):
    """
    Extracts solutions and usage stats from raw predictions.
    Returns: (solutions_list, usage_stats_dict)
    """
    solutions = preds_raw
    usage_stats = None
    
    if isinstance(preds_raw, tuple) and len(preds_raw) == 2 and isinstance(preds_raw[1], dict):
        solutions, usage_stats = preds_raw
        
    candidates = []
    if isinstance(solutions, list):
        for p in solutions:
            if isinstance(p, dict) and "grid" in p:
                candidates.append(p)
    elif isinstance(solutions, tuple) and len(solutions) == 2 and isinstance(solutions[0], list):
        # Fallback legacy checks
        pass
        
    return candidates, usage_stats
