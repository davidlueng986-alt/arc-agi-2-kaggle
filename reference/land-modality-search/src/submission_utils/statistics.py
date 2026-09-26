from typing import Dict, Any, List

def calculate_task_stats(task_results: Dict[str, Any]) -> Dict[str, Any]:
    task_stats = {
        "solved_tests": 0,
        "score_valid": True,
        "cost": 0.0,
        "output_tokens": 0,
        "total_tokens": 0,
        "duration": 0.0,
        "attempts_count": 0,
        "empty_attempts": 0
    }
    
    num_tests = len(task_results)
    
    for i, preds_raw in task_results.items():
        solutions = preds_raw
        usage_stats = None
        if isinstance(preds_raw, tuple) and len(preds_raw) == 2 and isinstance(preds_raw[1], dict):
            solutions, usage_stats = preds_raw
        
        test_solved = False
        test_result_known = True
        
        current_attempt_1 = [[0]]
        current_attempt_2 = [[0]]
        a1_correct = None
        a2_correct = None
        
        # Check candidates
        if solutions and isinstance(solutions, list) and len(solutions) > 0:
            cand1 = solutions[0]
            if isinstance(cand1, dict):
                if "grid" in cand1:
                    current_attempt_1 = cand1["grid"]
                a1_correct = cand1.get("is_correct")

        if solutions and isinstance(solutions, list) and len(solutions) > 1:
            cand2 = solutions[1]
            if isinstance(cand2, dict):
                if "grid" in cand2:
                    current_attempt_2 = cand2["grid"]
                a2_correct = cand2.get("is_correct")
        elif solutions and isinstance(solutions, list) and len(solutions) > 0:
            current_attempt_2 = current_attempt_1
            a2_correct = a1_correct

        # Evaluate logic
        if (a1_correct is True) or (a2_correct is True):
            test_solved = True
        
        if not test_solved:
            if (a1_correct is None) or (a2_correct is None):
                test_result_known = False

        if test_solved:
            task_stats["solved_tests"] += 1
        elif not test_result_known:
            task_stats["score_valid"] = False
        
        if current_attempt_1 == []:
            task_stats["empty_attempts"] += 1
        if current_attempt_2 == []:
            task_stats["empty_attempts"] += 1
        
        task_stats["attempts_count"] += 2
        
        if usage_stats:
            task_stats["cost"] += usage_stats.get("total_cost", 0.0) or 0.0
            task_stats["output_tokens"] += usage_stats.get("completion_tokens", 0) or 0
            task_stats["total_tokens"] += usage_stats.get("total_tokens", 0) or 0
            task_stats["duration"] += usage_stats.get("total_duration", 0.0) or 0.0
            
    return task_stats

def aggregate_results(all_task_results: Dict[str, Dict[int, Any]]) -> Dict[str, Any]:
    total_score = 0.0
    global_score_valid = True
    total_cost = 0.0
    total_attempts = 0
    total_output_tokens = 0
    total_tokens = 0
    total_duration = 0.0
    total_empty_attempts = 0
    
    task_results_map = {}
    unique_tasks = sorted(list(all_task_results.keys()))
    num_tasks = len(unique_tasks)
    
    for task_id in unique_tasks:
        tests = all_task_results[task_id]
        stats = calculate_task_stats(tests)
        
        num_tests = len(tests)
        task_score = (stats["solved_tests"] / num_tests) if num_tests > 0 else 0.0
        
        if not stats["score_valid"]:
            task_score = None
            global_score_valid = False
        elif global_score_valid:
            total_score += task_score
            
        total_cost += stats["cost"]
        total_attempts += stats["attempts_count"]
        total_output_tokens += stats["output_tokens"]
        total_tokens += stats["total_tokens"]
        total_duration += stats["duration"]
        total_empty_attempts += stats["empty_attempts"]
        
        task_results_map[task_id] = {
            "score": task_score,
            "cost": stats["cost"],
            "attempts": stats["attempts_count"],
            "output_tokens": stats["output_tokens"],
            "total_tokens": stats["total_tokens"],
            "duration": stats["duration"],
            "num_attempts_with_empty_list": stats["empty_attempts"]
        }
        
    avg_cost_per_task = (total_cost / num_tasks) if num_tasks > 0 else 0.0
    avg_cost_per_attempt = (total_cost / total_attempts) if total_attempts > 0 else 0.0
    avg_output_tokens_per_task = (total_output_tokens / num_tasks) if num_tasks > 0 else 0.0
    avg_total_tokens_per_task = (total_tokens / num_tasks) if num_tasks > 0 else 0.0
    avg_duration_per_task = (total_duration / num_tasks) if num_tasks > 0 else 0.0
    
    return {
        "score": total_score if global_score_valid else None,
        "total_tasks": num_tasks,
        "total_cost": total_cost,
        "total_attempts": total_attempts,
        "avg_cost_per_task": avg_cost_per_task,
        "avg_cost_per_attempt": avg_cost_per_attempt,
        "avg_output_tokens_per_task": avg_output_tokens_per_task,
        "avg_total_tokens_per_task": avg_total_tokens_per_task,
        "avg_duration_per_task": avg_duration_per_task,
        "task_results": task_results_map,
        "num_attempts_with_empty_list": total_empty_attempts
    }
