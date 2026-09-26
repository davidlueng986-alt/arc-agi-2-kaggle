from datetime import datetime
import json
import sys
from pathlib import Path

from src.submission_utils.common import numpy_converter
from src.submission_utils.formatting import (
    format_timestamp, 
    get_iso_now,
    build_usage_data, 
    build_cost_data, 
    create_metadata,
    extract_solution_candidates
)
from src.submission_utils.statistics import aggregate_results

def generate_submission(final_results, submission_dir_path: str, run_timestamp: str):
    submission_dir = Path(submission_dir_path)
    submission_dir.mkdir(parents=True, exist_ok=True)
    submission_file = submission_dir / "submission.json"
    
    # 1. Group results by task_id
    task_results = {}
    for task_id, test_idx, preds in final_results:
        if not preds:
            continue
        if task_id not in task_results:
            task_results[task_id] = {}
        task_results[task_id][test_idx] = preds

    formatted_submission = {}
    
    # 2. Process each task for the submission.json file
    for task_id, tests in task_results.items():
        max_idx = max(tests.keys())
        formatted_submission[task_id] = []
        task_aggregated_data = []
        
        for i in range(1, max_idx + 1):
            preds_raw = tests.get(i)
            
            candidates, usage_stats = extract_solution_candidates(preds_raw)
            
            attempt_1 = [[0]]
            attempt_2 = [[0]]
            correct_1 = None
            correct_2 = None
            reasoning_1 = None
            reasoning_2 = None
            
            if candidates:
                c1 = candidates[0]
                attempt_1 = c1.get("grid", [[0]])
                correct_1 = c1.get("is_correct")
                reasoning_1 = c1.get("reasoning_summary")
                
                if len(candidates) > 1:
                    c2 = candidates[1]
                    attempt_2 = c2.get("grid", [[0]])
                    correct_2 = c2.get("is_correct")
                    reasoning_2 = c2.get("reasoning_summary")
                else:
                    # Duplicate if only 1 attempt
                    attempt_2 = attempt_1
                    correct_2 = correct_1
                    reasoning_2 = reasoning_1
            
            # Formatted for submission.json (Kaggle format)
            formatted_submission[task_id].append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })
            
            # Prepare metadata for aggregated task file
            start_iso = format_timestamp(run_timestamp)
            end_iso = get_iso_now()
            usage_data = build_usage_data(usage_stats)
            cost_data = build_cost_data(usage_stats)
            
            metadata_template_1 = create_metadata(
                start_iso, end_iso, reasoning_1, usage_data, cost_data, task_id, i - 1
            )
            metadata_template_2 = create_metadata(
                start_iso, end_iso, reasoning_2, usage_data, cost_data, task_id, i - 1
            )

            task_aggregated_data.append({
                "attempt_1": {
                    "answer": attempt_1, 
                    "correct": correct_1,
                    "metadata": metadata_template_1
                },
                "attempt_2": {
                    "answer": attempt_2, 
                    "correct": correct_2,
                    "metadata": metadata_template_2
                }
            })

        # Save per-task JSON
        task_file = submission_dir / f"{task_id}.json"
        try:
            with open(task_file, "w") as f:
                json.dump(task_aggregated_data, f, indent=2, default=numpy_converter)
        except Exception as e:
            print(f"Error saving task file {task_file}: {e}", file=sys.stderr)

    # Save main submission.json
    try:
        with open(submission_file, "w") as f:
            json.dump(formatted_submission, f, default=numpy_converter)
        print(f"Submission file saved to: {submission_file}")
    except Exception as e:
        print(f"Error saving submission file: {e}", file=sys.stderr)

    # 3. Calculate and save aggregate statistics (results.json)
    results_data = aggregate_results(task_results)
    
    results_file = submission_dir / "results.json"
    try:
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=4, default=numpy_converter)
        print(f"Results file saved to: {results_file}")
    except Exception as e:
        print(f"Error saving results file: {e}", file=sys.stderr)