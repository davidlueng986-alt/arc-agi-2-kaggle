import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.parallel.worker import run_single_model

def run_models_in_parallel(models_to_run, run_id_counts, step_name, prompt, test_example, openai_client, anthropic_client, google_keys, verbose, image_path=None, run_timestamp=None, task_id=None, test_index=None, completion_message: str = None, on_task_complete=None, use_background=False, execution_mode="grid", train_examples=None, all_test_examples=None, codegen_version: str = None):
    all_results = []
    
    # Wrapper for debugging queue times
    def debug_run_single_model(queue_time, *args, **kwargs):
        start_wait = time.time() - queue_time
        if start_wait > 0.1:  # Only print if waiting more than 100ms
            import sys
            run_id = args[1] if len(args) > 1 else kwargs.get('run_id', 'unknown')
            print(f"DEBUG: Task {run_id} waited in queue for {start_wait:.2f}s", file=sys.stderr)
        return run_single_model(*args, **kwargs)

    with ThreadPoolExecutor(max_workers=20) as executor:
        
        # Generate unique run IDs
        run_list = []
        for model_name in models_to_run:
            count = run_id_counts.get(model_name, 0) + 1
            run_id_counts[model_name] = count
            run_id = f"{model_name}_{count}_{step_name}"
            
            # Per-model prompt generation if codegen_version is provided
            current_prompt = prompt
            if codegen_version:
                from src.tasks import build_prompt_codegen
                current_prompt = build_prompt_codegen(train_examples, test_examples=all_test_examples, version=codegen_version, model_name=model_name)
                # Debug print for specific models if needed (we'll skip general printing to keep logs clean)

            run_list.append({"name": model_name, "run_id": run_id, "prompt": current_prompt})

        future_to_run_id = {
            executor.submit(
                debug_run_single_model,
                time.time(), # Capture queue time
                run["name"], run["run_id"], run["prompt"], test_example, openai_client, anthropic_client, google_keys, verbose, image_path, run_timestamp, task_id, test_index, step_name, use_background, execution_mode, train_examples, all_test_examples
            ): run["run_id"]
            for run in run_list
        }

        total_tasks = len(future_to_run_id)
        completed_count = 0

        for future in as_completed(future_to_run_id):
            completed_count += 1
            run_id = future_to_run_id[future]
            try:
                res = future.result()
                if res:
                    all_results.append(res)
                
                # Handle progress updates
                if on_task_complete:
                    on_task_complete()
                elif completion_message:
                    remaining = total_tasks - completed_count
                    print(f"{completion_message}: {remaining} left")
                    
            except Exception as e:
                print(f"Model run {run_id} failed: {e}")
                
    return all_results
