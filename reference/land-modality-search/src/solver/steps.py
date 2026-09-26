import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.logging import PrefixedStdout
from src.tasks import build_prompt, build_prompt_codegen
from src.image_generation import generate_and_save_image
from src.hint_generation import generate_hint
from src.parallel import run_models_in_parallel
from src.selection import is_solved
from src.solver.pipelines import run_objects_pipeline_variant

def run_step_1(state, standard_models, codegen_params):
    state.set_status(step=1, phase="Shallow search")
    
    n_std = len(standard_models)
    
    # Parse Codegen Configuration from params string
    # Format: "model=version,model=version"
    codegen_jobs = []
    if codegen_params:
        try:
            for item in codegen_params.split(","):
                if "=" in item:
                    model, version = item.strip().split("=", 1)
                    model = model.strip()
                    version = version.strip()
                    # Determine execution mode: v4 enables python execution in LLM, others use local sandbox
                    exec_mode = "v4" if version == "v4" else "code"
                    
                    codegen_jobs.append({
                        "models": [model],
                        "version": version,
                        "exec_mode": exec_mode
                    })
        except Exception as e:
            print(f"Error parsing codegen_params '{codegen_params}': {e}", file=sys.stderr)
            # Fallback to safe default or empty? Let's fallback to empty to avoid crashing loop
            codegen_jobs = []
    
    n_code = sum(len(job["models"]) for job in codegen_jobs)
    total_models = n_std + n_code
    
    # Build descriptive breakdown string
    breakdown_parts = []
    if n_std > 0:
        breakdown_parts.append(f"{n_std} std")
    
    # Group codegen by version for summary
    version_counts = {}
    for job in codegen_jobs:
        v = job["version"]
        version_counts[v] = version_counts.get(v, 0) + len(job["models"])
    for v, count in version_counts.items():
        breakdown_parts.append(f"{count} {v}")
    
    print(f"Search: {' + '.join(breakdown_parts)}")

    step_1_log = {}
    if state.verbose >= 1:
        print(f"Running {total_models} models...")
    
    prompt_step1 = build_prompt(state.task.train, state.test_example)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        
        # 1. Standard Search
        if standard_models:
            f_std = executor.submit(run_models_in_parallel, standard_models, state.run_id_counts, "step_1", prompt_step1, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, completion_message="Search std", use_background=state.openai_background)
            futures.append(f_std)

        # 2. Codegen Jobs
        for i, job in enumerate(codegen_jobs):
            prompt_codegen = build_prompt_codegen(state.task.train, test_examples=state.task.test, version=job["version"])
            job_name = f"step_1_codegen_{job['version']}_{i}"
            
            f_code = executor.submit(
                run_models_in_parallel, 
                job["models"], 
                state.run_id_counts, 
                job_name, 
                prompt_codegen, 
                state.test_example, 
                state.openai_client, 
                state.anthropic_client, 
                state.google_keys, 
                state.verbose, 
                run_timestamp=state.run_timestamp, 
                task_id=state.task_id, 
                test_index=state.test_index, 
                completion_message=f"Search {job['version']}", 
                use_background=state.openai_background, 
                execution_mode=job["exec_mode"], 
                train_examples=state.task.train, 
                all_test_examples=state.task.test, 
                codegen_version=job["version"]
            )
            futures.append(f_code)
        
        all_results = []
        for f in futures:
            all_results.extend(f.result())
    
    state.process_results(all_results, step_1_log)
    state.log_step("step_1", step_1_log)

def run_step_3(state, models):
    state.set_status(step=3, phase="Extended search")
    print(f"Narrow search: {len(models)} left")
    step_3_log = {}
    if state.verbose >= 1:
        print(f"Running {len(models)} models...")
    prompt_step3 = build_prompt(state.task.train, state.test_example)
    results_step3 = run_models_in_parallel(models, state.run_id_counts, "step_3", prompt_step3, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, completion_message="Narrow search", use_background=state.openai_background)
    state.process_results(results_step3, step_3_log)
    state.log_step("step_3", step_3_log)

def check_is_solved(state, step_name, force_finish=False, continue_if_solved=False):
    state.set_status(phase="Eval")
    solved = is_solved(state.candidates_object)
    log = {"candidates_object": {str(k): v for k, v in state.candidates_object.items()}, "is_solved": solved}
    state.log_step(step_name, log)

    if solved:
        if continue_if_solved:
             print("Likely solution, continuing")
             return False, True
        else:
            print("Likely solution, exiting")
            return True, True

    if force_finish:
        print("No solution, exiting (forced)")
        return True, True # (finish_now, success) - success status is determined by finalize
    
    print("No solution, continuing")
    return False, False

def run_step_5(state, deep_models, image_models, codegen_params, hint_model, enable_hints=False, enable_objects=False, objects_only=False):
    state.set_status(step=5, phase="Full search")
    
    # Calculate tries
    if state.is_testing:
        gen_object_extraction = "claude-opus-4.5-no-thinking"
        gen_opus = "claude-opus-4.5-thinking-4000"
        gen_transformation = "gpt-5.1-low"
        unique_solvers = ["claude-opus-4.5-no-thinking", "gpt-5.1-none"]
    else:
        gen_object_extraction = "gemini-3-high"
        gen_opus = "claude-opus-4.5-thinking-60000"
        gen_transformation = "gpt-5.2-xhigh"
        unique_solvers = ["gemini-3-high"] * 2 + ["claude-opus-4.5-thinking-60000"] * 1 + ["gpt-5.2-xhigh"] * 3
        
    # Parse Codegen Configuration
    codegen_jobs = []
    if codegen_params:
        try:
            for item in codegen_params.split(","):
                if "=" in item:
                    model, version = item.strip().split("=", 1)
                    model = model.strip()
                    version = version.strip()
                    exec_mode = "v4" if version == "v4" else "code"
                    codegen_jobs.append({
                        "models": [model],
                        "version": version,
                        "exec_mode": exec_mode
                    })
        except Exception as e:
            print(f"Error parsing codegen_params '{codegen_params}': {e}", file=sys.stderr)
            codegen_jobs = []

    n_deep = len(deep_models)
    n_image = len(image_models)
    n_codegen = sum(len(job["models"]) for job in codegen_jobs)
    
    # Hints and Objects count
    # Hint: 1 run (which internally runs models, but here we count the main task? No, existing code counted `n_models + 1`. 
    # Actually existing code ran `run_hint_step` which runs `run_models_in_parallel` with `models`.
    # Let's assume hints use deep_models if enabled for now, or we need a hint_models list? 
    # The prompt implies we just disable them. If enabled, we'd need to know which models.
    # For now, if enabled, let's just use `deep_models` as a fallback or assume it's disabled.
    # Wait, the previous signature had `models`. I'll assume if hints are enabled they use `deep_models` for consistency or I should add `hint_models`. 
    # Given the instruction "Disable hints... we may reactivate them later", I will leave the logic mostly disabled but if active use deep_models.
    n_hint = len(deep_models) if enable_hints else 0 
    
    n_objects_models = len(unique_solvers)
    n_objects = ((n_objects_models + 2) * 1) if enable_objects else 0

    counters = {
        'deep': n_deep,
        'image': n_image,
        'codegen': n_codegen,
        'hint': n_hint,
        'objects': n_objects
    }
    
    if objects_only:
        # Override for objects-only mode
        counters = {k: 0 for k in counters}
        counters['objects'] = (n_objects_models + 2) * 1
        enable_objects = True # Force enable
        
    lock = threading.Lock()
    
    def update_progress(key):
        with lock:
            if key in counters:
                counters[key] -= 1
            # Deep/Image/Codegen/Hint/Objects
            d = counters['deep']
            i = counters['image']
            c = counters['codegen']
            h = counters['hint']
            o = counters['objects']
            print(f"Going DEEP: {d}/{i}/{c}/{h}/{o} left")

    # Initial Print
    d = counters['deep']
    i = counters['image']
    c = counters['codegen']
    h = counters['hint']
    o = counters['objects']
    print(f"Going DEEP: {d}/{i}/{c}/{h}/{o} left")

    step_5_log = {"trigger-deep-thinking": {}, "image": {}, "codegen": {}, "generate-hint": {}, "objects_pipeline": {}}

    # Generate image once for both visual and hint steps to avoid matplotlib race conditions
    common_image_path = f"logs/{state.run_timestamp}_{state.task_id}_{state.test_index}_step_5_common.png"
    if n_image > 0 or (enable_hints and n_hint > 0):
        generate_and_save_image(state.task, common_image_path)

    def run_deep_thinking_step(on_complete=None):
        if not deep_models: return "trigger-deep-thinking", [], None
        if state.verbose >= 1:
            print(f"Running {len(deep_models)} models with deep thinking...")
        prompt_deep = build_prompt(state.task.train, state.test_example, trigger_deep_thinking=True)
        results_deep = run_models_in_parallel(deep_models, state.run_id_counts, "step_5_deep_thinking", prompt_deep, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
        return "trigger-deep-thinking", results_deep, None

    def run_image_step(img_path, on_complete=None):
        if not image_models: return "image", [], None
        if state.verbose >= 1:
            print(f"Running {len(image_models)} models with image...")
        prompt_image = build_prompt(state.task.train, state.test_example, image_path=img_path)
        results_image = run_models_in_parallel(image_models, state.run_id_counts, "step_5_image", prompt_image, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, image_path=img_path, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
        return "image", results_image, None

    def run_hint_step(img_path, on_complete=None):
        # Use deep_models for hints if enabled
        models_for_hint = deep_models 
        if not models_for_hint: return "generate-hint", [], None
        
        if state.verbose >= 1:
            print(f"Running {len(models_for_hint)} models with generated hint...")
        
        hint_data = generate_hint(state.task, img_path, hint_model, state.verbose)
        extra_log = {}
        if hint_data and hint_data["hint"]:
            extra_log = {
                "model": hint_model,
                "requested_model": hint_data.get("requested_model"),
                "actual_model": hint_data.get("actual_model"),
                "Full raw LLM call": hint_data["prompt"],
                "Full raw LLM response": hint_data["full_response"],
                "Extracted hint": hint_data["hint"],
                "duration_seconds": round(hint_data.get("duration", 0), 2),
                "total_cost": hint_data.get("cost", 0),
                "input_tokens": hint_data.get("input_tokens", 0),
                "output_tokens": hint_data.get("output_tokens", 0),
                "cached_tokens": hint_data.get("cached_tokens", 0),
            }
            prompt_hint = build_prompt(state.task.train, state.test_example, strategy=hint_data["hint"])
            results_hint = run_models_in_parallel(models_for_hint, state.run_id_counts, "step_5_generate_hint", prompt_hint, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_complete, use_background=state.openai_background)
            return "generate-hint", results_hint, extra_log
        
        # If no hint generated, manually drain counter
        if on_complete:
            for _ in range(len(models_for_hint)):
                on_complete()
        return "generate-hint", [], extra_log

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = []
        
        if objects_only:
             futures.append(executor.submit(run_objects_pipeline_variant, state, gen_object_extraction, gen_transformation, "gpt_gen", unique_solvers, lambda: update_progress('objects'), use_background=state.openai_background))
        else:
            # 1. Deep Thinking
            if deep_models:
                futures.append(executor.submit(run_deep_thinking_step, lambda: update_progress('deep')))
            
            # 2. Image
            if image_models:
                futures.append(executor.submit(run_image_step, common_image_path, lambda: update_progress('image')))
            
            # 3. Hints
            if enable_hints:
                futures.append(executor.submit(run_hint_step, common_image_path, lambda: update_progress('hint')))
                
            # 4. Objects
            if enable_objects:
                futures.append(executor.submit(run_objects_pipeline_variant, state, gen_object_extraction, gen_transformation, "gpt_gen", unique_solvers, lambda: update_progress('objects'), use_background=state.openai_background))

            # 5. Codegen
            for i, job in enumerate(codegen_jobs):
                prompt_codegen = build_prompt_codegen(state.task.train, test_examples=state.task.test, version=job["version"])
                job_name = f"step_5_codegen_{job['version']}_{i}"
                
                # We need a wrapper to return the standard (name, results, log) format expected by the loop
                def run_codegen_wrapper(j_models, j_name, j_prompt, j_mode, j_ver, on_comp):
                    res = run_models_in_parallel(
                        j_models, 
                        state.run_id_counts, 
                        j_name, 
                        j_prompt, 
                        state.test_example, 
                        state.openai_client, 
                        state.anthropic_client, 
                        state.google_keys, 
                        state.verbose, 
                        run_timestamp=state.run_timestamp, 
                        task_id=state.task_id, 
                        test_index=state.test_index, 
                        completion_message=f"S5 codegen {j_ver}", 
                        use_background=state.openai_background, 
                        execution_mode=j_mode, 
                        train_examples=state.task.train, 
                        all_test_examples=state.task.test, 
                        codegen_version=j_ver,
                        on_task_complete=on_comp
                    )
                    return "codegen", res, {"version": j_ver}

                futures.append(executor.submit(
                    run_codegen_wrapper,
                    job["models"],
                    job_name,
                    prompt_codegen,
                    job["exec_mode"],
                    job["version"],
                    lambda: update_progress('codegen')
                ))

        for future in as_completed(futures):
            try:
                step_name, results, extra_log = future.result()
                
                # Handle logging
                if step_name.startswith("objects_pipeline_"):
                    state.process_results(results, step_5_log["objects_pipeline"])
                    step_5_log["objects_pipeline"][step_name.replace("objects_pipeline_", "")] = extra_log
                elif step_name == "codegen":
                    # For codegen, we might run multiple jobs, so we just append/merge to step_5_log["codegen"]
                    # Actually process_results handles the results list.
                    # extra_log contains version info if needed.
                    state.process_results(results, step_5_log["codegen"])
                else:
                    if step_name in step_5_log:
                        state.process_results(results, step_5_log[step_name])
                    
                    if step_name == "generate-hint" and extra_log:
                         step_5_log["generate-hint"]["hint_generation"] = extra_log
            except Exception as e:
                import traceback
                print(f"ERROR: A Step 5 parallel substep failed: {e}", file=sys.stderr)
                traceback.print_exc()

    state.log_step("step_5", step_5_log)
