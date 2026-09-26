from src.tasks import build_prompt, build_objects_extraction_prompt, build_objects_transformation_prompt
from src.parallel import run_single_model, run_models_in_parallel, extract_tag_content

def run_objects_pipeline_variant(state, generator_model_extract, generator_model_transform, variant_name, solver_models, on_task_complete=None, use_background=False):
    if state.verbose >= 1:
        print(f"Running Objects Pipeline ({variant_name}) with extract={generator_model_extract}, transform={generator_model_transform}...")
    pipeline_log = {}
    
    # Phase A: Extraction
    prompt_A = build_objects_extraction_prompt(state.task.train, state.test_example)
    res_A = run_single_model(generator_model_extract, f"step_5_{variant_name}_extract", prompt_A, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, use_background=use_background)
    if on_task_complete:
        on_task_complete()

    text_A_full = res_A.get("full_response", "")
    text_A = extract_tag_content(text_A_full, "objects_summary")
    if not text_A:
        text_A = text_A_full
    
    pipeline_log["extraction"] = {
        "model": res_A.get("model"),
        "requested_model": res_A.get("requested_model"),
        "actual_model": res_A.get("model"),
        "prompt": prompt_A,
        "response": text_A_full,
        "extracted_summary": text_A,
        "duration_seconds": round(res_A.get("duration", 0), 2),
        "total_cost": res_A.get("cost", 0),
        "input_tokens": res_A.get("input_tokens", 0),
        "output_tokens": res_A.get("output_tokens", 0),
        "cached_tokens": res_A.get("cached_tokens", 0),
        "timing_breakdown": res_A.get("timing_breakdown"),
    }

    # Phase B: Transformation
    prompt_B = build_objects_transformation_prompt(state.task.train, state.test_example, text_A)
    res_B = run_single_model(generator_model_transform, f"step_5_{variant_name}_transform", prompt_B, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, use_background=use_background)
    if on_task_complete:
        on_task_complete()

    text_B_full = res_B.get("full_response", "")
    text_B = extract_tag_content(text_B_full, "transformation_summary")
    if not text_B:
        text_B = text_B_full

    pipeline_log["transformation"] = {
        "model": res_B.get("model"),
        "requested_model": res_B.get("requested_model"),
        "actual_model": res_B.get("model"),
        "prompt": prompt_B,
        "response": text_B_full,
        "extracted_summary": text_B,
        "duration_seconds": round(res_B.get("duration", 0), 2),
        "total_cost": res_B.get("cost", 0),
        "input_tokens": res_B.get("input_tokens", 0),
        "output_tokens": res_B.get("output_tokens", 0),
        "cached_tokens": res_B.get("cached_tokens", 0),
        "timing_breakdown": res_B.get("timing_breakdown"),
    }

    # Phase C: Solution
    insertion_text = f"## Objects Description\n\n{text_A}\n\n## Transformation Description\n\n{text_B}"
    prompt_C = build_prompt(state.task.train, state.test_example, objects_insertion=insertion_text)
    
    pipeline_log["solution_prompt"] = prompt_C
    
    # We return the log data to be merged by the caller
    results_C = run_models_in_parallel(solver_models, state.run_id_counts, f"step_5_{variant_name}_sol", prompt_C, state.test_example, state.openai_client, state.anthropic_client, state.google_keys, state.verbose, run_timestamp=state.run_timestamp, task_id=state.task_id, test_index=state.test_index, on_task_complete=on_task_complete, use_background=use_background)
    
    return f"objects_pipeline_{variant_name}", results_C, pipeline_log
