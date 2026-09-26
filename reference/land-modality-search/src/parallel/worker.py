import sys
import os
import traceback
from typing import Optional, List, Dict

from src.grid import parse_grid_from_text, verify_prediction
from src.logging import log_failure
from src.parallel.codegen import extract_and_run_solver

# Refactored modules
from src.parallel.worker_utils.model_execution import execute_model_call, ExecutionContext
from src.parallel.worker_utils.v3_pipeline import run_v3_pipeline
from src.parallel.worker_utils.results import format_worker_result

def run_single_model(
    model_name, 
    run_id, 
    prompt, 
    test_example, 
    openai_client, 
    anthropic_client, 
    google_keys, 
    verbose, 
    image_path=None, 
    run_timestamp=None, 
    task_id=None, 
    test_index=None, 
    step_name=None, 
    use_background=False, 
    execution_mode="grid", 
    train_examples=None, 
    all_test_examples=None
):
    original_model_name = model_name
    prefix = f"[{run_id}]"
    if task_id is not None:
         prefix = f"[{run_id}|{task_id}:{test_index}]"
    
    if verbose:
        print(f"{prefix} Initiating call...")
        if image_path:
            print(f"{prefix} Including image: {image_path}")

    context = ExecutionContext()
    client_config = {
        'openai_client': openai_client,
        'anthropic_client': anthropic_client,
        'google_keys': google_keys
    }

    v3_details = None
    verification_details = None
    detailed_logs = None

    try:
        # 1. Execute Main Model Call
        response = execute_model_call(
            client_config=client_config,
            prompt=prompt,
            model_name=model_name,
            context=context,
            verbose=verbose,
            prefix=prefix,
            image_path=image_path,
            task_id=task_id,
            test_index=test_index,
            step_name=step_name,
            use_background=use_background,
            run_timestamp=run_timestamp,
            execution_mode=execution_mode
        )
        detailed_logs = getattr(response, "detailed_logs", None)

        # Handle fallback
        if response.model_name and response.model_name != model_name:
            if verbose:
                print(f"{prefix} Model fallback occurred: {model_name} -> {response.model_name}")
            run_id = run_id.replace(model_name, response.model_name, 1)
            model_name = response.model_name

        if verbose:
            print(f"{prefix} Response received.")

        grid_text = response.text

        # 2. V3 Pipeline (Optional)
        if execution_mode == "v3":
            grid_text, v3_details = run_v3_pipeline(
                hypothesis_plan=grid_text,
                train_examples=train_examples,
                all_test_examples=all_test_examples,
                client_config=client_config,
                model_name=model_name,
                context=context,
                verbose=verbose,
                prefix=prefix,
                image_path=image_path,
                task_id=task_id,
                test_index=test_index,
                step_name=step_name,
                use_background=use_background,
                run_timestamp=run_timestamp,
                execution_mode=execution_mode
            )

        # 3. Extraction & Execution
        predicted_grid = None
        
        if execution_mode in ("code", "v3", "v4"):
            try:
                predicted_grid, verification_details = extract_and_run_solver(
                    grid_text, 
                    test_example.input, 
                    train_examples=train_examples, 
                    task_id=task_id, 
                    test_index=test_index
                )
            except Exception as e:
                if verbose:
                    print(f"{prefix} Code Execution Failed: {e}")
                if verification_details is None:
                    verification_details = {
                        "status": "FAIL_EXTRACTOR_CRASH",
                        "error": str(e),
                        "traceback": traceback.format_exc()
                    }
        else:
            try:
                predicted_grid = parse_grid_from_text(grid_text)
            except ValueError as e:
                if verbose:
                    print(f"{prefix} Result: FAIL (Parse Error: {e})")

        # 4. Verification
        is_correct = False
        try:
            is_correct = verify_prediction(predicted_grid, test_example.output)
            
            if verbose:
                result_str = "PASS" if is_correct else ("FAIL" if is_correct is False else "UNKNOWN")
                print(f"{prefix} Result: {result_str}")
                
        except ValueError:
            is_correct = False

        return format_worker_result(
            model_name=model_name,
            requested_model=original_model_name,
            run_id=run_id,
            grid=predicted_grid,
            is_correct=is_correct,
            context=context,
            prompt=prompt,
            verification_details=verification_details,
            v3_details=v3_details,
            detailed_logs=detailed_logs
        )

    except Exception as e:
        # Check for concise error types
        error_str = str(e)
        error_lower = error_str.lower()
        concise_msg = None
        
        if "openai" in error_lower and ("max_output_tokens" in error_lower or "hit token limit" in error_lower):
            concise_msg = "Err: FAIL: OpenAI Max Tokens"
        elif "openai" in error_lower and "timed out after" in error_lower:
            concise_msg = "Err: FAIL: OpenAI Timeout 3300s"
        elif "violating our usage policy" in error_lower:
            concise_msg = "Err: FAIL: OpenAI Policy Violation"
        elif "server_error" in error_lower:
            concise_msg = "Err: FAIL: OpenAI Server Error"
        elif "claude-opus" in error_lower and ("peer closed connection" in error_lower or "incomplete chunked read" in error_lower):
            concise_msg = "Err: FAIL: Claude Connection Closed"
        elif "gemini" in error_lower and ("499" in error_lower or "cancelled" in error_lower):
            concise_msg = "Err: FAIL: Gemini Cancelled (499)"

        if concise_msg:
             # Brief summary to stdout
             print(concise_msg)
        else:
            # Full critical error dump to stderr
            error_msg = f"\n!!! CRITICAL ERROR in {model_name} ({run_id}) !!!\n{str(e)}\n{traceback.format_exc()}\n"
            try:
                os.write(2, error_msg.encode('utf-8', errors='replace'))
            except OSError:
                pass

            print(f"Error during execution: {e}", file=sys.stderr)
        
        if run_timestamp:
             log_failure(
                run_timestamp=run_timestamp,
                task_id=task_id if task_id else "UNKNOWN",
                run_id=run_id,
                error=e,
                model=model_name,
                test_index=test_index
            )
            
        return format_worker_result(
            model_name=model_name,
            requested_model=original_model_name,
            run_id=run_id,
            grid=None,
            is_correct=False,
            context=context, # May be partial
            prompt=prompt,
            verification_details=verification_details,
            v3_details=v3_details,
            detailed_logs=detailed_logs,
            error_message=str(e)
        )
