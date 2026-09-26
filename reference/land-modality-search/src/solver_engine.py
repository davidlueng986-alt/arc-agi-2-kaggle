import sys
from pathlib import Path

from src.logging import log_failure, set_log_dir
from src.solver.state import SolverState
from src.solver.steps import run_step_1, run_step_3, run_step_5, check_is_solved

# Re-export run_solver_mode for backward compatibility if imported elsewhere
def run_solver_mode(task_id: str, test_index: int, verbose: int, is_testing: bool = False, run_timestamp: str = None, task_path: Path = None, answer_path: Path = None, step_5_only: bool = False, objects_only: bool = False, force_step_5: bool = False, force_step_2: bool = False, judge_model: str = "gpt-5.2-xhigh", old_pick_solution: bool = False, task_status=None,     openai_background: bool = True, enable_step_3_and_4: bool = False, judge_consistency_enable: bool = False, judge_duo_pick_enable: bool = True, codegen_params: str = "gpt-5.2-low=v1b,gpt-5.2-low=v4,gemini-3-low=v4", step1_models: str = "gpt-5.2-none,claude-opus-4.5-no-thinking", disable_step_1_standard_models: bool = False, logs_directory: str = "logs/", task_data: dict = None):
    
    set_log_dir(logs_directory)

    # Initialize State
    try:
        state = SolverState(task_id, test_index, verbose, is_testing, run_timestamp, task_path, answer_path, judge_model, old_pick_solution=old_pick_solution, task_status=task_status, openai_background=openai_background, judge_consistency_enable=judge_consistency_enable, judge_duo_pick_enable=judge_duo_pick_enable, codegen_prompt=None, logs_directory=logs_directory, task_data=task_data)
    except Exception as e:
        print(f"Error initializing solver state: {e}", file=sys.stderr)
        raise e

    try:
        # Determine Step 1 standard models
        if disable_step_1_standard_models:
            models_step1_standard = []
            if verbose >= 1:
                print("Step 1 Standard Models disabled by user.")
        elif step1_models:
            models_step1_standard = [m.strip() for m in step1_models.split(",")]
        else:
            models_step1_standard = ["gpt-5.2-none", "claude-opus-4.5-no-thinking"]

        if is_testing:
            # Models for --solver-testing
            # models_step1 = ["claude-opus-4.5-thinking-4000", "gpt-5.1-none"] # Overridden by codegen_models argument
            models_step3 = ["claude-sonnet-4.5-no-thinking", "gpt-5.1-low"]
            
            # Step 5 - Testing
            models_step5_deep = ["gpt-5.2-low"]
            models_step5_image = ["gpt-5.2-low"]
            params_step5_codegen = "gemini-3-low=v4,gpt-5.2-low=v1b,gpt-5.2-low=v4"
            
            hint_generation_model = "gpt-5.1-low"
        else:
            # Models for --solver
            # models_step1 = ["gemini-3-high"] * 2 + ["claude-opus-4.5-thinking-60000"] * 2 + ["gpt-5.2-xhigh"] * 6
            models_step3 = ["claude-opus-4.5-thinking-60000", "gemini-3-high", "gemini-3-high", "gpt-5.2-xhigh", "gpt-5.2-xhigh"]
            
            # Step 5 - Production
            models_step5_deep = ["gpt-5.2-xhigh"] * 3
            models_step5_image = ["gpt-5.2-xhigh"] * 3 + ["gemini-3-high"] * 1
            
            # Construct large codegen string
            # 1x gemini-3-high v4 + 6x gpt-5.2-xhigh v1b
            parts = []
            parts.extend(["gemini-3-high=v4"] * 1)
            parts.extend(["gpt-5.2-xhigh=v1b"] * 6)
            params_step5_codegen = ",".join(parts)

            hint_generation_model = "gpt-5.2-xhigh"

        # Skip logic
        should_run_early_steps = not (step_5_only or objects_only)

        if should_run_early_steps:
            # STEP 1
            run_step_1(state, models_step1_standard, codegen_params)

            # STEP 2
            state.set_status(step=2)
            finish, _ = check_is_solved(state, "step_2", force_finish=force_step_2, continue_if_solved=force_step_5)
            if finish:
                return state.finalize("step_finish")

            if enable_step_3_and_4:
                # STEP 3
                run_step_3(state, models_step3)

                # STEP 4
                state.set_status(step=4)
                finish, _ = check_is_solved(state, "step_4", continue_if_solved=force_step_5)
                if finish:
                    return state.finalize("step_finish")
            else:
                if verbose >= 1:
                    print("Skipping Steps 3-4 (Disabled by default)")
        else:
             print("\nSkipping Steps 1-4 (Deep Search Only Mode)")

        # STEP 5
        run_step_5(state, models_step5_deep, models_step5_image, params_step5_codegen, hint_generation_model, enable_hints=False, enable_objects=False, objects_only=objects_only)

        # STEP FINISH
        return state.finalize("step_finish")
        
    except Exception as e:
        error_msg = str(e)
        if len(error_msg) > 50:
            error_msg = error_msg[:47] + "..."
        log_failure(
            run_timestamp=run_timestamp if run_timestamp else "unknown_timestamp",
            task_id=task_id,
            run_id="SOLVER_ENGINE_MAIN_LOOP",
            error=e,
            model="SYSTEM",
            step="MAIN",
            test_index=test_index,
            log_dir=logs_directory
        )
        print(f"CRITICAL ERROR in run_solver_mode: {e}", file=sys.stderr)
        raise e
