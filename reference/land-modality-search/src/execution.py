import sys
import time
import signal
import os
from pathlib import Path
from src.solver_engine import run_solver_mode
from src.logging import PrefixedStdout
from src.parallel import set_rate_limit_scaling
from src.llm_utils import set_retries_enabled

def _hard_timeout_handler(signum, frame):
    print(f"\n!!! CRITICAL WATCHDOG TIMEOUT !!!\nProcess {os.getpid()} exceeded global time limit. Killing.", file=sys.stderr)
    os._exit(1) # Hard kill process, skipping cleanup handlers

def execute_task(args, task_path: Path, test_index: int, run_timestamp: str, rate_limit_scale: float = 1.0, answer_path: Path = None, status_counters=None, task_data: dict = None):
    if status_counters:
        running, remaining, finished, lock = status_counters
        with lock:
            running.value += 1
            remaining.value -= 1

    # Install Watchdog (8 hours hard limit per task)
    old_handler = signal.signal(signal.SIGALRM, _hard_timeout_handler)
    signal.alarm(28800) 

    try:
        # Propagate settings to worker process
        if args.disable_retries:
            set_retries_enabled(False)

        # Apply rate limit scaling (only affects this process)
        if rate_limit_scale != 1.0:
            set_rate_limit_scaling(rate_limit_scale)
            
        task_id = task_path.stem if task_path else "unknown"
        
        predictions = None

        # Standard verbose mode
        task_status = {
            "step": "0",
            "phase": "Init",
            "start_time": time.time()
        }
        
        def get_prefix():
            # Column widths
            w_status = 10
            w_task = 10
            w_step = 5
            w_time = 6
            
            # 1. Status Column
            if status_counters:
                r = status_counters[0].value
                q = status_counters[1].value
                d = status_counters[2].value
                # Format: ⚡3 ⏳1 ✅1
                # We remove brackets to save space and fit in column
                stat_text = f" ⚡{r} ⏳{q} ✅{d}"
            else:
                stat_text = " Single Task"
            
            col_status = stat_text.ljust(w_status)

            # 2. Task Column
            task_str = f"{task_id}:{test_index}"
            col_task = task_str.ljust(w_task)

            # 3. Step Column
            step = task_status.get("step", "0")
            phase = task_status.get("phase", "Init")
            
            if phase == "Finished":
                step_str = "DONE"
            else:
                step_str = f"{step}/5"
            col_step = step_str.center(w_step)

            # 5. Time Column
            elapsed = time.time() - task_status.get("start_time", time.time())
            time_str = f"{elapsed:.1f}s"
            col_time = time_str.rjust(w_time)

            return f"| {col_status} | {col_task} | {col_step} | {col_time} | "

        with PrefixedStdout(get_prefix, message_width=50):
            try:
                predictions = run_solver_mode(
                    task_id, test_index, args.verbose, 
                    is_testing=args.solver_testing, 
                    run_timestamp=run_timestamp, 
                    task_path=task_path, 
                    answer_path=answer_path,
                    step_5_only=args.step_5_only,
                    objects_only=args.objects_only,
                    force_step_5=args.force_step_5,
                    force_step_2=args.force_step_2,
                    judge_model=args.judge_model,
                    old_pick_solution=args.old_pick_solution,
                    task_status=task_status,
                    openai_background=args.openai_background,
                    enable_step_3_and_4=args.enable_step_3_and_4,
                    judge_consistency_enable=args.judge_consistency_enable,
                    judge_duo_pick_enable=args.judge_duo_pick,
                    codegen_params=args.codegen_params,
                    step1_models=args.step1_models,
                    disable_step_1_standard_models=args.disable_step_1_standard_models,
                    logs_directory=args.logs_directory,
                    task_data=task_data
                )
            except Exception as e:
                raise e
        
        return task_id, test_index, predictions
    finally:
        # Disable Watchdog
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

        if status_counters:
            running, remaining, finished, lock = status_counters
            with lock:
                running.value -= 1
                finished.value += 1
