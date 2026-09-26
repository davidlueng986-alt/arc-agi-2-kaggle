import sys
import warnings
import os
import json
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

from src.tasks import load_task
from src.run_utils import find_task_path
from src.execution import execute_task
from src.submission import generate_submission
from src.batch_processing import run_batch_execution
from src.llm_utils import set_retries_enabled



def run_app(
    task=None,
    task_directory=None,
    task_file=None,
    answer_file=None,
    test=1,
    task_workers=60,
    startup_delay=90.0,
    task_limit=None,
    task_selection=None,
    task_test_selection=None,
    objects=False,
    step_5_only=False,
    objects_only=False,
    force_step_5=False,
    force_step_2=False,
    verbose=0,
    models=None,
    step1_models=None,
    hint=None,
    image=False,
    codegen_params=None,
    disable_retries=False,
    disable_step_1_standard_models=False,
    trigger_deep_thinking=False,
    generate_hint=False,
    generate_hint_model="gpt-5.1-medium",
    judge_model=None,
    old_pick_solution=False,
    logs_directory="logs/",
    submissions_directory="submissions/",
    answers_directory=None,
    solver=False,
    solver_testing=False,
    openai_background=True,
    enable_step_3_and_4=False,
    judge_consistency_enable=False,
    judge_duo_pick=True,
):
    # Set default values based on mode if not provided
    if step1_models is None:
        if solver_testing:
            step1_models = "gpt-5.2-low,claude-opus-4.5-thinking-4000"
        else:
            step1_models = "claude-opus-4.5-thinking-60000,claude-opus-4.5-thinking-60000,claude-opus-4.5-thinking-60000,gemini-3-high,gemini-3-high,gemini-3-high,gemini-3-high,gpt-5.2-xhigh,gpt-5.2-xhigh,gpt-5.2-xhigh,gpt-5.2-xhigh,gpt-5.2-xhigh,gpt-5.2-xhigh"

    if codegen_params is None:
        if solver_testing:
            codegen_params = "gpt-5.2-low=v1b,gpt-5.2-low=v4,gemini-3-low=v4"
        else:
            codegen_params = "gemini-3-high=v4,gpt-5.2-xhigh=v1b"

    # Construct args namespace to pass around internally as many legacy functions expect it
    args = SimpleNamespace(
        task=task,
        task_directory=task_directory,
        task_file=task_file,
        test=test,
        task_workers=task_workers,
        task_limit=task_limit,
        task_selection=task_selection,
        task_test_selection=task_test_selection,
        objects=objects,
        step_5_only=step_5_only,
        objects_only=objects_only,
        force_step_5=force_step_5,
        force_step_2=force_step_2,
        verbose=verbose,
        models=models,
        step1_models=step1_models,
        hint=hint,
        image=image,
        codegen_params=codegen_params,
        disable_retries=disable_retries,
        disable_step_1_standard_models=disable_step_1_standard_models,
        trigger_deep_thinking=trigger_deep_thinking,
        generate_hint=generate_hint,
        generate_hint_model=generate_hint_model,
        judge_model=judge_model,
        old_pick_solution=old_pick_solution,
        logs_directory=logs_directory,
        submissions_directory=submissions_directory,
        answers_directory=answers_directory,
        solver=solver,
        solver_testing=solver_testing,
        openai_background=openai_background,
        enable_step_3_and_4=enable_step_3_and_4,
        judge_consistency_enable=judge_consistency_enable,
        judge_duo_pick=judge_duo_pick
    )
    print("THIS IS THE OLD VERSION, USE THE V7 BRANCH")
    return

    if args.disable_retries:
        print("Retries disabled.")
        set_retries_enabled(False)

    if os.getenv("ARC_AGI_INSECURE_SSL", "").lower() == "true":
        print("WARNING: SSL verification disabled (ARC_AGI_INSECURE_SSL=true)")
        
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set default judge model if not specified
    if args.judge_model is None:
        if args.solver_testing:
            args.judge_model = "gpt-5.1-low"
        else:
            args.judge_model = "gpt-5.2-xhigh"
    
    # If no specific solver mode is chosen, default to --solver
    if not args.solver and not args.solver_testing:
        args.solver = True
    
    if args.solver_testing:
        print("Solver testing mode activated.")
        print(f"Judge model: {args.judge_model}")
        print()
    else:
        print("Solver mode activated.")
        print(f"Judge model: {args.judge_model}")
        print()
    
    warnings.filterwarnings("ignore", message=r"Pydantic serializer warnings:", category=UserWarning)

    # Ensure logs directory exists
    logs_dir = Path(args.logs_directory)
    if not logs_dir.exists():
        logs_dir.mkdir(parents=True, exist_ok=True)

    answers_dir = Path(args.answers_directory) if args.answers_directory else None
    if answers_dir and not answers_dir.exists():
         print(f"Error: Answers directory '{answers_dir}' does not exist.", file=sys.stderr)
         sys.exit(1)

    if args.task_file:
        # MONOLITHIC FILE MODE
        file_path = Path(args.task_file)
        if not file_path.exists():
            print(f"Error: Task file '{file_path}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        print(f"Loading tasks from monolithic file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                all_tasks = json.load(f)
        except Exception as e:
            print(f"Error reading task file: {e}", file=sys.stderr)
            sys.exit(1)
            
        # Filter tasks
        task_ids = sorted(list(all_tasks.keys()))
        
        if args.task_selection:
            selected_ids = [t.strip() for t in args.task_selection.split(",")]
            task_ids = [t for t in task_ids if t in selected_ids]
            if not task_ids:
                 print(f"No tasks found matching selection: {selected_ids}", file=sys.stderr)
                 sys.exit(0)
            print(f"Filtered to {len(task_ids)} tasks based on selection.")

        if args.task_limit:
            print(f"Limiting execution to first {args.task_limit} tasks.")
            task_ids = task_ids[:args.task_limit]

        # Prepare tasks_to_run
        tasks_to_run = []
        
        if args.task_test_selection:
            # Format: 'de809cff:1,faa9f03d:1'
            try:
                for item in args.task_test_selection.split(","):
                    parts = item.strip().split(":")
                    if len(parts) != 2:
                        raise ValueError(f"Invalid format: {item}")
                    tid, tidx = parts[0], int(parts[1])
                    if tid in all_tasks:
                        tasks_to_run.append((tid, tidx, all_tasks[tid]))
                    else:
                        print(f"Warning: Task {tid} from selection not found in monolithic file.", file=sys.stderr)
            except Exception as e:
                print(f"Error parsing --task-test-selection: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            for task_id in task_ids:
                task_data = all_tasks[task_id]
                # Count test cases
                num_tests = len(task_data.get("test", []))
                for i in range(num_tests):
                    test_idx = i + 1
                    # Add tuple (task_id, test_idx, task_data)
                    tasks_to_run.append((task_id, test_idx, task_data))
        
        total_tasks = len(tasks_to_run)
        print(f"Loaded {len(task_ids)} tasks. Total test cases: {total_tasks}")
        print(f"Starting batch execution with {args.task_workers} parallel task workers...")
        print()
        
        rate_limit_scale = 1.0 / max(1, args.task_workers)
        
        final_results = run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, answers_dir, startup_delay=startup_delay)
                        
        generate_submission(final_results, args.submissions_directory, run_timestamp)

    elif args.task_directory:
        if args.test != 1:
             print(f"Warning: --test argument ({args.test}) is ignored when using --task-directory. Running all tests for all tasks.", file=sys.stderr)

        directory = Path(args.task_directory)
        if not directory.exists() or not directory.is_dir():
            print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
            sys.exit(1)
            
        task_files = list(directory.glob("*.json"))
        if not task_files:
            print(f"No JSON files found in '{directory}'.", file=sys.stderr)
            sys.exit(0)
            
        task_files = sorted(task_files)
        
        if args.task_selection:
            selected_ids = [t.strip() for t in args.task_selection.split(",")]
            task_files = [f for f in task_files if f.stem in selected_ids]
            if not task_files:
                 print(f"No JSON files found in '{directory}' matching selection: {selected_ids}", file=sys.stderr)
                 sys.exit(0)
            print(f"Filtered to {len(task_files)} tasks based on selection.")

        if args.task_limit:
            print(f"Limiting execution to first {args.task_limit} tasks.")
            task_files = task_files[:args.task_limit]

        task_test_selection_list = None
        if args.task_test_selection:
            try:
                task_test_selection_list = []
                for item in args.task_test_selection.split(","):
                    parts = item.strip().split(":")
                    if len(parts) != 2:
                        raise ValueError(f"Invalid format for task:test selection: {item}")
                    task_test_selection_list.append((parts[0], int(parts[1])))
                print(f"Preparing {len(task_test_selection_list)} specific task:test pairs.")
            except Exception as e:
                print(f"Error parsing --task-test-selection: {e}", file=sys.stderr)
                sys.exit(1)

        # Prepare tasks
        tasks_to_run = []
        if task_test_selection_list:
            for task_id, test_idx in task_test_selection_list:
                try:
                    # We can use find_task_path or manually construct path since we have directory
                    task_path = directory / f"{task_id}.json"
                    if not task_path.exists():
                         # Fallback to general find_task_path if not in that specific directory
                         task_path = find_task_path(task_id)
                    
                    tasks_to_run.append((task_path, test_idx))
                except Exception as e:
                    print(f"Error resolving task {task_id}: {e}", file=sys.stderr)
        else:
            for task_file in task_files:
                try:
                    # Load task briefly just to count tests, we load again in worker
                    # We don't need answer path here strictly, but good to know
                    task = load_task(task_file)
                    num_tests = len(task.test)
                    for i in range(num_tests):
                        test_idx = i + 1
                        tasks_to_run.append((task_file, test_idx))
                except Exception as e:
                    print(f"Error loading task {task_file}: {e}", file=sys.stderr)

        total_tasks = len(tasks_to_run)
        print(f"Found {len(task_files)} task files. Total test cases: {total_tasks}")
        print(f"Starting batch execution with {args.task_workers} parallel task workers...")
        print()
        
        rate_limit_scale = 1.0 / max(1, args.task_workers)
        
        final_results = run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, answers_dir, startup_delay=startup_delay)
                        
        # Generate Submission File
        generate_submission(final_results, args.submissions_directory, run_timestamp)

    else:
        # Single task mode
        try:
            task_path = find_task_path(args.task)
            answer_path = None
            if answers_dir:
                 answer_path = answers_dir / task_path.name
            
            # Print Table Header
            print("Legend: ⚡ Running   ⏳ Queued   ✅ Done")
            print()
            print("| Status        | Task:Test  | Step  | Time   | Message")
            print("|---------------|------------|-------|--------|--------------------------------------------------")

            execute_task(args, task_path, args.test, run_timestamp, answer_path=answer_path)
        except FileNotFoundError as e:
             print(f"Error: {e}", file=sys.stderr)
             sys.exit(1)
        except SystemExit as se:
            sys.exit(se.code)
