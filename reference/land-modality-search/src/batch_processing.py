import concurrent.futures
import multiprocessing
import sys
import time
import threading
import os
import signal
from pathlib import Path

from src.execution import execute_task

# 11 hours 45 minutes = 42300 seconds
GLOBAL_TIMEOUT_SECONDS = 42300

def _monitor_timeout(start_time, executor_processes_func):
    """
    Background daemon that monitors total execution time.
    If GLOBAL_TIMEOUT_SECONDS is exceeded, it terminates all worker processes.
    """
    while True:
        elapsed = time.time() - start_time
        if elapsed > GLOBAL_TIMEOUT_SECONDS:
            print(f"\n{'!'*60}", file=sys.stderr)
            print(f"!!! GLOBAL TIMEOUT REACHED ({elapsed:.0f}s > {GLOBAL_TIMEOUT_SECONDS}s) !!!", file=sys.stderr)
            print("!!! TERMINATING ALL WORKER PROCESSES TO ENSURE GRACEFUL SUBMISSION !!!", file=sys.stderr)
            print(f"{'!'*60}\n", file=sys.stderr)
            
            # Identify and kill active children (workers)
            active_children = multiprocessing.active_children()
            for p in active_children:
                try:
                    p.terminate()
                except Exception:
                    pass
            return # Exit monitor thread
        
        time.sleep(10)

def run_batch_execution(args, tasks_to_run, run_timestamp, rate_limit_scale, answers_directory=None, startup_delay=0.0):
    final_results = []
    start_time = time.time()

    # Standard logging (interleaved)
    manager = multiprocessing.Manager()
    running = manager.Value('i', 0)
    remaining = manager.Value('i', len(tasks_to_run))
    finished = manager.Value('i', 0)
    lock = manager.Lock()
    status_counters = (running, remaining, finished, lock)

    # Print Table Header
    print("Legend: ⚡ Running   ⏳ Queued   ✅ Done")
    print()
    print("| Status        | Task:Test  | Step  | Time   | Message")
    print("|---------------|------------|-------|--------|--------------------------------------------------")

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.task_workers) as executor:
            # Start Global Timeout Monitor
            monitor_thread = threading.Thread(
                target=_monitor_timeout, 
                args=(start_time, None), 
                daemon=True
            )
            monitor_thread.start()

            future_to_task = {}
            for i, item in enumerate(tasks_to_run):
                # Check timeout before submitting next task (avoids starting new ones if close to limit)
                if time.time() - start_time > GLOBAL_TIMEOUT_SECONDS:
                    print("Global timeout reached during submission. Stopping new tasks.", file=sys.stderr)
                    break

                if i > 0 and startup_delay > 0:
                    time.sleep(startup_delay)

                if len(item) == 3:
                    task_id_str, test_idx, task_data = item
                    # Create a dummy Path that looks like a filename so .stem works and answers check works (maybe)
                    # Ideally answers match the ID. 
                    # If answers_directory is used, we assume standard naming.
                    task_path = Path(f"{task_id_str}.json") 
                else:
                    task_path, test_idx = item
                    task_data = None

                answer_path = answers_directory / task_path.name if answers_directory else None
                future = executor.submit(execute_task, args, task_path, test_idx, run_timestamp, rate_limit_scale, answer_path, status_counters, task_data)
                future_to_task[future] = (task_path, test_idx)
            
            # Process results
            # We wrap this in a try/except to catch the BrokenProcessPool if workers are killed
            try:
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        res = future.result()
                        final_results.append(res)
                    except concurrent.futures.process.BrokenProcessPool:
                        # This happens when we kill the workers
                        print("Worker process terminated (Global Timeout). Task incomplete.", file=sys.stderr)
                    except Exception as e:
                        print(f"Task failed: {e}", file=sys.stderr)
            except concurrent.futures.process.BrokenProcessPool:
                print("\nBatch execution interrupted: Process pool broken due to global timeout kill.", file=sys.stderr)
            except Exception as e:
                print(f"\nBatch execution interrupted: {e}", file=sys.stderr)

    except Exception as e:
        print(f"Global execution handler error: {e}", file=sys.stderr)

    return final_results
