import sys
import os

# Force stable, line-buffered stdout/stderr
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

import argparse
from src.runner import run_app
from src.logging import StderrToStdoutRedirector

def main():
    # Redirect stderr to stdout using our smart wrapper that handles table prefixes
    sys.stderr = StderrToStdoutRedirector()

    parser = argparse.ArgumentParser(description="Run ARC task test cases with multiple models in parallel.")
    
    # Task selection group
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task", help="Task ID (e.g., 38007db0) or path to JSON file")
    task_group.add_argument("--task-directory", help="Directory containing task JSON files to run in batch")
    task_group.add_argument("--task-file", help="Path to a monolithic JSON file containing multiple tasks (Kaggle format)")
    
    parser.add_argument("--task-selection", help="Comma-separated list of Task IDs to select from the directory (e.g. '08ed6ac7,15696249').")
    parser.add_argument("--task-test-selection", help="Comma-separated list of TaskID:TestIndex pairs (e.g. 'de809cff:1,faa9f03d:1').")
    parser.add_argument("--test", type=int, default=1, help="Test case index (1-based, default: 1). Ignored if --task-directory is used.")
    parser.add_argument("--task-workers", type=int, default=60, help="Number of tasks to run in parallel (default: 60). CAUTION: Divides global rate limits by this factor.")
    parser.add_argument("--startup-delay", type=float, default=90.0, help="Delay in seconds between starting parallel tasks in batch mode to avoid rate limits (default: 90.0).")
    parser.add_argument("--task-limit", type=int, default=None, help="Limit the number of tasks to run (useful for testing batch mode).")
    parser.add_argument("--objects", action="store_true", help="Run a 3-step pipeline: Extraction -> Transformation -> Solution.")
    parser.add_argument("--step-5-only", action="store_true", help="Run only Step 5 (Full Search) of the solver.")
    parser.add_argument("--objects-only", action="store_true", help="Run only the Objects Pipeline sub-strategy in Step 5.")
    parser.add_argument("--force-step-5", action="store_true", help="Force execution of Step 5 even if a solution is found in earlier steps.")
    parser.add_argument("--force-step-2", action="store_true", help="Force the pipeline to stop after Step 2 (First Check), proceeding to Step Finish regardless of outcome.")
    parser.add_argument("--verbose", type=int, default=0, help="Verbosity level: 0=Quiet (Default), 1=Standard (Old Default), 2=Debug (Old Verbose).")
    parser.add_argument("--models", type=str, help="Comma-separated list of models to run")
    parser.add_argument("--step1-models", type=str, default=None, help="Comma-separated list of models to use for Step 1 Standard Search (default depends on mode).")
    parser.add_argument("--hint", type=str, default=None, help="Optional hint to provide to the model")
    parser.add_argument("--image", action="store_true", help="Generate an image for the task and include it in the prompt.")
    parser.add_argument("--codegen-params", type=str, default=None, help="Comma-separated list of model=prompt_version pairs for codegen (default depends on mode).")
    parser.add_argument("--disable-retries", action="store_true", help="Disable all retries for LLM calls.")
    parser.add_argument("--disable-step-1-standard-models", action="store_true", help="Disable standard (non-codegen) models in Step 1.")
    parser.add_argument("--trigger-deep-thinking", action="store_true", help="Append a deep thinking procedure to the prompt.")
    parser.add_argument("--generate-hint", action="store_true", help="Generate a hint for the task using a separate model call.")
    parser.add_argument("--generate-hint-model", type=str, default="gpt-5.1-medium", help="Model to use for generating hints.")
    parser.add_argument("--judge-model", type=str, default=None, help="Model to use for the auditing judges (Logic & Consistency).")
    parser.add_argument("--old-pick-solution", action="store_true", help="Use the legacy pick_solution() logic instead of the multi-judge pick_solution_v2().")
    parser.add_argument("--disable-openai-background", dest="openai_background", action="store_false", default=os.getenv("OPENAI_BACKGROUND", "true").lower() == "true", help="Disable OpenAI Background (Responses) API (Default: Enabled).")
    parser.add_argument("--enable-step-3-and-4", action="store_true", help="Enable Steps 3 and 4 (Narrow Search and Extended Check) which are disabled by default.")
    parser.add_argument("--judge-consistency-enable", action="store_true", help="Enable the Consistency Judge in the advanced solution picker (Disabled by default).")
    parser.add_argument("--disable-judge-duo-pick", dest="judge_duo_pick", action="store_false", default=True, help="Disable the Duo Pick (Meta-Conclusion) Judge in the advanced solution picker (Enabled by default).")
    parser.add_argument("--logs-directory", type=str, default="logs/", help="Directory to save log files (default: logs/).")
    parser.add_argument("--submissions-directory", type=str, default="submissions/", help="Directory to save submission files (default: submissions/).")
    parser.add_argument("--answers-directory", type=str, help="Optional directory containing answer files (with 'output' for test cases).")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--solver", action="store_true", help="Enable solver mode.")
    mode_group.add_argument("--solver-testing", action="store_true", help="Enable solver testing mode with a smaller set of models.")
    
    args = parser.parse_args()

    # Pass args as keyword arguments to run_app
    run_app(**vars(args))

if __name__ == "__main__":
    main()