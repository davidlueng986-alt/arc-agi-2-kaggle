#!/usr/bin/env python3
"""Run the Duck-REPL harness against a local ARC-AGI-2 split and score it.

Examples
--------
Plumbing check with no GPU and no server (deterministic mock agent):

    python scripts/eval_local.py --mock --limit 10

Real run against a local vLLM server:

    python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1 ^
        --model qwen3.8-27b --workers 8 --limit 20

Score a single task verbosely:

    python scripts/eval_local.py --tasks 0934a4d8 --transcript

The `--project` flag turns the measured per-turn token cost into a wall-clock
forecast for the 240-task rerun, which is the number that decides whether this
hypothesis fits 12h at all.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.budget import Deadline  # noqa: E402
from harness.config import HarnessConfig  # noqa: E402
from harness.dataio import (  # noqa: E402
    find_data_dir,
    is_competition_rerun,
    iter_tasks,
    load_challenges,
    load_solutions,
    pick_challenge_file,
    task_shapes,
)
from harness.llm import VLLMClient  # noqa: E402
from harness.mock_agent import build_mock_llm  # noqa: E402
from harness.runner import Runner, write_run_report  # noqa: E402

SPLIT_FILES = {
    "eval": "arc-agi_evaluation_challenges.json",
    "test": "arc-agi_test_challenges.json",
    "train": "arc-agi_training_challenges.json",
}

# Conservative serving-side throughput assumptions for 4x L4 (96GB aggregate,
# PCIe, no NVLink) running Qwen3.8-27B FP8 under vLLM TP=4.  Derivation and the
# sensitivity table are in HYPOTHESIS.md section 5.
DEFAULT_PREFILL_TOK_S = 2000.0
DEFAULT_DECODE_TOK_S = 500.0
RERUN_TASKS = 240
USABLE_SECONDS = 12 * 3600 - 600 - 1200


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=None, help="directory holding the ARC-AGI-2 JSON files")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default=None,
                        help="which challenge file (default: test at rerun, else eval)")
    parser.add_argument("--limit", type=int, default=None, help="solve only the first N tasks")
    parser.add_argument("--tasks", default=None, help="comma-separated task ids")
    parser.add_argument("--mock", action="store_true", help="use the deterministic MockLLM (no GPU, no server)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-key", default=os.environ.get("ARC26_API_KEY", "EMPTY"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--context-budget", type=int, default=None)
    parser.add_argument("--reasoning-effort", default=None, choices=["low", "medium", "xhigh"],
                        help="pin every turn to one effort instead of using the schedule")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-image", action="store_true", help="disable the multimodal panel")
    parser.add_argument("--image-vision-tokens", type=int, default=None)
    parser.add_argument("--no-early-stop", action="store_true")
    parser.add_argument("--time-budget", type=float, default=None,
                        help="override the wall-clock budget in seconds (default: 12h minus buffers)")
    parser.add_argument("--out", default=None, help="submission.json path")
    parser.add_argument("--report", default=None, help="per-task JSON report path")
    parser.add_argument("--transcript", action="store_true", help="keep per-task transcripts in the report")
    parser.add_argument("--project", action="store_true", help="print the 240-task wall-clock forecast")
    parser.add_argument("--prefill-tok-s", type=float, default=DEFAULT_PREFILL_TOK_S)
    parser.add_argument("--decode-tok-s", type=float, default=DEFAULT_DECODE_TOK_S)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> HarnessConfig:
    config = HarnessConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        workers=args.workers,
    )
    if args.max_turns is not None:
        config.budget.max_turns = args.max_turns
        config.budget.hard_max_turns = args.max_turns + 4
    if args.context_budget is not None:
        config.budget.context_budget = args.context_budget
    if args.no_image:
        config.budget.use_image = False
    if args.image_vision_tokens is not None:
        config.budget.image_max_vision_tokens = args.image_vision_tokens
    if args.temperature is not None:
        config.sampling.temperature = args.temperature
    if args.seed is not None:
        config.sampling.seed = args.seed
    if args.reasoning_effort is not None:
        config.sampling.effort_first = args.reasoning_effort
        config.sampling.effort_default = args.reasoning_effort
        config.sampling.effort_on_shape_fail = args.reasoning_effort
        config.sampling.effort_final = args.reasoning_effort
    if args.no_early_stop:
        config.stop_on_verified = False
    if args.time_budget is not None:
        config.budget.total_seconds = args.time_budget
    return config


def build_llm(args: argparse.Namespace, config: HarnessConfig):
    if args.mock:
        return build_mock_llm()
    return VLLMClient(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout=config.request_timeout_s,
        max_retries=config.max_retries,
    )


def project(result, args: argparse.Namespace) -> None:
    """Turn measured per-turn token costs into a 240-task wall-clock forecast.

    Total time is independent of worker count once the GPU is saturated:
        time = tasks * turns * (prompt_tokens / prefill_rate
                                + completion_tokens / decode_rate)
    Workers only decide how fast you reach saturation.
    """
    total = result.ledger.get("total", {})
    calls = total.get("calls") or 0
    if not calls:
        print("projection: no LLM calls recorded")
        return
    prompt_per_turn = total.get("prompt_tokens", 0) / calls
    completion_per_turn = total.get("completion_tokens", 0) / calls
    turns_per_task = calls / max(1, result.tasks_attempted)
    per_turn_seconds = prompt_per_turn / args.prefill_tok_s + completion_per_turn / args.decode_tok_s
    forecast = RERUN_TASKS * turns_per_task * per_turn_seconds

    print("\n=== 240-task projection (from measured per-turn cost) ===")
    print(f"prompt tokens / turn      : {prompt_per_turn:,.0f}")
    print(f"completion tokens / turn  : {completion_per_turn:,.0f}")
    print(f"turns / task (measured)   : {turns_per_task:.2f}")
    print(f"GPU-seconds / turn        : {per_turn_seconds:.2f}s "
          f"(prefill {prompt_per_turn / args.prefill_tok_s:.2f}s + decode {completion_per_turn / args.decode_tok_s:.2f}s)")
    print(f"assumed aggregate rates   : prefill {args.prefill_tok_s:,.0f} tok/s, decode {args.decode_tok_s:,.0f} tok/s")
    print(f"forecast for 240 tasks    : {forecast:,.0f}s = {forecast / 3600:.2f}h")
    print(f"usable wall clock         : {USABLE_SECONDS:,.0f}s = {USABLE_SECONDS / 3600:.2f}h")
    headroom = USABLE_SECONDS / forecast if forecast > 0 else float("inf")
    print(f"headroom                  : {headroom:.2f}x")
    max_turns = USABLE_SECONDS / (RERUN_TASKS * per_turn_seconds) if per_turn_seconds > 0 else 0
    print(f"max affordable turns/task : {max_turns:.1f}")
    total_tokens = RERUN_TASKS * turns_per_task * (prompt_per_turn + completion_per_turn)
    print(f"total tokens (240 tasks)  : {total_tokens / 1e6:.1f}M")
    if headroom < 1.0:
        print("!! DOES NOT FIT -- reduce turns/task, context, or reasoning effort")
    elif headroom < 1.5:
        print("! tight -- the adaptive scheduler will be shrinking turns continuously")
    else:
        print("=> fits with margin")

    print("\n--- sensitivity: forecast hours at other serving rates ---")
    print("prefill\\decode " + "".join(f"{d:>10,.0f}" for d in (250.0, 500.0, 900.0)))
    for prefill in (1000.0, 2000.0, 3500.0):
        row = f"{prefill:>10,.0f}    "
        for decode in (250.0, 500.0, 900.0):
            seconds = RERUN_TASKS * turns_per_task * (prompt_per_turn / prefill + completion_per_turn / decode)
            row += f"{seconds / 3600:>9.2f}h"
        print(row)


def main(argv=None) -> int:
    args = parse_args(argv)
    log = (lambda *_: None) if args.quiet else print

    data_dir = find_data_dir(args.data_dir)
    if args.split:
        challenge_file = SPLIT_FILES[args.split]
    else:
        challenge_file = pick_challenge_file(data_dir, prefer_test=is_competition_rerun())
    log(f"data dir      : {data_dir}")
    log(f"challenge file: {challenge_file}")

    challenges = load_challenges(data_dir, challenge_file)
    solutions = load_solutions(data_dir, challenge_file)
    log(f"tasks         : {len(challenges)}   solutions: {'yes' if solutions else 'no'}")

    wanted = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    tasks = iter_tasks(challenges, limit=args.limit, task_ids=wanted)
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 2
    log(f"solving       : {len(tasks)} task(s) with {args.workers} worker(s), mock={args.mock}")

    config = build_config(args)
    out_path = args.out or os.path.join(
        os.path.abspath(os.getcwd()),
        "output",
        f"submission_{'mock' if args.mock else 'live'}_{challenge_file.split('_')[1]}_{len(tasks)}.json",
    )
    report_path = args.report or (os.path.splitext(out_path)[0] + "_report.json")

    deadline = Deadline(config.budget.usable_seconds)
    started = time.perf_counter()
    runner = Runner(
        build_llm(args, config),
        config,
        tasks,
        submission_path=out_path,
        task_shapes=task_shapes(challenges),
        solutions=solutions,
        total_tasks=len(tasks),
        deadline=deadline,
        workers=args.workers,
        keep_transcript=args.transcript,
        log=log,
    )
    result = runner.run()
    wall = time.perf_counter() - started

    print(f"\n=== summary ({wall:.1f}s wall) ===")
    print(f"submission    : {out_path}")
    print(f"tasks         : {result.tasks_attempted}/{len(tasks)}   verified: {result.tasks_verified}")
    if result.schema_problems:
        print(f"SCHEMA PROBLEMS ({len(result.schema_problems)}): {result.schema_problems[:5]}")
    else:
        print("schema        : valid")
    total = result.ledger.get("total", {})
    print(f"llm calls     : {total.get('calls')}   python calls: {total.get('python_calls')}   "
          f"verify calls: {total.get('verify_calls')}")
    print(f"tokens        : prompt {total.get('prompt_tokens', 0):,} + completion "
          f"{total.get('completion_tokens', 0):,} = {total.get('total_tokens', 0):,}")
    print(f"tokens / turn : {result.ledger.get('mean_turn_tokens', 0):,.0f}")
    if result.score:
        print(f"SCORE         : {result.score['score'] * 100:.2f}%  "
              f"({result.score['n_correct']}/{result.score['n_outputs']} test outputs)")
        solved = sum(1 for v in result.score["per_task"].values() if v == 1.0)
        print(f"tasks fully solved: {solved}/{len(result.score['per_task'])}")
    if args.mock:
        print("note          : --mock validates plumbing and schema only; its score is not a prediction")

    if args.project:
        project(result, args)

    write_run_report(report_path, result)
    print(f"report        : {report_path}")
    return 0 if not result.schema_problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
