#!/usr/bin/env python3
"""Run the H1 harness against the 120-task ARC-AGI-2 evaluation split and score it.

The evaluation split ships with ground truth, so this is the instrument that decides
whether H1 ("pure xhigh reasoning reaches >=50%") is alive or dead. Point it at a
live vLLM endpoint once a GPU is available:

    python scripts/eval_local.py --limit 40 --base-url http://127.0.0.1:8000/v1

Logic validation with no GPU at all:

    python scripts/eval_local.py --mock --limit 10

Budget arithmetic only (no model, no data):

    python scripts/eval_local.py --budget-table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from harness.budget import BudgetConfig, BudgetGovernor, ManualClock, SystemClock  # noqa: E402
from harness.config import HarnessConfig, SamplingConfig, ServingConfig  # noqa: E402
from harness.llm import HTTPChatClient, MockLLM  # noqa: E402
from harness.prompts import prompt_token_estimate  # noqa: E402
from harness.runner import budget_table, default_logger, load_tasks, mock_answers, run_harness  # noqa: E402


def parse_wave_samples(text: str) -> tuple:
    values = tuple(int(x) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("--wave-samples must contain at least one integer")
    if any(v < 0 for v in values):
        raise argparse.ArgumentTypeError("--wave-samples values must be non-negative")
    return values


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--split", default="evaluation", choices=["evaluation", "training", "test"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--limit", type=int, default=None, help="only run the first N tasks")
    p.add_argument("--task-ids", default=None, help="comma-separated task ids to run")
    p.add_argument("--seed", type=int, default=20260927)

    p.add_argument("--mock", action="store_true", help="use MockLLM instead of a real endpoint")
    p.add_argument("--mock-correct-prob", type=float, default=0.45)
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument(
        "--effort-mechanism",
        default="chat_template_kwargs",
        choices=["chat_template_kwargs", "top_level", "none"],
    )
    p.add_argument("--probe", action="store_true", help="check endpoint health then exit")

    p.add_argument("--phase-a-samples", type=int, default=2)
    p.add_argument("--phase-a-effort", default="low", choices=["low", "medium", "xhigh"])
    p.add_argument("--wave-samples", type=parse_wave_samples, default="3,3,3,3")
    p.add_argument("--max-xhigh-samples", type=int, default=12)
    p.add_argument("--xhigh-max-tokens", type=int, default=16384)
    p.add_argument("--phase-a-max-tokens", type=int, default=2560)
    p.add_argument("--xhigh-temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--concurrency", type=int, default=48)
    p.add_argument("--shuffle-train", action="store_true")
    p.add_argument("--consensus-threshold", type=float, default=0.7)
    p.add_argument("--consensus-min-candidates", type=int, default=4)

    p.add_argument("--total-hours", type=float, default=12.0)
    p.add_argument("--submission-buffer", type=float, default=600.0)
    p.add_argument("--startup-seconds", type=float, default=1020.0)
    p.add_argument("--planned-tok-s", type=float, default=450.0)
    p.add_argument("--pessimistic-tok-s", type=float, default=280.0)
    p.add_argument("--optimistic-tok-s", type=float, default=700.0)
    p.add_argument("--safety-reserve", type=float, default=0.10)
    p.add_argument("--xhigh-tokens-per-sample", type=int, default=10000)
    p.add_argument("--low-tokens-per-sample", type=int, default=1300)
    p.add_argument("--sim-speed", type=float, default=None,
                   help="mock only: pretend the clock advances this many seconds per request")

    p.add_argument("--budget-table", action="store_true", help="print the compute budget model and exit")
    p.add_argument("--prompt-stats", action="store_true", help="print prompt token statistics and exit")
    p.add_argument("--quiet", action="store_true")
    return p


def make_config(args: argparse.Namespace) -> HarnessConfig:
    task_ids = tuple(t.strip() for t in args.task_ids.split(",")) if args.task_ids else None
    config = HarnessConfig(
        data_dir=args.data_dir,
        split=args.split,
        output_dir=args.output_dir,
        limit=args.limit,
        task_ids=task_ids,
        seed=args.seed,
    )
    config.serving = ServingConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        effort_mechanism=args.effort_mechanism,
    )
    config.sampling = SamplingConfig(
        phase_a_effort=args.phase_a_effort,
        phase_a_samples=args.phase_a_samples,
        phase_a_max_tokens=args.phase_a_max_tokens,
        phase_b_effort="xhigh",
        wave_samples=args.wave_samples,
        max_xhigh_samples=args.max_xhigh_samples,
        xhigh_max_tokens=args.xhigh_max_tokens,
        xhigh_temperature=args.xhigh_temperature,
        top_p=args.top_p,
        consensus_threshold=args.consensus_threshold,
        consensus_min_candidates=args.consensus_min_candidates,
        concurrency=args.concurrency,
        shuffle_train=args.shuffle_train,
    )
    config.budget = BudgetConfig(
        total_seconds=args.total_hours * 3600.0,
        submission_buffer_seconds=args.submission_buffer,
        startup_seconds=args.startup_seconds,
        safety_reserve_fraction=args.safety_reserve,
        planned_tok_per_s=args.planned_tok_s,
        pessimistic_tok_per_s=args.pessimistic_tok_s,
        optimistic_tok_per_s=args.optimistic_tok_s,
        xhigh_tokens_per_sample=args.xhigh_tokens_per_sample,
        low_tokens_per_sample=args.low_tokens_per_sample,
    )
    config.log_every = 25
    return config


def print_budget_table(args: argparse.Namespace) -> None:
    config = make_config(args)
    n_tasks = 240
    rows_list = budget_table(config.budget, n_tasks=n_tasks, phase_a_samples=args.phase_a_samples)
    rows = {r["scenario"]: r for r in rows_list}
    c = config.budget
    print("H1 compute budget model")
    print("=" * 104)
    print(f"hardware            : 4 x NVIDIA L4 (24 GB, ~300 GB/s, sm_89 Ada), TP=4, no NVLink")
    print(f"model               : Qwen3.8-27B FP8 (~27 GB weights, 6.75 GB/GPU under TP=4)")
    print(f"runtime limit       : {args.total_hours} h = {c.total_seconds:.0f} s")
    print(f"  - submission buf  : {c.submission_buffer_seconds:.0f} s")
    print(f"  - startup/load    : {c.startup_seconds:.0f} s")
    print(f"  - finalize reserve: {c.finalize_reserve_seconds:.0f} s")
    print(f"  = usable          : {rows['central']['usable_seconds']:.0f} s")
    print(f"  - safety reserve  : {c.safety_reserve_fraction:.0%}")
    print(f"  = planned window  : {rows['central']['planned_seconds']:.0f} s")
    print()
    header = f"{'scenario':<12}{'tok/s':>8}{'completion budget':>20}{'tok/task':>12}{'xhigh samples/task':>22}{'phase B hours':>16}"
    print(header)
    print("-" * len(header))
    for row in rows.values():
        print(
            f"{row['scenario']:<12}{row['tok_per_s']:>8.0f}"
            f"{row['completion_token_budget']:>20,.0f}"
            f"{row['tokens_per_task']:>12,.0f}"
            f"{row['xhigh_samples_per_task_uniform']:>22.2f}"
            f"{row['xhigh_wave_hours_uniform']:>16.2f}"
        )
    print()
    print(f"phase A ({args.phase_a_effort}, {args.phase_a_samples}/task): "
          f"{rows['central']['phase_a_tokens']:,.0f} tokens = "
          f"{rows['central']['phase_a_tokens'] / rows['central']['completion_token_budget']:.1%} of central budget")
    print(f"xhigh per-sample cost assumption: {c.xhigh_tokens_per_sample:,} completion tokens "
          f"(cap {args.xhigh_max_tokens:,})")
    print(f"{args.phase_a_effort} per-sample cost assumption: {c.low_tokens_per_sample:,} completion tokens "
          f"(cap {args.phase_a_max_tokens:,})")
    print()
    print(json.dumps(rows, indent=2))


def print_prompt_stats(args: argparse.Namespace) -> None:
    config = make_config(args)
    tasks = load_tasks(config)
    stats = prompt_token_estimate(tasks)
    print(f"split={config.split} tasks={stats['n']}")
    print(json.dumps(stats, indent=2))
    print(f"total test outputs: {sum(t.n_test for t in tasks)}")


def probe(args: argparse.Namespace) -> int:
    client = HTTPChatClient(base_url=args.base_url, model=args.model, api_key=args.api_key)
    ok = client.health()
    print(f"endpoint {args.base_url} healthy: {ok}")
    if not ok:
        return 1
    from harness.llm import ChatRequest

    for effort in ("low", "medium", "xhigh"):
        started = time.monotonic()
        resp = client.complete(
            ChatRequest(
                messages=[{"role": "user", "content": "Reply with the single digit 7."}],
                max_tokens=256,
                temperature=0.0,
                reasoning_effort=effort,
            )
        )
        elapsed = time.monotonic() - started
        status = "OK" if resp.ok else f"FAIL {resp.error}"
        print(
            f"  effort={effort:<6} {status} in {elapsed:.1f}s "
            f"completion_tokens={resp.usage.completion_tokens} "
            f"reasoning_tokens={resp.usage.reasoning_tokens} "
            f"content={resp.content[:60]!r}"
        )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.budget_table:
        print_budget_table(args)
        return 0
    if args.prompt_stats:
        print_prompt_stats(args)
        return 0
    if args.probe:
        return probe(args)

    config = make_config(args)
    log = (lambda _m: None) if args.quiet else default_logger("eval_local")

    config.data_dir = config.data_dir
    tasks = load_tasks(config)
    if not tasks:
        log("no tasks loaded; check --data-dir / --split")
        return 2
    log(
        f"loaded {len(tasks)} tasks ({sum(t.n_test for t in tasks)} test outputs) from split={config.split}"
    )

    if args.mock:
        client = MockLLM(
            answers=mock_answers(tasks),
            correct_prob=args.mock_correct_prob,
            malformed_prob=0.03,
            seed=args.seed,
            tok_per_sec=0,
        )
        clock = ManualClock(0.0) if args.sim_speed else SystemClock()
        if args.sim_speed:
            client = _ClockAdvancingMock(client, clock, args.sim_speed)
        log(f"MOCK mode: correct_prob={args.mock_correct_prob}")
    else:
        client = HTTPChatClient(
            base_url=config.serving.base_url,
            model=config.serving.model,
            api_key=config.serving.api_key,
            timeout=config.sampling.request_timeout,
            max_retries=config.serving.max_retries,
            effort_mechanism=config.serving.effort_mechanism,
        )
        clock = SystemClock()
        if not client.health():
            log(f"endpoint {config.serving.base_url} is not healthy; use --mock or start vLLM first")
            return 3

    started = time.monotonic()
    outcome = run_harness(config, client=client, clock=clock, logger=log, tasks=tasks)
    wall = time.monotonic() - started

    report = outcome.report
    print()
    print("=" * 78)
    print(f"H1 eval run: {report.n_tasks} tasks / {report.n_test_outputs} test outputs")
    print(f"wall clock            : {wall:.1f} s")
    print(f"samples/task          : {report.samples_per_task:.2f} "
          f"(xhigh {report.xhigh_samples_per_task:.2f})")
    print(f"completion tokens     : {report.tokens['completion_tokens']:,}")
    print(f"reasoning tokens      : {report.tokens['reasoning_tokens']:,}")
    print(f"requests / failures   : {report.tokens['requests']} / {report.tokens['failures']}")
    print(f"measured tok/s        : {report.measured_tok_per_s}")
    print(f"consensus-stopped     : {report.consensus_stopped_tasks}")
    print(f"tasks with no parse   : {report.tasks_with_no_parse}")
    print(f"parse methods         : {report.parse_method_counts}")
    print(f"submission errors     : {len(report.submission_errors)}")
    print(f"submission            : {report.submission_path}")
    for wave in report.waves:
        print(
            f"  wave {wave.phase:<4} effort={wave.effort:<6} tasks={wave.tasks:>3} "
            f"reqs={wave.requests:>4} tok={wave.completion_tokens:>10,} "
            f"{wave.seconds:>7.1f}s consensus={wave.consensus_after:>3}"
            + (f" [{wave.trimmed_reason}]" if wave.trimmed_reason else "")
        )
    if outcome.score is not None:
        s = outcome.score
        print("-" * 78)
        print(f"SCORE                 : {s.score * 100:.2f}%  ({s.total_correct}/{s.total_outputs} outputs)")
        print(f"  attempt_1 only      : {s.attempt_1_score * 100:.2f}%")
        print(f"  2nd-attempt gain    : {s.second_attempt_gain * 100:.2f} pts")
    if report.coverage:
        cov = report.coverage
        print("-" * 78)
        print(f"pass@1 (per sample)   : {cov['pass_at_1'] * 100:.2f}%")
        print(f"coverage (pass@N)     : {cov['coverage_pass_at_n'] * 100:.2f}%  <- HARD CEILING")
        print(f"uncovered outputs     : {cov['uncovered_outputs']}")
        print(f"selection efficiency  : {cov['selection_efficiency'] * 100:.1f}%")
        print(f"VERDICT               : {cov['verdict']}")
    print("=" * 78)

    return 0 if not report.submission_errors else 1


class _ClockAdvancingMock:
    """Wraps MockLLM and advances a ManualClock so budget logic is exercised offline."""

    name = "mock-clock"

    def __init__(self, inner, clock, seconds_per_request: float) -> None:
        self.inner = inner
        self.clock = clock
        self.seconds_per_request = seconds_per_request

    def complete(self, request):
        self.clock.advance(self.seconds_per_request)
        return self.inner.complete(request)

    def health(self) -> bool:
        return True


if __name__ == "__main__":
    raise SystemExit(main())
