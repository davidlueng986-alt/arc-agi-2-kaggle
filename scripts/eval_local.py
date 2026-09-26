"""Local eval-split runner with ground-truth scoring and ablation switches.

Examples:
  python scripts/eval_local.py --mock --limit 10
  python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1 --limit 20
  python scripts/eval_local.py --mock --no-judge            # judge ablation
  python scripts/eval_local.py --mock --no-diversity        # pass@2 selection ablation
  python scripts/eval_local.py --mock --experts 1 --pool-size 1   # chain-refinement ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.budget import Deadline  # noqa: E402
from harness.config import PipelineConfig  # noqa: E402
from harness.llm import MockLLM, OpenAICompatClient  # noqa: E402
from harness.runner import TaskRunner  # noqa: E402

DEFAULT_DATA = r"C:\Users\daive\Downloads\arc-agi-2\data"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=os.environ.get("ARC_DATA_DIR", DEFAULT_DATA))
    p.add_argument("--split", default="evaluation", choices=["evaluation", "test", "training"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--task-ids", nargs="*", default=None)
    p.add_argument("--out", default=os.path.join("output", "submission_local.json"))
    p.add_argument("--mock", action="store_true", help="use the deterministic MockLLM")
    p.add_argument("--mock-fail-every", type=int, default=0,
                   help="mock: inject a malformed solver response every k calls")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    p.add_argument("--model", default="qwen3.8-27b")
    # ablations / config
    p.add_argument("--experts", type=int, default=2, choices=[0, 1, 2])
    p.add_argument("--iters-a", type=int, default=8)
    p.add_argument("--iters-b", type=int, default=4)
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--text-candidates", type=int, default=3)
    p.add_argument("--no-text", action="store_true")
    p.add_argument("--judges", type=int, default=3)
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--no-diversity", action="store_true")
    p.add_argument("--effort-solver", default="low", choices=["xhigh", "medium", "low"])
    p.add_argument("--effort-judge", default="medium", choices=["xhigh", "medium", "low"])
    p.add_argument("--token-budget", type=int, default=120_000)
    p.add_argument("--time-budget", type=float, default=41_700.0)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metas-out", default=None, help="optional per-task meta JSON path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ch_path = os.path.join(args.data_dir, f"arc-agi_{args.split}_challenges.json")
    sol_path = os.path.join(args.data_dir, f"arc-agi_{args.split}_solutions.json")
    with open(ch_path, encoding="utf-8") as f:
        challenges = json.load(f)
    solutions = None
    if os.path.exists(sol_path):
        with open(sol_path, encoding="utf-8") as f:
            solutions = json.load(f)

    if args.task_ids:
        challenges = {k: v for k, v in challenges.items() if k in set(args.task_ids)}
    elif args.limit:
        challenges = dict(sorted(challenges.items())[: args.limit])

    cfg = PipelineConfig(
        experts=args.experts,
        iters_a=args.iters_a,
        iters_b=args.iters_b,
        pool_size=args.pool_size,
        text_candidates=0 if args.no_text else args.text_candidates,
        use_judge=not args.no_judge,
        judges=args.judges,
        diversity=not args.no_diversity,
        effort_solver=args.effort_solver,
        effort_judge=args.effort_judge,
        token_budget_per_task=args.token_budget,
        total_time_s=args.time_budget,
        task_concurrency=args.concurrency,
        seed=args.seed,
        model=args.model,
        base_url=args.base_url,
    )

    if args.mock:
        client = MockLLM.from_tasks(challenges, solutions, fail_every=args.mock_fail_every, seed=args.seed)
    elif args.experts == 0 and cfg.text_candidates == 0:
        print("nothing to run: 0 experts and 0 text candidates", file=sys.stderr)
        return 2
    else:
        client = OpenAICompatClient(base_url=args.base_url, model=args.model)

    deadline = Deadline(cfg.total_time_s)
    runner = TaskRunner(
        challenges, client, cfg, args.out, solutions=solutions, deadline=deadline
    )
    t0 = time.time()
    summary = asyncio.run(runner.run())
    summary["client"] = "mock" if args.mock else args.base_url

    print(json.dumps(summary, indent=2))
    if solutions is not None and summary.get("pass2") is not None:
        print(f"\npass@2 (competition metric): {summary['pass2'] * 100:.2f}%")
        print(f"pass@1 (attempt_1 only):     {summary['pass1'] * 100:.2f}%")
    if args.metas_out:
        os.makedirs(os.path.dirname(os.path.abspath(args.metas_out)), exist_ok=True)
        with open(args.metas_out, "w", encoding="utf-8") as f:
            json.dump(runner.metas, f, indent=1, default=str)
    print(f"wall: {time.time() - t0:.1f}s -> wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
