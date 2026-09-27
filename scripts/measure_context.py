"""Measure the real per-turn context cost across the whole eval split.

Writes harness/docs/measured_context_budget.json, cited by HYPOTHESIS.md section 5.
Run:  python scripts/measure_context.py
"""

from __future__ import annotations

import json
import os
import statistics
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from harness.context import estimate_message_tokens
from harness.dataio import find_data_dir, iter_tasks, load_challenges
from harness.grids import compute_stats, text_tokens
from harness.prompts import TOOLS, build_system_prompt, build_task_block
from harness.render import render_task_panel

OUT = os.path.join(os.path.dirname(__file__), "..", "harness", "docs", "measured_context_budget.json")


def pct(values, q):
    values = sorted(values)
    if not values:
        return 0
    index = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[index]


def main() -> int:
    data_dir = find_data_dir(os.environ.get("ARC_DATA_DIR"))
    challenges = load_challenges(data_dir, "arc-agi_evaluation_challenges.json")
    tasks = iter_tasks(challenges)
    system_prompt = build_system_prompt()
    system_tokens = text_tokens(system_prompt)
    tools_tokens = estimate_message_tokens([], tools=TOOLS)

    rows = []
    for task in tasks:
        stats = compute_stats(task)
        block = build_task_block(task, stats, token_budget=4000)
        try:
            _image, info = render_task_panel(task, max_vision_tokens=160)
            vision = info["vision_tokens"]
            cell_px = info["cell_px"]
        except Exception as exc:
            vision, cell_px = 0, 0
            print("render failed", task.task_id, exc)
        grids = stats.n_train * 2 + stats.n_test
        cells = stats.max_cells
        rows.append(
            {
                "task_id": task.task_id,
                "n_train": stats.n_train,
                "n_test": stats.n_test,
                "grids": grids,
                "max_cells": cells,
                "task_block_tokens": text_tokens(block),
                "vision_tokens": vision,
                "cell_px": cell_px,
                "fixed_prefix_tokens": system_tokens + text_tokens(block) + vision + tools_tokens,
                "elided": "elided" in block,
            }
        )

    def col(name):
        return [r[name] for r in rows]

    summary = {
        "n_tasks": len(rows),
        "system_prompt_chars": len(system_prompt),
        "system_prompt_tokens": system_tokens,
        "tools_spec_tokens": tools_tokens,
        "task_block_tokens": {
            "mean": round(statistics.mean(col("task_block_tokens")), 1),
            "median": statistics.median(col("task_block_tokens")),
            "p90": pct(col("task_block_tokens"), 0.90),
            "max": max(col("task_block_tokens")),
        },
        "vision_tokens": {
            "mean": round(statistics.mean(col("vision_tokens")), 1),
            "median": statistics.median(col("vision_tokens")),
            "max": max(col("vision_tokens")),
        },
        "cell_px": {
            "mean": round(statistics.mean(col("cell_px")), 2),
            "min": min(col("cell_px")),
            "max": max(col("cell_px")),
        },
        "fixed_prefix_tokens": {
            "mean": round(statistics.mean(col("fixed_prefix_tokens")), 1),
            "median": statistics.median(col("fixed_prefix_tokens")),
            "p90": pct(col("fixed_prefix_tokens"), 0.90),
            "max": max(col("fixed_prefix_tokens")),
        },
        "n_grids_per_task": {
            "mean": round(statistics.mean(col("grids")), 2),
            "max": max(col("grids")),
        },
        "max_cells_per_grid": {
            "mean": round(statistics.mean(col("max_cells")), 1),
            "max": max(col("max_cells")),
        },
        "tasks_with_elided_pairs": sum(1 for r in rows if r["elided"]),
        "tasks_with_more_than_one_test_input": sum(1 for r in rows if r["n_test"] > 1),
    }

    # What the whole-turn prompt looks like once a rolling window is added.
    window_per_turn = 860  # assistant text+code ~290, tool result ~500, status ~70
    for keep in (4, 6, 8):
        summary[f"projected_prompt_tokens_keep{keep}"] = round(
            summary["fixed_prefix_tokens"]["mean"] + window_per_turn * keep
        )

    # Cell-level image cost, to justify NOT doing it.
    summary["vision_tokens_if_one_token_per_cell"] = round(
        statistics.mean([r["max_cells"] for r in rows])
    )

    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "per_task": rows}, handle, indent=1)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {os.path.abspath(OUT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
