"""Local evaluation harness: score a solver on the public evaluation split.

Usage:
    python evaluate.py                # score baseline_solver on evaluation split
    python evaluate.py --split training
"""
import sys
import time
import argparse

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from arc import load_split
from baseline_solver import heuristic_attempts


def evaluate(solver_attempts, split="evaluation", verbose=True):
    challenges, solutions = load_split(split)
    total, correct, solved_tasks = 0, 0, []
    t0 = time.time()
    for task_id, task in challenges.items():
        outs = solutions.get(task_id, [])
        if not outs:
            continue
        try:
            attempts = solver_attempts(task_id, task)
        except Exception as e:
            attempts = [[[t["input"], t["input"]] for t in task.get("test", [])]][0]
        for i, gt in enumerate(outs):
            total += 1
            pair = attempts[i] if i < len(attempts) else None
            hit = False
            if pair:
                for a in pair[:2]:
                    if a == gt:
                        hit = True
                        break
            if hit:
                correct += 1
                solved_tasks.append(task_id)
    score = correct / total if total else 0.0
    if verbose:
        print(f"[{split}] score={score:.4%} ({correct}/{total} outputs, "
              f"{len(solved_tasks)} tasks fully/partially hit) in {time.time()-t0:.1f}s")
    return score, solved_tasks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="evaluation", choices=["training", "evaluation"])
    args = ap.parse_args()
    evaluate(heuristic_attempts, args.split)
