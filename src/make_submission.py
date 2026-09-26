"""Build submission.json for the (placeholder) test challenges.

Usage:
    python make_submission.py [--challenges path] [--out path]

At competition rerun, the test file is swapped for the real 240 tasks.
"""
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent))
from arc import DATA_DIR, make_submission
from baseline_solver import heuristic_attempts


def attempts_to_preds(attempts):
    return [{"attempt_1": a[0], "attempt_2": a[1]} for a in attempts]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--challenges", default=str(DATA_DIR / "arc-agi_test_challenges.json"))
    ap.add_argument("--out", default=str(DATA_DIR.parent / "submissions" / "submission.json"))
    args = ap.parse_args()

    def solver(task_id, task):
        return heuristic_attempts(task_id, task)

    sub = make_submission(solver, args.challenges, args.out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"wrote {len(sub)} tasks -> {args.out}")
