# --- validate the submission and forecast the 240-task cost --------------------
import json
import os
from pathlib import Path

from harness.dataio import load_challenges, task_shapes
from harness.submission import validate_submission

if not Path(SUBMISSION_PATH).is_file():
    # Only reachable when no data mounted at all: there is nothing to solve and
    # no task ids to build a placeholder from. Report it plainly rather than
    # raising, so the kernel log stays readable.
    print(f"!! {SUBMISSION_PATH} does not exist -- no data was mounted, nothing to validate")
    print(f"total notebook elapsed: {time.time() - NOTEBOOK_START:.0f}s")
    SUBMISSION = {}
    FRESH_SHAPES = {}
    PROBLEMS = []
else:
    with open(SUBMISSION_PATH, encoding="utf-8") as handle:
        SUBMISSION = json.load(handle)
    if CHALLENGE_FILE is None:
        # Placeholder path: validate against sample_submission.json, which still
        # proves the file on disk is structurally what the scorer expects.
        REFERENCE = json.load(open(SAMPLE_SUBMISSION, encoding="utf-8"))
        FRESH_SHAPES = {task_id: len(entries) for task_id, entries in REFERENCE.items()}
        print("validating against sample_submission.json (no challenge file was reachable)")
    else:
        # Re-validate against the challenge file itself, not the in-memory copy,
        # so a stale or partially written file cannot slip through.
        FRESH_SHAPES = task_shapes(load_challenges(str(DATA_DIR), CHALLENGE_FILE))
    PROBLEMS = validate_submission(SUBMISSION, FRESH_SHAPES)

if Path(SUBMISSION_PATH).is_file():
    print(f"submission      : {SUBMISSION_PATH}")
    print(f"size            : {os.path.getsize(SUBMISSION_PATH) / 1e6:.2f} MB")
    print(f"task ids        : {len(SUBMISSION)} / {len(FRESH_SHAPES)} required")
    print(f"test outputs    : {sum(len(v) for v in SUBMISSION.values())} / {sum(FRESH_SHAPES.values())} required")
    print(f"schema problems : {len(PROBLEMS)}")
    for problem in PROBLEMS[:10]:
        print(f"  - {problem}")

    if PROBLEMS:
        # Repair rather than fail: a Kaggle rerun scores what is on disk, and a
        # repaired file beats a valid-looking exception.
        from harness.submission import build_submission, write_submission

        print("REPAIRING submission in place")
        repaired = build_submission(SUBMISSION, FRESH_SHAPES)
        for task_id, entries in repaired.items():
            for entry in entries:
                for key in ("attempt_1", "attempt_2"):
                    if not entry.get(key):
                        entry[key] = [[0, 0], [0, 0]]
        write_submission(SUBMISSION_PATH, repaired)
        PROBLEMS = validate_submission(json.load(open(SUBMISSION_PATH, encoding="utf-8")), FRESH_SHAPES)
        print(f"schema problems after repair: {len(PROBLEMS)}")

    assert not PROBLEMS, f"submission is invalid: {PROBLEMS[:5]}"

    if FRESH_SHAPES:
        sample_id = next(iter(FRESH_SHAPES))
        print(f"\nexample entry ({sample_id}):")
        print(json.dumps(SUBMISSION[sample_id])[:300])

# --- 240-task forecast from measured per-turn cost ----------------------------
if RESULT is not None:
    total = RESULT.ledger.get("total", {})
    calls = total.get("calls") or 0
    if calls and RESULT.tasks_attempted:
        prompt_per_turn = total.get("prompt_tokens", 0) / calls
        completion_per_turn = total.get("completion_tokens", 0) / calls
        turns_per_task = calls / RESULT.tasks_attempted
        PREFILL, DECODE, N_RERUN = 2000.0, 400.0, 240
        per_turn = prompt_per_turn / PREFILL + completion_per_turn / DECODE
        forecast = N_RERUN * turns_per_task * per_turn
        print("\n=== 240-task forecast from measured cost ===")
        print(f"prompt/turn {prompt_per_turn:,.0f}  completion/turn {completion_per_turn:,.0f}  turns/task {turns_per_task:.2f}")
        print(f"GPU-seconds/turn {per_turn:.2f}  (prefill {prompt_per_turn / PREFILL:.2f}s + decode {completion_per_turn / DECODE:.2f}s)")
        print(f"forecast {forecast:,.0f}s = {forecast / 3600:.2f}h vs usable {USABLE_SECONDS / 3600:.2f}h -> headroom {USABLE_SECONDS / max(forecast, 1):.2f}x")
        print(f"max affordable turns/task at these rates: {USABLE_SECONDS / (N_RERUN * max(per_turn, 1e-9)):.1f}")
        if forecast > USABLE_SECONDS:
            print("!! DOES NOT FIT: lower max_turns or reasoning effort before submitting")

print(f"\ntotal notebook elapsed: {time.time() - NOTEBOOK_START:.0f}s of {TOTAL_SECONDS:.0f}s")
print("done")
