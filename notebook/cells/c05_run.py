# --- run the Duck-REPL agent over the test challenges -------------------------
# Writes /kaggle/working/submission.json cumulatively: the file is pre-filled
# with valid fallbacks for EVERY task id before the first task starts and is
# rewritten atomically after each task, so a kill at any instant still yields a
# schema-valid submission. A watchdog thread finalises it before the buffer.
import json
import os
import time
from pathlib import Path

from harness.budget import Deadline
from harness.config import HarnessConfig
from harness.dataio import (
    is_competition_rerun,
    iter_tasks,
    load_challenges,
    load_solutions,
    pick_challenge_file,
    task_shapes,
)
from harness.llm import VLLMClient
from harness.mock_agent import build_mock_llm
from harness.runner import Runner, write_run_report
from harness.submission import validate_submission

NO_DATA = DATA_DIR is None
if NO_DATA:
    # No challenge file is reachable. In a real competition rerun Kaggle injects
    # /kaggle/input/competitions/arc-prize-2026-arc-agi-2/ itself, so this path
    # only triggers on an interactive "Save & Run All" whose data sources did not
    # attach. We degrade loudly instead of raising: there is nothing to solve and
    # nothing to submit, and an exception would hide the diagnosis under a
    # papermill traceback. The task ids are deliberately NOT embedded in this
    # notebook -- competition data is not redistributed in the repo.
    print("!! DATA_DIR is None -- no ARC challenge file is reachable")
    if SAMPLE_SUBMISSION is not None and SAMPLE_SUBMISSION.is_file():
        # sample_submission.json carries every task id with the right number of
        # attempt slots, which is enough to leave a schema-valid (zero-scoring)
        # file rather than none at all.
        with open(SAMPLE_SUBMISSION, encoding="utf-8") as handle:
            SAMPLE = json.load(handle)
        FALLBACK = {
            task_id: [
                {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}
                for _ in entries
            ]
            for task_id, entries in SAMPLE.items()
        }
        with open(SUBMISSION_PATH, "w", encoding="utf-8") as handle:
            json.dump(FALLBACK, handle)
        print(f"wrote a schema-valid placeholder from {SAMPLE_SUBMISSION.name} "
              f"({len(FALLBACK)} task ids). Score will be ~0; fix the data mount.")
    else:
        print("no sample_submission.json either -- nothing to submit from this run.")
        print("TO FIX: in the Kaggle notebook UI use 'Add input' and attach")
        print(f"  * the competition  : {COMPETITION_SLUG}")
        print("  * the model weights : saltb0x/qwen3-8-27b-fp8")
        print("  * a vLLM wheelhouse dataset (offline install; no internet at rerun)")
        print("then re-run. At competition rerun the data is mounted automatically,")
        print("so this message does not indicate a problem with the rerun path.")
    RESULT = None
    CHALLENGE_FILE = None
    CHALLENGES = {}
    SOLUTIONS = None
    SHAPES = {}
    TASKS = []
else:
    CHALLENGE_FILE = os.environ.get("ARC26_CHALLENGE_FILE", "").strip() or pick_challenge_file(
        str(DATA_DIR), prefer_test=is_competition_rerun()
    )
    CHALLENGES = load_challenges(str(DATA_DIR), CHALLENGE_FILE)
    SOLUTIONS = load_solutions(str(DATA_DIR), CHALLENGE_FILE)
    SHAPES = task_shapes(CHALLENGES)
print(f"challenge file : {CHALLENGE_FILE}")
print(f"tasks          : {len(CHALLENGES)}   test outputs: {sum(SHAPES.values())}")
print(f"solutions      : {'yes (scoring enabled)' if SOLUTIONS else 'no (rerun mode)'}")

# ARC26_LIMIT / ARC26_TASKS exist so a real Kaggle run can be smoke-tested on a
# handful of tasks before burning GPU hours; they are unset in a competition rerun.
LIMIT = int(os.environ.get("ARC26_LIMIT", "0") or 0) or None
ONLY = [t.strip() for t in os.environ.get("ARC26_TASKS", "").split(",") if t.strip()] or None
if os.environ.get("ARC26_MAX_TURNS"):
    config_max_turns = int(os.environ["ARC26_MAX_TURNS"])
else:
    config_max_turns = 14  # design target, HYPOTHESIS.md 5.4; break-even is 24

TASKS = iter_tasks(CHALLENGES, limit=LIMIT, task_ids=ONLY)
print(f"loaded tasks   : {len(TASKS)}" + (f" (limited to {LIMIT})" if LIMIT else ""))

# --- config -------------------------------------------------------------------
config = HarnessConfig(
    model=SERVED_MODEL_NAME,
    base_url=VLLM_BASE_URL,
    api_key="EMPTY",
    workers=WORKERS,
)
config.budget.total_seconds = USABLE_SECONDS
config.budget.safety_buffer_seconds = 0.0      # already folded into the deadline
config.budget.startup_reserve_seconds = 0.0
config.budget.max_turns = config_max_turns
config.budget.hard_max_turns = config_max_turns + 6  # break-even is 24 at plan rates
config.budget.max_task_seconds = 900.0
config.budget.context_window = MAX_MODEL_LEN
config.budget.context_budget = 24_000
config.budget.keep_recent_turns = 6
config.budget.use_image = True
config.budget.image_max_vision_tokens = 160    # Rule 1 of HYPOTHESIS.md section 4
config.sampling.max_tokens = 3_072
# effort_first/default/on_shape_fail default to medium/low/medium: `xhigh` (the
# server default) would emit 8-16K thinking tokens per turn and cost 28h of
# decode alone. Never pass `high` -- it is not a valid value and returns HTTP 500.

elapsed = time.time() - NOTEBOOK_START
remaining = max(60.0, HARD_DEADLINE - time.time() - 120.0)
deadline = Deadline(remaining)
print(f"elapsed {elapsed:.0f}s, agent deadline in {remaining:.0f}s")

if SERVER_READY:
    llm = VLLMClient(
        base_url=config.base_url,
        model=config.model,
        api_key=config.api_key,
        timeout=config.request_timeout_s,
        max_retries=config.max_retries,
    )
    print(f"backend: vLLM at {config.base_url} model={config.model} workers={config.workers}")
else:
    llm = build_mock_llm()
    config.workers = 4
    print("backend: MockLLM (no server) -- schema validation only, score is meaningless")


def log_line(text):
    stamp = time.time() - NOTEBOOK_START
    print(f"[{stamp:7.0f}s] {text}", flush=True)
    try:
        with open(LOG_DIR / "run.log", "a", encoding="utf-8") as handle:
            handle.write(f"{stamp:.0f} {text}\n")
    except Exception:
        pass


RESULT = None
if DATA_DIR is not None and TASKS:
    runner = Runner(
        llm,
        config,
        TASKS,
        submission_path=str(SUBMISSION_PATH),
        task_shapes=SHAPES,
        solutions=SOLUTIONS,
        total_tasks=len(TASKS),
        deadline=deadline,
        workers=config.workers,
        keep_transcript=False,
        log=log_line,
    )
    try:
        RESULT = runner.run()
    finally:
        runner.finalize()
        if SERVER_PROC is not None and SERVER_PROC.poll() is None:
            try:
                SERVER_PROC.terminate()
            except Exception:
                pass
elif DATA_DIR is not None:
    print("!! no tasks to solve -- leaving the pre-filled submission in place")

if RESULT is not None:
    write_run_report(str(WORKING / "run_report.json"), RESULT)
    print(f"\n=== run complete in {RESULT.seconds:.0f}s ===")
    print(f"tasks attempted : {RESULT.tasks_attempted}/{len(TASKS)}")
    print(f"tasks verified  : {RESULT.tasks_verified}")
    total = RESULT.ledger.get("total", {})
    calls = total.get("calls") or 0
    print(f"llm calls       : {calls}   python: {total.get('python_calls')}   verify: {total.get('verify_calls')}")
    if calls:
        print(f"prompt tok/turn : {total.get('prompt_tokens', 0) / calls:,.0f}")
        print(f"compl  tok/turn : {total.get('completion_tokens', 0) / calls:,.0f}")
    print(f"total tokens    : {total.get('total_tokens', 0):,}")
    if RESULT.schema_problems:
        print(f"SCHEMA PROBLEMS : {len(RESULT.schema_problems)} -> {RESULT.schema_problems[:5]}")
    else:
        print("schema          : valid")
    if RESULT.score:
        print(f"SCORE           : {RESULT.score['score'] * 100:.2f}%  "
              f"({RESULT.score['n_correct']}/{RESULT.score['n_outputs']} test outputs)")
