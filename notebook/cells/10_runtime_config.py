"""H1 runtime configuration: paths, deadline, and the knobs derived in HYPOTHESIS.md §4-5."""
import os
import sys
import time

RUN_START_MONO = time.monotonic()
RUN_START_WALL = time.time()

CWD = os.getcwd()
if CWD not in sys.path:
    sys.path.insert(0, CWD)

IS_RERUN = bool(os.environ.get("KAGGLE_IS_COMPETITION_RERUN"))
WORKING = "/kaggle/working" if os.path.isdir("/kaggle/working") else CWD
INPUT_ROOT = "/kaggle/input"
COMP_DIR = os.path.join(INPUT_ROOT, "competitions", "arc-prize-2026-arc-agi-2")

TOTAL_HOURS = 12.0
SUBMISSION_BUFFER_S = 600.0
FINALIZE_RESERVE_S = 120.0
SAFETY_RESERVE_FRACTION = 0.10
LAUNCH_RESERVE_S = 2700.0

SERVED_MODEL_NAME = "qwen3.8-27b"
VLLM_PORT = 8000
VLLM_PORT_ALT = 8001
BASE_URL = "http://127.0.0.1:%d/v1" % VLLM_PORT
BASE_URL_ALT = "http://127.0.0.1:%d/v1" % VLLM_PORT_ALT
VLLM_LOG = os.path.join(WORKING, "vllm.log")
VLLM_LOG_ALT = os.path.join(WORKING, "vllm_alt.log")

SPLIT = "test" if IS_RERUN else "evaluation"
DATA_DIR = COMP_DIR if os.path.isdir(COMP_DIR) else os.path.join(CWD, "data")
LIMIT = None

PHASE_A_EFFORT = "low"
PHASE_A_SAMPLES = 2
PHASE_A_MAX_TOKENS = 2560
WAVE_SAMPLES = (3, 3, 3, 3)
MAX_XHIGH_SAMPLES = 12
XHIGH_MAX_TOKENS = 16384
XHIGH_TEMPERATURE = 0.9
TOP_P = 0.95
CONCURRENCY = 48
CONSENSUS_THRESHOLD = 0.70
CONSENSUS_MIN_CANDIDATES = 4
SHUFFLE_TRAIN = False

PLANNED_TOK_S = 450.0
PESSIMISTIC_TOK_S = 280.0
OPTIMISTIC_TOK_S = 700.0
XHIGH_TOKENS_PER_SAMPLE = 10000
LOW_TOKENS_PER_SAMPLE = 1300


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def elapsed_s():
    return time.monotonic() - RUN_START_MONO


def remaining_s():
    return TOTAL_HOURS * 3600.0 - elapsed_s()


log("rerun=%s split=%s" % (IS_RERUN, SPLIT))
log("data_dir=%s" % DATA_DIR)
log("working=%s" % WORKING)
log("remaining=%.2f h" % (remaining_s() / 3600.0))
