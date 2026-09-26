"""Run the H1 harness over the competition test split with a hard watchdog armed.

The governor's horizon is rebased on time *already spent*: `total_seconds` is set to
what remains of the 12 h and `startup_seconds` to 0, so `t_gen`/`t_submit`/`t_hard`
land on the true absolute deadlines rather than 12 h from now.
"""
import json

from harness.budget import BudgetConfig, BudgetGovernor, SystemClock
from harness.config import HarnessConfig, SamplingConfig, ServingConfig
from harness.llm import HTTPChatClient
from harness.runner import default_logger, load_tasks, run_harness

HARNESS_LOG = default_logger("h1")
tasks = load_tasks(HarnessConfig(data_dir=DATA_DIR, split=SPLIT, limit=LIMIT))
HARNESS_LOG("loaded %d tasks / %d test outputs from split=%s" % (
    len(tasks), sum(t.n_test for t in tasks), SPLIT))

clock = SystemClock()
budget = BudgetConfig(
    total_seconds=max(600.0, remaining_s()),
    submission_buffer_seconds=SUBMISSION_BUFFER_S,
    startup_seconds=0.0,
    finalize_reserve_seconds=FINALIZE_RESERVE_S,
    safety_reserve_fraction=SAFETY_RESERVE_FRACTION,
    planned_tok_per_s=PLANNED_TOK_S,
    pessimistic_tok_per_s=PESSIMISTIC_TOK_S,
    optimistic_tok_per_s=OPTIMISTIC_TOK_S,
    xhigh_tokens_per_sample=CALIBRATED_XHIGH_TOKENS,
    low_tokens_per_sample=LOW_TOKENS_PER_SAMPLE,
)
governor = BudgetGovernor(budget, clock=clock)
HARNESS_LOG("deadline: t_gen=+%.0fs t_submit=+%.0fs planned_window=%.0fs projected=%.0f tok/s" % (
    governor.t_gen - governor.t0, governor.t_submit - governor.t0,
    governor.plan_seconds_left(), governor.projected_tok_per_s()))

config = HarnessConfig(data_dir=DATA_DIR, split=SPLIT, output_dir=WORKING, limit=LIMIT)
config.serving = ServingConfig(
    model=SERVED_MODEL_NAME,
    base_url=SERVING["base_url"] if SERVING else BASE_URL,
    effort_mechanism=EFFORT_MECHANISM or "chat_template_kwargs",
)
config.sampling = SamplingConfig(
    phase_a_effort=PHASE_A_EFFORT,
    phase_a_samples=PHASE_A_SAMPLES if client is not None else 0,
    phase_a_max_tokens=PHASE_A_MAX_TOKENS,
    phase_b_effort="xhigh",
    wave_samples=WAVE_SAMPLES if client is not None else (),
    max_xhigh_samples=MAX_XHIGH_SAMPLES,
    xhigh_max_tokens=XHIGH_MAX_TOKENS,
    xhigh_temperature=XHIGH_TEMPERATURE,
    top_p=TOP_P,
    consensus_threshold=CONSENSUS_THRESHOLD,
    consensus_min_candidates=CONSENSUS_MIN_CANDIDATES,
    concurrency=CONCURRENCY,
    shuffle_train=SHUFFLE_TRAIN,
    request_timeout=min(2400.0, max(300.0, governor.plan_seconds_left())),
)
config.budget = budget

if client is None:
    HARNESS_LOG("no usable LLM client; writing a schema-valid placeholder submission for all tasks")

outcome = run_harness(
    config,
    client=client,
    clock=clock,
    logger=HARNESS_LOG,
    tasks=tasks,
    governor=governor,
    submission_path=os.path.join(WORKING, "submission.json"),
    use_watchdog=True,
    hard_exit=True,
)

REPORT = outcome.report
SCORE = outcome.score
HARNESS_LOG("run finished in %.0fs; submission=%s" % (REPORT.elapsed_s, REPORT.submission_path))
with open(os.path.join(WORKING, "run_report.json"), "w", encoding="utf-8") as fh:
    json.dump(REPORT.as_dict(), fh, indent=2, default=str)
with open(os.path.join(WORKING, "probe.json"), "w", encoding="utf-8") as fh:
    json.dump(PROBE_JSON, fh, indent=2)
