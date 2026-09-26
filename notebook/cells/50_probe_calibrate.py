"""Probe the live server: detect the reasoning_effort mechanism and calibrate token cost.

Two things cannot be known without a GPU and both are measured here instead of guessed:

1. *How* `reasoning_effort` reaches the Qwen3.8 chat template. The documented
   mechanism is `chat_template_kwargs`; some vLLM builds want it top-level. A wrong
   guess is an HTTP 500 on every request, so all three are tried once, in order.
2. *How long* one xhigh sample actually is. The budget assumes 10,000 completion
   tokens; the real mean over 2 probes replaces it, which directly rescales how many
   xhigh samples per task the governor will approve.
"""
import time

from harness.config import HarnessConfig, SamplingConfig, ServingConfig
from harness.llm import ChatRequest, HTTPChatClient
from harness.parse import merge_reasoning_and_content, parse_answers
from harness.prompts import build_messages
from harness.tasks import load_challenges, resolve_solutions_path

PROBE_EFFORTS = ("low", "medium", "xhigh")
MECHANISMS = ("chat_template_kwargs", "top_level", "none")

client = None
EFFORT_MECHANISM = None
CALIBRATED_XHIGH_TOKENS = XHIGH_TOKENS_PER_SAMPLE
probe_rows = []

if SERVING:
    challenges = os.path.join(DATA_DIR, "arc-agi_%s_challenges.json" % SPLIT)
    solutions = resolve_solutions_path(DATA_DIR, SPLIT)
    probe_tasks = load_challenges(challenges, solutions)[:1]

    for mechanism in MECHANISMS:
        candidate = HTTPChatClient(
            base_url=SERVING["base_url"], model=SERVED_MODEL_NAME,
            timeout=1800.0, max_retries=1, effort_mechanism=mechanism,
        )
        resp = candidate.complete(ChatRequest(
            messages=[{"role": "user", "content": "Reply with the single digit 7."}],
            max_tokens=128, temperature=0.0, reasoning_effort="low",
        ))
        if resp.ok:
            client = candidate
            EFFORT_MECHANISM = mechanism
            log("reasoning_effort mechanism = %s" % mechanism)
            break
        log("mechanism %s failed: %s" % (mechanism, (resp.error or "")[:200]))

    if client is None:
        log("no effort mechanism worked; the server is unusable")

if client is not None and probe_tasks:
    task = probe_tasks[0]
    messages = build_messages(task)
    for effort in PROBE_EFFORTS:
        started = time.monotonic()
        resp = client.complete(ChatRequest(
            messages=messages,
            max_tokens=XHIGH_MAX_TOKENS if effort == "xhigh" else PHASE_A_MAX_TOKENS,
            temperature=XHIGH_TEMPERATURE if effort == "xhigh" else 0.7,
            top_p=TOP_P, seed=1, reasoning_effort=effort,
            meta={"task_id": task.task_id, "probe": True},
        ))
        wall = time.monotonic() - started
        parsed = parse_answers(merge_reasoning_and_content(resp.reasoning, resp.content), task.n_test)
        tps = resp.usage.completion_tokens / wall if wall > 0 else 0.0
        probe_rows.append({
            "effort": effort, "ok": resp.ok, "error": resp.error, "wall_s": round(wall, 1),
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "reasoning_tokens": resp.usage.reasoning_tokens,
            "single_stream_tok_s": round(tps, 1),
            "finish_reason": resp.finish_reason,
            "parse_method": parsed.method, "n_grids": len(parsed.grids),
        })
        log("probe %-6s ok=%s wall=%6.1fs completion=%6d reasoning=%6d %.1f tok/s parse=%s(%d)" % (
            effort, resp.ok, wall, resp.usage.completion_tokens,
            resp.usage.reasoning_tokens, tps, parsed.method, len(parsed.grids)))
        if effort == "xhigh" and resp.ok and resp.usage.completion_tokens > 0:
            measured = resp.usage.completion_tokens
            CALIBRATED_XHIGH_TOKENS = int(min(max(measured, 4000), XHIGH_MAX_TOKENS))
            log("calibrated xhigh_tokens_per_sample: %d -> %d" % (XHIGH_TOKENS_PER_SAMPLE, CALIBRATED_XHIGH_TOKENS))
            if resp.finish_reason == "length":
                log("WARNING: xhigh hit max_tokens=%d; expect truncated answers" % XHIGH_MAX_TOKENS)

PROBE_JSON = probe_rows
log("probe complete; elapsed %.0fs, remaining %.2f h" % (elapsed_s(), remaining_s() / 3600.0))
