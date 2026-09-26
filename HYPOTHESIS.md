# H1 — "Pure Reasoning Ceiling"

**Branch:** `h1/pure-xhigh` · **Competition:** `arc-prize-2026-arc-agi-2` · **Date:** 2026-09-27

---

## 1. The claim

> Qwen3.8-27B served locally on Kaggle L4x4 with `reasoning_effort=xhigh`, inside a
> **minimal clean harness** — no judge model, no code-refinement loop, no test-time
> training, no fine-tuning — reaches **≥ 50%** on ARC-AGI-2. Pure model reasoning
> ability + good prompting + enough samples.

Two sub-claims must both hold:

- **H1a (capability):** xhigh reasoning on a 27B open model produces a per-sample
  ARC-AGI-2 success rate high enough that ~7 samples + 2 attempts clears 50%.
- **H1b (schedulability):** that many xhigh samples actually fit in 12 h / L4x4
  across 240 tasks.

**Verdict up front: H1b survives, H1a almost certainly does not.** §4 shows 4–11
xhigh samples per task are affordable (central ≈ 6.7). §6 shows the score is
dominated by a capability term that sampling cannot move, and predicts **18–30%,
central ≈ 22%**. §7 gives the measurement that settles it in ~2 GPU-hours.

---

## 2. Evidence for the claim

| Evidence | Number | What it licenses |
|---|---|---|
| GPT-6 Astra, ARC-AGI-2, reasoning effort, no ARC fine-tuning | **95.0%** | Reasoning effort alone can saturate ARC-AGI-2 — *at frontier scale* |
| arXiv 2607.15439, 4-variant × 2-model × 2-effort ablation on ARC-AGI-3 | "every variant scored higher as model capability and reasoning effort increased, and these gains often **exceeded** the within-setting differences among variants" | Effort > harness mechanism, *as a relative ordering* |
| Qwen3.8-27B HLE | **30.8** | Fluid reasoning near Gemini-2.5-Pro tier — the benchmark that correlates best with ARC |
| Qwen3.8-27B GPQA-D / LiveCodeBench | 89.2 / 90.3 | Frontier-class reasoning and coding for a 27B dense model |
| Native context | 262 K | Whole task + all samples fit in one prompt; no truncation pressure |
| Tunable `reasoning_effort` | `{xhigh, medium, low}`, default `xhigh` | A real compute knob, not a fixed cost |

## 3. Evidence against the claim

| Evidence | Number | Why it bites |
|---|---|---|
| Poetiq — GPT-5.x, 8 experts × 10 refinement iterations, $30.57/task | **54.0%** | A *frontier* model with a maximal harness only just clears 50% |
| Gemini 3 Deep Think / Gemini 3 Pro / GPT-5.2 Pro standalone | 45 / 54 / 54.2% | The 50% line sits at frontier scale, not 27B |
| Land's 29-candidate + 3-judge harness on frontier APIs | 72.9% public, **50.3%** on the harder subset | Even the best published harness is ~50% on hard tasks |
| Best open-weights, no ARC fine-tuning, any scale | ~20–30% | The band Qwen3.8-27B must break out of |
| Kaggle public best (Qwen3-4B + TTT) | 33.89 | Our floor to beat, but it uses ARC-specific training |
| ARC-AGI-2 design intent | "less brute-forcible" | Rewards program induction from 3 examples; punishes mid-size dense models harder than knowledge benchmarks do |
| Land, `ANALYZE_STRATEGY.md` | many tasks at **0/10** across every hint condition | Per-task pass rate is strongly bimodal — the "cannot solve" mass is insensitive to sampling |

The asymmetry is the whole argument: **every data point above 50% is a frontier
model.** H1 asks a 27B open model to match them on reasoning effort alone.

---

## 4. Compute budget model (the core engineering deliverable)

All numbers below are reproduced live by `python scripts/eval_local.py --budget-table`.

### 4.1 Hardware and model

| Item | Value |
|---|---|
| GPUs | 4 × NVIDIA L4 — 24 GB GDDR6, **~300 GB/s**, sm_89 (Ada), 242 TFLOPS FP8 dense each |
| Interconnect | PCIe Gen4 x16, **no NVLink** → TP all-reduce is latency-bound |
| Model | Qwen3.8-27B **FP8**, ~27 GB weights (`saltb0x/qwen3-8-27b-fp8`, verified HF layout) |
| Topology | `--tensor-parallel-size 4` → **6.75 GB weights per GPU** |
| Architecture | 64 layers, hybrid: 48 Gated DeltaNet (constant-size state) + **16 full attention** (4 KV heads, head_dim 256) |

### 4.2 KV cache — memory is *not* the binding constraint

```
KV bytes/token = 16 attn layers × 4 KV heads × 256 head_dim × 2 (K,V) × 2 B (bf16)
               = 65,536 B = 64 KB/token           (aggregate, all ranks)
per TP=4 rank  = 64 KB / 4 = 16 KB/token           (1 KV head per rank)
```

Per-GPU memory:

```
24.0 GB × 0.92 (--gpu-memory-utilization)          = 22.08 GB
− 6.75 GB FP8 weight shard                         = 15.33 GB
− ~2.0 GB  CUDA ctx + vLLM + activations
           + DeltaNet conv/recurrent state
           + cudagraph pool                        = 13.33 GB  → KV pool
```

```
KV capacity per rank = 13.33e9 B / 16,384 B/token  = 813,600 tokens
worst-case sequence  = 4,255 prompt + 16,384 completion ≈ 20,640 tokens
                     → 813,600 / 20,640             = 39 concurrent sequences
mean-case sequence   = 1,135 prompt + 10,000 completion ≈ 11,135 tokens
                     → 813,600 / 11,135             = 73 concurrent sequences
```

→ `--max-num-seqs 64` is right at the worst-case ceiling; the harness runs
`concurrency=48` to keep headroom. **Memory permits far more parallelism than
bandwidth can pay for**, so throughput is the constraint.

### 4.3 Decode throughput

Per decode step, per rank:

```
weight read   = 6.75 GB / 300 GB/s                        = 22.5 ms   (independent of batch)
KV read       = B × L̄ × 16 KB / 300 GB/s                             (grows with batch)
collectives   = 2 all-reduce/layer × 64 layers = 128 × ~10 KB msgs
                over PCIe at ~45 µs each                  =  6.0 ms   (amortises over B)
```

| B | L̄ | weights | KV | AR | step | raw tok/s | **derated ×0.55** | achieved BW/rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 K | 22.5 | 0.4 | 6.0 | 28.9 ms | 35 | 19 | 24% |
| 8 | 8 K | 22.5 | 3.4 | 6.0 | 31.9 ms | 251 | 138 | 51% |
| 16 | 8 K | 22.5 | 6.8 | 6.0 | 35.3 ms | 453 | 249 | 68% |
| 32 | 8 K | 22.5 | 13.6 | 6.0 | 42.1 ms | 760 | **418** | 87% |
| 48 | 11 K | 22.5 | 25.6 | 6.0 | 54.1 ms | 887 | **488** | 95% |
| 64 | 11 K | 22.5 | 34.1 | 6.0 | 62.6 ms | 1022 | 562 | 98% |

The 0.55 derate covers chunked-prefill interference, scheduler bubbles, DeltaNet
recurrent-state updates, cudagraph misses, and L4 achieving ~85–90% of nominal
bandwidth. Note B=48/L̄=11K already sits at **95% of spec bandwidth** — the derate
is honest rather than generous, and B=64 is not reachable in practice.

Cross-check on compute: 450 tok/s × 2 × 27e9 FLOP = 24.3 TFLOPS against 968 TFLOPS
FP8 peak = **2.5% MFU**. Decode is memory-bound, so a low MFU is expected and
confirms we are nowhere near compute-limited.

**Planning figures**

| Scenario | Aggregate decode | Justification |
|---|---:|---|
| Pessimistic | **280 tok/s** | B≈24, 0.50 derate, PCIe contention |
| **Central** | **450 tok/s** | B≈40, 0.55 derate |
| Optimistic | **700 tok/s** | B=64, 0.70 derate |

### 4.4 Wall clock

```
Kaggle runtime limit                       43,200 s   (12 h)
− submission buffer (competition rule)        −600 s
− finalize reserve (write + validate)         −120 s
                                           = 42,480 s   ← t_gen, generation must stop here
− startup: kernel boot, wheelhouse install,
  25 GB dataset page-in, vLLM engine init,
  TP=4 NCCL setup, DeltaNet kernel compile,
  cudagraph capture                          −1,020 s
                                           = 41,460 s   usable generation window
− 10% schedule safety reserve                −4,146 s
                                           = 37,314 s   ← t_plan, what we actually schedule against
```

### 4.5 Token budget

| Scenario | tok/s | × 37,314 s | per task (÷240) |
|---|---:|---:|---:|
| Pessimistic | 280 | **10.45 M** | 43,533 |
| Central | 450 | **16.79 M** | 69,964 |
| Optimistic | 700 | **26.12 M** | 108,832 |

### 4.6 Per-sample cost

`xhigh` thinking emits 8,000–16,000+ tokens. Planning mean **10,000** completion
tokens/sample with a hard `max_tokens=16,384` cap. `low` planning mean **1,300**
with a 2,560 cap.

Prompt cost is measured, not guessed (`--prompt-stats`):

| Split | mean prompt tokens | p50 | p90 | max |
|---|---:|---:|---:|---:|
| **test (240, what we are scored on)** | **1,135** | 925 | 2,067 | 4,255 |
| evaluation (120, our local validation) | 1,946 | 1,845 | 3,116 | 5,379 |

With `--enable-prefix-caching` a task's N samples share one prefill, so total
prefill ≈ 240 × 1,135 × 2 × 27e9 = 14.7 PFLOP ≈ **76 s** — 0.2% of the window.
Prompt tokens consume KV, not decode budget.

Worst case 4,255 prompt + 16,384 completion = 20,639 < 32,768, so
`--max-model-len 32768` is safe with 37% headroom. (262,144 is unnecessary and
would inflate the profiling reservation.)

### 4.7 Phase A (cheap triage over *all* tasks)

```
2 samples × low × 1,300 tokens = 2,600 tokens/task
240 tasks × 2,600              = 624,000 tokens  = 3.7% of the central budget
624,000 / 450 tok/s            = 1,387 s = 23 min   (37 min at 280 tok/s)
```

Buys three things for 3.7% of budget: (1) after 23 minutes **every one of the 240
tasks has a schema-valid two-attempt answer on disk**, converting an
all-or-nothing 12 h run into one whose worst case is "phase-A quality", not zero;
(2) a free `low`-effort baseline candidate in every vote pool; (3) a live
throughput measurement that calibrates the governor before any xhigh is spent.

### 4.8 Phase B — the affordable xhigh sample count

```
central:     (16,791,300 − 624,000) / 240 tasks / 10,000 tokens = 6.74 samples/task
pessimistic: (10,447,920 − 624,000) / 240 tasks / 10,000 tokens = 4.09 samples/task
optimistic:  (26,119,800 − 624,000) / 240 tasks / 10,000 tokens = 10.62 samples/task
```

> ### ➤ **Answer to the crux question: 4–11 xhigh samples per task are affordable,
> ### central estimate ≈ 6.7.** Pure xhigh-everywhere *does* fit, so no triage is
> ### strictly required to make the schedule close — but see §4.10 for why the
> ### batching structure is mandatory anyway.

### 4.9 Wall-clock cross-check (does it fit in hours, not just tokens?)

| Scenario | Phase A | Phase B tokens | Phase B time | Total | vs 37,314 s plan |
|---|---:|---:|---:|---:|---|
| Pessimistic (280) | 37 min | 9.82 M | 35,071 s = 9.74 h | **10.36 h** | fits, 4,146 s reserve intact |
| Central (450) | 23 min | 16.17 M | 35,926 s = 9.98 h | **10.35 h** | fits, 4,146 s reserve intact |
| Optimistic (700) | 15 min | 25.50 M | 36,424 s = 10.12 h | **10.37 h** | fits |

All three converge on ~10.4 h because the token budget scales with throughput.
The reserve is never touched; a 2× throughput miss degrades *samples per task*,
not the deadline.

### 4.10 The batching requirement — a real finding, and a precondition

Naive serial scheduling (finish task *i*, then task *i+1*) puts only ~6.7 requests
in flight:

```
B = 7  →  step ≈ 22.5 + 2.0 + 6.0 = 30.5 ms  →  7/0.0305 = 230 tok/s
16.17 M tokens / 230 tok/s = 70,300 s = 19.5 h   ✗ does not fit 12 h
```

**Serial execution of this exact hypothesis overruns the budget by ~1.9×.** The
wave structure — every task's *k*-th samples issued into one shared in-flight pool
of 48 — is what makes B≈40 and therefore 450 tok/s reachable. This is not an
optimisation; it is a precondition of H1b. It is also a mild self-contradiction
with "minimal harness": the hypothesis is only schedulable *because* of a
non-trivial scheduling mechanism.

---

## 5. Design decisions that follow from the budget

| Decision | Because |
|---|---|
| `--tensor-parallel-size 4`, **one** vLLM instance | FP8 27 GB > 24 GB, so TP≥2 is forced. One instance = one health check = one failure mode. TP=2×2 models ~14% faster (810 vs 711 tok/s at B=32) but doubles failure surface and halves KV per instance — wrong trade for an untestable one-shot rerun. Topology is a config flag for later A/B. |
| `--max-model-len 32768`, not 262144 | Measured worst case is 20,639 tokens. 262 K only inflates vLLM's profiling reservation and shrinks `num_gpu_blocks`. |
| `--enable-prefix-caching --mamba-cache-mode align` | Prefix caching makes N samples per task cost one prefill; `align` (not `all`) is required for caching on the hybrid DeltaNet architecture. |
| `--kv-cache-dtype` left at `auto` | fp8 KV would double capacity to ~1.6 M tokens, but capacity is already 2× what we need, and an unsupported flag on a hybrid model kills the kernel. Not worth the risk. |
| **Free-form generation + tolerant parsing**, no `guided_json` | Structured output is unreliable in vLLM thinking mode (vLLM#35700); MTP/speculative decoding breaks constrained decoding. |
| `reasoning_effort ∈ {xhigh, medium, low}` only, validated at construction | `high` raises HTTP 500 (QwenLM/Qwen3.8#217). `ChatRequest.__post_init__` rejects it before any network call. |
| **CSV** grid encoding | Best of 9 formats Land benchmarked; a bad format costs ~10%. |
| **Wave** scheduling, `concurrency=48` | §4.10. |
| **Phase A over all tasks first** | §4.7 — a complete valid submission exists after 23 minutes. |
| Per-test-index voting (not whole-vector) | Scoring is per test output; a sample that only parsed test 1 still contributes. |
| **Shape-diverse attempt_2** | Exact-match scoring means if the plurality *shape* is wrong, every same-shape candidate is wrong too. Measured on mock: **+25.0 points** (34.4% → 59.4%) in a low-agreement run. |
| Effort-weighted votes (`low`=1.0, `medium`=1.5, `xhigh`=2.0), truncated ×0.3 | Higher-effort samples are more likely right; a truncated answer is unreliable but not worthless. |
| Deterministic tie-break: weight → effort rank → first arrival | Makes selection reproducible and unit-testable. |
| Governor re-plans before **every** wave against *measured* tok/s | The throughput forecast is the least certain number in this document. |
| Cumulative atomic submission write after every wave + independent watchdog thread | A hard timeout must never cost the whole run. |
| 6-rung vLLM **flag fallback ladder** | `--mamba-cache-mode`, `--reasoning-parser` and the tool-call parsers are version-sensitive and cannot be tested locally without a GPU. |
| `stdlib`-only harness (urllib + threads, no aiohttp/httpx/openai) | Shrinks the offline wheelhouse to vLLM itself; removes a whole class of rerun-day dependency failures. |

---

## 6. Predicted score

### 6.1 Model

Score = mean over the **259 test outputs** of P(correct grid ∈ the 2 submitted attempts).
Per-task per-sample xhigh success probability `p_t` is strongly bimodal — Land's
own `ANALYZE_STRATEGY.md` pass rates cluster at 0/10, 1–4/10 and 6–10/10 — so a
single-bin binomial badly misestimates. Fit a 3-group mixture to that shape:

| Group | Share | `p_t` | P(k≥2 of 7) | Plurality win | P(scored) | Contribution |
|---|---:|---:|---:|---:|---:|---:|
| Model "gets" the rule | 17% | 0.72 | 0.973 | 0.93 (+0.02 attempt-2) | **0.95** | 16.2 |
| Partial | 10% | 0.20 | 0.423 | 0.55 (+0.09 from k=1 recovery) | **0.32** | 3.2 |
| Cannot solve | 73% | 0.02 | — | ~0.15 | **0.02** | 1.5 |
| **Total** | | **p̄ = 0.155** | | | | **≈ 20.9%** |

Worked binomials (N = 7 samples):

```
p=0.72: P(k≥2) = 1 − 0.28⁷ − 7(0.72)(0.28⁶) = 1 − 0.0014 − 0.0253 = 0.973
p=0.20: P(k≥2) = 1 − 0.80⁷ − 7(0.20)(0.80⁶) = 1 − 0.2097 − 0.3670 = 0.423
        P(k=1) = 0.367 → × ~0.25 attempt-2 recovery = 0.092
p=0.02: P(k≥1) = 1 − 0.98⁷ = 0.132 → × ~0.15 recovery = 0.020
```

### 6.2 Sensitivity — the score is a capability term, not a sampling term

| `p̄` (per-sample xhigh) | Predicted score | Interpretation |
|---:|---:|---|
| 0.08 | 12–16% | R1-0528 tier |
| **0.155** | **18–26%** | **central estimate for Qwen3.8-27B** |
| 0.25 | 30–36% | if HLE 30.8 translates unusually well |
| 0.35 | 43–49% | would already be the best open-weights ARC-AGI-2 result ever |
| **0.40** | **≈ 52%** | **the minimum H1 requires** |

And the **pass@∞ ceiling** (infinite samples, perfect selection) at `p̄ = 0.155` is
`0.17 + 0.10 + 0.73×0.02 ≈ 0.29`. **No harness whatsoever gets this model to 50%
under the central estimate** — the deficit is capability, not compute.

Marginal value of more samples: going 7 → 12 xhigh samples/task (the optimistic
scenario) moves the middle group from 0.32 → ~0.38 and the total from ~21% →
~23%. **≈ +2 points for a 60% budget increase.** For reference, Poetiq spent
8 experts × 10 iterations on a *frontier* model and got 54%.

### 6.3 Prediction

> **Predicted eval-split score: 18–30%, central ≈ 22%.**
> **P(score ≥ 50%) ≈ 10–15%.**
> **P(score > 33.89, the current public Kaggle best) ≈ 20–25%.**

The 10–15% tail is not negligible: it is the probability that HLE 30.8 + LCB 90.3
genuinely place Qwen3.8-27B at `p̄ ≈ 0.40` on ARC-AGI-2, which would make it a
massive outlier versus every published open-weights result. That is a real,
testable possibility — which is exactly why this branch is worth running.

---

## 7. What would confirm or refute H1

### 7.1 Primary criterion

```bash
python scripts/eval_local.py --split evaluation        # 120 tasks, ground truth
```

> **H1 CONFIRMED iff `SCORE ≥ 50.0%`.** Anything below 40% refutes it.

### 7.2 Cheap discriminator first — ~2 GPU-hours, 40 tasks

```bash
python scripts/eval_local.py --split evaluation --limit 40
```

The harness auto-reports the decomposition `score = coverage × selection_efficiency`
(`harness/diagnostics.py`). Read `pass@1` and `coverage (pass@N)`:

| Observation | Conclusion |
|---|---|
| `pass@1 ≥ 0.35` | **H1a alive.** Binding constraint is sample count → argue for a bigger budget, not a bigger harness. |
| `pass@1 ∈ [0.15, 0.35]` | **H1a marginal.** Expect 25–40%. |
| `pass@1 ≤ 0.12` | **H1a REFUTED.** Coverage ceiling implies <30% at any N. Stop; pivot to a code-refinement branch. |
| `coverage ≥ 0.50` but `score < 0.40` | Model is fine, **selection** is the bottleneck. Refutes "minimal harness", *not* "pure reasoning" → invest in a judge, a different branch. |
| `selection_efficiency ≥ 0.90` | Voting is near-optimal; extra harness machinery cannot help. |

### 7.3 Simultaneously validates the budget model

The same run logs, with no extra work:

| Logged quantity | Tests | Threshold |
|---|---|---|
| `measured tok/s` | §4.3 throughput model | < 200 → the pessimistic case is optimistic; expect ~3 samples/task |
| mean xhigh `reasoning_tokens` | the 10,000-token assumption | > 14,000 → budget shrinks ~30%; lower `max_tokens` or wave count |
| `parse methods` histogram | prompt-format compliance | `marker` should be > 90%; heavy `last-block`/`none` means the ANSWER contract is being ignored |
| `tasks with no parse` | fallback pressure | should be ~0 |
| `2nd-attempt gain` | shape-diverse selection | mock measured +25 pts; on real data expect +3 to +8 pts |

### 7.4 Falsification of H1b (independent of score)

If `measured tok/s ≥ 280` **and** the run finishes inside 10.5 h **and** the
submission validates, H1b is confirmed: xhigh is schedulable on L4x4 at ~6.7
samples/task. H1b can be confirmed while H1a is refuted — that is the most likely
outcome, and it is still a useful result for the other branches.

---

## 8. Downside risk (honest)

| # | Risk | P | Impact | Mitigation actually implemented |
|---|---|---:|---|---|
| 1 | **Capability ceiling** — `p̄ ≈ 0.10`, score 12–18% | **high** | Fatal to H1 | None possible inside H1. This is the hypothesis being tested, not a bug. |
| 2 | TP=4 over PCIe with no NVLink underperforms → <280 tok/s | medium | −3 to −5 pts (samples trim to ~2.6/task) | Governor re-plans each wave against measured tok/s; graceful trim, deadline never breached |
| 3 | vLLM rejects a flag (`--mamba-cache-mode`, `--reasoning-parser qwen3`) | medium | Kernel dies at startup → score 0 | 6-rung flag fallback ladder, each rung health-checked; rung 6 is TP=2 minimal |
| 4 | xhigh runs past 16 K and truncates mid-answer | medium | Unparseable samples | `max_tokens=16384`; parser also searches the reasoning stream; truncated candidates down-weighted ×0.3 rather than discarded |
| 5 | Thinking-only model never emits the `ANSWER n:` contract | low–med | Lost samples | 5-tier parse fallback: marker → alias markers → ordered blocks → tail blocks → last block → deterministic heuristic grid |
| 6 | Consensus early-stop freezes a confidently-wrong answer | medium | −1 to −2 pts | Threshold 0.70 with ≥4 candidates; attempt_2 is still forced shape-diverse, so a wrong-shape consensus is hedged |
| 7 | Dataset mount path differs from expectation | low | Kernel dies | Runtime glob for `config.json` across 4 candidate slugs + clear error |
| 8 | 12 h hard timeout mid-run | low | Catastrophic without mitigation | Cumulative atomic write after every wave + independent watchdog thread + `os._exit(0)` at the buffer |

**Worst realistic case:** vLLM never comes up, the ladder exhausts, and the
submission is the phase-A-free placeholder `[[0,0],[0,0]]` for all 240 tasks →
**score ≈ 0**. Probability after the ladder: low, but non-zero, and it is the
reason the flag ladder exists.

---

## 9. What I found that contradicts the hypothesis

1. **The arXiv 2607.15439 citation does not support the extrapolation.** "Effort
   gains exceeded variant differences" is a statement about *relative ordering
   within a model tier*. It licenses "don't over-engineer the harness"; it does not
   license carrying a frontier-model result (Astra 95%) down to a 27B open model.
   Effort moves `p̄` from perhaps 0.08 → 0.155. It cannot move it to 0.40.

2. **The budget is not the binding constraint.** This is the most important
   negative finding. At central throughput, 6.7 xhigh samples/task is already past
   the point of diminishing returns (§6.2: +2 points for +60% budget). Even an
   *unlimited* budget caps this model at ≈29% under the central `p̄`. So the answer
   to "can xhigh fit in 12 h?" is **yes**, and the answer to "does that matter?" is
   **probably not**.

3. **`sergiodefreitas/qwen3-8-27b` (17.3 GB), listed in the brief, is an Ollama
   GGUF split archive** (`ollama_weights.part_aa`, `part_ab`, …), not an HF
   checkpoint. vLLM cannot load it. Removed from the candidate list. The two FP8
   mirrors (`saltb0x/…-fp8`, `jakobbrggen/…-fp8-hf-snapshot`) were verified to have
   a real HF layout (`config.json` + `generation_config.json` + per-layer
   `layers-N.safetensors`) and are interchangeable.

4. **"Minimal harness" is not actually achievable at this budget.** Serial
   per-task execution overruns 12 h by ~1.9× (§4.10). The hypothesis is only
   schedulable because of a wave scheduler, a budget governor, a consensus
   early-stop and a cumulative watchdog — i.e. four non-trivial harness mechanisms.
   The honest restatement of H1 is *"minimal **model-interaction** harness"*: no
   judge, no refinement, no code execution — but the scheduling layer is load-bearing.

5. **Selection, not sampling, is where the cheap wins are.** On a mock run tuned to
   low agreement, shape-diverse attempt_2 alone contributed **+25 points**
   (34.4% → 59.4%). On real data expect +3 to +8. That is a larger lever per unit
   of engineering than any additional xhigh sample, and it is already in the
   "minimal" harness.

6. **The watchdog itself was a single point of failure, and testing found it.** The
   cumulative writer is written from two threads: the scheduler after each wave and
   the watchdog daemon at `t_gen`. Unsynchronised, they collide at exactly the worst
   moment in the run — concurrent `os.replace` raised
   `PermissionError: [WinError 5]` on Windows (reproduced ~1 run in 8), and on Linux
   the same race makes `json.dump` iterate a dict that `update()` is mutating,
   raising *"dictionary changed size during iteration"*. Either would have killed the
   kernel at ~11 h 50 m and forfeited the entire run — the precise scenario the
   watchdog exists to prevent. Fixed with an `RLock` over `update()`/`write()`, a
   JSON snapshot taken under the lock, and a bounded retry on `PermissionError`.
   Covered by a 4-thread stress regression test; **15/15 clean suite runs** after the
   fix, versus ~1 failure in 8 before. Lesson: the safety mechanism needs its own
   failure modes tested, or it is just a second way to lose the run.

---

## 10. Bottom line

| Question | Answer |
|---|---|
| Does xhigh fit in 12 h / L4x4 for 240 tasks? | **Yes.** 4–11 samples/task (central **6.7**), total ≈ 10.4 h, reserve untouched. |
| Is the budget the binding constraint? | **No.** Capability is. |
| Predicted eval-split score | **18–30%, central ≈ 22%** |
| P(≥ 50%), i.e. P(H1 true) | **≈ 10–15%** |
| Cheapest decisive experiment | `--limit 40`, read `pass@1` and `coverage`. ~2 GPU-hours. |
| Biggest single risk | `p̄ ≈ 0.10` → 12–18%, indistinguishable from the public TTT baseline while costing 100× more compute. |
