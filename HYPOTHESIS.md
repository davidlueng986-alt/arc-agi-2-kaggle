# H4 — Verified Ensemble + Holistic Judge, pass@2-optimised

**Branch:** `h4/poetiq-judge` · **Competition:** `arc-prize-2026-arc-agi-2` (240 unseen tasks, L4×4, 12 h, offline rerun, 2 attempts per test output, exact-match scoring)

## 1. The claim

The highest-expected-value architecture that is *already proven* on ARC-AGI-2 is:

- **(a) Poetiq's code-refinement engine** — verified **54.0%** ARC-AGI-2 semi-private, ARC-Prize-verified, at **$30.57/task**, beating the previous best (Gemini 3 Deep Think, 45% @ $77.16/task) at half the cost. ARC Prize's 2025 technical report credits it as a verified harness raising Gemini 3 Pro from a 31% / $0.81-per-task baseline to 54% (**+23 points of pure harness gain**).
- **(b) Land's holistic trace judge** ("Modality-Driven Search with Holistic Trace Judging", arXiv 2606.31543) — verified **72.9%** semi-private @ $38.99/task, **76.1%** public eval @ $19.69/task; the top verified leaderboard entry, **+18.7 points** over standalone GPT-5.2 Pro (54.2%) / Gemini 3 Pro (54.0%). The judge alone contributed **+7 solved instances over majority vote at ~13% of system cost**.

Both were **ineligible for Kaggle** (external APIs; Poetiq states this explicitly). Porting them to a **locally-served Qwen3.8-27B (FP8, vLLM, TP=4)** inside the notebook, trimmed to Kaggle's 12 h / ~90–120K-token-per-task budget and explicitly optimised for **pass@2** rather than pass@1, is the novel contribution of this branch. Nobody has published either mechanism inside a Kaggle code competition with a local open-weight model.

This is the **de-risked branch**: its value is that every load-bearing component has an independently verified number attached, and the residual risk is concentrated in exactly one unmeasured quantity — the transfer of harness gains from frontier API models to a 27B open model.

## 2. Reconciliation: Poetiq's refinement vs Land's "refinement reduces diversity"

Land's documented negative result: *"prescriptive prompting templates and iterative refinement systematically reduce hypothesis diversity and degrade performance."* Poetiq's entire engine is 10 iterations of refinement — and it is the verified 54% system. The contradiction is only apparent; the two operated on **different objects**:

**Land refined a single chain.** Each refinement step conditions the next sample on *the previous answer*. The posterior concentrates on the first hypothesis (conditioning collapse); one wrong turn early makes every later sample a variant of the same wrong rule. Land's own `ANALYZE_STRATEGY.md` shows the same effect from the prompt side: injecting an explicit strategy lifts pass rate 0/10 → 9/10, but *generic* advice collapses it back to 0–1/10 — prescriptive content narrows the hypothesis distribution.

**Poetiq refines a population.** Every config knob in `arc_agi/config.py` is an anti-collapse mechanism:

| Poetiq knob | Anti-collapse function |
|---|---|
| `max_solutions: 5` (pool) | feedback context contains up to **5 distinct scored hypotheses**, not the last answer — the next sample is conditioned on a *distribution*, not a point |
| `soft_score` (partial credit) | near-miss alternative rules **survive** in the pool; a binary pass/fail filter would purge diversity before it can pay off |
| `shuffle_examples: True`, seed `seed+it` | train pairs re-ordered every iteration — different priming/order effects = free augmentation; breaks deterministic conditioning |
| `solver_temperature: 1.0` | maximum sampling entropy maintained across all iterations |
| `selection_probability: 1.0`, `improving_order: True` | *all* pool members always shown, worst→best — no elitist truncation, and the recency slot goes to the strongest exemplar without hiding the others |
| `NUM_EXPERTS ∈ {1,2,8}` | structurally **independent chains** with disjoint seed ranges (`solve_parallel_coding`: `cfg["seed"] += it * max_iterations`) — cross-expert diversity is immune to within-expert collapse |
| `use_new_voting`, diversity-first ordering | aggregation takes **one representative per distinct output grid** before duplicates — the final ranking rewards distinct hypotheses, not repetition |

**Resolution (design rule this branch enforces):** *refinement within an expert, diversity across experts, entropy at every step, and aggregation that is diversity-first.* Concretely:

1. Never feed the model only its last answer (pool ≥ 5, always fully shown).
2. Never filter the pool by binary pass/fail (soft-score ranking only; a failed-verification candidate is kept — Land measured verification precision ~97% but **recall ~43%**, so failing verification is a weak negative signal).
3. Keep ≥ 2 structurally independent experts with **disjoint seeds AND different prompt families** — Expert A uses Poetiq's rich prescriptive template (proven *in the pooled configuration*), Expert B uses Land's minimal codegen prompt (proven in the *single-shot* configuration). Each prompt family is used in the regime where it was verified to help.
4. Keep a non-code free-form text channel (3 samples, minimal prompt, no refinement) injecting Land-style raw diversity.
5. The holistic judge sees **all** traces including minority ones — it is itself an anti-consistency mechanism (*"designed to identify a correct minority hypothesis against a confidently wrong majority"*).

**Falsification handle:** the reconciliation predicts that ablating to a single expert with pool size 1 (pure chain refinement) degrades eval-split pass@2 versus the 2-expert/pool-5 configuration, while ablating *across-expert* diversity (identical seeds/prompts) degrades it less. `scripts/eval_local.py --experts 1 --pool-size 1` vs `--experts 2 --pool-size 5` measures exactly this.

## 3. Architecture (as implemented in `harness/`)

```
task (train pairs + test inputs)
   │
   ├─ Expert A — Poetiq engine: SOLVER_PROMPT_2, ≤8 iterations, temp 1.0, effort=low,
   │    shuffle_examples(seed+it), pool-5 feedback w/ soft scores + diff grids,
   │    sandbox after every sample, early-exit when all train pairs pass (verified)
   ├─ Expert B — Land minimal codegen (v1b-style, test inputs as probes), ≤4 iterations,
   │    disjoint seed, same pool/feedback machinery
   ├─ Text channel — 3× Land minimal free-form prompt (CSV answer), effort=low
   │
   ├─ early accept (skip judge): ≥2 independent verified votes on ONE grid for this test input
   │
   ├─ Holistic judge council — ONE prompt per test input containing the task, ALL candidate
   │    traces (code/reasoning truncated to 2.5K chars) + predicted grids in CSV;
   │    3 samples of the SAME model at reasoning_effort=medium, temp 1.0 (prefix cache
   │    ⇒ 1 prefill, 3 decodes); each judge outputs its top-2 grids (may SYNTHESIZE a
   │    novel grid); aggregate 2 pts per first choice, 1 pt per second choice per distinct grid
   │
   ├─ pass@2-aware attempt selection (shape-prior repair + source decorrelation, §5)
   │
   └─ cumulative submission.json after every task + watchdog finalize before the 600 s buffer
```

Verification semantics (Land, measured on 10 tasks × 10 iterations): pass rate 88%, verified rate 38%, 1 verified-but-wrong ⇒ **precision ~97%, recall ~43%**. Consequences: a verified program goes to the top of the pool (and gets a selection bonus), but *failing verification never discards a candidate* — the soft-score ranking and the judge keep minority hypotheses alive.

## 4. Trimmed budget allocation — full arithmetic

### 4.1 Wall clock

| Item | Value |
|---|---|
| Kaggle limit | 12 h = 43,200 s |
| Submission buffer (lb3389 pattern) | −600 s |
| vLLM load + CUDA graph capture (27 GB FP8 from dataset mount) | −900 s |
| **Usable** | **≈ 41,700 s** |
| Tasks at rerun | 240 ⇒ 173.75 s/task if strictly serial |

### 4.2 Serving arithmetic (FP8 27B, TP=4 on 4×L4)

- Per L4: 24 GB, ~300 GB/s bandwidth, ~121 TFLOPS FP8 dense.
- Weights: 27 GB ÷ 4 = **6.75 GB/GPU** → single-stream decode ≈ 300/6.75 ≈ **44 tok/s**; with continuous batching over B=6–10 concurrent streams (2 experts + 3 text + 3 judges overlap within/across 2 tasks), aggregate decode ≈ **200–350 tok/s**. Planning value: **200 tok/s aggregate (conservative)**.
- Prefill: 4 × 121 TFLOPS ÷ (2 × 27 GFLOP/token) ≈ 8,900 tok/s ideal; at 30% MFU ≈ 2,700 tok/s. Planning value: **2,000 tok/s**.
- KV: ~64 KB/token (only 16 of 64 layers are full attention); at TP=4 KV heads shard ⇒ ~16 KB/token/GPU. Per GPU: 24×0.90 − 6.75 (weights) − ~2 (activations/graphs) ≈ **12.8 GB KV** ⇒ ~800K tokens of sequence capacity (200K even if unsharded). Config: `--max-model-len 49152` (judge prompt ~15–19K + 4K out fits 2× over), `--max-num-seqs 16`, `--gpu-memory-utilization 0.90`, `--enable-prefix-caching --mamba-cache-mode align` (hybrid Gated-DeltaNet requires `align`), `--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`. No speculative decoding / MTP (breaks constrained decoding), no NVFP4 (Blackwell-only; L4 is Ada sm_89).
- `reasoning_effort ∈ {xhigh, medium, low}` only (`high` → HTTP 500, QwenLM/Qwen3.8#217); **default xhigh emits 8–16K thinking tokens/sample and would blow the budget** → pinned **low** for all generation, **medium** for judges. Hard `max_tokens` caps: 2,048 (solver) / 2,500 (text) / 4,096 (judge) guard against runaway thinking.

### 4.3 Per-task token allocation

| Component | Calls | Prefill/call | Decode/call | Billed tokens | GPU-effective tokens |
|---|---|---|---|---|---|
| Expert A (≤8 iters, effort=low) | 8 | ~4.2K (prompt 1.9K + problem 0.7K + pool feedback ≤1.6K) | ≤1.6K | 46.4K | 46.4K |
| Expert B (≤4 iters, minimal prompt + probes) | 4 | ~2.4K | ≤1.6K | 16.0K | 16.0K |
| Text channel (3 candidates) | 3 | ~1.3K | ≤2.0K | 9.9K | 9.9K |
| Judge council (15K prompt **once**, prefix-cached ×3; 4K decode ×3) | 3 | 45K billed / 15K effective | 12K | 57K | 27K |
| **Worst-case total** | **18** | | | **~129K billed** | **~99K effective** |

"GPU-effective" counts the shared judge prefill once (prefix cache hit). Land's full system: 29 candidates (8 text + 10 image + 11 code) + 3 judges ≈ **175K+ tokens/task ≈ 2× over budget**.

**What was cut and why:**
- **Image channel (10 candidates in Land) — cut entirely.** Land's own analysis (`ANALYZE_MULTIMODAL_IMAGES.md`): *"I have seen multimodality help in some cases, but it's not consistent and in somewhat structured testing I can't really recreate it."* Highest cost-per-verified-gain of any channel; also Qwen3.8 VL rendering adds a failure surface (image encoding inside vLLM) for unproven gain.
- **Code experts 11 → 2 (8+4 iterations, not 8×10).** Poetiq's published 54% is the 1-expert config; the 8-expert config multiplies cost ~8× for the "-c" variant. Two experts with disjoint seeds *and prompt families* preserve the across-expert diversity mechanism (§2) at ¼ the cost of Poetiq-8. Iterations 10 → 8: Poetiq early-exits on verification; the 9th/10th iterations fire only on tasks that resisted 8 samples, where marginal yield is lowest.
- **Text candidates 8 → 3.** Diversity insurance for the judge, not a primary channel; Qwen3.8-27B's coding (SWE-bench Pro **61.7** > Opus4.6 Max 53.4; Terminal Bench 73.0) is frontier-class while its abstract reasoning (HLE **30.8** vs ~40) is below frontier ⇒ weight the **code channel** heavily.
- **Judges 3 → 3: refused to cut.** The judge is the single highest-leverage component in Land's ablations (+7 solves at 13% of cost). With prefix caching the marginal cost of judges 2–3 is decode-only (~4K tokens each, ~4% of task decode). Cutting to 1 judge saves ~8K tokens/task and forfeits the council's noise-averaging — exactly the wrong trade for a weaker-than-frontier judge (§6).

**Expected (not worst-case) per-task cost:** verification early-exit truncates Expert A at ~4–5 iterations on average, and ~35–45% of tasks hit the early-accept path and skip the judge entirely ⇒ **expected ≈ 60–80K effective tokens/task**, inside the 90–120K envelope with margin.

### 4.4 Wall-clock roll-up (240 tasks)

| | Worst case | Expected (with early exits) |
|---|---|---|
| Decode tokens total | 240 × 31K = 7.4M | ~5.0M |
| Decode time @ 200–350 tok/s aggregate | 21,300–37,000 s | 14,000–25,000 s |
| Prefill tokens total (effective) | 240 × 62K = 14.9M | ~10M |
| Prefill time @ 2,000–2,700 tok/s | 5,500–7,500 s | 3,700–5,000 s |
| **Total** | **26,800–44,500 s** | **17,700–30,000 s** |
| vs 41,700 s usable | tight at the pessimistic corner | **28–58% margin** |

Sandbox CPU (≤18 runs × ≤5 pairs × ≤5 s per task) overlaps GPU decode and is not additive. Because the worst-case corner is tight, the runner implements a **degradation ladder** driven by the global deadline (§7): L1 @60% elapsed (iters 8→5, text 3→1), L2 @75% (iters→3, no text, 1 judge), L3 @88% (single-shot experts, no judge, consensus-only), L4 @93% (stop solving; fill remaining tasks with fallback grids; finalize). The submission is written cumulatively after every task, so any crash/timeout loses at most the in-flight task.

## 5. pass@2-aware attempt selection

The metric scores a point if **either** attempt matches exactly: `pass@2 = P(A1✓) + P(A1✗) · P(A2✓ | A1✗)`. Naive top-2 maximises the second-highest *score*; pass@2 demands attempt_2 maximise **conditional coverage** — it is worth exactly zero when it shares attempt_1's failure mode.

### 5.1 Measured shape statistics (120-task eval split, 172 test outputs — computed locally)

| Statistic | Value |
|---|---|
| P(shape(out) = shape(in)) over all test outputs | 71.5% |
| Tasks where the train-pair shape relation is consistent | **98.3%** |
| Train all-identity ⇒ test identity | **117/117 = 100%** |
| Train all-non-identity ⇒ test non-identity | 48/52 = 92.3% |
| Test inputs per task | 71×1, 46×2, 3×3 |

⇒ The **true output shape is a near-deterministic function of the train pairs** ("implied shape" S\*: identity, constant dims, or a per-axis linear fit `h' = a·h + b` validated on all train pairs). Shape errors are *systematic*: two candidates produced by mis-generalising the same shape rule fail **together**. This makes shape the right decorrelation axis — but only against the implied prior, not blindly (a naive "prefer a different shape for A2" rule would *lose* points, since P(S\* correct) ≈ 0.95–1.0).

### 5.2 The rule (implemented in `harness/selection.py`)

Per test input, over distinct candidate grids with metadata (judge points, verified flag, votes, best soft score, source channel, shape):

1. `score = judge_points + 2.0·verified + 0.25·ln(1+votes)` (no-judge mode: `3.0·verified + 1.0·ln(1+votes) + 0.5·soft`).
2. **attempt_1** = argmax score (deterministic tie-break: verified → votes → soft → grid key).
3. **Shape-repair hedge:** if S\* is known and `shape(attempt_1) ≠ S*`, attempt_2 = highest-scoring candidate **with shape S\*** (if any exists).
4. Otherwise attempt_2 = highest-scoring grid ≠ attempt_1, with tie-breaks preferring (a) a **different source** (expert/channel) than attempt_1 — independent failure modes, (b) larger content distance (normalised Hamming ≥ 0.5 when shapes match).
5. If the top-2 share both shape and source *and* a different-shape candidate with **support ≥ 2** (votes ≥ 2 or judge points ≥ 1) exists, swap it into attempt_2 (rare ambiguous-shape hedge; support-gated so a random singleton never displaces a corroborated candidate).
6. Hard guarantee: attempt_2 ≠ attempt_1 as grids; when fewer than 2 distinct grids exist, fall back to identity(test_input), then to the modal-train-output-colour grid with the implied shape. **Never lose the submission.**

The verified bonus is calibrated so that the shape prior and verification (both ~95–97% precise) dominate raw judge preference, while a genuine 2/3 judge majority (4+ points) can still overturn either — preserving Land's minority-rescue mechanism without letting one noisy judge sample misrank.

### 5.3 Quantified expected gain over naive top-2

Decompose by trigger case (per test output):

| Case | Frequency (measured/estimated) | Naive A2 hit rate | Diversity A2 hit rate |
|---|---|---|---|
| **B:** attempt_1 shape contradicts S\* (unverified top pick w/ wrong shape, or verified program mis-extrapolating content-dependent dims) | ~5–12% of outputs (shape surprises ≥2.6% measured even among train-consistent tasks; unverified-top-pick tasks push this up — the 27B verifies maybe half of tasks) | ~0.03 (A2 shape-correlated with A1) | ~0.30 (best S\*-shaped candidate ≈ conditional solve rate given right shape) |
| **C:** ambiguous S\* (train-non-identity extrapolation; 4/52 = 7.7% of non-identity outputs ⇒ ~2–3% of all) | ~3% | ~0.05 | ~0.12 (supported alternative shape family gets a slot) |
| **A:** unambiguous, attempt_1 shape = S\* (majority, ~85–92%) | ~87% | tie-break-only changes (source/content decorrelation on ties) | +0 to +0.01 |

- Case B contributes: 0.05–0.12 × (0.30 − 0.03) ≈ **+1.4 to +3.2 points**.
- Case C contributes: 0.03 × (0.12 − 0.05) ≈ **+0.2 points**.
- Case A (source decorrelation): ≈ **+0.1–0.3 points**.
- Downside: the rule mis-fires only when S\* itself is wrong (≤2–8% of tasks) *and* the naive A2 happened to be correct (≈0.03–0.05) ⇒ ≤ **−0.2 points**. Note the shape-repair rule *subsumes* the naive pick in case B (if the naive A2 already has shape S\*, it is selected unchanged), so the swap is close to strictly dominant.

**Net expected gain: ≈ +1.5 to +3.5 points of final score (median ≈ +2) over naive top-2**, i.e. roughly 6–12% relative on a ~28% baseline. All parameters above are ablatable (`--no-diversity` in `eval_local.py`) so the contribution is measurable on the 120-task eval split.

## 6. Judge-role decision (given a weaker-than-frontier judge)

Land judged with `gpt-5.2-xhigh`. We have one model: Qwen3.8-27B, coding frontier-class but abstract reasoning below frontier (HLE 30.8 vs ~40). Decision:

1. **Same model, `reasoning_effort=medium`, temperature 1.0, 3 samples of the identical prompt.** Homogeneous councils beat mixed-model ensembles in Land's measurement (and we have no second model offline anyway). `xhigh` for judges is rejected on budget: 8–16K thinking/sample × 3 judges = +12–42K decode/task, up to 40% of the whole task budget for one component; `medium` (~2–4K thinking) is the affordable tier and the council substitutes sample-count for per-sample strength.
2. **3 judges, not 1 — because the judge is weak.** Per-sample misranking variance is the main risk of a 27B judge; 2/1-point aggregation across 3 independent samples is self-consistency *for the judge itself*. Marginal cost ≈ 4K decode tokens/judge (prefix cache shares the 15K prefill) ≈ 4% of task decode. This is the component we refuse to cut.
3. **Recognition, not recall.** The judge selects among grids already in context (and may synthesize by *recombining* sub-components) — a recognition task, which partially offsets the HLE deficit versus free generation.
4. **Weak-judge guardrails.** (i) verified bonus 2.0 ⇒ a single judge first-choice (2 pts) cannot displace a verified grid (2.0 + votes); displacement requires a 2/3 first-choice majority (≥4 pts). (ii) A **synthesized** grid (never executed) is eligible for attempt_2 at council majority but takes attempt_1 over a verified grid only at ≥4 pts. (iii) Unanimous verified consensus with ≥2 independent votes **skips the judge** (its work is done; precision ~97%). (iv) Judge parse failures fall back to the no-judge scoring path — a dead judge degrades to consensus, never crashes the task.
5. **Anti-consistency instruction preserved verbatim** from Land's closing prompt (*"Often, new mechanics are introduced in the test example for which the solutions do not generalize well… study all the provided solutions and their reasoning to come up with a meta-conclusion"*), no vote counts revealed per candidate beyond the natural multiplicity of traces, and **CSV grid format** (best of 9 tested; a bad format costs ~10%) with tolerant parsing (`extract_all_grids`, last-2-distinct).

## 7. Local-model adaptation (27B on 4×L4, offline, hard 12 h cap)

Failure modes that frontier APIs made rare, and their handling:

| Failure | Handling |
|---|---|
| Malformed/missing code block | tolerant extraction: fenced ```python / ```py / bare fence / unfenced `def transform` / `def solver` (auto-bound to `transform` in the sandbox shim); sample retried within the iteration budget, iteration skipped on repeated failure (Poetiq semantics) |
| Refusals / empty output | refusal-pattern detection ⇒ treat as failed iteration, not fatal |
| Runaway thinking | `reasoning_effort` pinned (never `high` — HTTP 500; never default xhigh), hard `max_tokens` caps, per-request timeout with 2 retries |
| HTTP 500 / connection flakes | client-level retry with backoff; per-expert failure isolates (asyncio.gather with return_exceptions) — one dead expert never kills the task |
| Sandbox: infinite loop / fork bomb / FS writes | subprocess isolation, clean env (`PYTHONHASHSEED=0` + minimal platform vars), 5 s hard timeout + kill, stdout/stderr capped at 64 KB, temp cwd |
| Judge prompt overflow | ≤12 traces per test input, each truncated to 2.5K chars; 49K context cap leaves 2× margin |
| 12 h hard stop | deadline-aware degradation ladder (§4.4) + watchdog thread finalizing `submission.json` at deadline−60 s + cumulative write after every task + schema validation with fallback-fill for every missing task/test-output |
| vLLM missing at rerun (image change) | notebook probes `import vllm` → offline wheelhouse dataset → interactive pip; if serving never comes up, the fallback solver (identity + modal-colour grids) still writes a fully schema-valid submission |

## 8. Predicted score

Chain of evidence: Poetiq lifted Gemini 3 Pro 31% → 54% (harness gain ×1.74, +23 pts); Land's judge added +18.7 pts over the standalone model. Qwen3.8-27B single-shot ARC-AGI-2 code-solve rate is unmeasured; scaling Gemini-3-Pro's 31% raw by the HLE ratio (30.8/~40) suggests a ~20–25% raw code-channel base, but a 27B open model plausibly lands lower on ARC-AGI-2's anti-memorisation design.

| Scenario | Eval-split pass@2 |
|---|---|
| Optimistic (coding strength transfers; harness gain ×1.6–1.7) | **33–38%** |
| Central estimate | **24–30%** |
| Pessimistic (base-model coverage floor; weak judge adds ~0) | **15–20%** |
| Reference: best public Kaggle notebook (TTT, forbidden here) | 33.89% |
| Reference: Kaggle private #1 (Tufa Labs/MindsAI, program synthesis) | 83.06% |

**Predicted eval-split range: 18–35%, median ≈ 26–28%.** The rerun set is unseen and harder-skewed than the public eval split; expect the LB number to land a few points below the local measurement.

## 9. Confirmation / refutation criteria

**Confirms the claim** (all measurable locally via `scripts/eval_local.py` before spending a rerun):
1. Eval-split (120 tasks) pass@2 ≥ 25% with the real model at the default config.
2. Judge ablation: `--no-judge` loses ≥ 2 points vs judge-on, concentrated on tasks without verified consensus (if the judge contributes ≤ 0, the 27B judge mis-ranks and component (b) does not transfer — the claim is *half*-refuted).
3. Diversity ablation: `--no-diversity` loses ≥ 0.5 points (else the pass@2 angle is inert and §5's estimates are wrong).
4. Rerun completes 240 tasks inside 12 h with a schema-valid submission (watchdog never triggered at L4).

**Refutes the claim:**
1. Eval-split pass@2 < 15% ⇒ base-model coverage floor; harness gains do not transfer from frontier models to 27B (the one unmeasured quantity).
2. Verified-program rate < 20% of tasks ⇒ the code channel itself doesn't transfer (SWE-bench ≠ ARC).
3. Wall-clock blowup at L2+ degradation on a full 240-task rehearsal ⇒ the budget arithmetic in §4.4 is wrong.

## 10. What contradicts this hypothesis (honest register)

1. **Land's negative result is real evidence against part (a)** — reconciled in §2, but the reconciliation is an argument plus a falsifiable ablation, not a measured fact for *this* model.
2. **The judge was validated at gpt-5.2-xhigh strength.** A 30.8-HLE judge may mis-rank; our guardrails (§6.4) bound the damage but also bound the upside: if the judge can't overturn bad consensus, component (b) contributes ~0 and we are a trimmed Poetiq port.
3. **Poetiq's 54% was Gemini 3 Pro; Land's 72.9% was a frontier multi-model ensemble.** Every published number in §1 was produced by models stronger than ours at abstract reasoning. The harness-gain transfer coefficient is assumed, not measured.
4. **CSV-format and minimal-prompt wins were measured on gpt-5.1** — format sensitivity is model-specific; Qwen may prefer a different grid rendering (we keep tolerant parsing so a format miss degrades gracefully).
5. **Worst-case wall clock has only ~5% margin** (§4.4 pessimistic corner). The degradation ladder converts this from a crash risk into a score risk, but a full-batch 240-task rehearsal has not been run (no local GPU).
6. **The current public LB leader (33.89) uses TTT** — a technique this branch forbids. Our median prediction (~26–28%) is *below* it; this branch's thesis is that the pass@2 + verified-harness architecture has a higher ceiling once base-model coverage is confirmed, and that it is the only branch whose components carry published ARC-Prize-verified numbers.
