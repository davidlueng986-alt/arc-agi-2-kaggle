# Harness-First Architecture: Qwen3.8-27B, no SFT, no TTT

Decision record + implementation map. Supersedes the SFT/TTT scale-up plan in
`qwen38-27b-solution-map.md` (kept for history). Evidence gathered 2026-09-26.

## 0. Verdict on the two proposals

**Agreed, and the evidence is stronger than expected.** Both proposals are
independently published SOTA on ARC-AGI-2:

| Proposal | Published equivalent | Verified score |
|---|---|---|
| B: Two-turn pipeline via code (+ thinking) | **Poetiq** `poetiq-ai/poetiq-arc-agi-solver` (1K stars) | **54.0%** ARC-AGI-2 semi-private, ARC-Prize-verified, $30.57/task |
| A: Native tool / function calling | Land's **code channel** (11 of 29 candidates) + tool-use traces | part of **72.9%** semi-private / 76.1% public |

Reference: Poetiq beat the previous best (Gemini 3 Deep Think, 45% at $77.16/task)
at **half the cost**. Land's solver is the top verified leaderboard entry, beating
standalone GPT-5.2 Pro (54.2%) and Gemini 3 Pro (54.0%) by **+18.7 points**.

### Does Qwen3.8-27B still need SFT? — No. And it is also infeasible.

`sorokin/qwen3_4b_grids15_sft139` is the ARChitects/NVARC checkpoint (1st place
ARC Prize 2025). What it actually is:

- **Custom 16-token vocabulary** grafted onto Qwen3 (digits 0-9, newline, user,
  assistant, pad, eos) — a specialized seq2seq model, not a chat model.
- Trained on **3.2M augmented synthetic puzzles** (SDG pipeline: arc2_training,
  MINI-ARC, ConceptARC, RE-ARC, NVARC synthetic), 1 epoch, 12,716 steps.
- **32 GPUs** (4 nodes x 8), Megatron-LM, TP=8, CP=2.
- Its own model card: *"cannot solve ARC puzzles via simple inference... produces
  garbage without per-puzzle fine-tuning."*

So the SFT route requires a 32-GPU cluster + a synthetic-data pipeline. It is out
of reach on Kaggle, and it forces TTT. Dropping SFT is not just a preference — it
is the only viable path, and it happens to be the path the current SOTA takes.

### Does it need TTT? — No.

- TTT lineage on Kaggle: 2025 top = **24%**, 2026 best public notebook = **33.89**.
- Harness lineage (no weight updates at all): **54%** (Poetiq), **72.9%** (Land).
- Kaggle private #1 is **Tufa Labs (83.06)** = MindsAI, whose published artifact is
  `mindware/arc-codet5-660m`, a **program-synthesis** solver — i.e. the Kaggle
  winner is already code-generation based, not TTT.
- ARC Prize 2025 Technical Report names the **refinement loop** the defining theme
  of 2025-2026, and created a "Model Refinements" leaderboard category for
  application-layer harnesses.

## 1. Four corrections to the plan (evidence-based)

1. **Do NOT apply structured output globally.** Land's documented negative results:
   *"prescriptive prompting templates and iterative refinement systematically reduce
   hypothesis diversity and degrade performance."* Strict output constraints hurt.
   -> Use free-form generation + **tolerant CSV parsing** for the text/image channels
   (CSV beat 8 other grid formats; a bad format costs ~10%). Reserve
   `guided_json`/tool-calling for the **code channel**, where the contract is
   `def transform(grid: np.ndarray) -> np.ndarray` and structure genuinely helps.

2. **"Just enable thinking" will blow the budget.** Qwen3.8 open models are
   thinking-only and default to **`reasoning_effort=xhigh`**. The chat template
   accepts only `{xhigh, medium, low}` — passing `high` raises **HTTP 500**
   (QwenLM/Qwen3.8#217). xhigh thinking can emit 8-16K tokens/sample.
   -> Pin `low` for generation iterations, `medium` for the judge. `reasoning_effort`
   is our primary compute knob.

3. **Multimodal: yes, but not as the primary representation.** Land tested feeding
   rendered grids directly and concluded: *"I have seen multimodality help in some
   cases, but it's not consistent and in somewhat structured testing I can't really
   recreate it"* -> *"extract the hints in a separate prompt and stay text-only."*
   Yet the final system still runs an **image channel (10 of 29 candidates)**.
   -> Correct use of Qwen3.8's native VL: (a) image as an **independent diversity
   channel**, (b) **image -> visual-hint extraction -> text** two-stage
   (`hint_generation.py` / `ANALYZE_VISUAL_HINTS.md`). Do not replace CSV text.

4. **Iterative refinement must preserve a scored pool.** Land found refinement
   collapses diversity; Poetiq is 10 iterations of refinement and still wins. The
   difference is in Poetiq's config: `max_solutions=5` (keeps a pool), `soft_score`
   (partial credit, not binary), `shuffle_examples=True` with `seed+it` (re-orders
   train pairs every iteration = free augmentation), `improving_order=True`,
   `temperature=1.0`, and `NUM_EXPERTS` up to 8 independent experts.
   -> Design rule: refine **within** an expert, keep **many** experts/candidates,
   never let a single chain dominate.

## 2. Target architecture (fully local = Kaggle-eligible)

Poetiq and Land both used external APIs, so **both were ineligible for Kaggle**
(Poetiq states this explicitly). Porting the architecture to a local 27B inside the
notebook is the novel contribution — nobody has published it for this competition.

```
                 ARC task (train pairs + test inputs)
                              |
              render: CSV text (primary) + PNG (cartoon/precise)
                              |
   +--------------------------+---------------------------+
   |  CHANNEL C: code refinement expert (Poetiq engine)    |
   |    for it in 1..8:                                    |
   |      prompt = SOLVER_PROMPT + task(CSV, shuffled)     |
   |               + top-5 partial solutions w/ soft_score |
   |      -> model emits explanation + ONE python block    |
   |      -> sandbox: exec transform() on ALL train pairs  |
   |      -> soft_score = matched/total (+ shape credit)   |
   |      -> keep pool of best 5                           |
   |    verified programs = 100% train match               |
   +--------------------------+----------------------------+
   |  CHANNEL T: free-form text candidates (k=3, low effort)|
   |  CHANNEL I: image channel (k=2) + visual-hint->text    |
   +--------------------------+----------------------------+
                              |
        early stop if >=3 candidates agree on identical grid
                              |
   +--------------------------+----------------------------+
   |  HOLISTIC JUDGE (1 pass, medium effort)               |
   |   single long-context prompt: task + ALL traces        |
   |   (code + exec logs + reasoning) + predicted grids     |
   |   -> ranked top-2, may SYNTHESIZE a new grid           |
   |   anti-consistency: no instruction to prefer majority  |
   +--------------------------+----------------------------+
                              |
        attempt selection: judge #1, then #2 with a
        SHAPE-DIVERSITY constraint (binary exact-match scoring)
                              |
        cumulative submission.json write after every task
```

## 3. Compute budget (the binding constraint)

L4: 24GB, ~300 GB/s bandwidth, sm_89 (Ada). Qwen3.8-27B 4-bit = ~15GB weights.

| Item | Estimate |
|---|---|
| Single-stream decode | ~20 tok/s (15GB / 300GB/s) |
| Batched decode per L4 (B=8-16) | ~150-250 tok/s |
| 4 independent TP=1 instances | ~500-700 tok/s aggregate (conservative) |
| Usable wall clock | 12h - 600s buffer - ~15min load/kernel-compile = ~41,700s |
| **Total token budget** | **~21-29M tokens** |
| **Per task (240 tasks)** | **~90-120K tokens** |

Allocation per task (target ~95K):

| Component | Tokens |
|---|---|
| Code expert, 8 iterations (~4K prompt + 1.5K completion each, prefix-cached) | ~44K |
| Second code expert (different seed/prompt), 4 iterations | ~22K |
| Text channel k=3 | ~9K |
| Image channel k=2 + hint extraction | ~8K |
| Holistic judge (1 pass, ~25K trace context + 2K out) | ~27K |
| **Total** | **~110K** (tune down if over) |

Land's full config (29 candidates + 3 judges) costs ~175K+/task — **2x over budget**.
Poetiq's 8-expert config is also over. We run a trimmed version: **2 experts x
(8+4) iterations, 5 non-code candidates, 1 judge.**

## 4. vLLM serving config (with the traps)

```bash
# 4 independent instances, one per L4 (CUDA_VISIBLE_DEVICES=i, port 800i)
vllm serve /kaggle/input/qwen3-8-27b-awq-int4 \
  --served-model-name qwen3.8-27b \
  --tensor-parallel-size 1 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.92 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --enable-prefix-caching --mamba-cache-mode align \
  --kv-cache-dtype fp8 --calculate-kv-scales \
  --port 800$i
```

Traps (all verified against upstream issues):

| Trap | Detail | Fix |
|---|---|---|
| `reasoning_effort='high'` -> HTTP 500 | template accepts only xhigh/medium/low; default xhigh | pin `low`/`medium` explicitly |
| Structured output broken in thinking mode | vLLM #35700: qwen3_5 fails most of the time with thinking on; works with thinking off | use `guided_json` only in non-thinking branches, else tolerant parsing |
| MTP + speculative decoding breaks constrained decoding | draft tokens unconstrained until reasoning-end accepted | leave `--speculative-config` OFF |
| `response_format` emits doubled `{{` with reasoning | vLLM #48228 | prefer `guided_json` + `enable_thinking=false`, or tolerant parsing |
| Prefix caching unsupported with mamba cache 'all' | hybrid DeltaNet arch | `--mamba-cache-mode align` |
| NVFP4 on L4 | NVFP4 is Blackwell-native; L4 is Ada | use AWQ-INT4 / bnb-4bit only |
| FP8 27B = 27GB | exceeds one 24GB L4 | 4-bit, or TP=2 (halves worker count) |
| KV budget | 9GB free / 64KB per token (16 attn layers x 4 KV heads x 256) = ~140K tokens | `--kv-cache-dtype fp8` doubles it; serialize 25K-token judge calls |

Verified vLLM support: `vllm/model_executor/models/qwen3_5.py` implements
`Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration, IsHybrid)` —
native VL + hybrid DeltaNet + LoRA all supported. Official Qwen docs publish the
`--reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder`
invocation, so **native tool calling is first-class**.

## 5. Porting cost (this is the good news)

Poetiq's stack is almost backend-agnostic:

- `arc_agi/llm.py` -> **LiteLLM `acompletion`**. LiteLLM supports OpenAI-compatible
  endpoints: add `"openai/qwen3.8-27b"` to the `limiters`/`props` dicts and set
  `api_base=http://localhost:800X/v1`, `api_key="EMPTY"`. **~10 lines.**
- `arc_agi/sandbox.py` -> pure `asyncio.create_subprocess_exec` with numpy/scipy,
  clean env, 1.5-5s timeout. **Zero changes, works offline.**
- `main.py` -> already writes the cumulative submission after every task
  (**exactly** what a Kaggle rerun needs against a hard timeout) and already scores
  against a solutions file. Add: Kaggle path detection, wall-clock watchdog.
- `arc_agi/io.py::build_kaggle_two_attempts` -> already emits the competition's
  2-attempt schema.
- `arc_agi/prompts.py` (15.4KB) -> the full SOLVER_PROMPT_1/2/3 + FEEDBACK_PROMPT,
  reusable verbatim.

From Land's repo, reuse: `src/grid.py` (tolerant CSV parsing),
`src/image_generation.py` (cartoon/precise renderers), `src/hint_generation.py`,
`src/judges.py` + `src/selection_advanced.py` (holistic judge + 2pts/1pt aggregation),
`src/tasks/codegen_prompts/v1-v4.py`, `docs/ANALYZE_*.md` (the negative results).

## 6. Verification semantics (do not over-trust it)

Land's held-out verification experiment (10 tasks x 10 iterations):

- Pass rate **88%**, verified rate **38%**, verified-but-wrong **1**.
- => **precision ~97%, recall ~43%.**

So exact train-pair verification is a strong *positive* signal and a weak *negative*
one. Consequences for our design:

- A program matching all train pairs exactly is ~97% likely correct -> put it at the
  top of the candidate pool and let the judge confirm.
- Failing verification must NOT discard a candidate -> keep soft_score ranking.
- ARC-AGI-2 was explicitly designed to be *"less brute-forcible"*, and Land notes a
  candidate can be logical yet *overfit the train pairs* or miss a mechanic introduced
  only in the test input. Verification filters syntax/consistency, not generalization.

## 7. Implementation phases

| Phase | Work | Output | Est. |
|---|---|---|---|
| 0 | Attach `Qwen3.8-27B-AWQ-INT4` (or upload bnb-4bit) as Kaggle Model; pre-bake vLLM+litellm wheelhouse dataset | offline-ready inputs | 0.5d |
| 1 | vLLM 4-instance launcher cell + smoke test (reasoning_effort, tool parser, prefix cache) | serving baseline | 0.5d |
| 2 | Fork Poetiq repo -> swap LiteLLM backend to local vLLM, Kaggle paths, watchdog | runnable harness | 1d |
| 3 | **Measure on the 120-task public eval split** (solutions available) -> baseline number | first real score | 0.5d |
| 4 | Add Land's judge + text/image channels + shape-diverse attempt selection | +diversity | 1-2d |
| 5 | Budget tuning: experts x iterations x effort, A/B on eval split | cost/score curve | 1d |
| 6 | Submit; keep the 33.89 TTT fork as the safety submission | LB entry | - |

Local dev is possible before Kaggle: the same harness runs against any
OpenAI-compatible endpoint, so we can iterate on a single GPU and only pay Kaggle
quota for full runs.

## 8. Expected score (honest)

| Scenario | Estimate |
|---|---|
| Qwen3.8-27B + Poetiq-style code harness + 1 judge, trimmed budget | **40-55%** |
| If base-model knowledge coverage is the limiter (ARC Prize warns harness gains need it) | 25-35% |
| Public TTT baseline for comparison | 33.89 |
| Kaggle private #1 | 83.06 |
| Land's API-frontier reference | 72.9% |

Favourable signal: Qwen3.8-27B's **coding** is frontier-class (SWE-bench Pro 61.7,
above Opus4.6 Max's 53.4; Terminal Bench 73.0) and the code channel is precisely a
coding task. Unfavourable: abstract reasoning is below frontier (HLE 30.8 vs ~40).
Net: the code-heavy Poetiq engine is the right choice for *this* model's strength
profile, which is another argument for your plan over a TTT scale-up.

## 9. Risk register

| Risk | Mitigation |
|---|---|
| 27B lacks ARC knowledge coverage -> harness gains don't transfer | measure on eval split in Phase 3 before investing; keep TTT fork as fallback |
| Token budget overrun | reasoning_effort=low, iteration caps, per-task token accounting, earliest-first ordering |
| vLLM thinking+structured-output bugs | tolerant CSV parsing as default; guided_json only in non-thinking branches; MTP off |
| 4-bit quality loss vs bf16 | AWQ-INT4 (QAT variants exist: QUASAR W4A4 mirror) over naive bnb; A/B on eval split |
| Kaggle offline install | pre-baked wheelhouse dataset; test in a fresh notebook with internet disabled |
| Sandbox executing model-generated code | subprocess isolation + timeout + clean env already in Poetiq's sandbox; no network in rerun anyway |
| Judge context overflow | cap traces fed to judge (top 12), truncate reasoning, keep code + exec logs |
| Single point of failure at 12h | cumulative submission write after every task (already in main.py) |
