# Solution Map: Qwen3.8-27B on ARC-AGI-2 (Kaggle L4x4)

Target: replace the Qwen3-4B TTT pipeline (public LB 33.89) with Qwen3.8-27B and an
Astra-inspired harness. Document verified against sources as of 2026-09-26.

## 0. Why 27B now (verified facts)

| Evidence | Number | Source |
|---|---|---|
| GPT-6 Astra on ARC-AGI-2 | **95.0%** (max reasoning) | arcprize.org/results/openai-gpt-6-astra |
| GPT-6 Astra on ARC-AGI-3, Standard harness | 62.7% ($26K) | arcprize.org/blog/astra |
| GPT-6 Astra on ARC-AGI-3, Provider Adapter harness | **99.9%** ($19K) | same |
| Kaggle LB top (private) | 83.06 | kaggle leaderboard |
| Harness gap (same weights) | 62.7% -> 99.9% | ARC Prize |

Two conclusions: (1) ARC-AGI-2 is approaching saturation for frontier models
(92-95%), so the open-weights gap to close is ~60 points — model scale is the
biggest lever the public pipeline has not pulled. (2) The Astra result proves
**harness > raw model** — preserving reasoning state + compaction took the same
weights from 62.7% to 99.9%. We steal both levers: scale the model AND upgrade
the harness.

## 1. Verified enablers

- **Unsloth fine-tunes Qwen3.8-27B on 24GB VRAM (QLoRA 4-bit)** — unsloth.ai/docs/models/qwen3.8/train,
  release v0.1.800-beta (Aug 14 2026). Uses `qwen3_5` arch + Flash Linear Attention kernels
  for the Gated DeltaNet layers. Requires latest transformers v5.
- **KV cache is cheap**: only 16 of 64 layers are gated attention (4 KV heads, head_dim 256,
  GQA). Per token: 16 x 4 x 256 x 2 x 2B = 32KB (vs a pure-attention 27B at ~500KB+).
  8192-token beam = ~0.26GB; 8 DFS beams = ~2GB. DeltaNet layers carry constant-size state.
  **DFS beam decode is memory-feasible on a single L4.**
- **Kaggle mirrors exist** (FP8, NVFP4, official bf16): e.g. `michaelpoluektov/qwen3-8-27b-fp8`,
  `dangkhoa2016/qwen-qwen3-8-27b`, `impactganyu/qwen38-27b-fp8`. Apache 2.0.
  NOTE: FP8 (27GB) does NOT fit a single 24GB L4, and NVFP4 is Blackwell-native (L4 is
  Ada: FP8 yes, FP4 no). **Correct artifact: upload `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`
  (~15-16GB, NF4) as our own Kaggle Model** — one-time upload, attachable in rerun.
- **Caveat**: Qwen3.8 open models are **thinking-only**. The ARC grid format needs direct
  answers. Handle via SFT on direct-answer examples (grids15-style, empty thinking) and/or
  chat template `enable_thinking=False`. Unsloth docs: keep >=75% reasoning data to preserve
  ability — for our use the *direct* grid-completion format IS the target behaviour, so we
  SFT it in deliberately.

## 2. Architecture map

```
                        ARC task (train pairs + test input)
                                      |
        +-----------------------------+-----------------------------+
        |                    Phase 2 core (27B QLoRA TTT)           |
        |  augment n=12 (D8 x color perm x shuffle)                 |
        |  QLoRA r=128 (attn + DeltaNet proj + MLP, 4-bit base)     |
        |  1-2 epochs, lr 5e-5, bf16 compute, grad ckpt on          |
        +-----------------------------+-----------------------------+
                                      |
        turbo_dfs (unchanged algo, ported tokens)                    |
        16 augmented views x 12 ARC tokens, NLL < -log(0.15)         |
        KV-cache beams (cheap: 16/64 attention layers)               |
                                      |
        +--------------------- Phase 3 harness ---------------------+
        |  a) persistent scratchpad across views (KV prefix reuse)  |
        |  b) 27B CoT hypothesis -> DSL EXACT verifier on train     |
        |     pairs (ground truth is free) -> verified hypothesis    |
        |     emits test output as candidate                        |
        +-----------------------------------------------------------+
                                      |
        candidate pool = DFS beams + DSL-verified + few-shot SC votes
                                      |
        27B teacher-forced NLL re-rank over 8 augmentations (cached)
                                      |
        diverse-2 attempt selection (shape-disagreement bonus)
                                      |
        submission.json (all 240 task_ids, 2 attempts, schema-safe)
```

### Phase 1 — hybrid 4B+27B (risk floor, ship in ~1-2 days)

- Workers 0-1: keep 4B TTT + DFS exactly as-is (proven candidate generator).
- Workers 2-3: 27B bnb-4bit, two jobs:
  1. **Rescorer**: teacher-forced NLL of 4B candidates over 8 augmentations
     (replaces `calc_scores` ranking; inference-only, ~1-2 min/task).
  2. **Few-shot solver**: 262K context lets us pack ALL train pairs + D8
     augmented views in ONE prompt — no TTT — with self-consistency voting
     (k=8 samples). Catches tasks 4B TTT fails on.
- Merge pools, 27B-NLL ranks, diverse-2 attempts.
- Expected: +2-5 over 33.89 with LOW risk (4B path untouched as fallback).

### Phase 2 — full 27B QLoRA TTT (the scale-up)

1. **Pre-competition SFT ("grids15-on-27B")**: replicate sorokin's recipe on
   `unsloth/Qwen3.8-27B-unsloth-bnb-4bit`:
   - Corpus: ARC-AGI-1 400+ tasks + ARC-AGI-2 1000 training tasks, augmented
     (D8 symmetries x color permutations x example shuffle), direct-answer
     format, empty think blocks.
   - Compute: Kaggle L4x4 notebook (12h) x 2-3 sessions, LoRA r=64, export
     merged 4-bit model -> upload as Kaggle Model (private until submission).
   - This checkpoint is also our moat: no public 27B ARC-SFT exists today.
2. **Per-task TTT**: QLoRA r=128 (not 256 — 24GB ceiling), n=12 augmentations,
   1 epoch, adaptive per-task budget (300s easy / 1200s hard), easiest-first
   queue ordering.
3. **DFS port**: algorithm unchanged; token IDs re-derived at runtime
   (see port checklist). Decode step ~2x slower than 4B (memory-bandwidth:
   ~15GB weight reads/token vs 8GB) — compensate with the 27B's sharper NLL
   (fewer junk beams to explore) + slightly relaxed max_score.
4. Wall-clock: TTT ~4-6 min + DFS ~2-3 min + rescore ~1 min = 8-10 min/task
   -> 240 tasks / 4 workers = 8-10h. Fits 12h with the 600s buffer.

### Phase 3 — Astra-inspired harness (the 62.7->99.9 lesson)

The single biggest verified insight of Sep 2026: the same weights gained 37
points from harness alone. Open-weights analogues:

1. **Persistent reasoning state**: currently each augmented view decodes
   independently. Instead, carry a running "notes" prefix (model's own
   summary of the transformation hypothesis so far) + reuse DFS prefix KV
   cache across views of the same task. Cost: KV bytes are cheap (16/64
   layers). This is a bounded change to `turbo_dfs` bookkeeping.
2. **Hypothesis -> exact verification loop**: let 27B emit transformation
   hypotheses in text ("recolor 2->7 and rotate 90"), then EXECUTE them as
   DSL ops against ALL train pairs where we hold ground truth. Only
   hypotheses that reproduce every train pair exactly are allowed to emit
   the test output. The verifier is `src/baseline_solver.py`'s op set,
   extended. This converts the 27B's reasoning strength (its actual edge
   over 4B) into guaranteed-consistent candidates — something raw DFS
   cannot do.
3. **Agentic budget routing**: 27B scores each task's post-TTT train NLL;
   easy tasks get 1 decode pass, hard tasks get scratchpad + hypothesis
   loop + wider beams with the time saved.

## 3. Port checklist (4B -> 27B code changes)

| Item | 4B notebook | 27B port | Why |
|---|---|---|---|
| Loader | `FastLanguageModel.from_pretrained` | `FastModel.from_pretrained(..., load_in_4bit=True, offload_embedding=True)` | new unsloth API for Qwen3.8 |
| Model path | `sorokin/qwen3_4b_grids15_sft139` | our Kaggle Model mirror of `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` (+ our SFT adapter) | rerun has no internet |
| LoRA targets | manual `["q_proj",...,"lm_head"]` | unsloth `finetune_attention_modules=True, finetune_mlp_modules=True` + verify DeltaNet projection module names on `qwen3_5` arch | hybrid arch has new module types |
| LoRA rank | r=256, RSLoRA alpha=32 | r=64 (SFT) / r=128 (TTT) | 24GB ceiling |
| Token IDs | hardcoded ARC_VOCAB / USER=11 / ASSISTANT=12 / PAD=13 / EOS=15 | **re-derive at runtime** via `tokenizer.convert_tokens_to_ids(["0".."9","Ċ","<|im_end|>", ...])` | 248K-vocab tokenizer; IDs differ |
| Thinking | n/a | `enable_thinking=False` in chat template + SFT on direct answers | Qwen3.8 is thinking-only by default |
| Collator | `QwenDataCollatorForCompletionOnlyLM` | same, with re-derived IDs + Qwen3.8 chat template markers | template markers differ |
| max_seq | 8192 | 8192 (TTT) / up to 64K (few-shot Phase 1) | memory vs 262K capability |
| Multi-worker | 4x `CUDA_VISIBLE_DEVICES=rank` | unchanged (1 model per L4) | 4-bit fits |
| Deps | pip install at runtime | **pre-baked wheelhouse dataset** (transformers v5 + unsloth >= 0.1.800) pinned | rerun offline; kernel compile first-run |
| First-run | instant | DeltaNet kernel compile takes minutes — warm up before task loop | documented unsloth behaviour |

## 4. Memory budget (single L4, 24GB, Phase 2)

| Component | Bytes |
|---|---|
| 4-bit weights (27B) | ~15.0 GB |
| LoRA r=128 adapters (attn+MLP, bf16) | ~1.2 GB |
| Optimizer states (adamw 8-bit) | ~2.4 GB |
| Activations @ 8192, batch 1, grad ckpt | ~1.5-2.5 GB |
| DeltaNet constant state | tens of MB |
| DFS KV (8 beams x 8192) | ~2.1 GB |
| CUDA/context reserve | ~1.5 GB |
| **Total (TTT mode)** | **~21-23 GB (tight, fits)** |
| **Total (inference mode, no optimizer)** | **~18-19 GB (comfortable)** |

## 5. Risk register

| Risk | Mitigation |
|---|---|
| Thinking-only model emits `<think>` junk | SFT direct answers; template `enable_thinking=False`; decode rejects non-grid tokens (formatter already returns None) |
| Tokenizer ID mismatches | runtime re-derivation + unit assert in starter.py |
| Unsloth Qwen3.8 support is weeks old | pin versions in wheelhouse; Phase 1 keeps 4B fallback path |
| 24GB OOM during TTT | drop r to 64, n to 8, seq to 4096; offload_embedding=True |
| 27B slower than budget | adaptive budgets + easiest-first; worst case route only hard tasks to 27B |
| No public 27B ARC-SFT | build it (Phase 2.1) — also our edge |
| NVFP4/FP4 mirrors tempting but L4 is Ada | use bnb-4bit only on L4 |
| Kaggle 12h rerun | 8-10h projected + 600s buffer + watchdog cell from LB33.89 notebook |

## 6. Execution order

1. Upload `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` as private Kaggle Model (I can do this next).
2. Phase 1 fork of the LB 33.89 notebook: 2x4B + 2x27B workers, run on the
   120-task eval split (public solutions) -> measure 27B rescorer delta.
3. Phase 2.1 SFT run on Kaggle (2-3 x 12h sessions) -> eval-split ablation.
4. Phase 3 harness pieces, ablated one at a time on the eval split.
5. Submit best config; keep 33.89 fork as the safe fallback submission.
