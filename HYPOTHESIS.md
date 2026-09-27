# H2 — Duck-REPL Transfer

**Branch:** `h2/duck-repl` · **Competition:** `arc-prize-2026-arc-agi-2` · **Model:** Qwen3.8-27B FP8 on vLLM, 4× L4, 12 h, internet off at rerun

---

## 1. The claim

> The mechanism that won ARC-AGI-3 Milestone 1 — The Duck harness (Tufa Labs) running Qwen3.6-27B FP8 on vLLM — transfers to ARC-AGI-2 and beats prompt-only approaches, because it converts an abstract-reasoning task into an **interactive coding task with an exact oracle**.

Operationally, the claim decomposes into three testable sub-claims:

| # | Sub-claim | Measured by |
|---|---|---|
| C1 | Pointing an act–observe REPL loop at the **free exact oracle** (the ground-truth train outputs) produces more correct test grids than spending the same token budget on one-shot generation | A0 vs A3 ablation, token-matched |
| C2 | The Duck's *perception* elements (segmentation, upscaled image) survive the transfer, but its *every-turn* image injection does not, because our environment is static | A3 vs A4/A5 |
| C3 | The loop fits 12 h for 240 tasks on 4× L4 | §5 arithmetic + `scripts/eval_local.py --project` |

C3 is proven below from measurements. C1 and C2 are the empirical content and §7 states exactly what would refute them.

**The deepest objection, stated up front.** ARC-AGI-3 has no ground truth at inference time, so the Duck's loop buys *information it cannot otherwise get*. ARC-AGI-2 gives train-pair ground truth for free — so the loop buys feedback that is, in a sense, already available to any one-shot solver that could just… look. What the loop does **not** buy is anything that closes the *generalization* gap between train pairs and the unseen test input, which is the thing ARC-AGI-2 was explicitly designed to make hard. The honest framing of this hypothesis is therefore narrower than it first appears: **the Duck mechanism attacks search and self-correction, not generalization.** It should help to the extent that Qwen3.8-27B can *write* a correct rule but not *emit* it reliably in one shot, and it should not help at all where the model cannot form the rule. §7 makes that measurable.

---

## 2. The Duck mechanism, and precisely how each element maps

The Duck's core idea in its authors' words: *"the model never presses a key. It writes Python, and the Python presses keys."* Game state lives in the REPL as pre-loaded Python variables; the model inspects them, calls helpers, executes code, observes results, iterates.

The transfer key: **`action(actions)` becomes `verify(fn)`.** In the Duck, `action()` is a host round-trip that suspends the snippet, mutates the world, and refreshes the preloaded variables from the reply. We keep that exact protocol shape and retarget it at the oracle: `verify()` suspends the snippet, runs the candidate against every train pair *and* every test input, hands the result to the host (which registers it in a ranked candidate pool), and returns exact pass/fail with diff diagnostics to the model.

| The Duck (ARC-AGI-3) | This harness (ARC-AGI-2) | Notes |
|---|---|---|
| board preloaded as `current_frame`, `history`, `valid_actions` | task preloaded as `TRAIN`, `TEST`, `TRAIN_PAIRS`, `TEST_INPUTS` | fresh process per call ⇒ state reset, as in the Duck |
| `action(actions)` host round-trip | **`verify(fn)` host round-trip** | the oracle *is* the action |
| new frame + `last_action_result` (`board_changed`, `reward`, `level_completed`) | `passed/pairs/pass_rate/exact/soft_score`, per-pair `failures`, `aug`, `runs_without_labels`, test-output preview | exact, not noisy |
| `current_frame.ascii` (letter-coded, raw numeric grid **hidden**) | `Grid.ascii` **and** `Grid.csv` **and** the raw `list[list[int]]` | deliberate reversal — see §3.1 |
| `current_frame.segmentation`: 4-connected components, position-invariant hashes, boundaries, adjacency, parent/child | `segment(grid)` / `Grid.segmentation` / `Grid.seg(n)` — same algorithm, adapted from the Duck's MIT source | plus `bbox` and capped `cells`, because we write array code |
| 4×-upscaled board PNG injected **every turn** | token-budgeted panel PNG injected **once** | static environment — see §4 |
| continuous eviction of the oldest messages, keeping system prompt + recent history | `context.trim_to_budget`, protected `[system, task]` prefix, whole-block drops, image stripped last | plus orphan-`tool`-message repair |
| 30 s tool timeout, 4096-char output cap | identical defaults | |
| `UNDO` deliberately withheld | no undo; the pool is append-only and ranked, so there is nothing to undo | |
| one tool (`python`), no bespoke DSL | one tool (`python`), two host primitives (`verify`, `finish`), three pure views (`segment`, `show`, `shapes`) | |
| long system prompt: *"we had to encode a lot of specific knowledge into the prompts"* | 8,834 chars / **2,945 tokens**, encoding ARC-AGI-2 transformation families and diff-reading discipline | measured, §5 |
| world model restated every turn | `Hypothesis:` / `Next:` lines harvested into a compact ledger, re-emitted **only on change** | simplification pressure |
| `tool_steps` up to 12 per step | `max_python_calls_per_turn = 4`, batching encouraged | each call is a full prefill |

### 2.1 What the oracle gives that the Duck never had

Because *every* `verify` call runs on the test inputs too and reports exact train behaviour, the harness accumulates a **ranked pool of programs at zero marginal token cost**. That makes the competition's 2-attempt rule free:

- `attempt_1` = the best-ranked candidate's test output (verified-first, then `soft_score`, then simpler, then earlier);
- `attempt_2` = the highest-ranked candidate whose test output *differs*.

And the second-attempt policy is conditional, because the two cases have different dominant risks:

- **something verified on all train pairs** → Land measured precision ≈ 97 %, so the leader is probably right; `attempt_2` follows likelihood (`soft_score`), not diversity;
- **nothing verified** → the dominant risk *is* having the whole transformation family wrong (usually visible as a shape error), so `attempt_2` prefers a **different output shape**, covering two families instead of one.

A second structural gain: **grid-format parsing errors cease to exist.** Land's grid-format search found CSV beat 8 alternatives and that a bad format cost ~10 %. That entire failure class is about the model *emitting* a grid. Here the model emits a *program*; the grid is produced by executing it and read back as JSON. No tolerant parser is needed anywhere on the answer path.

---

## 3. Deliberate deviations from the Duck

### 3.1 The raw numeric grid is exposed (reversal)

The Duck hides the numeric grid: *"The raw numeric grid is intentionally hidden from the Python tool."* That is right for ARC-AGI-3 — a fixed 64×64 board is 4,096 integers, printing it pollutes context (the failure that motivated their segmentation tool), and the only available acts are symbolic (`UP`/`MOUSE(row, col)`), so the model never needs `int` values.

ARC-AGI-2 is the opposite on both counts: grids are ≤ 900 cells (measured mean of the largest grid per task: 565 cells, max 900) and **the task is literally numeric array manipulation**. A model that cannot index `g[r][c]` cannot write a transform. So `Grid` is a `list[list[int]]` subclass: it *is* the raw grid, with cheap views bolted on. The Duck's anti-pollution intent is preserved by a different means — a hard output cap plus a prompt rule ("NEVER print or echo a full grid"), which is what actually caused the pollution.

### 3.2 Elements dropped, and why

| Dropped | Why |
|---|---|
| every-turn image | environment is static; measured 138 vision tokens/turn would cost 463 K tokens over the run for zero new information (§4) |
| `history`, `transitions`, `previous_frame`, `last_transition`, `last_action_frame`, `last_action_result` | all exist to reason about *change between frames*. Nothing changes. Removing them also removes the Duck's most intricate prompt section (the `history[-1].frame` vs `previous_frame` confusion it spends 6 lines disambiguating) |
| `valid_actions` / action normalisation / `MOUSE(row, col)` | no action space |
| `UNDO` | the Duck withheld it; our pool is append-only and ranked so there is nothing to undo |
| persistent 30-assistant-turn history + every-turn world-model restatement | replaced by a ~700-token ledger emitted only when something changed; the arXiv ablation found simplification pressure helped in 3 of 4 settings |
| video / multi-frame animation | the Duck found more frames did not help ("small models struggle with extracting details and acting on them") |
| any ARC-specific DSL or solver library | the Duck's documented negative: *"hand-crafting specific tools for the model did not help, as it seems to hinder the creative abilities of the model."* Our tool surface is 1 tool + 2 host primitives + 3 pure data views |
| `tool_steps = 12` | capped at 4 tool calls per turn with explicit batching instructions: every extra call is another full prefill of a longer context |

### 3.3 Elements added

| Added | Why |
|---|---|
| **augmentation probe** (`aug`: transpose / rot180 / flipud pass rates) | a cheap overfitting signal that costs CPU, not tokens (one line in the report). Reported as a *hint*, never a filter: many ARC tasks are genuinely not symmetry-equivariant (gravity, corner crops, reading order), and the prompt says so |
| **label-dependence probe** (`runs_without_labels`) | re-runs an exactly-verified candidate with the ground-truth outputs blanked. Distinguishes a rule from a lookup table. Also reported, never filtered — legitimate programs do read labels (e.g. palette remapping) |
| **compact diff diagnostics** | `kind`, shape `delta_rows/delta_cols`, `n_diff`, `diff_frac`, `bbox`, up to 8 `[row, col, expected, got]` samples, value histograms at the mismatching cells, and a 5×5 ASCII crop of *both* grids centred on the first mismatch with the cell upper-cased. This is the "explicit mismatch diagnostics" the arXiv ablation ranked first |
| **ranked candidate pool + conditional 2-attempt selection** | §2.1 |
| **adaptive scheduler** | per-task time and turn allowances recomputed from `remaining × workers / remaining_tasks` before each dispatch, so a slow start shortens later tasks instead of running out of clock |
| **escalating context-overflow recovery** | drop oldest block → strip image → halve the task block (the grids are all still in the REPL) → give up cleanly with a valid fallback |
| **`finish(reason)`** | a control-flow primitive, not a solver tool. Lets the model end early and bank the tokens |

---

## 4. Duck vs Land on images: the contradiction and its resolution

**The Duck:** a 4×-upscaled image of the board injected *every turn*, because *"Qwen's encoder processes images in 16×16 pixel patches"*, so upscaling gives the best visual understanding. Perception that survived their ablations.

**Land** (`reference/land-modality-search/docs/ANALYZE_MULTIMODAL_IMAGES.md`): *"I have seen multimodality help in some cases, but it's not consistent and in somewhat structured testing I can't really recreate it"* → *"We probably instead should extract the hints in a separate prompt and stay text-only."* Yet Land's final 72.9 % system still ran an image channel (10 of 29 candidates).

**A competitor's A/B on Qwen specifically:** *"vision information helps Qwen get to the intuition about the problem faster… However, compared to [frontier models], qwen cannot accurately infer coordinates based on image and is also worse in reasoning."* And: *"Tokens are the real currency, not actions."* Their animation feature raised cost from 384 → 449 tokens/action (+17 %) and mechanically cost them moves.

These are not actually in conflict once you separate **what the image is for** from **what it is asked to do**:

1. **Land asked the image to be the answer channel.** The model looked at rendered grids and was expected to emit a grid back. That requires cell-level *coordinate read-out*, which is exactly the capability the A/B says Qwen lacks — and which no VLM has reliably, since one vision token covers a 32×32-px block. Hence "inconsistent": it works when the task is gestalt-shaped and fails when it needs a coordinate. Land's conclusion is correct *for that use*.
2. **The Duck asks the image to be a prior, alongside an exact symbolic channel.** The model never has to read a cell out of the picture, because `segmentation` and `.ascii` give exact values. The image only accelerates hypothesis formation ("that looks like 4-fold symmetry"). Under that division of labour the A/B's own first clause applies — *"vision helps Qwen get to the intuition faster"*.

**Resolution adopted here — three rules, each with a number attached:**

- **Rule 1 — the image is a gestalt prior, never a data source.** It is injected at "gestalt resolution" under an explicit **vision-token budget of 160**, not at cell resolution. Measured over all 120 eval tasks: mean **137.7** vision tokens, max 160, at a mean **6.57 px per cell** (range 3–13). One-token-per-cell would cost ~565 tokens — **4.1× more** — and would still not be reliably readable. The prompt states outright: *"use it for shape, symmetry and layout intuition, never to read a cell value."*
- **Rule 2 — inject once, not every turn.** This is the real Duck→ARC-AGI-2 correction. The Duck's board *changes every action*, so a fresh image carries new information every turn; re-sending it is how the model perceives its own action. Our task bundle is **static**: turns 2..T would receive byte-identical pixels. Sending them anyway would cost 138 × 13 × 240 ≈ **430 K tokens** for exactly zero bits. The image is attached to the task block on turn 1 and is the *first* thing eviction strips under pressure.
- **Rule 3 — the exact channels stay text.** ASCII in the prompt, full numeric grids in the REPL, coordinates in the diff diagnostics. Land's finding that direct image prompting is unreliable is respected by never making the image load-bearing; the Duck's finding that vision accelerates intuition is respected by still sending it.

This also reconciles Land's *own* inconsistency: Land kept an image channel in a 72.9 % system while reporting it as unhelpful, because in that system the image was an **independent diversity channel** (10 of 29 candidates) rather than a read-out device. Rule 1 is the same insight, made explicit and cheap.

Net effect versus the Duck: **the same perceptual benefit at 1/14 of the image cost**, and versus Land: an image channel that cannot corrupt the answer path.

---

## 5. Quantitative budget

### 5.1 Measured context cost (all 120 eval tasks)

`python scripts/measure_context.py` → `harness/docs/measured_context_budget.json`:

| Component | Mean | Median | p90 | Max |
|---|---|---|---|---|
| system prompt (8,834 chars) | 2,945 | — | — | — |
| tool spec | 286 | — | — | — |
| task block (ASCII, ≤4,000 budget) | 1,093 | 1,019 | 1,695 | 2,997 |
| panel image (vision tokens) | 138 | 140 | — | 160 |
| **fixed prefix, total** | **4,462** | **4,389** | **5,078** | **6,386** |

Supporting measurements: 7.4 grids/task mean (max 14); largest grid per task 565 cells mean (max 900); **0/120 tasks needed a train pair elided** at the 4,000-token task-block budget; 49/120 tasks have >1 test input.

End-to-end mock run over all 120 tasks (`scripts/eval_local.py --mock`, 8 workers, 15.2 s wall): 600 LLM calls, 600 python calls, 360 verify calls, **schema valid**, 6,553 prompt tokens/turn with mock-sized content.

Rolling window per turn (design): assistant text ~40 + tool-call code ~250 + tool result ~500 + amortised status ~70 = **~860 tokens**. With `keep_recent_turns = 6`: **5,160**.

> **Prompt per turn ≈ 4,462 + 5,160 = 9,620 tokens.** (The mock's 6,553 is lower only because mock code and tool results are smaller than a real agent's.)

**Reasoning tokens are never replayed.** Only `response.text` and `tool_calls` go back into the message list; `reasoning_content` is dropped. Thinking is paid once as *completion* and never re-prefilled — at ~1.5 K reasoning tokens/turn this is the largest single cost leak available, and it is closed by construction (unit-tested).

### 5.2 Completion cost and why `xhigh` is infeasible

`reasoning_effort` accepts only `{xhigh, medium, low}` (`high` → HTTP 500, QwenLM/Qwen3.8#217) and **defaults to `xhigh`**, which emits 8–16 K thinking tokens per call. At 12,000 completion tokens/turn:

```
decode alone = 240 tasks × 14 turns × 12,000 tok / 400 tok/s = 100,800 s = 28.0 h
```

**28 h against an 11.5 h budget — over by 2.4× on decode before counting a single prefill token.** Effort pinning is therefore mandatory, not a tuning nicety. Schedule used:

| Turn | Effort | Rationale |
|---|---|---|
| 0 | `medium` | hypothesis formation is where reasoning pays |
| after a **shape** mismatch | `medium` | a shape error means the transformation *family* is wrong — a big jump, not a tweak |
| otherwise | `low` | diff-driven refinement is mechanical |
| forced-final | `medium` | last chance |

Expected completion ≈ (3 × 1,800 + 11 × 700) / 14 ≈ **950 tokens/turn**.

### 5.3 Serving model for 4× L4, TP=4, FP8 27 GB

**Prefill (compute-bound).** 2 FLOP/param/token × 27 × 10⁹ = 54 GFLOP/token. 4× L4 ≈ 242 TFLOPS dense BF16 (484 FP8). With PCIe all-reduce (2 per layer × 64 layers, no NVLink) at 25–40 % MFU → 120–190 TFLOPS → 2,200–3,500 tok/s. **Plan at P = 2,000 tok/s.**

**Decode (bandwidth-bound).** 4 × 300 GB/s = 1,200 GB/s aggregate. Per step: 27 GB of weights + KV. This model has only 16 of 64 layers at full attention (4 KV heads × head_dim 256), so

```
KV per token = 16 layers × 4 heads × 256 dim × 2 (K,V) × 2 bytes = 65,536 B = 64 KB
KV per sequence at 9,620-token context = 0.61 GB
```

| concurrent seqs N | bytes/step | step time | aggregate decode |
|---|---|---|---|
| 1 | 27.6 GB | 23.0 ms | 43 tok/s |
| 8 | 31.9 GB | 26.6 ms | 301 tok/s |
| **16** | **36.8 GB** | **30.7 ms** | **521 tok/s → ~390 at 75 % efficiency** |
| 32 | 46.5 GB | 38.8 ms | 825 tok/s |

**Plan at D = 400 tok/s with N = 16 workers.** KV memory at N=16 is 9.8 GB total, sharded to 2.5 GB/GPU on top of 6.75 GB/GPU of weights — 9.3 GB of 24 GB, so `--gpu-memory-utilization 0.90` with `--max-num-seqs 32` is comfortable.

Note the structural consequence of the hybrid architecture: **at 64 KB/token, a 9.6 K-token context costs 0.61 GB of KV, i.e. 2 % of the weight read.** Long prompts here cost *prefill FLOPs*, not KV memory. That is why the budget is dominated by prefill and why concurrency (not context length) is the memory constraint.

### 5.4 The arithmetic

```
usable wall clock = 43,200 s (12 h) − 600 s (Kaggle buffer) − 1,200 s (vLLM load + graph capture)
                  = 41,400 s = 11.50 h

GPU-seconds per turn = prompt/P + completion/D
                     = 9,620/2,000 + 950/400
                     = 4.81 s     + 2.38 s
                     = 7.19 s

total = 240 tasks × T turns × 7.19 s        (independent of N once the GPU is saturated)

T = 10  →  17,256 s =  4.79 h     headroom 2.40×
T = 14  →  24,158 s =  6.71 h     headroom 1.71×   ← design target
T = 20  →  34,512 s =  9.59 h     headroom 1.20×
T = 24  →  41,414 s = 11.50 h     headroom 1.00×

max affordable turns/task = 41,400 / (240 × 7.19) = 24.0
total tokens at T = 14    = 240 × 14 × (9,620 + 950) = 35.5 M
```

**Concurrency is mandatory, not an optimisation.** At N=1 the aggregate decode rate is 43 tok/s, so

```
decode alone = 240 × 14 × 950 / 43 = 74,200 s = 20.6 h  →  total 25.1 h, over by 2.2×
```

The floor is set by requiring decode ≤ 41,400 − 16,128 = 25,272 s, i.e. D ≥ 126 tok/s, which the table puts at **N ≥ 4 theoretical, ~6 with overhead**. Default `workers = 16` sits 2.7× above that floor.

### 5.5 Sensitivity — forecast hours for 240 tasks at T = 14

| P \ D | 200 tok/s | 400 tok/s | 700 tok/s |
|---|---|---|---|
| **1,000 tok/s** | 13.4 h ✗ | 11.2 h ✗ | 10.2 h ✓ |
| **2,000 tok/s** | 8.9 h ✓ | **6.7 h ✓** | 5.7 h ✓ |
| **3,500 tok/s** | 7.0 h ✓ | 4.8 h ✓ | 3.8 h ✓ |

Max affordable turns at the pessimistic corners: (P=1000, D=400) → 14.4; (P=1000, D=200) → **12.0**. So even in the worst modelled case the loop fits at T ≈ 12, and the adaptive scheduler is what finds that automatically: it recomputes `max_task_seconds = remaining × workers / remaining_tasks` before each dispatch and shrinks `max_turns` proportionally when the fraction drops below 0.75.

### 5.6 Cost comparison to the published ablation

arXiv 2607.15439 ablated exactly this on ARC-AGI-3 (4 variants × 2 models × 2 reasoning efforts) and found the **verification variant ranked first in every setting** — at 90.75 M cost-tokens vs 32.32 M for the baseline at xhigh, **2.8×**.

| | cost-tokens | includes exact verification? |
|---|---|---|
| arXiv baseline (xhigh) | 32.32 M | no |
| arXiv verification variant (xhigh) | 90.75 M | yes |
| **this design (T=14)** | **35.5 M** | **yes** |

We deliver full exact verification at **1.10× the baseline and 0.39× the verification variant**. Three reasons, all quantified above: the oracle is *free and exact*, so it **terminates** search instead of extending it (early stop on verification, 2 distinct verified outputs, or `finish()`); the context is 9.6 K/turn against the Duck's ~95 KB prompt logs (≈25–30 K tokens/turn); and effort is pinned off xhigh. This is the explicit answer to "design for that trade-off".

### 5.7 What we spend it on versus prompt-only

A token-matched prompt-only baseline (16 samples × (4,500 prompt + 800 completion), no oracle) costs 240 × (16×4,500/2,000 + 16×800/400) = 16,320 s = **4.5 h**. We spend **6.7 h**. The hypothesis is that the extra 1.5× buys an exact oracle, a ranked program pool, free second attempts, and the elimination of grid-format parsing errors — not that it is cheaper.

---

## 6. Predicted score

Built up from measured components rather than asserted:

| Step | Value | Source |
|---|---|---|
| verification **precision** (a program passing all train pairs is actually right) | ~97 % | Land's held-out experiment: 10 tasks × 10 iterations, pass 88 %, verified 38 %, verified-but-wrong 1 |
| verification **recall** with frontier models | 38–43 % | same experiment |
| recall with Qwen3.8-27B | **20–30 %** | discounted: the A/B reports Qwen is *"worse in reasoning"* than frontier models, and rule inference is the reasoning-bound half of program synthesis |
| hit rate on unverified tasks (best `soft_score` + shape-diverse `attempt_2`) | 8–12 % | 2 diverse attempts, no oracle confirmation |

```
score ≈ P(verified) × precision + P(not verified) × unverified_hit
      ≈ 0.25 × 0.97            + 0.75 × 0.10
      ≈ 0.243                  + 0.075
      ≈ 0.32
```

**Predicted eval-split score: 25–40 %, central estimate ~32 %.**

Reference points: best public notebook (TTT, specialised SFT'd ARC model) **33.89**; Poetiq **54.0 %** (frontier API, ineligible for Kaggle); Land **72.9 %** (frontier API, ineligible); Kaggle private #1 **83.06**.

**This is an uncomfortable prediction and I am not going to dress it up.** The central estimate lands *at* the 33.89 public baseline, not above it — and that baseline uses a purpose-built ARC model with a 16-token vocabulary trained on 3.2 M synthetic puzzles plus test-time training. Reaching parity with a *general* 27B model and a harness, with no weight updates at all, would be a meaningful result about mechanism, but it is not a win on the leaderboard. The sub-claim C1 — beating *prompt-only* Qwen3.8-27B, which should sit at 5–15 % — is very likely to hold. The stronger reading of the hypothesis ("beats the best public approach") is a coin flip.

---

## 7. Confirmation and refutation

All criteria are computable on the 120-task eval split by `scripts/eval_local.py` (which scores against `arc-agi_evaluation_solutions.json`) plus the per-task report.

### Ablation ladder (this is what isolates the Duck mechanism)

| ID | Variant | Isolates |
|---|---|---|
| A0 | prompt-only, k samples + majority vote, **token-matched to A3** | the baseline the claim is made against |
| A1 | A0 + code channel, one shot, no oracle | value of "write a program" alone |
| A2 | A1 + `verify` once, keep-if-passes, no loop | value of the oracle *without* iteration |
| **A3** | **full REPL loop (this harness)** | **value of the act–observe loop** |
| A4 | A3, `--no-image` | Rule 1/2 of §4 |
| A5 | A3 with `segment()` removed from the sandbox | the Duck's surviving perception tool |

**A2 → A3 is the Duck transfer claim.** If the loop is worth anything, A3 > A2 at matched tokens. A3 vs A4/A5 tests C2.

### CONFIRM the hypothesis if all of:

1. **eval score ≥ 25 %**, and
2. **verified rate ≥ 20 %** (fraction of tasks with ≥ 1 program passing all train pairs exactly), and
3. **verification precision ≥ 90 %** (of verified tasks, fraction scoring 1.0), and
4. **A3 − A0 ≥ +5 absolute points at matched tokens**, and **A3 − A2 ≥ +3 points**.

### REFUTE the hypothesis if any of:

- **verified rate < 12 %** → the oracle is not *reachable* by this model. The Duck's mechanism has nothing to bite on, because in ARC-AGI-3 the loop can always make progress by probing, and here it cannot. This kills C1 outright.
- **precision < 70 %** → train-pair overfitting dominates. ARC-AGI-2's "less brute-forcible" design defeats exact verification as a *positive* signal, and the ranked pool is actively misleading. This kills the 2-attempt policy too.
- **A3 ≤ A0 at matched tokens** → the loop is not worth its prefill. This is the cleanest single refutation, and it is the one the competitor's "tokens are the real currency" warning predicts if per-turn cost is underestimated.
- **the run does not complete 240 tasks in 11.5 h at T ≥ 8** → C3 fails and the whole design is infeasible on this hardware regardless of score.

### AMBIGUOUS — harness sound, model too weak

If **verified rate ≥ 20 %** and **precision ≥ 90 %** but **score < 25 %**, the bottleneck is candidate *generation*, not the harness. The correct response is to re-point the same harness at a stronger base model, not to redesign it — which is exactly the Duck authors' stated philosophy: *"a lightweight harness… built to be replaced under a stronger model rather than to compensate for a weaker one."* This branch must be reported as such and not counted as confirmation.

### Diagnostic instrumentation already in place

Per task the report records `turns`, `python_calls`, `verify_calls`, `n_verified`, `pool_size`, `best_soft_score`, `prompt_tokens`, `completion_tokens`, `seconds`, `stop_reason`, and (with `--transcript`) the full message list. `scripts/eval_local.py --project` converts the measured per-turn cost into the 240-task forecast with a sensitivity table, so C3 is re-checked from live numbers rather than from this document.

---

## 8. Risk register

| Risk | P | Impact | Mitigation | Detected by |
|---|---|---|---|---|
| verification recall too low — the model cannot write a program that passes all train pairs | **high** | fatal to C1 | `soft_score` ranking still yields an answer from partial credit; A1/A2/A3 ladder localises it | verified rate < 12 % |
| train-pair overfitting (2–4 pairs, ARC-AGI-2 is "less brute-forcible") | high | precision collapse | augmentation probe + label-dependence probe + explicit prompt rule ("a RULE, not a lookup; never key on absolute coordinates") + shape-diverse `attempt_2` when unverified | precision < 70 % |
| serving throughput worse than modelled (PCIe TP=4 all-reduce) | medium | run does not finish | adaptive scheduler shrinks `max_turns`/`max_task_seconds` from measured progress; `--project` re-forecasts from live numbers; fallback TP=2 × 2 instances (halves all-reduce participants, 2 independent queues) | forecast > 11.5 h |
| vLLM tool-calling unreliable in thinking mode (vLLM#35700 affects structured output; tool parsing is a different path but shares the reasoning parser) | medium | lost turns | never use `response_format`/`guided_json`; recover Qwen-coder tool-call markup from text (the Duck's own fallback, implemented + tested); exactly one tool | tool-call parse failure rate |
| `reasoning_effort` 500s | low | every turn lost | validate locally against `{low, medium, xhigh}`; on HTTP 500 drop the field for the rest of the run | `client.stats["effort_dropped"]` |
| prefix caching ineffective on the GatedDeltaNet hybrid (`--mamba-cache-mode align` required) | medium | +2 h | the budget **assumes no caching benefit**; caching is upside only | server cache-hit metrics |
| offline install of vLLM at rerun (no internet) | medium | kernel cannot run | wheelhouse dataset + `pip install --no-index --find-links`; CPU/mock fallback so the kernel still writes a valid submission | notebook smoke cell |
| sandbox executing model-generated code | low | run destroyed | fresh OS process per call, `-I`, empty env, `CUDA_VISIBLE_DEVICES=""`, temp cwd, RLIMIT_CPU/AS/FSIZE/NOFILE, blocked imports, host-side process-group kill, 30 s cap | tests: timeout kill, blocked imports, output cap, memory bomb |
| 30×30 × 6-pair task overflows context | low | task lost | task-block budget with explicit elision + full data still in the REPL; escalating overflow recovery | measured: max task block 2,997 tokens, 0/120 elided |
| single point of failure at 12 h | low | zero score | submission pre-filled with valid fallbacks for all 240 ids **before the first task**, rewritten atomically after every task, watchdog finalises 90 s before the deadline | `test_submission_file_is_valid_before_any_task_runs` |

---

## 9. Evidence that contradicts this hypothesis

Recorded because a hypothesis branch that only lists its own support is worthless.

1. **The generalization objection (§1).** The Duck's loop buys feedback. ARC-AGI-2 train-pair feedback is already available to a one-shot solver that reads the pairs carefully; what is *not* available is any signal about the test input. The mechanism optimises the cheap part of the problem.
2. **Land's verification recall is only 38–43 % with frontier models.** If GPT-5-class models only produce a train-passing program on ~4 tasks in 10, Qwen3.8-27B will do worse. The oracle's ceiling may be far lower here than in ARC-AGI-3, where the loop could always make progress by probing. **This is the strongest single piece of counter-evidence.**
3. **arXiv 2607.15439 found a persistent *executable world model* was NOT uniformly beneficial** — textual beat flexible-interface executable for gpt-5.5. Our design is an executable interface. Partial defence: ours is not a world model but a *program under test against an exact oracle*, which is a different cognitive task. The defence is plausible, not proven.
4. **The same ablation put verification's cost at 2.8×.** We argue (§5.6) that our 35.5 M escapes that multiplier, but the argument rests on throughput estimates that have not been measured on the target hardware.
5. **Qwen is specifically weaker at the reasoning half.** *"qwen cannot accurately infer coordinates based on image and is also worse in reasoning."* Program synthesis for ARC is ~70 % rule inference and ~30 % code. The model's strength profile (SWE-bench Pro 61.7, Terminal Bench 73.0) matches the 30 %, not the 70 %. HLE 30.8 vs ~40 for frontier is the number that should worry us.
6. **The Duck's own negative on hand-crafted tools.** *"hand-crafting specific tools for the model did not help, as it seems to hinder the creative abilities of the model."* We ship `verify`, `finish`, `segment`, `show`, `shapes`, `augment`, `grid_diff`, `cell_accuracy`, `normalize_grid` and a `Grid` class with 9 members. That is a *lot* more surface than "one tool". Defence: they are data views and control flow, not solvers, and `verify` is the hypothesis itself. But the tension is real and A5 exists to test it.
7. **Land's "prescriptive prompting templates and iterative refinement systematically reduce hypothesis diversity and degrade performance."** Our system prompt is 2,945 tokens of prescriptive method ("the loop that works", 6 numbered steps). It may collapse the model onto the families we listed and away from the ones we did not. Mitigation not yet implemented: an A/B against a minimal prompt.
8. **The 33.89 public baseline is a specialised ARC model.** Matching it with a general model would be a good mechanistic result but not a competitive one, and the competition is scored on the leaderboard.

---

## 10. Reproducing

```bash
# 350 tests, ~25 s, no GPU, no torch, no network
python -m pytest tests -q

# measure the real context cost over all 120 eval tasks
python scripts/measure_context.py

# plumbing + schema + scoring + 240-task forecast, no server needed
python scripts/eval_local.py --mock --project

# real run against a local vLLM server
python scripts/eval_local.py --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b --workers 16 --limit 20
```
