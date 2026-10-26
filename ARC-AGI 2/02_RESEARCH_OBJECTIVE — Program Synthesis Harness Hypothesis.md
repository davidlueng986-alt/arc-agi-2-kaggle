# ARC-AGI 2 — Research Objective & Core Hypothesis

Initialized: 2026-10-24

# 1\. North-star objective

Build an open, Kaggle-compatible ARC-AGI-2 system whose core intelligence lives in code/harness-level test-time reasoning:

LLM / coding model  
→ propose executable transformation program  
→ run program  
→ exact verifier on all train pairs  
→ counterexample  
→ repair / branch / search  
→ maintain diverse surviving hypotheses  
→ execute on test input  
→ select two final outputs.

The project is explicitly NOT centered on conventional TTA tuning. TTA systems remain useful baselines and optional fallback modules, but the primary research object is the executable reasoning harness.

# 2\. Core hypothesis H0

H0: ARC-AGI-2 can be reframed as specification-by-example program induction. A strong coding model inside a lightweight executable harness should be able to convert a large fraction of tasks into small deterministic programs, and exact verification should make iterative search far more reliable than direct grid generation.

Working prior: \~90% subjective confidence that the top ARC-AGI-2 Kaggle systems use this broad family of ideas or something functionally equivalent. This prior is motivated by the leaderboard gap and Tufa Labs’ prior work, but it is NOT evidence that Tufa’s hidden ARC-2 solution is Duck-derived.

# 3\. Why this direction

## Reason A — Frontier capability already exists.

ARC-AGI-2 scores of roughly 89–95% are already observed from top frontier reasoning models. This means the underlying abstraction tasks are often within current model capability. Our problem is to capture that capability efficiently, reliably and offline inside Kaggle constraints.

## Reason B — Code converts reasoning into exact execution.

ARC grading is exact-match. Direct model generation can fail even after correct conceptual inference because of coordinate, counting, recoloring or copy errors. Executable code separates rule inference from deterministic execution and gives us automatic unit tests on every train pair.

## Reason C — ARC-AGI-3 already validated the coding-harness paradigm.

Tufa Labs’ Duck won ARC-AGI-3 Milestone \#1 by treating interactive environments as coding problems and letting an LLM write/run Python in a live REPL. ARC-AGI-2 is structurally easier for this paradigm: no unknown action dynamics are required and every demonstration provides a direct exact-output verifier.

## Reason D — Tufa Labs is the highest-value clue.

The current ARC-AGI-2 public leaderboard leader, Tufa Labs, is the same team behind Duck and the earlier Stochastic Goose. Their ARC-2 method is undisclosed, so no direct transfer can be claimed. But the team identity makes Duck-style transfer the first architecture to investigate.

## Reason E — Public reproducibility gap is too large for small tuning alone.

Publicly reproducible systems are still clustered around \~30, while the leading Kaggle score is \~88. This motivates looking for an architectural regime change rather than another small TTA tweak.

# 3A. Model policy — Qwen3.8 family first

The research will be built and evaluated primarily around two models only:  
• Qwen3.8-27B — bring-up / debugging / reproducible baseline. Use it to validate the harness, verifier, repair loop, search logic and ablations quickly. Do not over-optimize the architecture around its capability ceiling.  
• Qwen3.8-Flash-Next — primary target and intended competition model. The harness, quantization strategy, context budget, multi-GPU serving, candidate count and runtime scheduler should be designed around making Flash-Next run reliably on the Kaggle L4×4 environment while preserving as much reasoning quality as possible.

# 

## Model-selection rule:

Do not expand the project into broad model shopping. Stay on Qwen3.8-27B \+ Qwen3.8-Flash-Next unless a newly released model is demonstrably in the same capability class or better AND is at least as viable under the Kaggle runtime/memory constraints. Qwen4-27B / Qwen4-Flash are explicit drop-in candidates if released and competitive, but they must beat the current model on a fixed ARC-AGI-2 harness benchmark before replacing it.

# 

## Competition objective:

Optimize for verified ARC solutions per GPU-hour, not generic benchmark prestige alone. However, because latent-rule induction quality is likely the dominant bottleneck, preserve the strongest model that can physically run before sacrificing model capability merely to increase sampling throughput.

# 

# 4\. Two primary research tracks

## Track A — Duck-style generic coding harness \[Priority 1\]

Keep the environment minimal and model-driven:  
• Represent ARC task data as Python objects / NumPy arrays.  
• Give the model a REPL / code execution loop.  
• Let the model write temporary helpers and inspect intermediate states.  
• Verify candidate transformation code against all training pairs.  
• Feed exact failures / diffs back as counterexamples.  
• Allow repeated modification until a program fits.  
• Keep the harness generic; avoid a large handcrafted DSL initially.

## Track B — Explicit program-synthesis search \[Priority 1B\]

Make search state explicit:  
• Generate N independent candidate programs / hypotheses.  
• Execute all candidates against train examples.  
• Reject non-fitting programs.  
• Repair promising partial programs using counterexamples.  
• Branch when ambiguity remains.  
• Deduplicate semantically equivalent programs.  
• Rank exact-fit survivors by simplicity, robustness and hypothesis diversity.  
• Produce two test predictions from genuinely different surviving hypotheses.

The two tracks can converge: Track A supplies the flexible code-writing agent; Track B supplies orchestration, search, memory and selection.

# 5\. Minimal target architecture

## Module 1 — Task loader

Load ARC JSON and expose train/test grids in a clean immutable structure.

## Module 2 — Representations

Always provide raw integer grid.  
Optionally compute object segmentation, connected components, bounding boxes, color statistics and transformations. These are helpers, not hardcoded solution rules.

## Module 3 — Code-generation agent

Input: task examples \+ current candidate state \+ failure evidence.  
Output: executable Python transformation, preferably a pure solve(grid) \-\> grid function plus optional helper functions.

## Module 4 — Sandbox / executor

Run candidate safely with time/memory limits.  
Capture stdout, exceptions, intermediate artifacts and output grids.

## Module 5 — Exact verifier

For every train pair:  
candidate(input) \== target exactly.  
Return structured counterexamples:  
• which examples fail,  
• wrong dimensions,  
• cell-level diff,  
• color/object mismatch summaries.

## Module 6 — Repair loop

Ask model to modify the program based on concrete failures instead of restarting blindly.  
Maintain parent/child lineage for candidate programs.

## Module 7 — Candidate archive

Store program source, train-fit score, exact-fit flag, complexity, lineage, runtime, output signature and failure summary.

## Module 8 — Search / scheduler

Allocate compute adaptively:  
• kill hopeless branches early;  
• spend more iterations on nearly-correct programs;  
• generate fresh independent hypotheses when repair stagnates;  
• preserve diversity instead of converging all agents onto the same wrong rule.

## Module 9 — Final selector

ARC allows exactly two predictions per test input.  
Select two outputs with maximum posterior usefulness:  
• exact-fit candidates first;  
• prefer different hypotheses/output signatures when ambiguity exists;  
• use simplicity / robustness priors as tie-breakers.

## Module 10 — Fallback layer

Only after code-first solver is measured:  
• public TTA / Qwen ARC baseline,  
• deterministic geometric primitives,  
• nearest solved-task retrieval if competition rules/data allow,  
• ensemble arbitration.  
Fallbacks must not dominate architecture complexity.

# 6\. Critical research questions

## Q1. Can a generic Python coding loop beat public \~30-point baselines without a handcrafted ARC DSL?

## Q2. How much gain comes from executable verification alone?

## Q3. Is iterative repair better than independent sampling at equal compute?

## Q4. What candidate count gives the best score/runtime trade-off?

## Q5. Does maintaining diverse exact-fit programs materially improve pass@2?

## Q6. Which verifier feedback is most useful: raw diff, object diff, natural-language diagnosis, or all three?

## Q7. Does a light object API help, or does it constrain the model like the hand-built tools that hurt Duck?

## Q8. When should the scheduler abandon repair and restart with a new hypothesis?

## Q9. Can one small/medium local coding model generate hypotheses while a second model/ranker selects them?

## Q10. How much of the public-to-LB gap is architecture versus model strength?

# 7\. First ablation matrix

## A0 — Direct model output, no executable code.

## A1 — One generated program, one execution, no repair.

## A2 — One program \+ iterative counterexample repair.

## A3 — Multiple independent programs \+ exact verification.

## A4 — Multiple programs \+ repair \+ candidate archive.

## A5 — A4 \+ diversity-aware pass@2 selection.

## A6 — A5 \+ lightweight object/geometry helpers.

## A7 — A5 \+ public TTA solver as fallback.

The first goal is not a leaderboard submission. It is to identify which architectural step produces the first discontinuous gain.

# 8\. Evaluation protocol

Primary offline metric:  
Accuracy on the official 120-task public eval set, pass@2.

Secondary metrics:  
• pass@1.  
• fraction of tasks where at least one exact-fit training program is found.  
• train-fit → test-fit generalization rate.  
• candidates generated per task.  
• repair iterations.  
• wall-clock runtime.  
• GPU/CPU usage.  
• failure categories.  
• score variance across seeds/reruns.

Competition metric:  
Kaggle public LB, recorded with exact notebook/code hash and runtime.  
Do not tune repeatedly against LB without logging; treat LB as a sparse external validation signal.

# 9\. Failure taxonomy to record

F1 — wrong rule / semantic abstraction failure.  
F2 — correct idea, buggy code.  
F3 — train-overfit program.  
F4 — underdetermined task; wrong surviving hypothesis selected.  
F5 — candidate collapse / insufficient diversity.  
F6 — verifier feedback fails to guide repair.  
F7 — context pollution / long-session degradation.  
F8 — timeout / execution inefficiency.  
F9 — representation/tool bias prevents discovery.  
F10 — fallback solver conflict.

# 10\. Immediate R\&D plan

## Step 0 — Freeze references and code.

Acquire:  
• official ARC-AGI-2 data/guide,  
• current public \~30-point notebook baselines,  
• Tufa Duck / Stochastic Goose code and write-ups,  
• Milestone \#1 Duck description,  
• any clean ARC program-synthesis frameworks worth borrowing infrastructure from.  
Store immutable source snapshots under 20\_Public-Baselines and 30\_References.

## Step 1 — Build minimal verifier harness.

No fancy tools:  
task loader \+ Python executor \+ exact verifier \+ structured diff \+ candidate storage.

## Step 2 — Run a coding model on a small representative public subset.

Goal: prove the full propose → execute → verify → repair loop works end-to-end.

## Step 3 — Add multi-candidate search.

Parallel independent hypotheses and repair branches; measure diversity and marginal gains.

## Step 4 — Compare generic Python against helper-rich version.

Only add segmentation/object primitives if the ablation demonstrates improvement.

## Step 5 — Scale to all 120 public eval tasks.

Produce per-task traces and failure taxonomy, not just an aggregate score.

## Step 6 — Package a Kaggle-safe offline notebook.

Respect 12-hour runtime, no internet and available L4×4 compute. Then submit sparingly and log the result.

# 11\. Success criteria

Minimum scientific success:  
Demonstrate a reproducible architecture gain over a strong public baseline and identify why.

Competitive success:  
Cross the public \~30-point regime decisively using the code-first harness.

Stretch target:  
Approach the hidden-leaderboard regime (70–85+) under Kaggle constraints.

Ultimate target:  
A reproducible open system at or above the 85% private threshold.

# 12\. Non-negotiable research discipline

• Every leaderboard claim must be timestamped.  
• Separate confirmed fact from inference.  
• Every experiment gets a code hash and fixed configuration.  
• Never hide failed runs.  
• Do not optimize solely for public LB.  
• Prefer simple generic mechanisms before ARC-specific handcrafted rule libraries.  
• Architecture changes require ablation evidence.  
