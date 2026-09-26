# H3 — Program Synthesis over an Exact Oracle + Verified-Set Disambiguation

**Branch:** `h3/code-verify-search` · **Competition:** ARC Prize 2026 (`arc-prize-2026-arc-agi-2`) · **Model:** Qwen3.8-27B (FP8, vLLM TP=4, L4x4, offline rerun)

---

## 1. The claim

**For Qwen3.8-27B specifically, the optimal ARC-AGI-2 architecture is to stop asking the model for answer grids and ask it for programs, then exploit the train pairs as a free, exact, deterministic oracle — and to spend the branch's real intellectual effort not on generation but on DISAMBIGUATION: given several train-verified programs that disagree on the test input, decide which two grids to submit.**

### 1.1 Why program synthesis matches THIS model's strength profile

Qwen3.8-27B's published benchmark vector is sharply asymmetric:

| Capability | Qwen3.8-27B | Opus4.6 Max (reference frontier) |
|---|---|---|
| SWE-bench Pro (coding) | **61.7** | 53.4 |
| Terminal Bench 2.1 (agentic coding) | **73.0** | — |
| QwenSWEBench | **79.0** | — |
| NL2Repo-Bench | 42.3 | — |
| HLE (multidisciplinary abstract reasoning) | **30.8** | 40.0 |

Its *coding* is above the frontier reference; its *abstract reasoning* is ~9 points below it. Asking the model to emit an answer grid is a pure abstract-reasoning call — the weak axis, with no verification. Asking it to emit `def transform(grid: np.ndarray) -> np.ndarray` is a coding call — the strong axis — and the emitted artifact is **machine-checkable against ground truth we already have** (the train pairs). The harness converts an HLE-shaped problem into a SWE-bench-shaped problem plus a search problem the CPU can own.

Precedent: Poetiq's code-refinement engine verified **54.0%** on ARC-AGI-2 semi-private at half the cost of the previous best; the Kaggle-private #1 lineage (Tufa Labs/MindsAI, 83.06) ships a program-synthesis artifact; the ARC Prize 2025 report names the *refinement loop* the defining theme of the year. None of these systems, however, treat the **disagreement set** as the primary object — Poetiq returns at the first train-verified program (`solve_coding.py` early-returns when `all(r["success"] for r in train_res)`). That early return is precisely the gap this branch attacks.

### 1.2 Where the difficulty actually lives

Land's held-out-verification measurement (`reference/land-modality-search/docs/ANALYZE_VERIFICATION.md`): pass rate 88%, verified rate 38%, verified-but-wrong 1 → **verification precision ≈ 97%, recall ≈ 43%**. Two design consequences, adopted verbatim:

1. Verification is a **strong positive** signal: a program that matches all train pairs exactly is ~97% likely to be broadly correct → verified programs always outrank unverified ones.
2. Verification is a **weak negative** signal: most correct-in-spirit programs fail some pair for incidental reasons → **nothing is ever discarded for failing verification**; unverified programs remain in the pool, ranked by soft score (mean cell accuracy), and can occupy attempt_2 when no verified output exists.

But 97% precision is measured *per verified program on tasks that get verified*. The competition-relevant failure is different: **among the tasks where ≥2 distinct behaviors verify, which behavior is the truth?** That conditional error rate is what the disambiguation layer exists to reduce, and it is the number the ARC-AGI-2 design explicitly inflates ("logical yet overfit to training pairs", "miss a newly introduced latent concept in the test case" — Land).

---

## 2. Architecture

```
task (train pairs + test inputs)
  │
  ├─ SYNTHESIS (LLM, the only token-spending stage)
  │    2 experts × ≤5 iterations, temp 1.0, reasoning_effort=low
  │    prompt variants rotated (full-guidance / minimal — Land's diversity finding)
  │    pair order shuffled per iteration (seed+it, Poetiq's free augmentation)
  │    feedback = per-pair diff grids (`pred/truth` cells) + soft scores
  │    after first verification: prompts ask for a DIFFERENT consistent hypothesis
  │    early stop: ≥4 distinct verified behaviors, or patience exhausted, or budget
  │
  ├─ ORACLE (exact, free)
  │    every program runs on every train pair in a subprocess sandbox
  │    exact-match verification; soft score; diff diagnostics
  │
  ├─ POOL
  │    dedupe by normalized source; keep behavioral duplicates (independent evidence)
  │    group verified programs by test-output behavior
  │
  ├─ DISAMBIGUATION (zero LLM tokens — numpy + sandbox re-execution only)
  │    §3: equivariance suite · shape law · memorization screen · MDL ·
  │        behavioral clustering · determinism · soft fallback
  │    gated escalation (only while disagreement persists & budget allows):
  │      S7 adjudication: 1 LLM call — "these verified programs disagree; which
  │         rule generalizes? write the final program" → verified → +vote
  │      S2 LOO: ≤3 held-out-fold resyntheses; a fold program that passes its
  │         WITHHELD pair demonstrates generalization; its test behavior
  │         corroborates the matching cluster
  │
  └─ PASS@2 SELECTION
       attempt_1 = top cluster's grid; attempt_2 = best DIFFERENT grid,
       +0.15 bonus for a different SHAPE; cumulative atomic submission write
```

---

## 3. The disambiguation layer (the contribution)

Setup: verified set V = programs passing **all** train pairs exactly. Group V by test-output behavior → clusters C₁…C_k. If k=1 there is nothing to decide (consensus). If k≥2 the clusters disagree and every member of each cluster is, by construction, equally consistent with all labeled data. **The ranker must therefore score them by evidence the synthesis never saw.** Each signal below is exactly such evidence.

### S1 — Symmetry equivariance (D4 + color permutations)

**Mechanism.** For symmetry candidate s (a D4 element, or a color permutation σ), the augmented train pairs `(s·xᵢ → s·yᵢ)` are *pseudo-labeled held-out data*: no LLM produced them, and the synthesis prompts never saw them. A program P scores on:
- `equiv_train(P)`: fraction of kept s where P(s·xᵢ) == s·yᵢ for **all** pairs — a falsification battery with induced ground truth;
- `equiv_test(P)`: fraction of kept s where P(s·x_test) == s·P(x_test) — pure self-consistency (no labels needed; catches rules keyed to absolute position/orientation, which is the single most common overfitting mode we can name a priori).

**Two-stage gating (this is what makes the signal sound rather than a heuristic).**

1. *Data-level plausibility.* s survives only if `{(xᵢ→yᵢ)} ∪ {(s·xᵢ→s·yᵢ)}` is a consistent relation (no input mapped to two outputs). A symmetry the data itself contradicts is discarded before any program is scored.
2. *Pool-majority resolvability.* Among surviving s, keep only those respected by a **support-weighted majority (≥50%) of distinct verified behaviors**. Rationale: pseudo-labels are an *assumption* (the rule commutes with s), not an entailment. If most independent hypotheses that explain the data commute with s, s-commutation is a property of the posterior mass and likely of the truth (a Condorcet-flavored argument); if the majority *breaks* s, the true rule almost surely breaks s (e.g. gravity-down tasks break rot90) and scoring on s would penalize exactly the correct programs.

**Known pathology, handled.** A trivially-equivariant wrong program (the identity map commutes with *everything*) can out-score a truly-correct non-commuting rule (fliplr) *when the train pairs are individually symmetric* — then the data cannot distinguish them at all (Bayes-irreducible; fixture F5 documents this, and the MDL prior's choice of the simpler rule is the defensible bet). The pool-majority gate contains the damage: identity-like programs only win the gate when they are a real majority of the verified posterior. Residual risk is hedged structurally by pass@2 (F5 test asserts attempt_2 carries the other behavior) and by the gated adjudication call, which is the one component that can bring semantic judgment to bear.

**Color permutations** are generated only between colors with *identical usage profiles* across train pairs (sorted per-pair (count_in, count_out) signatures equal) and pass the same two gates. This targets the "color-identity overfit" mode: `recolor 2→3` vs the role-based truth, when the data treats 2 and 3 interchangeably (fixture F2: the overfit passes all train pairs; the (2 3)-permuted pseudo-pairs kill it).

**Why it generalizes rather than overfits.** Every point of `equiv_train` is a passed falsification test on data *not* shown during synthesis. A program can only score high by having learned a rule with the symmetry structure the task actually exhibits — memorized position/color accidents fail under transform. The signal never *adds* evidence for a program that merely agrees with the train pairs; it only separates programs that agree on train but differ in structure. Cost: ~24 extra sandbox executions per program per task, zero tokens.

### S1c — Shape law

Fit a small interpretable family of input→output shape laws on train pairs (identity, constant, rational per-axis scale, exact small-affine). A program is penalized **only when its test output shape contradicts every law consistent with the train data** (unanimity requirement → no false penalties when the data underdetermines the law). This catches size-constant memorizers (`if shape==(4,4): return TABLE[...]`) that pass train and return the wrong shape on an unseen test size (fixture F3). Shape errors are unforced errors: exact-match scoring gives zero credit for a right pattern in a wrong frame, and the shape relation is the most reliably extrapolable property of the task.

### S2 — Leave-one-out corroboration (LLM, gated)

When disagreement persists after offline signals: for up to 3 folds, resynthesize a program with pair *i* withheld (Land's protocol). A fold program that then passes pair *i* **exactly** has demonstrated out-of-sample generalization — the only *direct* measurement of it available. Its test-input behavior is compared to the clusters: `loo(cluster) = fraction of held-out-verified fold programs whose test behavior matches the cluster`, weighted 0.6. Because the fold program never saw pair i, matching it is independent evidence, not train-fit. Gated: only if ≥3 train pairs (a fold on a 2-pair task leaves one pair — no statistical meaning) and budget remains.

**2-pair tasks (15.8% of the distribution)** get no LOO; they are the most overfit-prone cohort, so the config multiplies the equivariance weight ×1.25 when n_pairs ≤ 2 — substituting free symmetry evidence for missing held-out evidence.

### S3 — MDL prior + S3b memorization screen

`mdl(P)` = min-max-normalized inverse AST size (docstrings/comments stripped — formatting-immune). Justification: CompressARC reached 20% on ARC-AGI-1 by *MDL-based selection alone* with 76K parameters and no pretraining; among hypotheses equally consistent with the data, the shorter description has more prior mass (Solomonoff-flavored; practically: fewer degrees of freedom to have fit noise). Weight deliberately small (0.25) — MDL is a tiebreaker, not a discriminator; a short wrong program beats a long right one only when nothing else separates them.

`memorization(P)` (weight −1.5, the largest single penalty): AST scan for numeric literal sequences (len ≥ 4) equal to a flattened train grid, CSV serializations of train grids inside string constants, or ≥2 distinct full rows (len ≥ 6) of one train grid. A program containing the literal answer to a training input is not a hypothesis about the task; it is a lookup table that happens to compile. This is sound to check statically and is exactly the "logical yet overfit" candidate Land warns about. (The synthesis prompt also tells the model such programs are rejected — screening changes the sampling distribution, not just the ranking.)

### S4 — Behavioral clustering on perturbed inputs

Every battery program runs on: x_test (weight 1.0), its 7 non-identity D4 transforms, data-plausible color-permuted variants, 2 single-cell random recolors, and a background-padded variant (weights 0.25 each). Pairwise weighted agreement → greedy clustering at θ=0.75; `support_norm(P) = |cluster(P)| / |battery|` (weight 1.0 — the largest single term after the verified bonus). This is majority-vote self-consistency, but *deeper* than voting on the test output alone: two programs that agree on x_test yet diverge under perturbation are less mutually corroborating than two that agree everywhere, and the agreement matrix records that. Independent derivation matters — the pool keeps behaviorally-duplicate programs from different experts/iterations precisely so support counts independent discoveries, and dedupes source-identical ones so a chatty expert cannot manufacture consensus.

### S5 — Soft fallback (verification is a weak negative)

If V = ∅: no discards. Rank by soft score with the soft weight tripled (`fallback_soft_boost=3.0`) so the train likelihood dominates the priors — an unverified program that explains 95% of train cells beats a symmetric, short program that explains 56%. Equivariance is **disabled** in this mode: without verified behaviors the pool-majority gate is meaningless and trivially-equivariant junk (identity) would collect the symmetry bonus. attempt_2 still takes the next distinct output → pass@2 keeps a live round on ~soft-ranked guesses.

### S6 — Determinism

Every battery program runs the test input `det_runs+1` times; any mismatch ⇒ penalty 1.0. A contract-violating program (hidden RNG, clock reads) is not a hypothesis about the task either; its output is not even a fixed function of the input, so no other signal computed on it can be trusted. Cheap (3 extra runs) and catches a real failure mode of temperature-1.0 code generation.

### S7 — Adjudication (LLM, gated, ≤1 call)

When ≥2 clusters survive the offline layer with close scores, one medium-effort call shows the model the disagreeing programs *and their conflicting test outputs* and asks which rule is the intended generalization, then to write the final program. The answer re-enters the oracle: only if it verifies does its behavior receive a +0.6 vote. This is the refinement-loop theme pointed at disambiguation itself, and it spends the model's *coding-strength* on a well-posed analytical sub-problem ("find which of these two generalizations is spurious") rather than on raw abstraction. It is also the designated escape hatch for the S1 irreducible-ambiguity case.

### Combination

```
score(P) = 2.0·verified(P) + w_soft·soft(P) + 1.0·support_norm(P)
         + 0.8·[0.6·equiv_train + 0.4·equiv_test]·boost₂ₚₐᵢᵣ
         + 0.6·loo(P) + 0.6·adj(P) + 0.25·mdl_norm(P) + 0.2·robust(P)
         − 1.5·shape_violation(P) − 1.5·memorization(P) − 1.0·nondet(P)

w_soft = 0.5 (verified present) | 1.5 (fallback mode)     boost₂ₚₐᵢᵣ = 1.25 if n_pairs ≤ 2
```

Ordering rationale: the verified bonus (2.0) implements Land's 97% precision as a near-gate; support (1.0) is the dominant *within-verified* term because mutual agreement of independent derivations is the most broadly reliable self-supervised signal we have; equivariance (0.8) is the primary generalization discriminator but capped below support because its pseudo-labels are assumptions, not entailments; LOO/adjudication (0.6 each) are strong but expensive and sparse; penalties (1.5) are set so a single memorization/shape hit outweighs any combination of the positive priors — being a lookup table is disqualifying, not merely disfavored; MDL (0.25) only ever breaks ties. Weights are fixed a priori from this reasoning (not fit to eval-split scores — fitting 10 weights on 120 tasks would itself be overfitting); `scripts/eval_local.py` reports the ablation table so the user can see what each earns.

### pass@2-aware selection (quantified)

Under exact-match scoring, duplicating attempt_1 is **strictly dominated**: P(win) = p₁ + (1−p₁)·p₂·(1−ρ), where ρ is the failure-mode correlation between the two attempts. With p₁=0.5, p₂=0.4: ρ=0.9 (near-duplicate hypotheses) → P=0.52; ρ=0.3 (structurally different hypotheses) → P=0.64 — a **+12-point** swing on such tasks, and `d(test)/dρ < 0` always. Hence: attempt_2 is always a *different grid*, with a +0.15 bonus for a different *shape* (shape-law errors are the most correlated failure mode — if the winning cluster has the frame wrong, its same-shape runner-up is wrong the same way, while a different-shape cluster is wrong differently). Across the full distribution, if ~35% of tasks reach a verified disagreement with ≥2 clusters and diversity moves ρ from ~0.9 to ~0.5 on them, the expected gain over naive top-2 is ≈ 0.35 × (1−p₁)·p₂·0.4 ≈ **+2–4 points** — for zero tokens.

---

## 4. The "less brute-forcible" objection, head-on

The ARC-AGI-2 report (arXiv 2505.11831) states tasks were designed to be *"less brute-forcible… to minimize susceptibility to naive or computationally intensive brute-force program search"*, emphasizing multi-rule composition, multi-step composition, contextual rule application, and in-context symbol definition. Three responses:

1. **We do not run program search.** There is no DSL, no enumeration, no Icecuber-style lattice (and this branch bans it explicitly). The hypothesis proposal distribution is the LLM's own prior — the abductive leap from 3 examples to "stack shapes on the side indicated by the outline color" is made by the model in one shot, in a universal hypothesis space (arbitrary Python over numpy/cv2). Brute force searches a *syntax* space; we sample a *semantic* prior and use computation only to check and to compare. Every composition depth, contextual gate, or symbol-binding the report names is expressible in the hypothesis space — the limitation is proposal quality, not representation.
2. **The oracle's job is filtering, not finding.** Verification never fabricates a hypothesis; it kills spurious ones at ~zero marginal cost. What "less brute-forcible" defeats is *search without a proposer*; a frontier-class coder (SWE-bench Pro 61.7) proposing 8–12 programs per task with diff-driven repair is on the other side of that design decision.
3. **The overfitting warning is aimed exactly where we aim the layer.** "Logical yet overfit to training pairs" and "miss a latent concept introduced in the test case" describe programs that pass the oracle and fail the test — i.e., the disagreement-set problem. Train verification is *necessary, not sufficient*; §3 is the sufficiency apparatus: pseudo-labeled symmetry tests the synthesis never saw (S1), shape-law extrapolation (S1c), literal-memorization execution (S3b), held-out-fold generalization measured directly (S2), perturbation stability (S4), and — for the residual that is genuinely undecidable offline — one semantic judgment call (S7) plus a structurally diverse second attempt (pass@2). No published ARC harness combines the exact oracle with a *principled disagreement-resolution layer*; that combination is this branch's bet.

---

## 5. Token budget arithmetic

Envelope: 12h − 600s Kaggle buffer − ~900s model load/compile ≈ **41,400s** usable. FP8 27B on 4×L4 (TP=4, aggregate ~1.2TB/s, weights ~27GB): single-stream decode ≈ 40–45 tok/s; with 8–16 concurrent requests, decode is bandwidth-shared — assume a **conservative 250 tok/s aggregate sustained** (60% of the ~440 tok/s weight-streaming ceiling; prefill of cached prefixes is nearly free and the solver prompt is a shared ~1.3K-token prefix across every call, which `--enable-prefix-caching` reuses).

Total decode capacity: 41,400s × 250 tok/s ≈ **10.3M completion tokens** ≈ **43K/task** over 240 tasks. Per-call plan (reasoning_effort pinned **low**; `high` is an HTTP-500 trap and default `xhigh` would emit 8–16K thinking tokens and blow the envelope ~4×):

| Stage | Calls/task | Prompt tok (raw) | Completion tok | Notes |
|---|---|---|---|---|
| Synthesis (2 experts × ≤5 iters) | ≤10 | ~4.5K each (prompt 1.3K + task CSV 0.5–2K + ≤5 feedback blocks ~1.5K); ~85% prefix-cached | ~1.0–1.3K each (thinking-low + program) | early-stops at 4 distinct verified behaviors or patience 3 |
| Adjudication (gated: disagreement) | ≤1 on ~30–40% of tasks | ~6K | ~1.5K (effort=medium) | only when clusters remain close |
| LOO folds (gated: disagreement, ≥3 pairs) | ≤3 on ~20–30% of tasks | ~4K each | ~1.0K each | withheld-pair protocol |
| **Expected** | **~11.5** | **~57K raw (~10K novel)** | **~13.5K** | |

Wall-clock check: 240 tasks × 13.5K completion = 3.2M completion tokens ÷ 250 tok/s ≈ **12,900s ≈ 3.6h** of decode; prefill of ~10K novel tokens/task ≈ 2.4M tokens at ~2–4K tok/s ≈ **0.3h**; tasks run 4–8-way concurrently against one server, so queueing overlaps. Sandbox/disambiguation CPU: ~200–600 grid-evaluations/task in one warm subprocess ≈ 1–4s/task ≈ **0.2h** on the CPU side (no GPU, no tokens — the disambiguation layer being free is a deliberate budget asymmetry we exploit: it costs ~0.5% of the wall clock and 0% of the token budget, so we can afford to run the *full* battery on every disagreement rather than triage it). Total ≈ **4.5–6h** of the 11.5h — a ~2× headroom that absorbs slow tasks, retries, and the 30×30-grid tail. At the observed mock-run rate (~10 calls/task, 4.3M raw prompt tokens per 120 tasks) the real run lands inside the 75K/task cap enforced in code (`Budget.can_call` refuses calls past the cap; the deadline watchdog stops synthesis and finalizes the submission 420s before the wall).

Comparison: Land's 29-candidate + 3-judge system ≈ 175K tokens/task — ~2.3× over this plan; Poetiq-8-experts is similarly over. The trimmed 2-expert pool is affordable *because* selection quality is bought with numpy instead of judges.

---

## 6. Offline evidence (measured on this machine, no GPU)

**Fixture suite** (`tests/fixtures/synthetic.py`, 7 tasks, 22+ assertions): every fixture is constructed so ≥2 programs pass ALL train pairs and only one behavior is truly correct. Results (all green, `pytest tests` = **212 passed**):

| Fixture | Overfit mode defeated | Decisive signal(s) (measured) |
|---|---|---|
| F1 position overfit (3×3 window) | hardcoded absolute position | equiv_train 1.00 vs 0.25 |
| F2 color-identity overfit | hardcoded color identity | perm{2↔3} pseudo-labels kept by both gates; equiv_train 1.00 vs 0.22 |
| F3 shape memorizer | lookup table keyed by shape | memorization 1.0 + unanimous shape law (−3.0 score); equiv_train 1.00 vs 0.40 |
| F4 consensus (3 correct impls vs 1 deviant) | lone deviant | cluster support 0.75 vs 0.25 |
| F5 irreducible ambiguity (symmetric train) | *undecidable offline* | documented limitation; pass@2 hedge asserted |
| F6 no-verified fallback + hidden RNG | nondeterministic higher-soft program | soft×3 dominance + determinism penalty (det flag fired) |
| F8 three-way, train-invisible divergence | conditional recolors inert on train | equiv_train blind (all 1.00) → equiv_**test** 1.00 vs 0.25 + MDL (3 clusters) |

The fixture run also exercises the gated LLM-in-the-loop mechanics end-to-end: on F1/F8 the mock adjudicator and the LOO folds return the correct rule, and the injected votes (`adj=1.0`, `loo=1.0`) land on the correct behavior group exactly as designed.

**Fixture-split ablation** (full pipeline through MockLLM, `scripts/eval_local.py --split fixtures`; a1 = attempt_1 exact-match accuracy, the discriminating metric — pass@2 saturates when only 2 clusters exist because both fill the two slots):

```
config                    pass@2       a1_acc
full                      1.0000     0.8571   (only miss: F5, Bayes-irreducible)
no_determinism            1.0000     0.7143   (F6 nondet program elected)
all_offline_priors_off    1.0000     0.7143
first_verified_poetiq     0.7143     0.0000   (returns at first verified → picks the overfit)
random_baseline           1.0000     0.4286
```

Single-signal ablations mostly do not move fixture scores — by construction each fixture is catchable by ≥2 independent signals (defense in depth), so removals are masked; the per-signal evidence is the unit assertions above (e.g. F8's deviants are separated *only* by test-side consistency and MDL once train pseudo-labels are blind). The policy-level contrasts are the loud ones: **first-verified (the Poetiq early-return policy) scores 0/7 on attempt_1 and loses 2 points of pass@2 on these fixtures**; full disambiguation loses only the case that is provably undecidable from the data.

**Scale/plumbing run** (`--mock --limit 120`, template-library MockLLM, real subprocess sandbox, 4 workers): 120/120 tasks, submission schema valid, ~4s/task wall, ~10 calls/task, 0 verified programs (16 generic templates cannot fit real ARC-AGI-2 tasks — the run validates plumbing, budget gating, fallback selection and the writer/watchdog, not intelligence; scores on this configuration are 0 as expected).

---

## 7. Predicted score

Honest decomposition for the real rerun (240 unseen ARC-AGI-2 tasks):

- **Verified-set coverage** (≥1 train-verified program): Poetiq with Gemini-3-Pro verified ~54%; Qwen3.8-27B's abstract reasoning is below that frontier (HLE 30.8 vs 40) while its coding is above it, and hypothesis *finding* is the reasoning-bound step. Assume **25–40%** of tasks reach a verified program (temperature-1.0 pooling + 10–12 calls + diff-repair is a stronger search than a single chain, which pushes up; ARC-AGI-2's compositional/contextual designs push down).
- **Disambiguation precision** (given verified, truth in attempt_1∪attempt_2): consensus tasks (single behavior) are ~free; on disagreements the fixture work + Land's 97% per-program precision suggest **85–95%**.
- **Unverified fallback**: soft-scored guesses convert ~2–5% of the remaining tasks.

**Predicted eval-split-equivalent score: 0.30×0.90 + 0.70×0.03 ≈ 0.25–0.35**, central estimate **~0.28–0.32** (public TTT notebook baseline: 33.89; verified API-frontier harnesses: 54–73). A 27B local model matching the 33.89 public baseline would already validate the architecture-per-model-strength argument; the ceiling is set by proposal quality, not by the disambiguation layer.

**Confirm/refute.** Run `scripts/eval_local.py --base-url … --limit 120` against the served model on the eval split:
- *Confirms* if: verified-set coverage ≥ 25%; disambiguation precision on the verified set ≥ 85%; total ≥ 0.22.
- *Refutes the disambiguation thesis* (keep synthesis, replace selection) if: verified coverage ≥ 25% but precision < 70%, or ablations show `no_equiv`/`no_support` ≥ `full` on ≥120 real tasks.
- *Refutes the model-fit thesis* (wrong model for this architecture) if: verified coverage < 15% at full budget — the coding prior is not compensating for the reasoning gap, and no selection layer can rescue an empty verified set.

---

## 8. Biggest risk

**Proposal starvation on ARC-AGI-2's compositional tasks.** If Qwen3.8-27B's abstraction gap (HLE 30.8) means it rarely forms the *right* hypothesis in ≤10 tries, the verified set is empty on most tasks and the entire disambiguation layer — however principled — operates on nothing. Mitigations in place: diff-diagnostic repair (grounding the next proposal in exact per-cell error maps), prompt-variant rotation to preserve hypothesis diversity (Land's negative result on prescriptive prompts), pool diversity with early-stop only at 4 behaviors, per-task token headroom (~2× under envelope) to raise iterations if eval-split coverage comes in low, and the soft-fallback path that still extracts ranked guesses from partial programs. The observable that decides this is the verified-coverage number in §7 — measurable locally on the 120-task eval split the moment a vLLM server exists, before spending any Kaggle quota.

## 9. What would contradict this document

- Verified coverage on the eval split ≥ 50% (the model is better than its HLE suggests; budget should shift to more candidates/task and the score projection rises toward Poetiq's 54%).
- Disagreement rate among verified tasks < 10% (consensus is near-universal; the layer's value collapses to the pass@2 shape hedge — still positive, but the thesis shrinks).
- Equivariance ablation *negative* on real tasks at scale (the pool-majority gate fails to filter non-equivariant true rules often enough to hurt) → drop S1's train half, keep test-side consistency and shape law.
- Any verified-but-wrong cluster that memorized a train grid *without* literal constants (semantic memorization, e.g. reconstructed via arithmetic) — static screening cannot see it; only S1/S4/S2 can, and a measurable rate >5% would mean the screen is security theater.

---

### Provenance

Prompts adapted from the Poetiq ARC solver (MIT License, © 2025 Poetiq, Inc., `reference/poetiq-solver`), with CSV grid rendering and the anti-overfit contract added; sandbox isolation pattern from `poetiq-solver/arc_agi/sandbox.py`, re-engineered into a persistent batch worker with parent-side per-job deadlines; verification semantics (precision/recall asymmetry, never-discard rule) and the codegen-prompt diversity finding from Land's modality-search docs (`reference/land-modality-search/docs/ANALYZE_VERIFICATION.md`, `ANALYZE_CODEGEN_PROMPT.md`); cumulative-submission + watchdog pattern from the lb3389 public notebook source. Benchmark figures for Qwen3.8-27B from the model card as recorded in `reference/harness-first-qwen38-27b.md`.
