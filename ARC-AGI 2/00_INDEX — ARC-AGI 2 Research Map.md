# ARC-AGI 2 Research Map

Working research workspace — 2026-10-24

# Research thesis

ARC-AGI-2 should now be treated primarily as a constrained program-induction / coding-agent problem, not as a conventional visual-puzzle benchmark. Our highest-priority hypothesis is that a Duck-style generic coding harness or a closely related executable-program synthesis system can turn each ARC task into: examples → hypothesis → executable program → exact verification → repair/search → two final predictions.

# Current working prior

\~90% subjective prior that the strongest Kaggle ARC-AGI-2 systems belong to this broad family (coding/program synthesis \+ executable verification/search), while the exact Tufa Labs implementation remains undisclosed. This is a research prior, not a factual claim.

# Workspace index

01\_BACKGROUND — ARC-AGI 2 & Current State (2026-10-24)  
Purpose: benchmark definition, competition constraints, frontier-model saturation, current Kaggle leaderboard, public reproducibility gap, and evidence around Tufa Labs / Duck.  
https\://docs.google.com/document/d/1Ob\_n-ekijiiyb4TA7nNwOyP4O6OSDznmtfIWrZ8ZGcU/edit?usp=drivesdk

02\_RESEARCH\_OBJECTIVE — Program Synthesis Harness Hypothesis  
Purpose: lock the research direction, define hypotheses, target architecture, evaluation criteria, ablations, and immediate R\&D sequence.  
https\://docs.google.com/document/d/1SZv6U-MeSpbTLqncIGpal6C1L1SIo9lxgHU53FFFjIk/edit?usp=drivesdk

03\_RESEARCH\_LOG — Experiments & Findings  
Purpose: one chronological source of truth for every experiment, score, runtime, failure mode, code version, and decision.  
https\://docs.google.com/document/d/1scr5ifvLyUXI8M27\_zciihxipvxOy43IDcxLCn60n5Q/edit?usp=drivesdk

# Drive structure

10\_Code-Harness — active implementation, harness snapshots, architecture notes.  
20\_Public-Baselines — public Kaggle notebooks / reproducible baselines and clean-room reproductions.  
30\_References — ARC Prize docs, Tufa/Duck resources, papers, write-ups, relevant repos.  
40\_Experiments — experiment outputs, traces, ablations, score tables.  
50\_Submissions — Kaggle-ready notebook/package snapshots and submission records.

# Research scope lock

IN SCOPE:

# 1\. Generic coding-agent harness transfer from ARC-AGI-3 / Duck philosophy.

# 2\. LLM-generated executable Python programs.

# 3\. Exact train-pair verifier.

# 4\. Multi-candidate generation, diversity, search and iterative repair.

# 5\. Counterexample-guided refinement.

# 6\. Program ranking / simplicity priors / pass@2 final selection.

# 7\. Compute scheduling under Kaggle limits.

# 8\. Hybrid fallback only when it improves the code-first architecture.

OUT OF PRIMARY SCOPE:

# 1\. Spending most R\&D time tuning conventional TTA as the main solver.

# 2\. Hand-building a huge brittle ARC DSL before testing generic Python.

# 3\. Treating leaderboard score gains as evidence without reproducible mechanism.

# 4\. Overfitting the 120 public evaluation tasks.

# Immediate sequence

R0 — Ground truth: freeze benchmark rules, current scores, public baselines and Duck resources.  
R1 — Minimal ARC-2 coding harness: solve(grid)-\>grid programs \+ exact verifier \+ repair loop.  
R2 — Candidate search: multiple independent programs, counterexample repair, candidate archive.  
R3 — Selection: rank surviving programs and emit exactly two predictions.  
R4 — Compare generic Python vs structured object helpers / light DSL.  
R5 — Integrate only useful components from public \~30-point systems as fallbacks.  
R6 — Optimize runtime / GPU usage, package for Kaggle, submit, log every result.

# Decision rule

Do not add complexity because it sounds intelligent. Add a component only if it measurably improves public-eval accuracy, Kaggle score, stability, or compute efficiency.

# Status

Workspace initialized. Background, objective and research-log documents created. Next engineering action is to acquire/freeze public baselines and Duck harness/code resources, then build the minimal ARC-2 executable-program verifier loop.  
