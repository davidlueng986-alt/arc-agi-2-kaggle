# ARC-AGI 2 — Research Log

Single source of truth for experiments and decisions.

# Logging rule

Every meaningful run must be recorded here before the next architecture decision. Do not rely on memory or scattered chat history.

# Entry template

Date / time:  
Experiment ID:  
Code / notebook hash:  
Dataset / task subset:  
Model / checkpoint:  
Compute:  
Time budget:  
Random seed(s):

Hypothesis:  
What exact claim is this run testing?

Change from previous run:  
One concise description of the architectural or parameter change.

Configuration:  
Harness version:  
Candidate count:  
Max repair iterations:  
Verifier feedback:  
Representations/helpers:  
Selection rule:  
Fallbacks:

Results:  
Public-eval pass@1:  
Public-eval pass@2:  
Kaggle public LB, if submitted:  
Runtime:  
Peak memory / GPU:  
Exact-fit-train rate:  
Train-fit → test-fit generalization:  
Variance across reruns:

Failure breakdown:  
F1 wrong rule:  
F2 buggy code:  
F3 train overfit:  
F4 ambiguity / wrong selection:  
F5 candidate collapse:  
F6 ineffective repair:  
F7 context degradation:  
F8 timeout:  
F9 representation/tool bias:  
F10 fallback conflict:

Qualitative observations:  
What did the traces reveal?

Decision:  
KEEP / REJECT / MODIFY / NEEDS MORE EVIDENCE

Next experiment:  
The smallest next test that can change the decision.

\----------------------------------------------------------------

# Research baseline snapshot — 2026-10-24

Frontier model capability: top ARC-AGI-2 verified model results are already \~89–95%.  
Kaggle public leader: Tufa Labs \~88.06 (provisional).  
Public reproducible notebook regime: roughly \~29–32.  
Primary hypothesis: code/harness-level executable program induction \+ verifier/search can bridge a large part of this gap.  
Primary architecture tracks:  
A. Duck-style generic Python coding harness.  
B. Explicit multi-candidate program synthesis \+ verification \+ iterative repair.  
