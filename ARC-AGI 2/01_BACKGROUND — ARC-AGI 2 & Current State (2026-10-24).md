# ARC-AGI 2 — Background & Current State

Last verified: 2026-10-24

# 1\. What ARC-AGI-2 is

ARC-AGI-2 is the second static benchmark in the Abstraction and Reasoning Corpus family. It preserves the ARC input/output-grid format: a task provides only a small number of demonstration pairs, and the system must infer the latent transformation rule and apply it to unseen test inputs.

The benchmark was introduced in 2025 to push beyond ARC-AGI-1. ARC Prize describes the main difficulty classes as symbolic interpretation, compositional reasoning, and contextual rule application. The intended target is fluid reasoning: acquire a new task-specific skill from very little data rather than retrieve a memorized solution.

Dataset structure:  
• 1,000 public training tasks.  
• 120 public evaluation tasks.  
• 120 semi-private evaluation tasks.  
• 120 private evaluation tasks.  
All calibrated evaluation tasks were solved by at least two humans within at most two attempts.

# 2\. 2026 Kaggle competition setting

The 2026 ARC-AGI-2 competition asks for at least 85% accuracy on the private evaluation set under Kaggle efficiency constraints.

Important mechanics:  
• Code Competition; submissions run as Kaggle notebooks.  
• No internet during evaluation.  
• CPU and GPU notebook runtime limit: 12 hours.  
• Competition provides L4×4 machines with 96 GB total GPU memory.  
• For each test input, the solver outputs exactly two candidate outputs; either exact match earns the task point.  
• Prize-eligible systems must open-source their methods.  
• ARC-AGI-2 does not have the ARC-AGI-3-style mid-competition Milestone \#1 / \#2 open-source checkpoints. The 2026 schedule lists those milestones only for ARC-AGI-3.

This last point matters: the best ARC-AGI-2 Kaggle methods can remain opaque until the final competition/write-up window.

# 3\. The benchmark is near-saturated for frontier models — but not for constrained open systems

At the model-capability level, ARC-AGI-2 is no longer an unsolved frontier in the same sense it was in 2025\.

ARC Prize verified results currently include:  
• GPT-6 Astra: 95.0% best ARC-AGI-2 score.  
• GPT-6.1 Sol: 94.2%.  
• Claude Opus 5.5: 93.3%.  
• Gemini 4 Argon: 91.3%.  
• GPT-6 Sol: 89.6%.  
• Gemini 3.8 Flash: 89.2%.  
Even lower-cost / smaller frontier systems increasingly land in the 60–80% range, although it would be inaccurate to say that every new model automatically exceeds 70%.

Interpretation:  
The top end of ARC-AGI-2 is becoming saturated as a raw frontier-model capability test. The interesting unsolved problem has shifted toward efficiency, open weights, restricted compute, robust inference architecture, and turning model reasoning into reliable exact outputs.

# 4\. Kaggle leaderboard versus publicly reproducible systems

The live 2026 Kaggle system track shows a dramatically different picture.

Current observed public leaderboard:  
• Tufa Labs — 88.06.  
• rabbithole — 80.56.  
• Yi-Chia Chen — 77.22.  
• nvbanana — 77.08.  
The leaderboard is provisional until the private evaluation / final competition results.

In contrast, public reproducible notebook anchors audited in the community remain around 30 points:  
• arc2-vanilla-exact — 31.39.  
• NVARC/TRM evidence variant — 31.11.  
• PerfPatch evidence lab — \~29.03.  
Other independent experiment records report reproducible public results around 30–32.

Therefore the important empirical gap is not “AI can only solve \~30% of ARC-AGI-2.” Frontier models can already solve \~90% at high reasoning effort. The gap is:  
high-capability / expensive model inference ≈ 80–95%  
versus  
publicly reproducible Kaggle-constrained open systems ≈ 30%  
versus  
top opaque Kaggle systems ≈ 80–88%.

That gap strongly suggests that architecture, test-time compute allocation, search, executable verification, or another unpublished systems technique is doing substantial work.

# 5\. Why Tufa Labs is the strongest clue

Tufa Labs currently leads the ARC-AGI-2 public Kaggle leaderboard at 88.06. Their exact ARC-AGI-2 method has not been disclosed; reporting says the team plans to share more after the deadline. Therefore we must not state that their ARC-2 solver is Duck.

However, Tufa Labs is also the team behind the first-place ARC-AGI-3 Milestone \#1 system “The Duck.” ARC Prize describes The Duck as a small open-source LLM that treats each ARC-AGI-3 environment as an interactive coding problem: it writes and runs Python in a live REPL, inspects results, invokes helpers, takes actions, observes consequences, and iterates. It uses multiple representations and a lightweight generic harness, and Tufa reported that hand-crafted tools could hurt compared with letting the model improvise.

This creates a highly informative prior:  
If a generic coding harness worked on the harder interactive ARC-AGI-3 setting, transferring the same philosophy to static ARC-AGI-2 is structurally easier because ARC-AGI-2 provides direct input/output examples and an exact verifier.

# 6\. Why ARC-AGI-2 is naturally a program-induction problem

For each task we are given examples:  
(x1, y1), (x2, y2), …, (xn, yn).

Instead of asking an LLM to directly paint the final output grid, ask it to synthesize a program f such that:  
f(xi) \= yi for every training pair.

Then:

# 1\. Generate a candidate transformation hypothesis.

# 2\. Express it as executable Python.

# 3\. Run it against every training example.

# 4\. Use exact-match failures as counterexamples.

# 5\. Repair or replace the program.

# 6\. Keep multiple independent programs that fit all demonstrations.

# 7\. Execute surviving programs on the test grid.

# 8\. Use the two allowed outputs to preserve hypothesis diversity.

This removes a major failure mode of direct LLM prediction: the model may understand the rule but make coordinate, counting, copying, or grid-construction errors. Execution delegates deterministic mechanics to code.

# 7\. Working interpretation of the current landscape

Capability question: largely answered at the frontier. Multiple models exceed 89–95%.  
Open efficient-system question: still unsolved and scientifically interesting.  
Competition question: can an open Kaggle-constrained system approach the reasoning capability of expensive frontier inference?  
Our research bet: yes, primarily by converting ARC-AGI-2 into an executable program-induction and verifier-guided search problem.

# 8\. Epistemic status

CONFIRMED:  
• ARC-AGI-2 structure and 2026 competition rules.  
• Frontier model scores listed above.  
• Tufa Labs public Kaggle score around 88.06.  
• Tufa Labs authored the ARC-AGI-3 Milestone \#1 winning Duck coding harness.  
• Publicly reproducible Kaggle notebooks around \~30 exist.

HIGH-CONFIDENCE INFERENCE:  
• A coding/program-synthesis harness is a strong candidate explanation for the hidden leaderboard jump and is a rational research direction.

NOT CONFIRMED:  
• Tufa Labs uses Duck directly on ARC-AGI-2.  
• Tufa Labs uses Python REPL program synthesis specifically.  
• The full 88.06 score can be explained by a single mechanism.

# 9\. Sources

ARC Prize — ARC-AGI-2 overview:  
https\://arcprize.org/arc-agi/2

ARC Prize — ARC-AGI-2 2026 competition:  
https\://arcprize.org/competitions/2026/arc-agi-2

Kaggle — ARC Prize 2026 ARC-AGI-2:  
https\://www\.kaggle.com/competitions/arc-prize-2026-arc-agi-2

ARC Prize — current verified model results:  
https\://arcprize.org/results

ARC Prize — GPT-6 Astra result:  
https\://arcprize.org/results/openai-gpt-6-astra

ARC Prize — GPT-6.1 Sol result:  
https\://arcprize.org/results/openai-gpt-6-1-sol

ARC Prize — Gemini 4 Argon result:  
https\://arcprize.org/results/google-gemini-4-argon

ARC Prize — ARC-AGI-3 Milestone \#1 / The Duck:  
https\://arcprize.org/blog/arc-prize-2026-milestone-1

ARC Prize — ARC-AGI-3 Preview competition / Tufa Labs history:  
https\://arcprize.org/competitions/arc-agi-3-preview-agents

Current ARC-AGI-2 Kaggle leaderboard tracker:  
https\://arc.markbarney.net/

Community audit of public reproducible ARC-AGI-2 notebook anchors:  
https\://github.com/OpenKaggle/arc-agi-2-paper-track-research

Note: third-party leaderboard trackers and community audits are used only for current competition-state observations; benchmark definitions and model results are grounded in ARC Prize / Kaggle primary sources.  
