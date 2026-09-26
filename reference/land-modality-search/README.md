

# Overview

## Algorithm

Paper: {tbd}

Approach: what I'd call Multi-Model Reflective Reasoning
- Using GPT-5.2, Gemini-3, Opus 4.5
- Long-horizon/multi-step reasoning (~6hrs/problem)
- Agentic codegen (>100,000 python calls)
- Visual reasoning
- Council of judges

Four key solvers:
* Multimodal: Generate an image of problem and use as part of the prompt to solve the problem
* Hint: Extract key hints about how to solve the problem separately, and then supply hints to key models to guide them in the right direction
* Three step search: In first step, label all objects in the input/output (as opposed to a regular grid representation). In a second step, label all potential transformations involved in the input-output pairing. In third steps, use the extracted objects and the transformations to attempt at finding a solution
* Deep-search: Use specialized prompt to ensure triggering of maximum depth of search

All solvers output a full reasoning trace as part of their solution. Then all solvers are "scored" by two judges:
* Logic Judge: Does the reasoning produce the claimed result
* Consistency Judge: Is the reasoning coherent and consistent
Based on the judges assessment all candidates are scored, and a decision is made on whether further "search" is needed or whether a likely solution has been reached.

I've done a bunch of analysis to come to this algorithm that may be informative to others trying to solve ARC AGI. Please see below for some of that analysis, or just hook me up on the ARC AGI discord channel.

## Current performance (Jan 5, 2026)

76.11% on ARC AGI 2 eval: https://x.com/LandJohan/status/2008197725263716589

Latest solver on the V7 branch: https://github.com/beetree/ARC-AGI/tree/Johan_Land_Solver_V7


## Historical performance (Dec 15 2025)

The performance is: 70.7% on [arc agi 2 eval dataset](https://github.com/arcprize/ARC-AGI-2/tree/main/data/evaluation)

Full results here: https://www.kaggle.com/code/johanland/johan-land-solver-v6-public/output?scriptVersionId=286318109&select=submissions.tgz


## Historical performance

I did a run on Dec 1 on the Eval 2 dataset which yielded a 50.3% solved problems (each sub task measured independently). 

Full results here: https://github.com/beetree/ARC-AGI/blob/main/docs/RESULTS.md


# How to get started

```bash
git clone https://github.com/beetree/ARC-AGI
cd ARC-AGI/
# Ensure you have at least python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp config/api_keys.env.example config/api_keys.env
# Add your keys
set -a && source config/api_keys.env && set +a
git clone https://github.com/arcprize/ARC-AGI-2.git
mv ARC-AGI-2/data/evaluation .
rm -rf ARC-AGI-2
python run.py --task-directory evaluation --task-limit 1
```
You should see something like this, and it should complete in a few minutes only

```
Solver testing mode activated.
Judge model: gpt-5.1-low

Limiting execution to first 1 tasks.
Found 1 task files. Total test cases: 1
Starting batch execution with 60 parallel task workers...

Legend: ⚡ Running   ⏳ Queued   ✅ Done

| Status        | Task:Test  | Step  | Phase           | Time   | Message
|---------------|------------|-------|-----------------|--------|------------------------------
|  ⚡1 ⏳0 ✅0  | 0934a4d8:1 |  1/5  | Shallow search  |   0.0s | Broad search: 2 left
|  ⚡1 ⏳0 ✅0  | 0934a4d8:1 |  1/5  | Shallow search  |  36.7s | Broad search: 1 left
|  ⚡1 ⏳0 ✅0  | 0934a4d8:1 |  1/5  | Shallow search  |  49.3s | Broad search: 0 left
|  ⚡1 ⏳0 ✅0  | 0934a4d8:1 |  2/5  | Eval            |  49.3s | No solution, exiting (forced)
|  ⚡1 ⏳0 ✅0  | 0934a4d8:1 |  DONE | Finished        |  93.1s | FAIL ($0.1587)
Submission file saved to: submissions/submission.json
Results file saved to: submissions/results.json
```
