
# Conclusion

A more extensive prescriptive prompt vastly reduces performance. We saw this previously when requiring/forcing JSON output, or forcing too extensive explanations/motivations. It seems that the models need to freely choose their own path, and it's better to "clean it up" afterwards than prescribe to them what to do.

# Prompts

I started off with a very `basic prompt` that I wrote quickly:

```
Below is an ARC AGI task. You're given the training input/output pairs in python. Your task is to write a python function solver(input) that returns the output grid. The solver() function must solve all the input/output pairs

Solved examples:
Example 1:
input:
[[2, 0, 2, 0, 6, 0 ... 0, 0, 0]]
output:
[[0, 5, 0, 0, 0, 0, 0], [2, 2, ... 0, 0]]

Example 2:
input:
...
output:
...

...

Only output the python code for the solver() function
```

Through iteration with gemini 3 and gpt 5.2 pro (through the apps) I developed this prompt into a `advanced prompt`:

```
You are an expert ARC-AGI Solver Architect. You will be given ARC task data containing multiple training (input_grid → output_grid) pairs. Your job is to infer the single general transformation that maps EVERY training input to its output, then implement it as Python.

CRITICAL OUTPUT RULE (non-negotiable):
- When answering the ARC task, output ONLY raw Python code that defines `def solver(input_grid): ...` and returns the predicted output grid.
- Output NOTHING else: no markdown outside the code, no explanations outside the code, no extra top-level definitions, no prints/logging, no I/O.
- You MAY include detailed explanations inside the code as Python comments (including markdown-style headings/bullets), as long as the final output is still valid Python source.

FUNCTION CONTRACT:
- Signature: `def solver(input_grid: list[list[int]]) -> list[list[int]]:`
- `input_grid` is a rectangular list of lists of integers 0–9.
- Return a NEW rectangular list of lists of integers 0–9 (do not mutate `input_grid`).
- Deterministic and pure: no randomness, no external state, no side effects.

ALLOWED / DISALLOWED TOOLS:
- Do NOT use external libraries (NO numpy/pandas/cv2/etc.).
- If you need the standard library, import ONLY inside `solver` and keep it minimal (e.g., `collections`, `itertools`, `math`, `copy`).
- No file/network access, no reading/writing, no debugging output.

SILENT INTERNAL REASONING WORKFLOW (do this privately; never reveal chain-of-thought):
1) Extract multiple candidate hypotheses (rules) from the training pairs.
2) For each hypothesis, validate it against ALL training examples.
3) If a hypothesis fails even ONE example, reject it immediately (no patching, no special-casing).
4) Prefer the simplest consistent rule (Occam’s razor). Avoid complex multi-stage pipelines unless forced by the data.
5) Consider at least one plausible alternative rule and silently reject it by identifying which training pair it contradicts.
6) Implement the final rule cleanly in code.
7) Mentally test `solver()` on every training input to ensure it reproduces every training output exactly.

ANALYZE & DECOMPOSE (use only what is supported by ALL examples):
- Dimensions: same size vs crop/pad/expand; infer output size from evidence, not assumptions.
- Colors: sets and frequencies; background is often most frequent but MUST be verified (never assume background=0).
- Objects: connected components (usually 4-neighborhood), lines, rectangles, holes/enclosures, bounding boxes, masks, centroids, adjacency/touching, containment.
- Common transforms: translate, rotate, reflect, scale, crop, frame/border, fill, draw/extend rays until collision, copy/paste objects, pattern completion, recolor via consistent mapping, selection of specific object(s) by a consistent criterion.
- Useful priors (only if validated): geometry (symmetry/rotation), topology (enclosure/holes), arithmetic (counts/sorts by size/position).

FORBIDDEN ANTI-PATTERNS (must not appear in code):
- NO lookup tables or memorization (e.g., `if input_grid == train_input[i]: return train_output[i]`).
- NO hardcoding fixed grid sizes, absolute coordinates, specific example outputs, or per-example branches.
- NO assuming symmetry, tiling, repetition, or color semantics unless ALL training pairs prove it.
- NO introducing new colors unless training outputs prove new colors are required.

FAIL-FAST REQUIREMENT (IMPORTANT):
- The solver must NOT “fail safe” or silently guess when the input deviates from the inferred structure.
- Derive explicit preconditions from the training-consistent rule (e.g., “exactly one object”, “exactly two colors”, “object touches border”, “output size equals bounding box”, etc.).
- Enforce those preconditions with `assert ... , "clear message"` or `raise ValueError("clear message")`.
- If the test input violates the learned preconditions, explicitly FAIL (raise) rather than returning a plausible-looking grid.
- Do NOT wrap the whole solver in broad try/except that hides errors or returns a fallback.

IMPLEMENTATION STYLE:
- Write clear, general code. Use small helper functions INSIDE `solver()` if helpful (e.g., flood fill, bounding box, rotate/flip, overlay).
- Avoid global variables and caching across calls.
- You may include detailed explanations as comments inside `solver()` to document the inferred rule and the checks.

FINAL REMINDER:
- Your final answer to the ARC task must be ONLY the Python code defining `solver(input_grid)` (comments inside the code are allowed) and nothing else.

[ARC TASK DATA WILL BE INSERTED BELOW THIS LINE]

Solved examples:
Example 1:
input:
[[2, 0, 2, 0, 6 ... 0, 0, 0, 0, 0, 0, 0]]
output:
[[0, 5, 0, 0, 0, 0, 0], [2, 2, ... 0, 0], [0, 0, 0, 0, 0, 0, 0]]

Example 2:
input:
...
output:
...

...
```

# Performance

On the 10 easiest arc 2 eval problems (`136b0064:1,247ef758:1,7c66cb00:1,31f7f899:1,36a08778:1,e8686506:1,d8e07eb2:2,edb79dae:1,16de56c4:1,7b5033c1:1`) using gpt-5.2-medium the performance between these two were very surprising:

`basic prompt`:
- 5/2/3
- 5/2/3
- 5/2/3

`advanced prompt`:
- 2/2/6
- 1/2/7
- 1/2/7

(results denote solved/train-cases-solved-but-problem-wrong/train-cases-failed)

That is, the `basic prompt` outperformed the `advanced prompt` vastly.

TODO: Test on harder problem with higher reasoning

