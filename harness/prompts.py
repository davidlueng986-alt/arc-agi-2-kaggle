"""Prompt construction.

- Expert A: Poetiq SOLVER_PROMPT_2, reused nearly verbatim (verified 54% engine;
  prescriptive template is safe *because* it runs inside the pooled/shuffled
  refinement loop — see HYPOTHESIS.md §2).
- Expert B: Land-style minimal codegen prompt (v1b lineage) with test inputs as
  generalization probes — deliberately NOT prescriptive (Land's negative result).
- Feedback: Poetiq FEEDBACK_PROMPT verbatim + scored pool block (improving order).
- Text channel: Land's minimal free-form prompt (build_prompt lineage) + CSV answer.
- Judge: port of Land's build_duo_pick_prompt (holistic, all traces, top-2 grids,
  synthesis allowed, anti-consistency closing instructions preserved).
"""

from __future__ import annotations

from typing import Optional

from harness.grids import grid_to_csv, grid_to_diagram
from harness.types import Grid

# ---------------------------------------------------------------------------
# Expert A — Poetiq SOLVER_PROMPT_2 (verbatim from reference/poetiq-solver)
# ---------------------------------------------------------------------------

SOLVER_PROMPT_A = '''
You are a world-class expert in solving Abstract Reasoning Corpus (ARC) tasks. Your approach is methodical, creative, and highly effective. You are also a master Python coder, producing elegant, efficient, and well-documented solutions.

Your goal is to analyze a set of input-output examples and devise a Python function that accurately transforms any input grid into its corresponding output grid. The key is to identify a *single, consistent transformation rule* that generalizes across *all* examples. Do not give up until you find a correct solution.

Follow this iterative process:

**Part 1: Initial Analysis and Hypothesis Generation**

1.  **Example Inspection:** Carefully examine the input and output grids for each example. Note their dimensions, color palettes, and any prominent visual features (shapes, symmetries, patterns). Use visualization techniques to aid your analysis.
2.  **Transformation Hypotheses:** Formulate several candidate transformation rules. Start with simpler rules and gradually increase complexity. Consider these categories:
    *   **Color Transformations:** Replacing colors based on specific criteria (e.g., adjacency, frequency). For example, replace all 0s with 1s, or replace the most frequent color with the least frequent color.
    *   **Object Isolation:** Identifying and isolating objects based on color, shape, or position. For example, extract the largest connected component of a certain color, or isolate objects based on their spatial relationships.
    *   **Spatial Operations:** Rotating, reflecting, resizing, or moving objects. For example, rotate the grid by 90 degrees, reflect the grid horizontally or vertically, or resize the grid by a certain factor.
    *   **Pattern Generation:** Replicating or extending existing patterns. For example, repeat a certain pattern across the grid, or generate a new pattern based on the existing patterns.
3.  **Symmetry Analysis:** Identify any symmetries (rotational, reflectional) in the input and output grids. Determine if the transformation preserves or alters these symmetries.

**Part 2: Iterative Testing and Refinement**

1.  **Code Implementation:** Implement your strongest candidate rule as a Python function. The function *must* accept a 2D numpy array as input and return a 2D numpy array as output.
2.  **Rigorous Testing:** Test your code against *all* training examples. A single failure indicates an incorrect rule.
3.  **Feedback Analysis:** If your code fails, carefully analyze the feedback. Identify the specific examples that failed and the nature of the errors. Use print statements to debug intermediate values and verify your assumptions.
4.  **Hypothesis Refinement:** Based on the feedback, refine your transformation rule. This may involve adjusting parameters, adding new conditions, or discarding the rule altogether and starting with a new hypothesis.
5.  **Repeat:** Continue this iterative process of coding, testing, and refining until you find a rule that works for all training examples. Do not give up until you find a correct solution.

**Part 3: Coding Guidelines**

1.  **Available Libraries:** You can use `numpy`, `cv2` (OpenCV), and any library from the standard Python library.
2.  **Computer Vision Techniques:** Consider using `cv2` for tasks involving object detection, edge detection, or image filtering.
3.  **Utility Functions:** Write reusable utility functions to improve code modularity and readability.
4.  **Error Handling:** Implement robust error handling to gracefully manage edge cases and invalid inputs.
5.  **Code Clarity:** Write clean, well-documented code with meaningful variable names and comments.

**Part 4: Output Requirements**

1.  **Output Format:**
    *   Begin with a concise paragraph explaining the proposed solution, followed by a Python code section.
    *   You *must* provide a code output representing your best attempt. Do not give up or refuse to produce code.
    *   **The code section must be a single, valid Python code block in markdown fenced code block format and nothing else.**
    *   The main transform function must have the signature `def transform(grid: np.ndarray) -> np.ndarray`.
    *   Document the transformation rule implemented in the docstring of the transform function.
    *   Do not include any `__name__ == "__main__"` block. This will be added later by the user. You are writing a library function.

**Example:**

**Problem:**
Input:
<Diagram>
0 0 1
0 1 0
1 0 0
</Diagram>

Output:
<Diagram>
1 1 1
1 1 1
1 1 1
</Diagram>

**Explanation:**
Replace all 0s with 1s.

```python
import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    """Replace all 0s with 1s."""
    return np.where(grid == 0, 1, grid)
```

**PROBLEM:**

Below is a textual representation of the input-output examples and the challenge to be solved.

$$problem$$
'''

# ---------------------------------------------------------------------------
# Expert B — Land-style minimal codegen prompt (no hypothesis categories,
# no output template beyond the code contract; probes = test inputs)
# ---------------------------------------------------------------------------

SOLVER_PROMPT_B_HEADER = '''Below is an ARC AGI task. You're given the training input/output pairs. Your task is to write a python function transform(grid) that returns the output grid. The grid is a 2D numpy array. The transform() function must solve all the input/output pairs. You're also given the input-only test data to help you ensure your solution is generalizable.

You may use numpy, scipy, and cv2 (OpenCV) for grid manipulation.
'''

SOLVER_PROMPT_B_FOOTER = '''
Write a brief explanation of the rule you found, then output the python code for the transform() function in a single markdown code block. Do not include any `__name__ == "__main__"` block.
'''

# ---------------------------------------------------------------------------
# Feedback (Poetiq verbatim)
# ---------------------------------------------------------------------------

FEEDBACK_PROMPT = '''
**EXISTING PARTIAL/INCORRECT SOLUTIONS:**

Following are some of the best, though not completely correct, solutions so far. For each solution, its code, corresponding feedback regarding its output on the example problems, and a numeric score between 0. (worst) and 1. (best) indicating the quality of outputs is also provided. Study these solutions and corresponding feedback and produce a new solution fixing all the issues. Make sure to follow the output format specified earlier.

$$feedback$$
'''

FEEDBACK_SOLUTION_TEMPLATE = """
<solution_{index}>
<solution_code>
```python
{code}
```
</solution_code>
<solution_evaluation>
{feedback}
</solution_evaluation>
<solution_score>
{score}
</solution_score>
</solution_{index}>
"""

# ---------------------------------------------------------------------------
# Text channel — Land's minimal free-form prompt (prompts_standard.build_prompt
# lineage) + the CSV answer format (winner of Land's 9-format benchmark).
# ---------------------------------------------------------------------------

TEXT_PROMPT_HEADER = '''You are solving an ARC (Abstraction and Reasoning Corpus) task.
Each grid cell is an integer 0-9 representing a color.
Use the solved examples to infer the transformation and apply it to the test input.
'''

TEXT_PROMPT_FOOTER = '''
Respond with an explanation of your thinking that is detailed enough that someone can reconstruct your solution. Afterwards, you MUST also respond with the completed output grid.
Format every grid you output as comma-separated values (CSV) with each row on a new line, inside its own fenced code block, like this:
```
7,0,0,7
0,7,7,0
```
'''


def _diagram(grid: Grid) -> str:
    return grid_to_diagram(grid)


def build_problem_str(
    train_in: list[Grid],
    train_out: list[Grid],
    test_in: list[Grid],
    shuffle: bool = True,
    seed: int = 0,
) -> str:
    """Poetiq format_problem: <Diagram> examples + challenges, train order shuffled
    with `seed` (Poetiq passes seed+iteration each round: free augmentation)."""
    order = list(range(len(train_in)))
    if shuffle and len(order) > 1:
        import random

        rng = random.Random(seed)
        rng.shuffle(order)

    parts: list[str] = []
    for n, i in enumerate(order, start=1):
        parts.append(
            f"""
Example #{n}
Input:
<Diagram>
{_diagram(train_in[i])}
</Diagram>

Output:
<Diagram>
{_diagram(train_out[i])}
</Diagram>
"""
        )
    for n, tin in enumerate(test_in, start=1):
        parts.append(
            f"""
Challenge #{n}
Input:
<Diagram>
{_diagram(tin)}
</Diagram>
"""
        )
    return "".join(parts)


def build_solver_a_prompt(
    train_in: list[Grid], train_out: list[Grid], test_in: list[Grid],
    shuffle: bool = True, seed: int = 0,
) -> str:
    problem = build_problem_str(train_in, train_out, test_in, shuffle=shuffle, seed=seed)
    return SOLVER_PROMPT_A.replace("$$problem$$", problem)


def build_solver_b_prompt(
    train_in: list[Grid], train_out: list[Grid], test_in: list[Grid],
    shuffle: bool = True, seed: int = 0,
) -> str:
    order = list(range(len(train_in)))
    if shuffle and len(order) > 1:
        import random

        rng = random.Random(seed)
        rng.shuffle(order)
    lines = [SOLVER_PROMPT_B_HEADER, "", "Solved examples:"]
    for n, i in enumerate(order, start=1):
        lines.append(f"Example {n}:")
        lines.append("input:")
        lines.append(grid_to_csv(train_in[i]))
        lines.append("output:")
        lines.append(grid_to_csv(train_out[i]))
        lines.append("")
    lines.append("Input-only test data:")
    for n, tin in enumerate(test_in, start=1):
        lines.append(f"Probe {n}:")
        lines.append("input:")
        lines.append(grid_to_csv(tin))
        lines.append("")
    lines.append(SOLVER_PROMPT_B_FOOTER)
    return "\n".join(lines)


def build_feedback_block(
    solutions: list[dict], max_examples: int = 5, improving_order: bool = True
) -> str:
    """Poetiq create_examples: top-K pool members by score; improving_order shows
    worst->best so the strongest exemplar occupies the recency slot."""
    if not solutions:
        return ""
    ranked = sorted(range(len(solutions)), key=lambda i: solutions[i]["score"], reverse=True)
    ranked = ranked[:max_examples]
    if improving_order:
        ranked = ranked[::-1]
    blocks = []
    for k, idx in enumerate(ranked, start=1):
        s = solutions[idx]
        blocks.append(
            FEEDBACK_SOLUTION_TEMPLATE.format(
                index=k, code=s["code"], feedback=s["feedback"], score=f"{s['score']:.2f}"
            )
        )
    return "\n".join(blocks)


def build_feedback_prompt(solutions: list[dict], max_examples: int = 5, improving_order: bool = True) -> str:
    block = build_feedback_block(solutions, max_examples, improving_order)
    return FEEDBACK_PROMPT.replace("$$feedback$$", block)


def build_text_prompt(
    train_in: list[Grid], train_out: list[Grid], test_input: Grid
) -> str:
    lines = [TEXT_PROMPT_HEADER, "", "Solved examples:"]
    for n, (tin, tout) in enumerate(zip(train_in, train_out), start=1):
        lines.append(f"Example {n}:")
        lines.append("input:")
        lines.append(grid_to_csv(tin))
        lines.append("output:")
        lines.append(grid_to_csv(tout))
        lines.append("")
    lines.append("Test input:")
    lines.append(grid_to_csv(test_input))
    lines.append("")
    lines.append(TEXT_PROMPT_FOOTER)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Holistic judge — port of Land's build_duo_pick_prompt (audit_prompts.py)
# ---------------------------------------------------------------------------

JUDGE_CLOSING = """Your task is to understand these solutions, and assess how well they've understood the problem, and how likely their solutions are to provide the correct solution to the test input.
Often, new mechanics are introduced in the test example for which the solutions do not generalize well. Please output two solutions that you think represent the right mechanic for solving the problem.
Output your two solutions as grids (in code blocks). Format the grids as comma-separated values (CSV) with each row on a new line, like this:
```
7,0,0,7
0,7,7,0
```
Explain how you came to these two solutions being the two most likely. In coming up with your two solutions, study all the provided solutions and their reasoning to come up with a meta-conclusion about how to solve the problem.
If you believe none of the provided solutions is right, you may synthesize a new grid recombining the correct sub-components of the flawed solutions.
Put your two final grids at the very END of your answer, one per fenced code block, first choice first, second choice second."""


def build_judge_prompt(
    train_in: list[Grid],
    train_out: list[Grid],
    test_input: Grid,
    entries: list[dict],
    total_attempts: int,
    trace_cap_chars: int = 2500,
) -> str:
    """One long-context prompt containing the task and ALL candidate traces.

    entries: [{"content": trace/code text, "grid": predicted grid}] already
    deduplicated/capped by the caller.
    """
    from harness.parsing import truncate

    parts: list[str] = []
    parts.append(f"Below is a prompt that was run {total_attempts} times:")
    base = build_text_prompt(train_in, train_out, test_input)
    parts.append("\n<PROMPT START>")
    parts.append(base)
    parts.append("<PROMPT STOP>\n")
    parts.append(
        f"Solutions were generated {len(entries)} times, using different types of solvers. "
        "All solutions are represented below:\n"
    )
    for i, e in enumerate(entries, start=1):
        parts.append(f"<SOLUTION {i} START>")
        parts.append("<CONTENT>")
        parts.append(truncate(e.get("content", ""), trace_cap_chars))
        parts.append("</CONTENT>")
        parts.append("<PREDICTED_GRID>")
        parts.append(grid_to_csv(e.get("grid")))
        parts.append("</PREDICTED_GRID>")
        parts.append(f"<SOLUTION {i} STOP>\n")
    parts.append(JUDGE_CLOSING)
    return "\n".join(parts)
