"""Prompt construction for program synthesis.

Adapted from the Poetiq solver prompts (MIT License, Copyright 2025 Poetiq,
Inc. — see reference/poetiq-solver/LICENSE.txt), with three deliberate changes:
  1. grids rendered as CSV (`0,1,2`) — benchmarked best of 9 formats (Land);
  2. explicit anti-overfit contract (no memorized grids — the harness screens
     for literals; no hardcoded positions/colors/sizes; must generalize to
     unseen sizes) — this is the ARC-AGI-2 "less brute-forcible" response;
  3. extra compact structural analysis per grid (connected components,
     palettes) — the Gre24 finding that extra text representations help.

Land's documented negative result — "prescriptive prompting templates
systematically reduce hypothesis diversity and degrade performance" — is why
we keep TWO solver prompt variants of different verbosity and rotate them
across experts/iterations instead of one heavily-engineered template.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .grid import analysis_block, format_csv
from .types import Grid, Task

# ---------------------------------------------------------------------------
# solver prompts (variant A: fuller guidance; variant B: minimal — Land found
# minimal prompts preserve hypothesis diversity)
# ---------------------------------------------------------------------------

SOLVER_PROMPT_A = '''You are an expert at solving ARC-AGI abstract reasoning tasks by writing Python programs, and a master Python coder.

Analyze the input/output examples below, infer the SINGLE general transformation rule consistent with ALL of them, and implement it as a function `transform`.

Think about:
- Objects (connected components), their colors, sizes, positions, and relationships.
- Operations: moving, rotating, reflecting, scaling, cropping, recoloring, counting, completing patterns, gravity, tiling, drawing lines/rays, enclosing, symmetry.
- Whether the rule composes several steps, and whether some step is gated by a contextual cue (e.g. a color or position decides which branch applies).
- Whether the rule treats colors by ROLE (e.g. "the unique object", "the largest component", "the minority color") rather than by identity.

GENERALIZATION CONTRACT (critical — this is ARC-AGI-2, the test input differs from the training ones):
- Your program will be run on a NEW input grid, possibly of a different size, with objects in different positions, and possibly colors that never appear in the training examples. It must not crash and must apply the same rule.
- NEVER hardcode: no memorized grids, no absolute cell coordinates, no fixed grid sizes, no per-example special cases. Programs containing literal copies of training grids are automatically rejected.
- Prefer role-based color logic over color identity when the examples suggest it (e.g. "the least common color", "the color of the object touching the border").
- The rule should be geometrically natural: if the examples do not force a direction (up/down/left/right) or a specific color, your program should behave consistently under rotation, reflection, and color renaming.
- Output shape: derive it from the rule (same size, cropped, scaled, constant...), never assume.

IMPLEMENTATION:
- Signature: `def transform(grid: np.ndarray) -> np.ndarray`. `grid` is a 2-D int array (values 0-9). Return a NEW 2-D int array; do not modify the input in place.
- Available: numpy, cv2, Python standard library. Deterministic and pure: no randomness, no I/O, no global state.
- Handle any input size >= 1x1 gracefully.

OUTPUT FORMAT:
- A brief explanation, then the complete program in a SINGLE markdown ```python code block. No `__main__` block, no test code, no calls to transform.

**PROBLEM:**

$$problem$$
'''

SOLVER_PROMPT_B = '''Below is an ARC task: training input/output grid pairs (CSV) and test input(s). Write a Python function `transform(grid: np.ndarray) -> np.ndarray` (numpy available, deterministic, no hardcoded grids/positions/sizes — it must generalize to the test input, which may have a different size and unseen colors) that maps every input to its output according to the single rule consistent with all training pairs.

Reply with a short explanation and then exactly one markdown ```python code block containing the program (no `__main__`, no test calls).

$$problem$$
'''

FEEDBACK_PROMPT = '''**PREVIOUS ATTEMPTS:**

Below are the best programs so far. None is fully correct. Each shows its code, its execution feedback on the training examples (mismatching cells are shown as `predicted/expected`), and a score in [0, 1]. Study the failure patterns, form a better hypothesis, and produce a NEW program that fixes all the issues. Do not merely patch a failing example — find the rule that explains ALL examples.

$$feedback$$
'''

CONSISTENCY_NUDGE = '''**NOTE:** Your previous program passed ALL training pairs. It was kept. The training pairs are necessary but NOT sufficient evidence: the test input may differ in size, positions, or colors. Re-examine whether your rule is stated by role rather than by accident of the training grids (absolute positions, specific color identities, literal sizes). Now propose either (a) the same program if you are confident it generalizes, or (b) a DIFFERENT hypothesis that also explains all training pairs but generalizes better. Always output one ```python code block.
'''

ADJUDICATION_PROMPT = '''Two or more Python programs below ALL reproduce every training input/output pair exactly, yet they produce DIFFERENT outputs on the test input. At most one captures the intended rule.

Decide which program generalizes correctly, and write the final correct program. Reason about:
- Which rule is more natural/general (role-based colors, geometric consistency, no magic constants)?
- Which one would still be consistent if the training grids were rotated, reflected, or recolored?
- Does one of them hardcode accidents of the training set (positions, sizes, color identities)?
- The test input's specifics: sizes, colors, object counts that never appeared in training.

Programs and their test outputs:

$$candidates$$

Output a short justification, then the FINAL program (signature `def transform(grid: np.ndarray) -> np.ndarray`, numpy/cv2/stdlib, deterministic, no memorized grids) in a single markdown ```python code block.

$$problem$$
'''

LOO_PROMPT = '''You are an expert at solving ARC-AGI tasks by writing Python programs.

Below are SOME training pairs of an ARC task (one pair is withheld). Infer the general rule and implement `def transform(grid: np.ndarray) -> np.ndarray` (numpy available; deterministic; no hardcoding of grids, positions, sizes, or color identities — describe the rule by roles so that it applies to ANY pair from this task).

Reply with a short explanation, then exactly one markdown ```python code block.

$$problem$$
'''


# ---------------------------------------------------------------------------
# problem rendering
# ---------------------------------------------------------------------------


def render_task(
    task: Task,
    shuffle: bool = True,
    seed: int = 0,
    omit_pair: Optional[int] = None,
    include_test: bool = True,
    include_analysis: bool = True,
) -> str:
    """CSV rendering of train pairs (+ test inputs), optionally shuffled with
    `seed` (Poetiq's free augmentation: re-ordering pairs every iteration) and
    optionally omitting one pair (LOO folds)."""
    pairs = list(task.train)
    idxs = list(range(len(pairs)))
    if omit_pair is not None:
        idxs = [i for i in idxs if i != omit_pair]
    if shuffle and len(idxs) > 1:
        rng = np.random.default_rng(seed)
        idxs = [idxs[i] for i in rng.permutation(len(idxs))]

    parts = []
    for n, i in enumerate(idxs, start=1):
        x, y = pairs[i]
        parts.append(f"Example #{n}\nInput:\n{format_csv(x)}\n\nOutput:\n{format_csv(y)}")
    if include_test:
        for n, x in enumerate(task.test, start=1):
            parts.append(f"Test input #{n} (produce the output grid for this one):\n{format_csv(x)}")
    text = "\n\n".join(parts)
    if include_analysis:
        shown_pairs = [pairs[i] for i in idxs]
        tests = list(task.test) if include_test else []
        text += "\n\nStructural analysis:\n" + analysis_block(shown_pairs, tests)
    return text


def _fill(template: str, **fields: str) -> str:
    out = template
    for k, v in fields.items():
        out = out.replace(f"$${k}$$", v)
    return out


def build_solver_messages(
    task: Task,
    iteration: int,
    variant: str = "A",
    partials: Optional[Sequence[tuple[str, str, float]]] = None,
    max_partials: int = 5,
    improving_order: bool = True,
    shuffle_seed: int = 0,
) -> list[dict]:
    """Chat messages for one synthesis call.

    partials: (code, feedback_text, soft_score) of previous attempts.
    """
    template = SOLVER_PROMPT_A if variant == "A" else SOLVER_PROMPT_B
    problem = render_task(task, shuffle=True, seed=shuffle_seed + iteration)
    content = _fill(template, problem=problem)
    if partials:
        ranked = sorted(partials, key=lambda p: p[2])
        if not improving_order:
            ranked = ranked[::-1]
        ranked = ranked[-max_partials:]
        blocks = []
        for k, (code, fb, score) in enumerate(ranked, start=1):
            blocks.append(
                f"<solution_{k}>\n<score>{score:.2f}</score>\n<feedback>\n{fb}\n</feedback>\n"
                f"<code>\n```python\n{code}\n```\n</code>\n</solution_{k}>"
            )
        content += "\n\n" + _fill(FEEDBACK_PROMPT, feedback="\n\n".join(blocks))
    return [{"role": "user", "content": content}]


def build_verified_nudge_messages(
    task: Task, verified_code: str, iteration: int, seed: int = 0
) -> list[dict]:
    """When the pool already contains a verified program, later iterations ask
    for a DIVERSE alternative hypothesis (pool diversity feeds disambiguation)."""
    problem = render_task(task, shuffle=True, seed=seed + iteration)
    content = _fill(SOLVER_PROMPT_A, problem=problem)
    content += (
        "\n\n**A PROGRAM THAT PASSES ALL TRAINING PAIRS:**\n```python\n"
        + verified_code
        + "\n```\n\n"
        + CONSISTENCY_NUDGE
    )
    return [{"role": "user", "content": content}]


def build_adjudication_messages(
    task: Task, entries: Sequence[tuple[str, Optional[Grid]]], variant_seed: int = 0
) -> list[dict]:
    """entries: (code, test_output_of_first_test_input) of the disagreeing
    verified representatives."""
    from .grid import format_csv as _fc

    blocks = []
    for k, (code, out) in enumerate(entries, start=1):
        out_txt = _fc(out) if out is not None else "(program crashed on the test input)"
        blocks.append(
            f"<program_{k}>\n```python\n{code}\n```\n"
            f"<its_output_on_test_input_1>\n{out_txt}\n</its_output_on_test_input_1>\n</program_{k}>"
        )
    problem = render_task(task, shuffle=False, include_analysis=False)
    content = _fill(ADJUDICATION_PROMPT, candidates="\n\n".join(blocks), problem=problem)
    return [{"role": "user", "content": content}]


def build_loo_messages(task: Task, omit_pair: int, seed: int = 0) -> list[dict]:
    problem = render_task(task, shuffle=False, seed=seed, omit_pair=omit_pair, include_test=False)
    content = _fill(LOO_PROMPT, problem=problem)
    return [{"role": "user", "content": content}]
