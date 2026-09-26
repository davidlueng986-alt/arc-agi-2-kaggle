"""Prompt construction: minimal, clean, CSV-encoded.

H1 deliberately avoids prescriptive multi-section templates. Land's documented
negative result is that "prescriptive prompting templates and iterative
refinement systematically reduce hypothesis diversity and degrade performance",
so the prompt states the task contract and the output format and then gets out of
the model's way.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .grids import Grid, to_csv_indented
from .tasks import ARCTask

SYSTEM_PROMPT = """You are an expert at ARC-AGI-2 abstract reasoning puzzles.

You are shown input/output grid pairs from one puzzle. Grids are rectangular
matrices of integers 0-9 written as comma-separated rows; 0 is usually but not
always the background. Exactly one deterministic transformation maps every input
grid to its output grid. Infer that transformation from the training pairs, check
that it reproduces every one of them exactly, then apply it to the test inputs.

Be careful about these things, which decide most puzzles:
- The output size is set by the rule, not by the input size. Determine it explicitly.
- Distinguish the relevant objects from the background before transforming anything.
- A rule that explains all training pairs exactly is far more likely to be right
  than one that explains most of them.

Answer format. After your reasoning, write the answer for each test input like this:

ANSWER 1:
2,0,1
0,2,0
1,0,2

Use one comma-separated row per line with no spaces and no brackets. If there is
more than one test input, continue with ANSWER 2:, ANSWER 3:, and so on, in the
same order as the test inputs were given. Write nothing after the final answer."""

TRAIN_HEADER = "Training pair {i} input:\n{gi}\n\nTraining pair {i} output:\n{go}"
TEST_HEADER = "Test input {i}:\n{gi}"
FOOTER = (
    "Work out the transformation and give the output grid for each of the {n} test "
    "input{s}. Reply with ANSWER 1:{extra} in the exact format shown above."
)


@dataclass(frozen=True)
class Prompt:
    system: str
    user: str

    @property
    def approx_tokens(self) -> int:
        return int((len(self.system) + len(self.user)) / 3.2)


def build_prompt(
    task: ARCTask,
    shuffle_train: bool = False,
    seed: int = 0,
    max_train_pairs: Optional[int] = None,
) -> Prompt:
    pairs: List[tuple] = list(task.train)
    if max_train_pairs is not None and len(pairs) > max_train_pairs:
        pairs = pairs[:max_train_pairs]
    if shuffle_train and len(pairs) > 1:
        rng = random.Random(f"{seed}:{task.task_id}")
        rng.shuffle(pairs)

    parts: List[str] = []
    for i, (gi, go) in enumerate(pairs, start=1):
        parts.append(
            TRAIN_HEADER.format(
                i=i, gi=to_csv_indented(gi, "    "), go=to_csv_indented(go, "    ")
            )
        )
    for i, gi in enumerate(task.test, start=1):
        parts.append(TEST_HEADER.format(i=i, gi=to_csv_indented(gi, "    ")))

    n = task.n_test
    extra = "".join(f" ANSWER {k}:" for k in range(2, n + 1))
    parts.append(FOOTER.format(n=n, s="" if n == 1 else "s", extra=extra))
    return Prompt(system=SYSTEM_PROMPT, user="\n\n".join(parts))


def build_messages(task: ARCTask, shuffle_train: bool = False, seed: int = 0) -> List[dict]:
    prompt = build_prompt(task, shuffle_train=shuffle_train, seed=seed)
    return [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]


def prompt_token_estimate(tasks: Sequence[ARCTask]) -> dict:
    vals = sorted(t.approx_prompt_tokens() for t in tasks)
    if not vals:
        return {"n": 0, "mean": 0, "p50": 0, "p90": 0, "max": 0, "total": 0}
    return {
        "n": len(vals),
        "mean": int(sum(vals) / len(vals)),
        "p50": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, int(0.9 * len(vals)))],
        "max": vals[-1],
        "total": sum(vals),
    }
