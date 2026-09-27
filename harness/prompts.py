"""Prompts for the ARC-AGI-2 Duck-REPL agent.

The Duck's authors are explicit that their system prompt is *not* lightweight --
"we had to encode a lot of specific knowledge into the prompts".  The same is
true here, but the knowledge is different: theirs is about 64x64 game boards,
HUD bars and MOUSE coordinates; ours is about ARC-AGI-2 transformation families
and about how to use an exact oracle without overfitting to it.

Two deliberate economies relative to the Duck, both because our environment is
static rather than a changing game:

* the task data is emitted **once** as a fixed prefix (so it is prefix-cacheable
  and never re-derived), not restated per turn;
* the per-turn user message is emitted **only when something changed** (new best
  candidate, budget warning, no verification produced), not every turn.

And one deliberate economy relative to the Duck's world-model protocol: we ask
for at most two short labelled lines (``Hypothesis:`` / ``Next:``) instead of a
full restated world model each turn.  The arXiv 2607.15439 ablation found
"simplification" pressure helped in 3 of 4 settings; restating a large world
model every turn is the opposite of that and costs tokens we would rather spend
on turns.
"""

from __future__ import annotations

from typing import Sequence

from .grids import Task, TaskStats, compute_stats, shape_of, text_tokens, to_ascii
from .oracle import COLOR_LEGEND

TOOL_NAME = "python"

TOOL_CALL_FORMAT_GUIDANCE = (
    "When calling `python`, emit exactly the tool-call format shown elsewhere in this prompt. "
    "Use only that format; no markdown fences around the tool call, no prose wrappers, no "
    "alternate syntax. Do not quote tool-call markup inside explanatory text."
)

ROLE_ADDENDUM = (
    "You are a coding agent solving ARC-AGI-2 puzzles.\n"
    "You never emit an answer grid. You write Python, and the Python produces the answer.\n"
)

TASK_FORMAT_ADDENDUM = (
    "\nTask format:\n"
    "- Each task has TRAIN pairs (input grid AND its ground-truth output grid) and TEST "
    "inputs (no output given).\n"
    "- Grids are rectangular, values 0-9, sizes 1x1 to 30x30. Input and output shapes often "
    "DIFFER -- deriving the output shape is usually half the puzzle.\n"
    f"- Colour legend for the ASCII views: {COLOR_LEGEND}.\n"
    "- You must produce one output grid per TEST input, in order.\n"
)

ORACLE_ADDENDUM = (
    "\nThe oracle -- this is the most important thing you have:\n"
    "- `verify(fn)` runs `fn(grid) -> grid` against EVERY train pair and compares to the "
    "ground-truth outputs exactly. It returns `passed`, `pairs`, `pass_rate`, `exact`, "
    "`soft_score`, per-pair `failures` with precise diff diagnostics, `aug` symmetry probe, "
    "and a compact preview of the TEST predictions.\n"
    "- This is a free, exact, deterministic oracle. Use it. Do not reason about whether your "
    "rule is right -- execute it and read the diff.\n"
    "- `fn` must accept a list-of-lists of ints and return a list-of-lists of ints (numpy "
    "arrays are accepted and converted). It must be a pure function of its input grid.\n"
    "- Every `verify` call also REGISTERS the candidate with the harness. The best registered "
    "candidate's test predictions become the submission. You never print a final grid.\n"
    "- `finish(reason)` ends the task early and saves budget. Call it once a candidate "
    "verifies on all train pairs, or when you are certain nothing better is coming.\n"
)

DIFF_READING_ADDENDUM = (
    "\nReading the diff diagnostics:\n"
    "- `kind: shape` means your output dimensions are wrong. Fix this FIRST: shape errors tell "
    "you which transformation family you are in (crop-to-object, extract, tile, canvas resize). "
    "`delta_rows`/`delta_cols` say by how much.\n"
    "- `kind: cells` gives `n_diff`, `diff_frac`, `bbox` (bounding box of all mismatches), "
    "`samples` as `[row, col, expected, got]`, histograms of expected-vs-got values at the "
    "mismatching cells, and `expected_crop`/`got_crop` -- a 5x5 ASCII window around the first "
    "mismatch with the mismatching cell upper-cased.\n"
    "- `diff_frac` near 0 with a tight bbox means your rule is almost right and mis-handles a "
    "special case. `diff_frac` near 1 means the rule is wrong.\n"
    "- `expected_hist_at_diffs` vs `got_hist_at_diffs` often reveals a colour mapping you got "
    "backwards or a background you failed to fill.\n"
    "- `aug` re-runs your program on transpose/rot180/flipud variants of each train pair. Many "
    "ARC tasks are genuinely NOT symmetry-equivariant (gravity, cropping to a corner, reading "
    "order), so a low `aug` is a hint to check for accidental symmetry assumptions, never a "
    "rule and never a reason to discard a fully verified program.\n"
)

RUNTIME_STATE_ADDENDUM = (
    "\nVariables preloaded into EVERY `python` call:\n"
    "- `TRAIN` -- list of `Pair` objects with `.input`, `.output` (both `Grid`), `.index`.\n"
    "- `TEST` -- list of `Grid`, the test inputs, in submission order.\n"
    "- `TRAIN_PAIRS` / `TEST_INPUTS` -- the same data as plain `list[list[int]]`.\n"
    "- `TASK_ID` -- string.\n"
    "- `Grid` is a `list[list[int]]` subclass with `.shape`, `.rows`, `.cols`, `.ascii`, "
    "`.csv`, `.segmentation`, `.seg(max_nodes)`, `.colors()`, `.histogram()`, `.copy_grid()`.\n"
    "- `verify(fn, aug=True)` -- the oracle (see above).\n"
    "- `finish(reason='')` -- end the task.\n"
    "- `segment(grid)` / `grid.seg(max_nodes)` -- 4-connected components: per-object id, colour "
    "char, pixel count, bbox, a position-invariant `hash` (equal hash == same shape and colour "
    "wherever it sits, so you can match objects between input and output), clockwise `boundary` "
    "corner points, `children` (nesting), plus `adjacency_list` of touching object pairs.\n"
    "- `show(grid, mode='ascii'|'csv')`, `ascii_grid(grid)`, `shapes()`.\n"
    "- `augment(grid, name)`, `AUGMENTATIONS`, `grid_diff(expected, got)`, `cell_accuracy(a, b)`, "
    "`normalize_grid(x)`.\n"
    "- `numpy` is importable, as are the standard data/algorithm modules (itertools, functools, "
    "collections, math, re, random, heapq, bisect, statistics, copy, json, string, operator, "
    "fractions, dataclasses, typing, hashlib).\n"
    "- Process, filesystem, network and interpreter modules are blocked: no os, sys, subprocess, "
    "socket, pathlib, shutil, threading, pickle, importlib.\n"
)

ARC_KNOWLEDGE_ADDENDUM = (
    "\nARC-AGI-2 domain knowledge (encoded deliberately -- use it to form hypotheses fast):\n"
    "- The transformation is a RULE, not a lookup. It is applied to a test input that differs "
    "from every train input. Never key on absolute coordinates, on a specific grid size, or on "
    "the identity of a train input.\n"
    "- Recurring families: crop to the bounding box of a distinguished object or region; extract "
    "the unique/odd object (by colour, size or shape); complete a symmetry (reflection, rotation, "
    "4-fold); tile or repeat a pattern to fill a canvas; gravity / push objects to an edge; "
    "extend or connect lines and rays until they hit something; colour remapping driven by a key "
    "region or by object size; counting objects and emitting a bar chart or an n x n pattern; "
    "maze path following; denoising (delete the single differing pixel); fill enclosed holes; "
    "draw outlines/borders; boolean overlay of two halves of the input; fractal self-similar "
    "expansion where an object is replaced by a scaled copy of another; sorting or reordering "
    "objects by size or colour.\n"
    "- Very often the input contains TWO parts: a large field and a small 'key' or 'instruction' "
    "region that specifies what to do to the field. Check for a small separate object, a strip "
    "along an edge, or a sub-grid separated by a uniform line.\n"
    "- When input and output shapes differ, the output size is usually determined by something in "
    "the input: a bounding box, a count, a scale factor, or a fixed canvas implied by the train "
    "pairs. Compare output shapes ACROSS pairs before guessing.\n"
    "- Background is usually the most frequent colour, but not always 0. Derive it, don't assume.\n"
    "- Objects are usually 4-connected same-colour regions. `segment()` plus the position-"
    "invariant `hash` is the fastest way to match objects between input and output.\n"
)

METHOD_ADDENDUM = (
    "\nMethod -- the loop that works:\n"
    "1. INSPECT (one snippet): print shapes of all train inputs and outputs, the colour "
    "histograms, and `seg()` summaries of pair 0. Never print a whole grid.\n"
    "2. HYPOTHESISE: name the transformation family. Say which evidence supports it.\n"
    "3. IMPLEMENT + VERIFY in the SAME snippet: define `transform`, call `verify(transform)`, "
    "and print only the compact report. Do not waste a turn on each.\n"
    "4. READ THE DIFF and form the next hypothesis from it. Target the largest systematic error "
    "first (shape, then a whole region, then a special case).\n"
    "5. Repeat 3-4. Generalise across ALL train pairs -- a rule that only works on pairs 0 and 1 "
    "is wrong.\n"
    "6. When `exact` is true on all pairs, call `finish(...)`. If you have budget left and the "
    "task has few train pairs (2 or fewer), consider one alternative interpretation and verify "
    "that too, so a second distinct answer exists.\n"
    "- Batch aggressively: one snippet may inspect several things, define several candidate "
    "transforms and `verify` each of them. A turn is far more expensive than a line of code.\n"
    "- If two candidate rules both verify on the train pairs, prefer the simpler and more "
    "general one; the other is your second attempt anyway, so register both.\n"
)

TOOL_SESSION_ADDENDUM = (
    "\nTool session rules:\n"
    "- You have exactly one tool: `python`, called with a single ephemeral `code` string.\n"
    "- Every call starts from a FRESH process: nothing you defined earlier survives. Redefine "
    "`transform` (and any helpers) in the snippet that calls `verify`. The preloaded variables "
    "above are always there.\n"
    "- Hard limit {tool_timeout}s per call; output capped at ~{tool_output_chars} characters, "
    "after which it is truncated and you are told.\n"
    "- Keep output compact and decision-oriented: counts, coordinates, shapes, diffs, hashes. "
    "NEVER print or echo a full grid -- that is what `seg()`, `.shape`, `.histogram()` and the "
    "`verify` diff are for. Printing whole grids wastes your own context and is the single most "
    "common failure mode.\n"
    "- You may call `verify` and `finish` multiple times in one snippet, including inside loops "
    "(e.g. verify a family of parameterised candidates and print a ranked table).\n"
    "- It is fine to write a lot of code. It is not fine to print a lot of output.\n"
)

OUTPUT_STYLE_ADDENDUM = (
    "\nAssistant text:\n"
    "- Keep it to at most two short lines before a tool call, optionally labelled `Hypothesis:` "
    "and `Next:`. These lines are harvested into a persistent ledger that survives context "
    "eviction, so make them count and correct them when the evidence changes.\n"
    "- Do not restate the grids, the colour legend or these instructions.\n"
    "- Do not narrate code you are about to write; write it.\n"
)


def build_system_prompt(
    *, tool_timeout_s: float = 30.0, tool_output_chars: int = 4096, with_tool_guidance: bool = True
) -> str:
    parts = [
        ROLE_ADDENDUM,
        TASK_FORMAT_ADDENDUM,
        ORACLE_ADDENDUM,
        DIFF_READING_ADDENDUM,
        RUNTIME_STATE_ADDENDUM,
        ARC_KNOWLEDGE_ADDENDUM,
        METHOD_ADDENDUM,
        TOOL_SESSION_ADDENDUM.format(
            tool_timeout=int(tool_timeout_s), tool_output_chars=tool_output_chars
        ),
        OUTPUT_STYLE_ADDENDUM,
    ]
    if with_tool_guidance:
        parts.append("\n" + TOOL_CALL_FORMAT_GUIDANCE + "\n")
    return "".join(parts)


PYTHON_TOOL_SPEC: dict = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Run one ephemeral Python snippet against the preloaded task state. Globals: TRAIN, "
            "TEST, TRAIN_PAIRS, TEST_INPUTS, TASK_ID, Grid, verify(fn), finish(reason), "
            "segment(grid), show(grid), shapes(), augment(grid, name), grid_diff(a, b). "
            "Grid objects expose .shape, .ascii, .csv, .segmentation, .seg(n), .colors(), "
            ".histogram(). numpy is importable. State does NOT persist between calls. "
            "Print compact derived summaries only -- never a whole grid. "
            "verify(fn) is an exact oracle against all train pairs and registers the candidate "
            "as a possible answer. Use print(...) for output or assign a compact object to `result`."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to run. Ephemeral: not saved across calls.",
                }
            },
            "required": ["code"],
        },
    },
}

TOOLS = [PYTHON_TOOL_SPEC]


def _grid_block(label: str, grid, *, budget: int) -> str | None:
    text = f"{label} {shape_of(grid)[0]}x{shape_of(grid)[1]}:\n{to_ascii(grid)}"
    if text_tokens(text) > budget:
        return None
    return text


def build_task_block(task: Task, stats: TaskStats | None = None, *, token_budget: int = 4000) -> str:
    """Fixed task prefix: structural facts, then as many grids as fit the budget.

    ASCII (one symbol per cell) is used rather than CSV because it is roughly a
    third the tokens for the same information and the model never has to *emit*
    a grid, so Land's finding that CSV is the safest generation format does not
    bind.  Whatever does not fit is elided with an explicit note, and the REPL
    always has the full data.
    """
    stats = stats or compute_stats(task)
    lines = [
        f"TASK {task.task_id}",
        stats.one_line(),
        f"test input shapes: {stats.test_shapes}",
        "",
    ]
    per_grid_budget = max(200, token_budget // 4)
    used = text_tokens("\n".join(lines))
    included_pairs = 0
    for index, pair in enumerate(task.train):
        block_in = _grid_block(f"train[{index}] input", pair.input, budget=per_grid_budget)
        block_out = _grid_block(f"train[{index}] output", pair.output, budget=per_grid_budget)
        if block_in is None or block_out is None:
            break
        chunk = block_in + "\n" + block_out
        if used + text_tokens(chunk) > token_budget:
            break
        lines.append(chunk)
        used += text_tokens(chunk)
        included_pairs = index + 1

    for index, grid in enumerate(task.test_inputs):
        block = _grid_block(f"test[{index}] input", grid, budget=per_grid_budget)
        if block is None or used + text_tokens(block) > token_budget:
            break
        lines.append(block)
        used += text_tokens(block)

    elided_pairs = task.n_train - included_pairs
    if elided_pairs > 0:
        lines.append(
            f"[{elided_pairs} further train pair(s) elided to save context -- read them with "
            "TRAIN[i].input.ascii / .csv / .seg() inside `python`]"
        )
    lines.append("")
    lines.append(
        "Everything above is also preloaded in the REPL as TRAIN and TEST. Inspect there rather "
        "than asking for it to be repeated."
    )
    return "\n".join(lines)


def build_status_message(
    *,
    turn_index: int,
    max_turns: int,
    ledger_lines: Sequence[str],
    tokens_used: int,
    token_budget: int,
    seconds_left: float | None = None,
    headline: str | None = None,
    nudge: str | None = None,
) -> str:
    """Compact per-turn status. Emitted only when something changed."""
    lines: list[str] = []
    if headline:
        lines.append(headline)
    lines.append(
        f"turn {turn_index + 1}/{max_turns}; context {tokens_used}/{token_budget} tokens"
        + (f"; {int(seconds_left)}s left on this task" if seconds_left is not None else "")
    )
    if ledger_lines:
        lines.append("Ledger (survives context eviction):")
        lines.extend(f"- {line}" for line in ledger_lines)
    if nudge:
        lines.append(nudge)
    lines.append(
        "Call `python` now. Define `transform` and `verify(transform)` in the same snippet; "
        "print only compact summaries."
    )
    return "\n".join(lines)


FINAL_ANSWER_INSTRUCTION = (
    "TURN BUDGET REACHED. Make one last `python` call. If any candidate verified on all train "
    "pairs, call `finish('best registered')`. Otherwise register your single best-scoring "
    "candidate with `verify(transform)` and then `finish('best effort')`. Do not start a new "
    "line of investigation."
)
