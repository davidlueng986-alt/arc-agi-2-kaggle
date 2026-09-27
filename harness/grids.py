"""Grid / task data model and text renderings.

Grids are plain ``list[list[int]]`` everywhere (JSON-safe, matches the
competition schema exactly).  numpy is optional and only used by ``render.py``.

Two text renderings are provided because they do different jobs:

* ``csv``  -- comma-separated digits.  Land's grid-format search found CSV beat
  8 other formats by ~10%; it is the form the model reasons about numerically.
* ``ascii`` -- one colour symbol per cell, no separators.  Roughly a third the
  tokens of CSV for the same grid; it is the form for *shape* perception, and is
  what the Duck exposes as ``current_frame.ascii``.

Neither is ever parsed back: because the answer is produced by executing a
program rather than by the model emitting a grid, the whole "model produced a
malformed grid" failure class (~10% in Land's measurements) does not exist here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .oracle import COLOR_CHARS, COLOR_LEGEND, ascii_grid, normalize_grid, shape_of

Grid = list[list[int]]


@dataclass(frozen=True)
class Pair:
    input: Grid
    output: Grid

    @property
    def input_shape(self) -> tuple[int, int]:
        return tuple(shape_of(self.input))  # type: ignore[return-value]

    @property
    def output_shape(self) -> tuple[int, int]:
        return tuple(shape_of(self.output))  # type: ignore[return-value]


@dataclass(frozen=True)
class Task:
    task_id: str
    train: tuple[Pair, ...]
    test_inputs: tuple[Grid, ...]

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_test(self) -> int:
        return len(self.test_inputs)


@dataclass
class TaskStats:
    """Cheap structural facts about a task, computed once and reused in prompts."""

    n_train: int = 0
    n_test: int = 0
    input_shapes: list[list[int]] = field(default_factory=list)
    output_shapes: list[list[int]] = field(default_factory=list)
    test_shapes: list[list[int]] = field(default_factory=list)
    shape_preserved: bool = True
    colors_in: list[int] = field(default_factory=list)
    colors_out: list[int] = field(default_factory=list)
    same_palette: bool = True
    max_cells: int = 0

    def one_line(self) -> str:
        return (
            f"{self.n_train} train pairs, {self.n_test} test input(s); "
            f"input shapes {self.input_shapes}; output shapes {self.output_shapes}; "
            f"shape_preserved={self.shape_preserved}; "
            f"palette_in={self.colors_in} palette_out={self.colors_out}"
        )


def _colors(grids: Iterable[Grid]) -> list[int]:
    seen: set[int] = set()
    for grid in grids:
        for row in grid:
            seen.update(row)
    return sorted(seen)


def compute_stats(task: Task) -> TaskStats:
    in_shapes = [list(shape_of(p.input)) for p in task.train]
    out_shapes = [list(shape_of(p.output)) for p in task.train]
    test_shapes = [list(shape_of(g)) for g in task.test_inputs]
    colors_in = _colors([p.input for p in task.train])
    colors_out = _colors([p.output for p in task.train])
    all_grids = [p.input for p in task.train] + [p.output for p in task.train] + list(task.test_inputs)
    return TaskStats(
        n_train=task.n_train,
        n_test=task.n_test,
        input_shapes=in_shapes,
        output_shapes=out_shapes,
        test_shapes=test_shapes,
        shape_preserved=in_shapes == out_shapes,
        colors_in=colors_in,
        colors_out=colors_out,
        same_palette=set(colors_in) == set(colors_out),
        max_cells=max((g[0].__len__() * len(g) for g in all_grids if g), default=0),
    )


def to_csv(grid: Grid) -> str:
    return "\n".join(",".join(str(v) for v in row) for row in grid)


def to_ascii(grid: Grid, color_chars: str = COLOR_CHARS) -> str:
    return ascii_grid(grid, color_chars)


def parse_csv(text: str) -> Grid | None:
    """Tolerant CSV/whitespace grid parser (used for fixtures and debugging only)."""
    rows: Grid = []
    for raw in text.splitlines():
        line = raw.strip().strip(",").strip()
        if not line:
            continue
        parts = [p for p in line.replace(" ", ",").split(",") if p != ""]
        try:
            rows.append([int(p) for p in parts])
        except ValueError:
            return None
    if not rows:
        return None
    width = len(rows[0])
    if any(len(r) != width for r in rows):
        return None
    return rows


def task_from_dict(task_id: str, blob: dict[str, Any]) -> Task:
    train = []
    for example in blob.get("train", []) or []:
        grid_in = normalize_grid(example.get("input"))
        grid_out = normalize_grid(example.get("output"))
        if grid_in is None or grid_out is None:
            raise ValueError(f"{task_id}: malformed train pair")
        train.append(Pair(input=grid_in, output=grid_out))
    test_inputs = []
    for example in blob.get("test", []) or []:
        grid_in = normalize_grid(example.get("input"))
        if grid_in is None:
            raise ValueError(f"{task_id}: malformed test input")
        test_inputs.append(grid_in)
    if not train or not test_inputs:
        raise ValueError(f"{task_id}: needs at least one train pair and one test input")
    return Task(task_id=task_id, train=tuple(train), test_inputs=tuple(test_inputs))


def pair_to_payload(pair: Pair) -> dict[str, Any]:
    return {"input": pair.input, "output": pair.output}


def task_to_payload(task: Task) -> dict[str, Any]:
    """JSON payload handed to the sandbox on every tool call."""
    return {
        "task_id": task.task_id,
        "train": [pair_to_payload(p) for p in task.train],
        "test": [list(g) for g in task.test_inputs],
    }


def render_pair_block(index: int, pair: Pair, *, mode: str = "ascii") -> str:
    render = to_ascii if mode == "ascii" else to_csv
    return (
        f"train[{index}] input {pair.input_shape[0]}x{pair.input_shape[1]}:\n"
        f"{render(pair.input)}\n"
        f"train[{index}] output {pair.output_shape[0]}x{pair.output_shape[1]}:\n"
        f"{render(pair.output)}"
    )


def render_test_block(index: int, grid: Grid, *, mode: str = "ascii") -> str:
    render = to_ascii if mode == "ascii" else to_csv
    shape = shape_of(grid)
    return f"test[{index}] input {shape[0]}x{shape[1]}:\n{render(grid)}"


def estimate_tokens(value: Any) -> int:
    """Cheap chars/3 token estimate -- the Duck's ``_estimate_tokens``.

    Used for eviction and budgeting when the server does not report usage.  For
    ASCII grid art (one symbol per cell, long runs of repeats) chars/3 is close;
    for CSV it under-counts slightly, which is the safe direction for a *budget*.
    """
    import json

    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, (len(rendered) + 2) // 3)


def text_tokens(text: str) -> int:
    return max(0, (len(text) + 2) // 3)


def sequence_text(content: Any) -> str:
    """Flatten an OpenAI-style message ``content`` (str or list of parts) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                out.append(str(part.get("text", "")))
        return "\n".join(out)
    return str(content)
