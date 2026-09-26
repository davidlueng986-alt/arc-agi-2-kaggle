"""Tolerant extraction of ANSWER blocks from free-form thinking-model output.

Structured output (`guided_json` / `response_format`) is unreliable with vLLM
thinking mode (vLLM#35700) and the reasoning parser splits text into
`reasoning_content` and `content`, so we generate free-form and parse tolerantly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .grids import Grid, parse_grid_blocks

_MARKER_RE = re.compile(
    r"(?:^|\n)[^\n]{0,40}?\b(?:ANSWER|OUTPUT|RESULT|SOLUTION)\s*#?\s*(\d{1,2})\s*\**\s*[:\-]",
    re.IGNORECASE,
)


@dataclass
class AnswerParse:
    grids: Dict[int, Grid] = field(default_factory=dict)
    method: str = "none"
    n_blocks: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.grids)

    def indices(self) -> List[int]:
        return sorted(self.grids.keys())

    def as_list(self, n_test: int) -> List[Optional[Grid]]:
        return [self.grids.get(i + 1) for i in range(n_test)]


def _marker_positions(text: str) -> List[Tuple[int, int, int]]:
    """Return (index_label, slice_start, slice_end) for every answer marker."""
    hits = [(m.group(0), int(m.group(1)), m.end()) for m in _MARKER_RE.finditer(text)]
    out: List[Tuple[int, int, int]] = []
    for pos, (_full, label, end) in enumerate(hits):
        nxt = hits[pos + 1][2] if pos + 1 < len(hits) else len(text)
        out.append((label, end, max(end, nxt)))
    return out


def parse_answers(text: str, n_test: int) -> AnswerParse:
    if not text:
        return AnswerParse(method="empty")

    markers = _marker_positions(text)
    if markers:
        by_label: Dict[int, List[Tuple[int, int]]] = {}
        for label, start, end in markers:
            by_label.setdefault(label, []).append((start, end))

        grids: Dict[int, Grid] = {}
        for label in sorted(by_label):
            if label < 1 or label > n_test:
                continue
            for start, end in reversed(by_label[label]):
                blocks = parse_grid_blocks(text[start:end], allow_single_cell=True)
                if blocks:
                    grids[label] = blocks[0]
                    break
        if grids:
            return AnswerParse(
                grids=grids, method="marker", n_blocks=len(parse_grid_blocks(text))
            )

    blocks = parse_grid_blocks(text)
    if not blocks:
        return AnswerParse(method="none")

    if n_test == 1:
        return AnswerParse(grids={1: blocks[-1]}, method="last-block", n_blocks=len(blocks))

    if len(blocks) == n_test:
        return AnswerParse(
            grids={i + 1: g for i, g in enumerate(blocks)},
            method="ordered-blocks",
            n_blocks=len(blocks),
        )

    if len(blocks) >= n_test:
        tail = blocks[-n_test:]
        return AnswerParse(
            grids={i + 1: g for i, g in enumerate(tail)},
            method="tail-blocks",
            n_blocks=len(blocks),
        )

    return AnswerParse(grids={1: blocks[-1]}, method="last-block", n_blocks=len(blocks))


def merge_reasoning_and_content(reasoning: str, content: str) -> str:
    """Build the search space for the parser.

    With `--reasoning-parser qwen3` the visible answer lands in `content`, but a
    truncated generation (`finish_reason == "length"`) can leave the answer inside
    the reasoning stream, so both are searched, content first.
    """
    parts = [p for p in (content or "", reasoning or "") if p]
    return "\n\n".join(parts)
