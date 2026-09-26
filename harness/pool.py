"""Program pool: dedupe, behavior grouping, provenance tracking."""
from __future__ import annotations

import hashlib
import re
from typing import Optional, Sequence

import numpy as np

from .grid import grid_key
from .types import Candidate, Grid, ProgramEval


def source_hash(code: str) -> str:
    """Normalized-source identity: whitespace/comment-insensitive."""
    norm = re.sub(r"\s+", " ", re.sub(r"#.*", "", code)).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def behavior_key(eval_result: ProgramEval, test_outputs: Sequence[Optional[Grid]] = ()) -> tuple:
    """Two programs with the same behavior key are indistinguishable on all
    data we have (train outcomes incl. failures + test outputs)."""
    parts = []
    for ps in eval_result.per_pair:
        if ps.exact:
            parts.append(("exact",))
        elif ps.shape_ok:
            parts.append(("soft", round(ps.soft, 6)))
        else:
            parts.append(("err", (ps.error or "")[:40]))
    test_part = tuple(grid_key(o) if o is not None else None for o in test_outputs)
    return (tuple(parts), test_part)


class ProgramPool:
    """Keeps every behaviorally distinct program; groups by observable behavior."""

    def __init__(self) -> None:
        self.candidates: list[Candidate] = []
        self._by_source: dict[str, int] = {}

    def add(
        self,
        code: str,
        source: str,
        eval_result: ProgramEval,
        test_outputs: Optional[list[Optional[Grid]]] = None,
    ) -> tuple[int, bool]:
        """Returns (index, is_new). Source-identical duplicates are dropped;
        behavior duplicates are KEPT (independent corroborations matter for
        cluster support)."""
        sh = source_hash(code)
        if sh in self._by_source:
            return self._by_source[sh], False
        cand = Candidate(
            code=code,
            source=source,
            eval=eval_result,
            test_outputs=list(test_outputs) if test_outputs is not None else [],
        )
        idx = len(self.candidates)
        self.candidates.append(cand)
        self._by_source[sh] = idx
        return idx, True

    # -- views --------------------------------------------------------------

    def verified(self) -> list[int]:
        return [i for i, c in enumerate(self.candidates) if c.verified]

    def unverified(self) -> list[int]:
        return [i for i, c in enumerate(self.candidates) if not c.verified]

    def verified_behaviors(self) -> dict[tuple, list[int]]:
        """Group verified candidates by their test-output behavior.

        Candidates missing any test output form the None-keyed group and are
        excluded from attempt selection (but not from the pool).
        """
        groups: dict[tuple, list[int]] = {}
        for i in self.verified():
            c = self.candidates[i]
            if any(o is None for o in c.test_outputs):
                key = ("incomplete",)
            else:
                key = tuple(grid_key(o) for o in c.test_outputs)
            groups.setdefault(key, []).append(i)
        return groups

    def best_partial(self, n: int = 5) -> list[int]:
        """Top-n unverified candidates by soft score (never discard failures)."""
        unv = list(self.unverified())
        unv.sort(key=lambda i: (-self.candidates[i].soft_mean, len(self.candidates[i].code)))
        return unv[:n]

    def __len__(self) -> int:
        return len(self.candidates)
