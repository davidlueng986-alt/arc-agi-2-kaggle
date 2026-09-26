"""Grid utilities: parsing/formatting, the D4 symmetry group, color permutations,
shape-law inference, data-consistency checks and probe perturbations.

Everything here is pure numpy/stdlib so the whole disambiguation layer is
testable offline (no GPU, no LLM).
"""
from __future__ import annotations

import itertools
from collections import Counter, deque
from fractions import Fraction
from typing import Iterable, Optional, Sequence

import numpy as np

from .types import Grid

# ---------------------------------------------------------------------------
# basic grid handling
# ---------------------------------------------------------------------------


def to_grid(x: Sequence[Sequence[int]] | np.ndarray) -> Grid:
    """Convert nested lists / arrays to a 2-D int64 numpy grid."""
    arr = np.asarray(x)
    if arr.ndim != 2:
        if arr.ndim == 1 and arr.size == 0:
            arr = arr.reshape(0, 0)
        else:
            raise ValueError(f"expected 2-D grid, got shape {arr.shape}")
    if arr.size and (arr.min() < 0 or arr.max() > 9):
        raise ValueError("grid values must be in 0..9")
    return arr.astype(np.int64, copy=False)


def grid_key(g: Grid) -> tuple:
    """Hashable canonical key for a grid."""
    a = np.asarray(g)
    return (a.shape[0], a.shape[1], a.tobytes())


def grids_equal(a: Optional[Grid], b: Optional[Grid]) -> bool:
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def format_csv(g: Grid) -> str:
    """CSV rendering (benchmarked best of 9 formats; a bad format costs ~10%)."""
    return "\n".join(",".join(str(int(c)) for c in row) for row in np.asarray(g))


def parse_grid_lenient(text: str) -> Optional[Grid]:
    """Tolerantly parse a grid out of model output text.

    Accepts CSV rows, space-separated digits, bracketed JSON-ish rows. Returns
    the LAST plausible rectangular block (models tend to restate the final
    answer last), or None.
    """
    import json
    import re

    text = text.strip()
    # try whole-text JSON first
    for cand in (text, re.sub(r"^[^{\[]*", "", text)):
        try:
            obj = json.loads(cand)
            if isinstance(obj, list) and obj and isinstance(obj[0], list):
                return to_grid(obj)
        except Exception:
            pass

    # first balanced-looking nested JSON array anywhere in the text
    m = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if (
                isinstance(obj, list)
                and obj
                and all(isinstance(r, list) for r in obj)
                and len({len(r) for r in obj}) == 1
                and all(isinstance(v, int) and 0 <= v <= 9 for r in obj for v in r)
            ):
                return to_grid(obj)
        except Exception:
            pass

    rows: list[list[int]] = []
    blocks: list[list[list[int]]] = []
    for line in text.splitlines():
        s = line.strip().strip("`").strip()
        s = s.replace("[", " ").replace("]", " ").replace(",", " ")
        toks = s.split()
        if toks and all(re.fullmatch(r"\d", t) for t in toks):
            rows.append([int(t) for t in toks])
        else:
            if rows:
                blocks.append(rows)
                rows = []
    if rows:
        blocks.append(rows)

    # keep rectangular blocks; return the last one
    for blk in reversed(blocks):
        widths = {len(r) for r in blk}
        if len(widths) == 1 and blk:
            return to_grid(blk)
    return None


def background_color(g: Grid) -> int:
    vals, counts = np.unique(np.asarray(g), return_counts=True)
    return int(vals[int(np.argmax(counts))])


# ---------------------------------------------------------------------------
# D4 symmetry group
# ---------------------------------------------------------------------------
# An element is (k, f): apply horizontal-axis flip (up-down) first if f, then
# rotate 90 deg counter-clockwise k times. The 8 elements form the dihedral
# group D4 acting on rectangular grids.

D4_ELEMENTS: list[tuple[int, bool]] = [
    (0, False),  # identity
    (1, False),  # rot90 CCW
    (2, False),  # rot180
    (3, False),  # rot270 CCW (= rot90 CW)
    (0, True),   # flip up-down
    (1, True),   # flip then rot90 (= transpose)
    (2, True),   # flip then rot180 (= flip left-right)
    (3, True),   # flip then rot270 (= anti-transpose)
]

D4_NAMES = ["id", "rot90", "rot180", "rot270", "flipud", "transpose", "fliplr", "antitranspose"]


def d4_name(elem: tuple[int, bool]) -> str:
    return D4_NAMES[D4_ELEMENTS.index(elem)]


def apply_d4(g: Grid, elem: tuple[int, bool]) -> Grid:
    k, f = elem
    x = np.asarray(g)
    if f:
        x = x[::-1, :]
    if k:
        x = np.rot90(x, k)
    return np.ascontiguousarray(x)


def apply_d4_inv(g: Grid, elem: tuple[int, bool]) -> Grid:
    k, f = elem
    x = np.asarray(g)
    if k:
        x = np.rot90(x, -k)
    if f:
        x = x[::-1, :]
    return np.ascontiguousarray(x)


def compose_d4(a: tuple[int, bool], b: tuple[int, bool]) -> tuple[int, bool]:
    """Return c with apply(apply(x,b),a) == apply(x,c)."""
    ka, fa = a
    kb, fb = b
    # rot(k) o flip = flip o rot(-k);  (ka,fa)*(kb,fb)
    k = (ka + (-kb if fa else kb)) % 4
    return (k, fa ^ fb)


def invert_d4(elem: tuple[int, bool]) -> tuple[int, bool]:
    k, f = elem
    if not f:
        return ((-k) % 4, False)
    return (k, True)  # flip o rot(k) inverted = rot(-k) o flip = flip o rot(k)


# ---------------------------------------------------------------------------
# color permutations
# ---------------------------------------------------------------------------


def apply_color_perm(g: Grid, perm: dict[int, int]) -> Grid:
    x = np.asarray(g).copy()
    for a, b in perm.items():
        x[np.asarray(g) == a] = b
    return x


def color_profile(pairs: list[tuple[Grid, Grid]], color: int) -> tuple:
    """Usage signature of a color across train pairs: sorted (count_in, count_out)."""
    prof = []
    for xi, yi in pairs:
        prof.append((int(np.sum(xi == color)), int(np.sum(yi == color))))
    return tuple(sorted(prof))


def candidate_color_perms(
    pairs: list[tuple[Grid, Grid]], max_perms: int = 6
) -> list[dict[int, int]]:
    """Color transpositions that are plausible task symmetries.

    A transposition (c1 c2) is *candidate* when both colors have identical
    usage profiles across train pairs (they are interchangeable as far as the
    data can tell) and the permuted relation set stays consistent with the
    original one (checked by the caller via relations_consistent). Compositions
    of disjoint transpositions are added when cheap.
    """
    colors = set()
    for xi, yi in pairs:
        colors.update(np.unique(xi).tolist())
        colors.update(np.unique(yi).tolist())
    colors = sorted(int(c) for c in colors)
    profiles = {c: color_profile(pairs, c) for c in colors}

    groups: dict[tuple, list[int]] = {}
    for c, p in profiles.items():
        groups.setdefault(p, []).append(c)

    perms: list[dict[int, int]] = []
    for p, cs in groups.items():
        if len(cs) < 2:
            continue
        if all(sum(t) == 0 for t in p):
            # colors that never appear carry no information
            continue
        cs = sorted(cs)
        for a, b in itertools.combinations(cs, 2):
            perms.append({a: b, b: a})
            if len(perms) >= max_perms:
                return perms
        # 3-cycles for groups of >=3 interchangeable colors
        if len(cs) >= 3:
            a, b, c = cs[:3]
            perms.append({a: b, b: c, c: a})
            if len(perms) >= max_perms:
                return perms
    return perms[:max_perms]


# ---------------------------------------------------------------------------
# data-level consistency of pseudo-labeled (augmented) relations
# ---------------------------------------------------------------------------


def relations_consistent(relations: Iterable[tuple[Grid, Grid]]) -> bool:
    """False when the same input is related to two different outputs."""
    seen: dict[tuple, tuple] = {}
    for x, y in relations:
        kx = grid_key(x)
        ky = grid_key(y)
        if kx in seen and seen[kx] != ky:
            return False
        seen[kx] = ky
    return True


def plausible_d4(pairs: list[tuple[Grid, Grid]]) -> list[tuple[int, bool]]:
    """D4 elements whose pseudo-labels do not contradict the train data.

    Element s survives when {(s.x_i -> s.y_i)} unioned with the original pairs
    is a consistent relation. This is a *data-level* screen only; whether the
    true rule actually commutes with s is decided later by pool-majority
    resolvability (see disambiguate.py).
    """
    out = []
    for elem in D4_ELEMENTS:
        rel = list(pairs) + [(apply_d4(x, elem), apply_d4(y, elem)) for x, y in pairs]
        if relations_consistent(rel):
            out.append(elem)
    return out


def plausible_color_perms(
    pairs: list[tuple[Grid, Grid]], max_perms: int = 6
) -> list[dict[int, int]]:
    out = []
    for perm in candidate_color_perms(pairs, max_perms=max_perms * 2):
        rel = list(pairs) + [
            (apply_color_perm(x, perm), apply_color_perm(y, perm)) for x, y in pairs
        ]
        if relations_consistent(rel):
            out.append(perm)
        if len(out) >= max_perms:
            break
    return out


# ---------------------------------------------------------------------------
# shape laws
# ---------------------------------------------------------------------------


class ShapeLaw:
    """Predicts output shape from input shape. None => no prediction."""

    def __init__(self, name: str, fn):
        self.name = name
        self._fn = fn

    def predict(self, h: int, w: int) -> Optional[tuple[int, int]]:
        return self._fn(h, w)

    def __repr__(self) -> str:  # pragma: no cover
        return f"ShapeLaw({self.name})"


def _affine_laws(pairs: list[tuple[Grid, Grid]]) -> list[ShapeLaw]:
    """Enumerate small exact affine laws out_h = a*h + b*w + c (and for w)."""
    cand_vals = [Fraction(v) for v in (0, 0.5, 1, 1.5, 2, 3, -0.5, -1, -2)]
    laws_h: list[tuple[Fraction, Fraction, Fraction]] = []
    laws_w: list[tuple[Fraction, Fraction, Fraction]] = []
    shapes = [(x.shape[0], x.shape[1], y.shape[0], y.shape[1]) for x, y in pairs]
    for a in cand_vals:
        for b in cand_vals:
            for c in (Fraction(v) for v in range(-6, 7)):
                if all(a * h + b * w + c == oh for h, w, oh, _ in shapes):
                    laws_h.append((a, b, c))
                if all(a * h + b * w + c == ow for h, w, _, ow in shapes):
                    laws_w.append((a, b, c))
    laws = []
    for (ah, bh, ch) in laws_h:
        for (aw, bw, cw) in laws_w:
            def fn(h, w, _h=(ah, bh, ch), _w=(aw, bw, cw)):
                oh = _h[0] * h + _h[1] * w + _h[2]
                ow = _w[0] * h + _w[1] * w + _w[2]
                if oh <= 0 or ow <= 0 or oh.denominator != 1 or ow.denominator != 1:
                    return None
                if oh > 64 or ow > 64:
                    return None
                return (int(oh), int(ow))

            laws.append(ShapeLaw(f"affine(h:{ah},{bh},{ch};w:{aw},{bw},{cw})", fn))
    return laws


def consistent_shape_laws(pairs: list[tuple[Grid, Grid]]) -> list[ShapeLaw]:
    """All shape laws from a small interpretable family that fit every train pair.

    Family: identity, constant, rational per-dimension scale, affine. The
    violation test only fires when *every* consistent law disagrees with a
    program's output shape, which keeps the signal sound on small data.
    """
    if not pairs:
        return []
    laws: list[ShapeLaw] = []
    in_shapes = {(x.shape[0], x.shape[1]) for x, _ in pairs}
    out_shapes = {(y.shape[0], y.shape[1]) for _, y in pairs}

    if all(x.shape == y.shape for x, y in pairs):
        laws.append(ShapeLaw("identity", lambda h, w: (h, w)))
    if len(out_shapes) == 1:
        (oh, ow), = out_shapes
        laws.append(ShapeLaw(f"const({oh},{ow})", lambda h, w, _o=(oh, ow): _o))
    scales = []
    for qh in (Fraction(1), Fraction(1, 2), Fraction(2), Fraction(1, 3), Fraction(3),
                Fraction(2, 3), Fraction(3, 2), Fraction(1, 4), Fraction(4)):
        for qw in (Fraction(1), Fraction(1, 2), Fraction(2), Fraction(1, 3), Fraction(3),
                   Fraction(2, 3), Fraction(3, 2), Fraction(1, 4), Fraction(4)):
            if qh == 1 and qw == 1:
                continue  # covered by SameLaw
            ok = True
            for x, y in pairs:
                h, w = x.shape
                if h * qh != y.shape[0] or w * qw != y.shape[1]:
                    ok = False
                    break
            if ok:
                scales.append((qh, qw))
    if scales and len(in_shapes) >= 2:
        for qh, qw in scales[:4]:
            laws.append(
                ShapeLaw(
                    f"scale({qh},{qw})",
                    lambda h, w, _q=(qh, qw): (
                        (int(h * _q[0]), int(w * _q[1]))
                        if (h * _q[0]).denominator == 1 and (w * _q[1]).denominator == 1
                        and 0 < h * _q[0] <= 64 and 0 < w * _q[1] <= 64
                        else None
                    ),
                )
            )
    if len({(x.shape[0], x.shape[1]) for x, _ in pairs}) >= 3:
        laws.extend(_affine_laws(pairs))
    return laws


def shape_law_violation(
    laws: Sequence[ShapeLaw], in_shape: tuple[int, int], out_shape: Optional[tuple[int, int]]
) -> bool:
    """True when out_shape contradicts EVERY consistent law (unanimous vote)."""
    if out_shape is None or not laws:
        return False
    preds = []
    for law in laws:
        p = law.predict(*in_shape)
        if p is not None:
            preds.append(tuple(p))
    if not preds:
        return False
    return all(p != tuple(out_shape) for p in preds)


# ---------------------------------------------------------------------------
# probes / perturbations (behavioral clustering inputs)
# ---------------------------------------------------------------------------


def pad_border(g: Grid, color: Optional[int] = None, width: int = 1) -> Grid:
    x = np.asarray(g)
    c = background_color(x) if color is None else color
    return np.pad(x, width, mode="constant", constant_values=c).astype(x.dtype, copy=False)


def crop_border(g: Grid, width: int = 1) -> Optional[Grid]:
    x = np.asarray(g)
    if x.shape[0] <= 2 * width or x.shape[1] <= 2 * width:
        return None
    return x[width:-width, width:-width]


def perturb_cells(g: Grid, rng: np.random.Generator, n_cells: int = 1) -> Grid:
    """Recolor n random cells to a different random color."""
    x = np.asarray(g).copy()
    h, w = x.shape
    for _ in range(n_cells):
        r, c = int(rng.integers(0, h)), int(rng.integers(0, w))
        choices = [v for v in range(10) if v != x[r, c]]
        x[r, c] = int(rng.choice(choices))
    return x


# ---------------------------------------------------------------------------
# tiny structural analysis used in prompts (Gre24-style extra representation)
# ---------------------------------------------------------------------------


def components(g: Grid, include_background: bool = False) -> list[dict]:
    """4-connected components as dicts {color, size, bbox}."""
    x = np.asarray(g)
    h, w = x.shape
    seen = np.zeros_like(x, dtype=bool)
    bg = background_color(x)
    out = []
    for i in range(h):
        for j in range(w):
            if seen[i, j] or (x[i, j] == bg and not include_background):
                continue
            color = int(x[i, j])
            q = deque([(i, j)])
            seen[i, j] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and x[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            out.append(
                {
                    "color": color,
                    "size": len(cells),
                    "bbox": (min(rs), min(cs), max(rs), max(cs)),
                }
            )
    return out


def analysis_block(pairs: list[tuple[Grid, Grid]], test_inputs: list[Grid]) -> str:
    """Compact per-grid structural summary appended to synthesis prompts."""
    lines = []
    for idx, (x, y) in enumerate(pairs):
        for tag, g in (("in", x), ("out", y)):
            comps = components(g)
            palette = sorted(set(int(v) for v in np.unique(g)))
            lines.append(
                f"pair{idx + 1}.{tag}: {g.shape[0]}x{g.shape[1]} bg={background_color(g)} "
                f"palette={palette} objects={len(comps)} "
                f"obj_sizes={sorted((c['size'] for c in comps), reverse=True)[:6]}"
            )
    for idx, x in enumerate(test_inputs):
        comps = components(x)
        palette = sorted(set(int(v) for v in np.unique(x)))
        lines.append(
            f"test{idx + 1}.in: {x.shape[0]}x{x.shape[1]} bg={background_color(x)} "
            f"palette={palette} objects={len(comps)} "
            f"obj_sizes={sorted((c['size'] for c in comps), reverse=True)[:6]}"
        )
    return "\n".join(lines)
