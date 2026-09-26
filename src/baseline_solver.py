"""Heuristic baseline solver (classic 1st-place-2024 style DSL kernel).

Strategy: for each task, try to learn a deterministic input->output
transformation from the train pairs; apply it to the test input.
Falls back to heuristics (identity, color map, symmetry, crop) and
a small enumerate-and-vote over candidate grids for the 2 attempts.
"""
import numpy as np


def to_np(g):
    return np.array(g, dtype=int)


def to_list(a):
    return [[int(c) for c in row] for row in a]


# ---------- candidate generators ----------

def identity(x):
    return [to_list(x)]


def color_map_candidates(x, ys):
    """Global color permutation consistent with train pairs, applied to x."""
    cands = []
    for y in ys:
        xm, ym = to_np(y["input"]), to_np(y["output"])
        if xm.shape != ym.shape:
            return []
        mapping = {}
        ok = True
        for i in range(xm.shape[0]):
            for j in range(xm.shape[1]):
                a, b = int(xm[i, j]), int(ym[i, j])
                if mapping.setdefault(a, b) != b:
                    ok = False
                    break
        if not ok:
            return []
        m = mapping
        out = np.vectorize(lambda v: m.get(int(v), int(v)))(x)
        cands.append(to_list(out))
    return cands


def crop_to_content(x):
    """Crop grid to the bounding box of non-zero cells."""
    nz = np.argwhere(x != 0)
    if nz.size == 0:
        return to_list(x)
    r0, c0 = nz.min(axis=0)
    r1, c1 = nz.max(axis=0)
    return to_list(x[r0:r1 + 1, c0:c1 + 1])


def pad_crop_candidate(x, ys):
    """If train pairs are crops of inputs, crop the test input the same way."""
    outs = []
    for y in ys:
        xm, ym = to_np(y["input"]), to_np(y["output"])
        if xm.shape == ym.shape:
            continue
        # try bounding-box crop
        c = crop_to_content(xm)
        if c and to_np(c).shape == ym.shape and (to_np(c) == ym).all():
            outs.append(crop_to_content(x))
    return outs


def symmetry_candidates(x):
    """Flip / rotate the grid."""
    outs = []
    for fn in (np.flipud, np.fliplr, np.rot90, lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 3)):
        t = fn(x)
        outs.append(to_list(t))
    return outs


def tile_candidates(x, ys):
    """Output is the input tiled/repeated (train-consistent)."""
    outs = []
    for y in ys:
        xm, ym = to_np(y["input"]), to_np(y["output"])
        try:
            rh = ym.shape[0] // xm.shape[0]
            cw = ym.shape[1] // xm.shape[1]
        except ZeroDivisionError:
            continue
        if xm.shape[0] * rh == ym.shape[0] and xm.shape[1] * cw == ym.shape[1]:
            if rh > 0 and cw > 0 and (np.tile(xm, (rh, cw)) == ym).all():
                outs.append(to_list(np.tile(x, (rh, cw))))
    return outs


def fill_bg_candidates(x, ys):
    """Keep grid size, replace one color with another (train-consistent)."""
    outs = []
    for y in ys:
        xm, ym = to_np(y["input"]), to_np(y["output"])
        if xm.shape != ym.shape:
            continue
        diff = xm != ym
        if not diff.any():
            continue
        src_vals = set(xm[diff].tolist())
        dst_vals = set(ym[diff].tolist())
        if len(src_vals) == 1 and len(dst_vals) == 1:
            s, d = src_vals.pop(), dst_vals.pop()
            outs.append(to_list(np.where(x == s, d, x)))
    return outs


# ---------- solver core ----------

def solve_task(task):
    """Return a ranked list of candidate output grids for the test input."""
    train = task.get("train", [])
    test = task.get("test", [])
    if not test:
        return []
    x = to_np(test[0]["input"])

    cands = []
    seen = set()

    def add(cl):
        for c in cl:
            key = tuple(map(tuple, c)) if isinstance(c, list) else None
            if key is not None and key not in seen:
                seen.add(key)
                cands.append(c)

    # 1. identity (output == input) if it explains all train pairs
    if all(to_np(p["input"]).shape == to_np(p["output"]).shape
           and (to_np(p["input"]) == to_np(p["output"])).all() for p in train):
        add(identity(x))

    # 2. exact train-pair match: if test input equals a train input, reuse its output
    for p in train:
        if to_np(p["input"]).shape == x.shape and (to_np(p["input"]) == x).all():
            add([p["output"]])

    add(color_map_candidates(x, train))
    add(fill_bg_candidates(x, train))
    add(pad_crop_candidate(x, train))
    add(tile_candidates(x, train))
    if all(to_np(p["input"]).shape == to_np(p["output"]).shape for p in train):
        add(symmetry_candidates(x))

    # rank: identity-style first, then heuristics
    return cands if cands else [to_list(x)]


def heuristic_solver(task_id, task):
    cands = solve_task(task)
    x0 = task["test"][0]["input"]
    a1 = cands[0] if cands else x0
    a2 = cands[1] if len(cands) > 1 else x0
    outs = []
    for t in task.get("test", []):
        outs.append(a1 if t["input"] == task["test"][0]["input"] else t["input"])
    # per test-input guesses (this simple version handles the common 1-test case)
    return [{"attempt_1": a1, "attempt_2": a2}
            for _ in task.get("test", [])]


def heuristic_attempts(task_id, task):
    """Return list (per test input) of [attempt1_grid, attempt2_grid]."""
    per_test = []
    for t in task.get("test", []):
        x = t["input"]
        sub_task = {"train": task.get("train", []), "test": [t]}
        cands = solve_task(sub_task)
        per_test.append([cands[0] if cands else x,
                         cands[1] if len(cands) > 1 else x])
    return per_test


def solve_task_attempts(task_id, task):
    """Solver API for make_submission: list of 2 attempts per test input."""
    return heuristic_attempts(task_id, task)
