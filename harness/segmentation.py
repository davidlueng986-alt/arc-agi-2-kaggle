"""Connected-component segmentation of one ARC-AGI-2 grid.

Adapted from The Duck's ``inference/utils/segmentation.py`` (MIT, Tufa Labs) --
the one perception tool that survived their ablations, built because a common
failure was the model printing the whole board and polluting its own context.

Self-contained on purpose: standard library only, no project imports, no
``from __future__``, so the source can be spliced verbatim into the REPL sandbox
bootstrap where project packages are not importable.

Differences from the Duck version, all deliberate:

* ARC-AGI-2 grids are 10-colour and at most 30x30 (ARC-AGI-3 boards are fixed
  64x64 with 16 colours), so ``color_chars`` defaults to the 10-colour map.
* Adds ``bbox`` and ``cells`` (capped) per node: in ARC-AGI-2 the model writes
  array code, so absolute coordinates are useful in a way they are not when the
  only available act is ``MOUSE(row, col)`` on a 64x64 board.
* Adds a hard component cap.  The Duck's containment pass is O(n_components *
  H * W); on a pathological 30x30 checkerboard that is 900 * 900 cell visits.
  Above ``MAX_NODES`` we skip containment and say so, instead of burning the
  30s tool budget.
"""

import hashlib

ORTH = ((-1, 0), (1, 0), (0, -1), (0, 1))
# clockwise Moore-neighbour offsets, starting at NW
CW = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))
CW_INDEX = {off: i for i, off in enumerate(CW)}

DEFAULT_COLOR_CHARS = ".brgysmocd"
MAX_NODES = 256
MAX_CELLS_PER_NODE = 24


def _trace_outer_contour(cells, start):
    """Moore-neighbour trace of a 4-connected component's outer perimeter, clockwise."""
    if len(cells) == 1:
        return [start]

    contour = [start]
    b = start
    prev = (start[0], start[1] - 1)
    second = None
    for _ in range(8 * len(cells) + 16):
        idx = CW_INDEX[(prev[0] - b[0], prev[1] - b[1])]
        nxt = None
        new_prev = prev
        for k in range(1, 9):
            off = CW[(idx + k) % 8]
            cand = (b[0] + off[0], b[1] + off[1])
            if cand in cells:
                nxt = cand
                back = CW[(idx + k - 1) % 8]
                new_prev = (b[0] + back[0], b[1] + back[1])
                break
        if nxt is None:
            break
        if second is None:
            second = nxt
        elif b == start and nxt == second:  # Jacob's stopping criterion
            break
        contour.append(nxt)
        prev, b = new_prev, nxt

    if len(contour) > 1 and contour[-1] == contour[0]:
        contour.pop()
    return contour


def _corner_points(contour):
    """Reduce a traced contour loop to only the points where its direction changes."""
    if len(contour) <= 2:
        return list(contour)
    m = len(contour)
    corners = []
    for i in range(m):
        prev, cur, nxt = contour[i - 1], contour[i], contour[(i + 1) % m]
        d_in = (cur[0] - prev[0], cur[1] - prev[1])
        d_out = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d_in != d_out:
            corners.append(cur)
    return corners


def _object_hash(cells, color):
    """Translation-invariant signature: colour plus shape normalised to a top-left
    origin.  Equal hashes mean the same object regardless of where it sits, so a
    shape can be tracked across grids or duplicated shapes spotted in one grid."""
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return hashlib.sha1(repr((color, norm)).encode()).hexdigest()[:12]


def segment_grid(grid, color_chars=DEFAULT_COLOR_CHARS):
    """Segment one grid into 4-connected same-value components.

    Returns ``{"nodes": [...], "adjacency_list": [[i, j], ...], ...}``.  Each node
    carries ``id`` (reading-order index), ``color`` (integer), ``char`` (symbol),
    ``hash``, ``pixels``, ``bbox`` ``[r0, c0, r1, c1]`` inclusive, ``cells``
    (capped), ``boundary`` (clockwise corner points) and ``children`` (ids of the
    components it is the innermost encloser of).
    """
    height = len(grid)
    width = len(grid[0]) if height else 0
    if height == 0 or width == 0:
        return {
            "nodes": [],
            "adjacency_list": [],
            "shape": [0, 0],
            "truncated": False,
            "containment_computed": False,
        }

    comp_id = [[-1] * width for _ in range(height)]
    components = []
    for sr in range(height):
        for sc in range(width):
            if comp_id[sr][sc] != -1:
                continue
            value = grid[sr][sc]
            cid = len(components)
            cells = set()
            stack = [(sr, sc)]
            comp_id[sr][sc] = cid
            while stack:
                r, c = stack.pop()
                cells.add((r, c))
                for dr, dc in ORTH:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and comp_id[nr][nc] == -1
                        and grid[nr][nc] == value
                    ):
                        comp_id[nr][nc] = cid
                        stack.append((nr, nc))
            components.append({"value": int(value), "cells": cells, "start": (sr, sc)})

    n = len(components)
    truncated = n > MAX_NODES
    counted = min(n, MAX_NODES)

    adj_pairs = set()
    for r in range(height):
        for c in range(width):
            cid = comp_id[r][c]
            if r + 1 < height and comp_id[r + 1][c] != cid:
                other = comp_id[r + 1][c]
                adj_pairs.add((min(cid, other), max(cid, other)))
            if c + 1 < width and comp_id[r][c + 1] != cid:
                other = comp_id[r][c + 1]
                adj_pairs.add((min(cid, other), max(cid, other)))
    adjacency_list = sorted([a, b] for a, b in adj_pairs if a < counted and b < counted)

    children = [[] for _ in range(n)]
    containment_computed = False
    if not truncated:
        # For each component b, flood-fill its complement inward from the border;
        # any component never reached is enclosed by b.  Parent = innermost encloser.
        enclosers = [set() for _ in range(n)]
        for b in range(n):
            reached = [[False] * width for _ in range(height)]
            stack = []
            for r in range(height):
                for c in (0, width - 1):
                    if comp_id[r][c] != b and not reached[r][c]:
                        reached[r][c] = True
                        stack.append((r, c))
            for c in range(width):
                for r in (0, height - 1):
                    if comp_id[r][c] != b and not reached[r][c]:
                        reached[r][c] = True
                        stack.append((r, c))
            while stack:
                r, c = stack.pop()
                for dr, dc in ORTH:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and not reached[nr][nc]
                        and comp_id[nr][nc] != b
                    ):
                        reached[nr][nc] = True
                        stack.append((nr, nc))
            for a in range(n):
                if a == b:
                    continue
                ar, ac = components[a]["start"]
                if not reached[ar][ac]:
                    enclosers[a].add(b)
        for a in range(n):
            if enclosers[a]:
                parent = max(enclosers[a], key=lambda e: (len(enclosers[e]), -e))
                children[parent].append(a)
        for child_list in children:
            child_list.sort()
        containment_computed = True

    ncolors = len(color_chars)
    nodes = []
    for cid in range(counted):
        comp = components[cid]
        cells = comp["cells"]
        value = comp["value"]
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        ordered = sorted(cells)
        boundary = _corner_points(_trace_outer_contour(cells, comp["start"]))
        nodes.append(
            {
                "id": cid,
                "color": value,
                "char": color_chars[max(0, min(ncolors - 1, value))],
                "hash": _object_hash(cells, value),
                "pixels": len(cells),
                "bbox": [min(rows), min(cols), max(rows), max(cols)],
                "cells": [[r, c] for r, c in ordered[:MAX_CELLS_PER_NODE]],
                "cells_truncated": len(cells) > MAX_CELLS_PER_NODE,
                "boundary": [[r, c] for r, c in boundary],
                "children": children[cid],
            }
        )

    return {
        "nodes": nodes,
        "adjacency_list": adjacency_list,
        "shape": [height, width],
        "n_components": n,
        "truncated": truncated,
        "containment_computed": containment_computed,
    }


def segment_summary(grid, color_chars=DEFAULT_COLOR_CHARS, max_nodes=24):
    """One-line-per-object digest -- what you actually want in a tool result.

    The full ``segment_grid`` payload for a busy 30x30 grid is thousands of
    tokens; this is the compact form the model should print.
    """
    seg = segment_grid(grid, color_chars)
    lines = []
    for node in seg["nodes"][:max_nodes]:
        r0, c0, r1, c1 = node["bbox"]
        lines.append(
            "#{id} {char}{pixels}px {w}x{h}@({r0},{c0}) h={hash}{kids}".format(
                id=node["id"],
                char=node["char"],
                pixels=node["pixels"],
                w=c1 - c0 + 1,
                h=r1 - r0 + 1,
                r0=r0,
                c0=c0,
                hash=node["hash"][:8],
                kids=(" kids=%s" % node["children"]) if node["children"] else "",
            )
        )
    if len(seg["nodes"]) > max_nodes:
        lines.append("... %d more components" % (len(seg["nodes"]) - max_nodes))
    header = "shape=%dx%d components=%d%s" % (
        seg["shape"][0],
        seg["shape"][1],
        seg["n_components"],
        "" if seg["containment_computed"] else " (containment skipped: too many)",
    )
    return header + "\n" + "\n".join(lines)
