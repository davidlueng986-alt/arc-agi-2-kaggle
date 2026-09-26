import numpy as np
import pytest

from harness.grid import (
    D4_ELEMENTS,
    analysis_block,
    apply_color_perm,
    apply_d4,
    apply_d4_inv,
    background_color,
    candidate_color_perms,
    components,
    compose_d4,
    consistent_shape_laws,
    crop_border,
    format_csv,
    grid_key,
    grids_equal,
    invert_d4,
    pad_border,
    parse_grid_lenient,
    perturb_cells,
    plausible_color_perms,
    plausible_d4,
    relations_consistent,
    shape_law_violation,
    to_grid,
)

RNG = np.random.default_rng(0)


def rand_grid(h, w, k=4, seed=None):
    rng = np.random.default_rng(seed)
    return rng.integers(0, k, size=(h, w)).astype(np.int64)


class TestD4Group:
    def test_eight_elements(self):
        assert len(D4_ELEMENTS) == 8
        assert len(set(D4_ELEMENTS)) == 8

    @pytest.mark.parametrize("elem", D4_ELEMENTS)
    def test_inverse_roundtrip_square(self, elem):
        g = rand_grid(5, 5, seed=1)
        assert np.array_equal(apply_d4_inv(apply_d4(g, elem), elem), g)
        assert np.array_equal(apply_d4(apply_d4_inv(g, elem), elem), g)

    @pytest.mark.parametrize("elem", D4_ELEMENTS)
    def test_inverse_roundtrip_rect(self, elem):
        g = rand_grid(3, 7, seed=2)
        assert np.array_equal(apply_d4_inv(apply_d4(g, elem), elem), g)

    @pytest.mark.parametrize("a", D4_ELEMENTS)
    @pytest.mark.parametrize("b", D4_ELEMENTS)
    def test_closure(self, a, b):
        g = rand_grid(4, 4, seed=3)
        c = compose_d4(a, b)
        assert c in D4_ELEMENTS
        assert np.array_equal(apply_d4(apply_d4(g, b), a), apply_d4(g, c))

    @pytest.mark.parametrize("elem", D4_ELEMENTS)
    def test_invert_matches_apply_inv(self, elem):
        g = rand_grid(5, 5, seed=4)
        inv = invert_d4(elem)
        assert np.array_equal(apply_d4(g, inv), apply_d4_inv(g, elem))

    def test_four_applications_identity(self):
        g = rand_grid(4, 4, seed=5)
        x = g
        for _ in range(4):
            x = apply_d4(x, (1, False))
        assert np.array_equal(x, g)


class TestColorPerms:
    def test_apply_transposition(self):
        g = to_grid([[0, 2], [3, 2]])
        out = apply_color_perm(g, {2: 3, 3: 2})
        assert np.array_equal(out, to_grid([[0, 3], [2, 3]]))

    def test_profile_symmetric_task_generates_perm(self):
        # color 2 and 3 used with mirrored counts -> identical sorted profiles
        x1 = to_grid([[2, 2, 0], [3, 0, 0], [0, 0, 0]])
        y1 = to_grid([[3, 3, 0], [2, 0, 0], [0, 0, 0]])
        x2 = to_grid([[3, 3, 0], [2, 0, 0], [0, 0, 1]])
        y2 = to_grid([[2, 2, 0], [3, 0, 0], [0, 0, 1]])
        perms = candidate_color_perms([(x1, y1), (x2, y2)])
        assert {2: 3, 3: 2} in perms

    def test_asymmetric_usage_generates_no_perm(self):
        x1 = to_grid([[2, 2, 2], [0, 0, 0], [0, 0, 0]])
        y1 = to_grid([[8, 8, 8], [0, 0, 0], [0, 0, 0]])
        perms = candidate_color_perms([(x1, y1)])
        assert perms == []


class TestConsistency:
    def test_relations_consistent_basic(self):
        a = to_grid([[0, 1], [1, 0]])
        b = to_grid([[1, 1], [0, 0]])
        assert relations_consistent([(a, b), (b, a)])
        assert not relations_consistent([(a, b), (a, a)])

    def test_plausible_d4_excludes_contradicted_symmetry(self):
        # x2 == rot180(x1) but y2 != rot180(y1) -> rot180 contradicted
        x1 = to_grid([[1, 0], [0, 0]])
        y1 = to_grid([[2, 0], [0, 0]])
        x2 = to_grid([[0, 0], [0, 1]])  # rot180 of x1
        y2 = to_grid([[0, 0], [3, 0]])  # NOT rot180 of y1
        plaus = plausible_d4([(x1, y1), (x2, y2)])
        assert (2, False) not in plaus  # rot180 excluded
        assert (0, False) in plaus    # identity always plausible

    def test_plausible_d4_keeps_symmetric_task(self):
        x = to_grid([[1, 1], [1, 1]])
        y = to_grid([[2, 2], [2, 2]])
        assert len(plausible_d4([(x, y)])) == 8


class TestShapeLaws:
    def test_identity_law(self):
        pairs = [(rand_grid(4, 4, seed=1), rand_grid(4, 4, seed=2))]
        laws = consistent_shape_laws(pairs)
        names = [l.name for l in laws]
        assert any("identity" in n for n in names)
        assert not shape_law_violation(laws, (4, 4), (4, 4))
        assert shape_law_violation(laws, (4, 4), (2, 4))

    def test_scale_law_half_rows(self):
        pairs = [
            (rand_grid(4, 4, seed=1), rand_grid(2, 4, seed=2)),
            (rand_grid(6, 4, seed=3), rand_grid(3, 4, seed=4)),
        ]
        laws = consistent_shape_laws(pairs)
        assert laws, "scale law should fit"
        assert not shape_law_violation(laws, (8, 6), (4, 6))
        assert shape_law_violation(laws, (8, 6), (8, 6))
        assert shape_law_violation(laws, (8, 6), (2, 6))

    def test_no_laws_no_violation(self):
        pairs = [
            (rand_grid(4, 4, seed=1), rand_grid(3, 5, seed=2)),
            (rand_grid(5, 3, seed=3), rand_grid(7, 2, seed=4)),
        ]
        laws = consistent_shape_laws(pairs)
        # whatever fits, an unpredictable shape must never be flagged
        assert not shape_law_violation(laws, (9, 9), (1, 1)) or laws

    def test_const_law(self):
        y = to_grid([[1, 1], [1, 1]])
        pairs = [(rand_grid(3, 3, seed=1), y), (rand_grid(5, 4, seed=2), y)]
        laws = consistent_shape_laws(pairs)
        assert shape_law_violation(laws, (6, 6), (3, 3))
        assert not shape_law_violation(laws, (6, 6), (2, 2))


class TestGridIO:
    def test_csv_roundtrip(self):
        g = rand_grid(6, 7, seed=9)
        parsed = parse_grid_lenient(format_csv(g))
        assert parsed is not None and np.array_equal(parsed, g)

    def test_parse_fenced_block(self):
        text = "Here you go:\n```json\n[[1, 2], [3, 4]]\n```\nDone."
        parsed = parse_grid_lenient(text)
        assert np.array_equal(parsed, to_grid([[1, 2], [3, 4]]))

    def test_parse_space_separated(self):
        text = "output:\n1 2 3\n4 5 6\n"
        parsed = parse_grid_lenient(text)
        assert np.array_equal(parsed, to_grid([[1, 2, 3], [4, 5, 6]]))

    def test_parse_last_block_wins(self):
        text = "1,1\n1,1\n\nActually the final answer:\n2,2\n2,2\n"
        parsed = parse_grid_lenient(text)
        assert np.array_equal(parsed, to_grid([[2, 2], [2, 2]]))

    def test_parse_garbage_returns_none(self):
        assert parse_grid_lenient("no grid here at all") is None

    def test_to_grid_validation(self):
        with pytest.raises(ValueError):
            to_grid([1, 2, 3])
        with pytest.raises(ValueError):
            to_grid([[10, 0], [0, 0]])

    def test_grid_key_and_equality(self):
        a = to_grid([[1, 2], [3, 4]])
        b = to_grid([[1, 2], [3, 4]])
        assert grid_key(a) == grid_key(b)
        assert grids_equal(a, b)
        assert not grids_equal(a, to_grid([[1, 2], [3, 5]]))
        assert not grids_equal(a, None)
        assert grids_equal(None, None)


class TestProbes:
    def test_pad_and_crop_roundtrip(self):
        g = rand_grid(5, 6, seed=11)
        p = pad_border(g, color=0)
        assert p.shape == (7, 8)
        c = crop_border(p)
        assert np.array_equal(c, g)

    def test_perturb_changes_exactly_n_cells(self):
        g = rand_grid(6, 6, seed=12)
        rng = np.random.default_rng(0)
        p = perturb_cells(g, rng, n_cells=1)
        assert int(np.sum(p != g)) == 1

    def test_components_and_analysis(self):
        g = to_grid(
            [
                [int(c) for c in "11000"],
                [int(c) for c in "11002"],
                [int(c) for c in "00002"],
                [int(c) for c in "00000"],
            ]
        )
        comps = components(g)
        assert len(comps) == 2
        sizes = sorted(c["size"] for c in comps)
        assert sizes == [2, 4]
        block = analysis_block([(g, g)], [g])
        assert "objects=2" in block


def test_background_color():
    g = to_grid([[0, 0, 0], [0, 1, 2], [0, 0, 0]])
    assert background_color(g) == 0
