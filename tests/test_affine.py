"""Affine (a general map between index spaces) and axis_coords (the per-axis rule for any
coordinates, the one axis_table applies)."""
from types import SimpleNamespace

import numpy as np
import pytest

from labelfield import Affine, Mapping, axis_coords, build_tables
from labelfield.mapping import SEPARABLE_TOLERANCE


def geometry(directions, origin):
    """A geometry record as rankfield and duckn have it: direction rows per array axis, origin."""
    return SimpleNamespace(directions=np.asarray(directions, dtype=np.float64), origin=np.asarray(origin, float))


def rotation(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def test_apply_inverse_round_trip():
    a = Affine(rotation(17) * 1.3, (2.0, -1.5, 0.25))
    x = np.random.default_rng(0).normal(size=(10, 3)) * 20
    np.testing.assert_allclose(a.inverse().apply(a.apply(x)), x, atol=1e-10)


def test_between_goes_through_the_world():
    fine = geometry(np.diag([0.8, 0.7, 0.7]), (-100.0, 20.0, 5.0))
    coarse = geometry(rotation(10) @ np.diag([3.0, 3.0, 3.0]), (-98.0, 18.0, 3.0))
    aff = Affine.between(fine, coarse)
    idx = np.array([[0, 0, 0], [10, 20, 30], [3.5, 7.25, 1.0]])
    world = fine.origin + idx @ fine.directions
    np.testing.assert_allclose(coarse.origin + aff.apply(idx) @ coarse.directions, world, atol=1e-9)


def test_separable_when_the_grids_line_up():
    fine = geometry(np.diag([1.0, 0.7, 0.7]), (0.0, 0.0, 0.0))
    coarse = geometry(np.diag([3.0, 3.0, 3.0]), (-1.0, -1.15, -1.15))
    m = Affine.between(fine, coarse).separable
    assert isinstance(m, Mapping)
    np.testing.assert_allclose(m.a, (1 / 3, 0.7 / 3, 0.7 / 3))
    np.testing.assert_allclose(m.apply([4, 5, 6]), Affine.between(fine, coarse).apply([4, 5, 6]))


def test_rotation_flip_and_swap_are_not_separable():
    assert Affine(rotation(5)).separable is None
    assert Affine(np.diag([1.0, -1.0, 1.0])).separable is None                         # a flip
    assert Affine(np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, 1]])).separable is None     # an axis swap


def test_float_noise_is_not_a_rotation():
    """Two oblique grids of one orientation compose to off-diagonal terms of ~1e-17."""
    r = rotation(23.0)
    a = geometry(r @ np.diag([0.8, 0.8, 1.0]), (1.0, 2.0, 3.0))
    b = geometry(r @ np.diag([1.6, 1.6, 2.0]), (0.5, 1.5, 2.5))
    aff = Affine.between(a, b)
    m = np.asarray(aff.m)
    assert 0 < np.abs(m - np.diag(np.diag(m))).max() < SEPARABLE_TOLERANCE
    assert aff.separable is not None
    assert Affine(np.array([[1.0, 1e-3, 0], [0, 1, 0], [0, 0, 1]])).separable is None


def test_validation():
    with pytest.raises(ValueError):
        Affine(np.eye(2))
    with pytest.raises(ValueError):
        Affine(np.full((3, 3), np.nan))


@pytest.mark.parametrize("interp", ["linear", "nearest"])
@pytest.mark.parametrize("outside", ["background", "clamp"])
def test_axis_coords_is_the_rule_the_tables_apply(interp, outside):
    """For a per-axis mapping, axis_coords on each voxel's coordinate decides what the tables do."""
    out_shape, src = (23, 17, 11), (9, 12, 5)
    m = Mapping((0.43, 0.77, 0.61), (-1.3, -0.2, -0.9))
    tables = build_tables(out_shape, src, m, interp=interp, outside=outside)
    for ax in range(3):
        c = m.a[ax] * np.arange(out_shape[ax]) + m.b[ax]
        valid, i0, i1, f = axis_coords(c, src[ax], interp=interp, outside=outside)
        t = tables[ax]
        np.testing.assert_array_equal(valid, t.i0 >= 0)
        np.testing.assert_array_equal(i0[valid], t.i0[valid])
        np.testing.assert_array_equal(i1, t.i1)
        np.testing.assert_array_equal(f, t.f64)


def test_axis_coords_edges_and_ties():
    n = 5
    c = np.array([-0.51, -0.5, -0.49, 0.5, 1.5, 3.5, 4.0, 4.49, 4.5, 4.51])
    lin_valid, lin_i0, lin_i1, lin_f = axis_coords(c, n, interp="linear")
    near_valid, near_i0, _, near_f = axis_coords(c, n, interp="nearest")
    np.testing.assert_array_equal(lin_valid, [0, 1, 1, 1, 1, 1, 1, 1, 1, 0])
    np.testing.assert_array_equal(near_valid, [0, 1, 1, 1, 1, 1, 1, 1, 0, 0])   # 4.5 picks sample 5: none
    np.testing.assert_array_equal(near_i0[near_valid], [0, 0, 1, 2, 4, 4, 4])    # halves round UP
    assert (near_f == 0).all()
    np.testing.assert_array_equal(lin_i1, np.minimum(lin_i0 + 1, n - 1))
    assert axis_coords(np.zeros((2, 3)), n)[0].shape == (2, 3)                   # any shape
