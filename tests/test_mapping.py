"""The convention constructors against the resamplers they claim to invert."""
import numpy as np
import pytest
from scipy.ndimage import map_coordinates, zoom
from skimage.transform import resize

from labelfield import Mapping, build_tables, reference


def _vol(shape, seed=0):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float64)


def _values(vol, out_shape, mapping, interp="linear", outside="background"):
    tables = build_tables(out_shape, vol.shape, mapping, interp=interp, outside=outside)
    v, valid = reference.interpolate(vol[None], tables)
    assert valid.all()
    return v[0]


@pytest.mark.parametrize("factors", [(1.7, 2.3, 0.6), (0.45, 0.45, 3.1), (2.0, 0.5, 1.0)])
def test_corner_matches_scipy_zoom(factors):
    vol = _vol((7, 9, 11))
    want = zoom(vol, factors, order=1, mode="nearest", grid_mode=False)
    out_shape = want.shape
    got = _values(vol, out_shape, Mapping.corner(out_shape, vol.shape))
    np.testing.assert_allclose(got, want, atol=1e-12)


@pytest.mark.parametrize("out_shape", [(12, 21, 7), (3, 4, 25), (7, 9, 11)])
def test_center_matches_skimage_resize(out_shape):
    vol = _vol((7, 9, 11))
    want = resize(vol, out_shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True)
    got = _values(vol, out_shape, Mapping.center(out_shape, vol.shape))
    np.testing.assert_allclose(got, want, atol=1e-9)


@pytest.mark.parametrize("factors", [(1.7, 2.3, 0.6), (0.45, 0.45, 3.1), (2.5, 1.0, 0.5)])
def test_nearest_matches_scipy_zoom_order0(factors):
    lab = np.random.default_rng(1).integers(0, 50, size=(7, 9, 11)).astype(np.float64)
    want = zoom(lab, factors, order=0, mode="nearest", grid_mode=False)
    got = _values(lab, want.shape, Mapping.corner(want.shape, lab.shape), interp="nearest")
    np.testing.assert_array_equal(got, want)


def test_separate_z_matches_nnunet_style_reference():
    """linear in-plane (skimage, center), nearest along z (scipy order 0)."""
    vol = _vol((6, 9, 11))
    out_shape = (15, 17, 21)
    planes = np.stack([resize(vol[z], out_shape[1:], order=1, mode="edge", anti_aliasing=False,
                              preserve_range=True) for z in range(vol.shape[0])])
    cz = (np.arange(out_shape[0]) + 0.5) * vol.shape[0] / out_shape[0] - 0.5
    iz = np.clip(np.floor(np.clip(cz, 0, vol.shape[0] - 1) + 0.5), 0, vol.shape[0] - 1).astype(int)
    want = planes[iz]
    got = _values(vol, out_shape, Mapping.center(out_shape, vol.shape), interp=("nearest", "linear", "linear"))
    np.testing.assert_allclose(got, want, atol=1e-9)


def test_map_coordinates_with_explicit_affine():
    vol = _vol((8, 9, 10))
    m = Mapping((0.6, 1.3, 0.9), (0.4, -0.2, 1.1))
    out_shape = (9, 6, 8)
    tables = build_tables(out_shape, vol.shape, m, outside="clamp")
    got, _ = reference.interpolate(vol[None], tables)
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in out_shape], indexing="ij"), -1)
    coords = np.clip(m.apply(grid), 0, np.array(vol.shape) - 1)
    want = map_coordinates(vol, coords.reshape(-1, 3).T, order=1, mode="nearest").reshape(out_shape)
    np.testing.assert_allclose(got[0], want, atol=1e-12)


def test_spacing_rule_is_the_mlx_kernels():
    m = Mapping.spacing((1.5, 1.5, 1.5), (3.0, 3.0, 3.0))
    np.testing.assert_allclose(m.apply([10, 20, 30]), [5, 10, 15])
    m = Mapping.spacing((1.0, 0.651, 0.651), (3.0, 3.0, 3.0), shift=(1, 2, 3))
    np.testing.assert_allclose(m.apply([0, 0, 0]), [1, 2, 3])


def test_compose_and_inverse():
    m1 = Mapping((0.5, 2.0, 1.5), (1.0, -3.0, 0.25))
    m2 = Mapping((3.0, 0.25, 1.0), (-1.0, 2.0, 0.0))
    x = np.array([[1.0, 2.0, 3.0], [7.0, 0.0, 5.5]])
    np.testing.assert_allclose((m1 >> m2).apply(x), m2.apply(m1.apply(x)))
    np.testing.assert_allclose((m1 >> m1.inverse()).apply(x), x, atol=1e-12)
    with pytest.raises(ValueError):
        Mapping((-1, 1, 1))
    with pytest.raises(ValueError):
        Mapping.corner((5, 5, 5), (1, 5, 5)).inverse()


def test_corner_single_sample_axis():
    m = Mapping.corner((1, 5, 5), (4, 5, 5))
    assert m.a[0] == 0.0
    assert m.apply([0, 0, 0])[0] == 0.0


def test_center_nearest_picks_are_scipys_at_exact_ties():
    """The voxel-center rule evaluated as scipy does, (j + 0.5) * s - 0.5: a * j + b rounds the
    other way at exact half-way coordinates (1453 of these picks differed before 0.1.2)."""
    total = 0
    for n_src in range(3, 60):
        for n_out in range(3, 140, 5):
            want = zoom(np.arange(n_src, dtype=float), n_out / n_src, order=0, mode="nearest", grid_mode=True)
            if want.shape[0] != n_out:
                continue
            m = Mapping.center((n_out, 1, 1), (n_src, 1, 1))
            got = build_tables((n_out, 1, 1), (n_src, 1, 1), m, interp="nearest")[0].i0
            np.testing.assert_array_equal(got, want.astype(int), err_msg=f"{n_src} -> {n_out}")
            total += n_out
    assert total > 100000


def test_centered_flag_composition_and_inverse():
    c = Mapping.center((9, 9, 9), (6, 6, 6))
    assert c.centered
    np.testing.assert_allclose(c.b, 0.5 * np.asarray(c.a) - 0.5)                  # a, b as always
    assert (Mapping.identity() >> c) == c and (c >> Mapping.identity()) == c     # kept exact
    x = np.array([[0.0, 3.0, 8.0], [4.5, 1.0, 2.0]])
    np.testing.assert_allclose(c.apply(x), (x + 0.5) * 6 / 9 - 0.5, atol=1e-12)
    shift = Mapping((1.0, 1.0, 1.0), (-2.0, 0.0, -1.0))
    assert not (shift >> c).centered and not c.inverse().centered
    np.testing.assert_allclose((shift >> c).apply(x), c.apply(shift.apply(x)), atol=1e-12)
    np.testing.assert_allclose((c >> c.inverse()).apply(x), x, atol=1e-12)
    assert "centered" in repr(c)
    with pytest.raises(ValueError):
        Mapping((2.0, 2.0, 2.0), (0.0, 0.0, 0.0), centered=True)


@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_center_slabs_with_out_start_equal_one_call(interp):
    import torch
    import labelfield as lf
    src, out_shape = (6, 9, 8), (57, 17, 15)                    # 6 -> 57 along z: exact ties
    logits = torch.from_numpy(np.random.default_rng(3).normal(size=(4, *src)).astype(np.float32))
    m = Mapping.center(out_shape, src)
    whole = lf.to_labels(logits, out_shape, m, interp=interp).numpy()
    out = torch.zeros(out_shape, dtype=torch.uint8)
    for z0 in range(0, out_shape[0], 5):
        z1 = min(z0 + 5, out_shape[0])
        lf.to_labels(logits, (z1 - z0, *out_shape[1:]), m, interp=interp, out=out[z0:z1], out_start=(z0, 0, 0))
    np.testing.assert_array_equal(out.numpy(), whole)
