"""TotalSegmentator's compositor, reproduced.

TotalSegmentator (v3 branch, ``nnunet.py``) runs a multi-model task part by part on the model
grid: argmax, map through the part's label table, ``np.copyto(combined, lut[seg],
where=lut[seg] != 0)``; then one ``change_spacing(order=0)`` - ``scipy.ndimage.zoom`` with the
voxel-corner rule - back to the input shape. ``to_labels`` with ``interp="nearest"``,
``Mapping.corner``, ``paint=True`` and ``transparent="zero"`` must give the same labels exactly,
and with ``interp="linear"`` the same compositing of smooth per-part labels (the float64
reference computes it the same way).
"""
import numpy as np
import pytest
import torch
from scipy.ndimage import zoom

import labelfield as lg
from labelfield import Mapping, build_tables, reference

from conftest import voronoi_logits

# (model shape, input shape): up in plane, down or level in z, odd sizes, a single-sample axis
SHAPES = [((13, 17, 19), (31, 40, 23)), ((9, 11, 7), (9, 25, 18)), ((1, 12, 14), (3, 29, 33))]


def _parts(model_shape, seed=0):
    """Three parts with their own classes; part 2 has a class whose label is 0 (an auxiliary
    class the task does not report), which TotalSegmentator's compositor leaves transparent."""
    ks = [(6, [0, 1, 2, 3, 4, 5]), (5, [0, 10, 0, 12, 13]), (4, [0, 20, 21, 22])]
    return [(voronoi_logits(K=k, shape=model_shape, seed=seed + 7 * i), lut) for i, (k, lut) in enumerate(ks)]


def _totalsegmentator(parts, model_shape, input_shape):
    """The v3 branch's path, in numpy and scipy."""
    combined = np.zeros(model_shape, dtype=np.uint8)
    for logits, lut in parts:
        mapped = np.asarray(lut, dtype=np.uint8)[logits.argmax(0)]
        np.copyto(combined, mapped, where=mapped != 0)
    factors = np.asarray(input_shape) / np.asarray(model_shape)
    out = zoom(combined, factors, order=0, mode="nearest")
    assert out.shape == tuple(input_shape)
    return out


@pytest.mark.parametrize("model_shape,input_shape", SHAPES)
def test_nearest_composite_is_totalsegmentators_exactly(device, model_shape, input_shape):
    parts = _parts(model_shape)
    want = _totalsegmentator(parts, model_shape, input_shape)
    assert len(np.unique(want)) > 4
    mapping = Mapping.corner(input_shape, model_shape)
    out = torch.zeros(input_shape, dtype=torch.uint8, device=device)
    for logits, lut in parts:
        lg.to_labels(torch.from_numpy(logits).to(device), input_shape, mapping, interp="nearest",
                     lut=lut, paint=True, transparent="zero", out=out)
    np.testing.assert_array_equal(out.cpu().numpy(), want)


@pytest.mark.parametrize("model_shape,input_shape", SHAPES)
def test_linear_composite_matches_the_reference(device, model_shape, input_shape):
    parts = _parts(model_shape, seed=3)
    mapping = Mapping.corner(input_shape, model_shape)
    tables = build_tables(input_shape, model_shape, mapping)
    want = np.zeros(input_shape, dtype=np.int64)
    margins = []
    for logits, lut in parts:
        v, valid = reference.interpolate(logits, tables)
        reference.decide(v, valid, lut=lut, paint=True, transparent="zero", out=want)
        margins.append(reference.margins(v))
    out = torch.zeros(input_shape, dtype=torch.uint8, device=device)
    for logits, lut in parts:
        lg.to_labels(torch.from_numpy(logits).to(device), input_shape, mapping, lut=lut, paint=True,
                     transparent="zero", out=out)
    got = out.cpu().numpy().astype(np.int64)
    near = np.zeros(input_shape, dtype=bool)
    for m in margins:
        near |= m < 1e-4
    assert not ((got != want) & ~near).any()
    # smooth, not blocky: the linear composite differs from the nearest one at boundaries
    blocky = _totalsegmentator(parts, model_shape, input_shape)
    assert (got != blocky).any()


def test_zero_differs_from_background_only_where_a_label_is_zero(device):
    model_shape, input_shape = (10, 12, 14), (19, 23, 27)
    logits, lut = _parts(model_shape)[1]                      # the part with an unreported class
    mapping = Mapping.corner(input_shape, model_shape)
    t = torch.from_numpy(logits).to(device)
    a = torch.full(input_shape, 99, dtype=torch.uint8, device=device)
    b = torch.full(input_shape, 99, dtype=torch.uint8, device=device)
    lg.to_labels(t, input_shape, mapping, lut=lut, paint=True, out=a)                       # background
    lg.to_labels(t, input_shape, mapping, lut=lut, paint=True, transparent="zero", out=b)
    a, b = a.cpu().numpy(), b.cpu().numpy()
    tables = build_tables(input_shape, model_shape, mapping)
    v, _ = reference.interpolate(logits, tables)
    won_by_2 = (v.argmax(0) == 2) & (reference.margins(v) > 1e-4)   # channel 2's label is 0
    assert won_by_2.any()
    assert (a[won_by_2] == 0).all()                   # "background" writes its label, 0
    assert (b[won_by_2] == 99).all()                  # "zero" leaves the voxel to earlier parts
    np.testing.assert_array_equal(a[~won_by_2 & (reference.margins(v) > 1e-4)],
                                  b[~won_by_2 & (reference.margins(v) > 1e-4)])


def test_transparent_validation():
    logits = voronoi_logits(K=3, shape=(4, 5, 6))
    m = Mapping.corner((8, 9, 10), (4, 5, 6))
    with pytest.raises(ValueError, match="transparent"):
        lg.to_labels(logits, (8, 9, 10), m, transparent="nothing")
    with pytest.raises(ValueError, match="argmax paint"):
        lg.to_labels(logits, (8, 9, 10), m, transparent="zero")                       # not a paint
    with pytest.raises(ValueError, match="argmax paint"):
        lg.to_labels(logits, (8, 9, 10), m, mode="regions", paint=True, transparent="zero",
                     out=torch.zeros((8, 9, 10), dtype=torch.uint8))


def test_kernel_lut():
    lut = [0, 5, 0, 7]
    np.testing.assert_array_equal(lg.kernel_lut(lut, mode="argmax", paint=False, transparent="background"), lut)
    np.testing.assert_array_equal(lg.kernel_lut(lut, mode="argmax", paint=True, transparent="background"),
                                  [-1, 5, 0, 7])
    np.testing.assert_array_equal(lg.kernel_lut(lut, mode="argmax", paint=True, transparent="zero"),
                                  [-1, 5, -1, 7])
    np.testing.assert_array_equal(lg.kernel_lut(lut, mode="regions", paint=True, transparent="background"), lut)


def test_nearest_past_a_cropped_edge_is_outside():
    """A label map that is background beyond a crop, resampled nearest: a coordinate exactly
    half-way past the last sample picks the (background) voxel beyond it, so it is outside -
    not clamped onto the edge voxel. Input y = 35 of 37 lands on model 17.5 of 19 here."""
    model_shape, input_shape = (4, 19, 5), (4, 37, 5)
    logits = voronoi_logits(K=3, shape=model_shape, seed=1)
    lab = np.zeros(model_shape, np.uint8)
    lab[:, :18] = logits[:, :, :18].argmax(0)                  # a crop that ends at y = 18
    want = zoom(np.where(lab == 0, 9, lab).astype(np.uint8), np.asarray(input_shape) / model_shape,
                order=0, mode="nearest")
    out = torch.full(input_shape, 9, dtype=torch.uint8)
    lg.to_labels(torch.from_numpy(np.ascontiguousarray(logits[:, :, :18])), input_shape,
                 Mapping.corner(input_shape, model_shape), interp="nearest", paint=True,
                 transparent="zero", out=out)
    np.testing.assert_array_equal(out.numpy(), want)


@pytest.mark.parametrize("interp", ["nearest", "linear"])
def test_slabs_with_out_start_equal_one_call(device, interp):
    """Slab by slab with out_start makes the decisions of a single call - including the
    exact half-way coordinates (65 * 69 / 130 = 34.5) that folding the offset into b flips."""
    model_shape, input_shape = (70, 9, 8), (131, 17, 15)
    logits = torch.from_numpy(voronoi_logits(K=5, shape=model_shape, seed=4)).to(device)
    mapping = Mapping.corner(input_shape, model_shape)
    whole = lg.to_labels(logits, input_shape, mapping, interp=interp).cpu().numpy()
    out = torch.zeros(input_shape, dtype=torch.uint8, device=device)
    for z0 in range(0, input_shape[0], 7):
        z1 = min(z0 + 7, input_shape[0])
        lg.to_labels(logits, (z1 - z0, *input_shape[1:]), mapping, interp=interp,
                     out=out[z0:z1], out_start=(z0, 0, 0))
    np.testing.assert_array_equal(out.cpu().numpy(), whole)
