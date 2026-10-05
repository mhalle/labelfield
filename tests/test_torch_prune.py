"""The torch backend's argmax path prunes channels per model cell (a channel whose largest corner
value is below another's smallest cannot win in that cell). It must give the dense path's labels
bit for bit: same blend arithmetic, same first-maximum rule at ties."""
import numpy as np
import pytest
import torch

from labelfield import Mapping, build_tables
from labelfield.backends import torch_gather


def smooth_logits(K, shape, seed, scale=8.0, integer=False, dtype=torch.float32):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(K, *shape))
    for axis in (1, 2, 3):
        for _ in range(2):
            x = (np.roll(x, 1, axis) + x + np.roll(x, -1, axis)) / 3
    x = x * scale
    if integer:                                   # exact ties everywhere
        x = np.round(x)
    return torch.from_numpy(x.astype(np.float32)).to(dtype)


def both(logits, out_shape, mapping, *, interp="linear", lut=None, paint=False, skip=None,
         out_start=(0, 0, 0), prefill=None):
    K = logits.shape[0]
    tables = build_tables(out_shape, tuple(logits.shape[1:]), mapping, interp=interp, out_start=out_start)
    lut = np.arange(K, dtype=np.int32) if lut is None else np.asarray(lut, dtype=np.int32)
    res = []
    for prune in (False, True):
        out = torch.zeros(out_shape, dtype=torch.int32) if prefill is None else prefill.clone()
        torch_gather.run(logits, out, tables, lut, mode="argmax", paint=paint, background=0,
                         threshold=0.0, skip=skip, prune=prune)
        res.append(out)
    return res


CASES = [  # (K, source, output, mapping)
    (5, (7, 9, 11), (20, 23, 19), Mapping.corner((20, 23, 19), (7, 9, 11))),
    (12, (9, 8, 10), (17, 31, 25), Mapping.center((17, 31, 25), (9, 8, 10))),
    (3, (12, 12, 12), (6, 7, 5), Mapping.center((6, 7, 5), (12, 12, 12))),              # downsampling
    (40, (6, 10, 9), (13, 22, 18), Mapping((0.41, 0.43, 0.47), (-0.8, -1.3, 0.2))),     # outside parts
    (300, (5, 6, 7), (9, 11, 13), Mapping.corner((9, 11, 13), (5, 6, 7))),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("integer", [False, True])
@pytest.mark.parametrize("interp", ["linear", "nearest", ("nearest", "linear", "linear")])
def test_pruned_equals_dense(case, integer, interp):
    K, src, out_shape, mapping = CASES[case]
    logits = smooth_logits(K, src, seed=case * 7 + integer, integer=integer)
    dense, pruned = both(logits, out_shape, mapping, interp=interp)
    assert torch.equal(dense, pruned)
    assert len(torch.unique(dense)) > 1


def test_paint_lut_skip_slab_and_fp16():
    K, src, out_shape = 9, (8, 9, 10), (14, 17, 16)
    logits = smooth_logits(K, src, seed=3, dtype=torch.float16)
    lut = [0, 4, 4, 7, 0, 9, 2, 2, 5]                     # channels sharing labels; label 0 twice
    skip = np.array([True, False, False, False, True, False, False, False, False])
    prefill = torch.from_numpy(np.random.default_rng(1).integers(0, 3, size=(6, 17, 16)).astype(np.int32))
    m = Mapping.corner(out_shape, src)
    dense, pruned = both(logits, (6, 17, 16), m, lut=lut, paint=True, skip=skip, out_start=(5, 0, 0),
                         prefill=prefill)
    assert torch.equal(dense, pruned)
    assert not torch.equal(dense, prefill)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), float("-inf")])
def test_finite_finds_every_non_finite_value(device, dtype, bad, monkeypatch):
    """Checked in slabs of Z planes (a small chunk here forces several), a NaN or infinity is found
    wherever it is: first, middle and last plane."""
    monkeypatch.setattr(torch_gather, "FINITE_CHUNK", 3 * 4 * 5 * 2)            # two planes per slab
    for where in [(0, 0, 0, 0), (1, 3, 2, 4), (2, 6, 3, 4)]:
        logits = torch.zeros((3, 7, 4, 5), dtype=dtype, device=device)
        if bad is not None:
            logits[where] = bad
        assert torch_gather.finite(logits) is (bad is None)
        assert torch_gather.finite(logits) == bool(torch.isfinite(logits).all())


def test_non_finite_logits_take_the_dense_path():
    logits = smooth_logits(4, (5, 5, 5), seed=0)
    logits[1, 2, 2, 2] = float("nan")
    tables = build_tables((8, 8, 8), (5, 5, 5), Mapping.corner((8, 8, 8), (5, 5, 5)))
    out_auto, out_dense = torch.zeros((8, 8, 8), dtype=torch.int32), torch.zeros((8, 8, 8), dtype=torch.int32)
    lut = np.arange(4, dtype=np.int32)
    torch_gather.run(logits, out_auto, tables, lut, mode="argmax", paint=False, background=0, threshold=0.0)
    torch_gather.run(logits, out_dense, tables, lut, mode="argmax", paint=False, background=0, threshold=0.0,
                     prune=False)
    assert torch.equal(out_auto, out_dense)
