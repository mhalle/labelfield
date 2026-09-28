"""Clear errors and fallbacks for inputs the backends cannot take."""
import warnings
from unittest import mock

import numpy as np
import pytest
import torch

import labelfield as lf
from labelfield import Affine, Mapping, backends
from labelfield.backends import metal, torch_gather, triton_gpu

from conftest import voronoi_logits


def test_an_affine_is_refused_with_a_pointer_to_separable():
    logits = voronoi_logits(K=3, shape=(4, 5, 6))
    with pytest.raises(TypeError, match="separable"):
        lf.to_labels(logits, (8, 9, 10), Affine(np.eye(3)))
    with pytest.raises(TypeError, match="Mapping"):
        lf.to_labels(logits, (8, 9, 10), (1.0, 1.0, 1.0))
    ok = Affine(np.diag([0.5, 0.5, 0.5])).separable                  # the per-axis form works
    assert lf.to_labels(logits, (8, 9, 10), ok).shape == (8, 9, 10)


def test_transparency_mask_refuses_an_unknown_mode():
    with pytest.raises(ValueError, match="transparent"):
        lf.transparency_mask([0, 1, 2], transparent="nothing")


@pytest.mark.parametrize("fused,device", [("metal", "mps"), ("triton", "cuda")])
def test_auto_falls_back_for_a_dtype_or_layout_the_kernel_cannot_take(fused, device):
    dev = torch.device(device)
    module = {"metal": metal, "triton": triton_gpu}[fused]
    with mock.patch.object(module, "available", lambda: True):
        shape, out = (3, 8, 8, 8), (16, 16, 16)
        assert backends.select("auto", dev, shape, out, logits_dtype=torch.float16).name == fused
        c = backends.select("auto", dev, shape, out, logits_dtype=torch.float64)
        assert (c.name, c.module) == ("torch", torch_gather) and "float64" in c.fallback
        c = backends.select("auto", dev, shape, out, logits_dtype=torch.float32, out_contiguous=False)
        assert c.name == "torch" and "contiguous" in c.fallback
        assert backends.select("auto", dev, shape, out).name == fused           # no dtype given: as before


def test_to_labels_falls_back_on_a_non_contiguous_view_and_matches_torch():
    if not metal.available():
        pytest.skip("no Metal kernel here")
    logits = torch.from_numpy(voronoi_logits(K=4, shape=(6, 7, 8))).to("mps")
    m = Mapping.corner((11, 13, 15), (6, 7, 8))
    big = torch.zeros((11, 20, 15), dtype=torch.uint8, device="mps")
    view = big[:, 2:15]
    assert not view.is_contiguous()
    with pytest.warns(RuntimeWarning, match="contiguous"):
        lf.to_labels(logits, (11, 13, 15), m, out=view)
    want = lf.to_labels(logits, (11, 13, 15), m, backend="torch")
    np.testing.assert_array_equal(view.cpu().numpy(), want.cpu().numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        lf.to_labels(logits, (11, 13, 15), m)                                  # contiguous: no warning
