"""Parity with the nnunet-inference-mlx fused kernel, on frozen fixtures."""
import os

import numpy as np
import pytest
import torch

import labelfield as lg
from labelfield import Mapping
from labelfield.backends import metal

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "mlx_fused_oracle.npz")


def _fixture_cases():
    import json
    data = np.load(FIXTURE)
    cases = json.loads(str(data["cases"]))
    return [(name, c, data[f"{name}_logits"], data[f"{name}_labels"]) for name, c in cases.items()]


@pytest.mark.parametrize("coord_dtype", [np.float32, np.float64])
@pytest.mark.parametrize("backend", ["torch", "metal"])
def test_small_fixtures_match_mlx_fused_kernel(backend, coord_dtype):
    """Labels produced by nnunet-inference-mlx's fused Metal kernel
    (`inverse_resample_argmax`, oracle ref main@40ebe55) on two synthetic
    cases, generated 2026-08-22 and frozen here. Expected: identical."""
    if backend == "metal" and not metal.available():
        pytest.skip("no MPS")
    device = "mps" if backend == "metal" else "cpu"
    for name, c, logits, want in _fixture_cases():
        mapping = Mapping.spacing(c["acq"], c["target"])      # the MLX kernel's s2t = acq / target, clamped
        got = lg.to_labels(torch.from_numpy(logits).to(device), tuple(c["out"]), mapping, outside="clamp",
                           coord_dtype=coord_dtype, backend=backend).cpu().numpy()
        mism = int((got != want).sum())
        assert mism == 0, f"{name}/{backend}/{coord_dtype.__name__}: {mism} voxels differ from the MLX kernel"
