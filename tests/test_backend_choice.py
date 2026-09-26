"""Which restore backend "auto" takes, decided by what each fused kernel can address.

Until 2026-09-11 "auto" took the device's fused kernel by device type alone, and the Triton
kernel refused any field of K x Zs x Ys x Xs >= 2^31 with a ValueError nothing caught: the
open CADS head-and-neck model (K=30) on a whole-body PET/CT's attenuation CT (334x334x678 at
1.5 mm, 2.27e9 logits) failed outright on an A10G (in haversack, where this code came from). The kernel
now takes a 64-bit channel base, as the Metal one always has, so that field stays on it; what
a fused kernel still cannot address - a channel, or for Triton the output, of 2^31 voxels or
more - "auto" hands to the torch backend, and ``to_labels`` warns.

The decision is a function of shapes alone, so it is tested here without a GPU and without
the memory such a field needs.
"""
import unittest
import warnings
from unittest import mock

import numpy as np
import pytest
import torch

import labelfield as lg
from labelfield import Mapping, backends
from labelfield.backends import metal, torch_gather, triton_gpu

from conftest import voronoi_logits

CUDA, MPS, CPU = torch.device("cuda"), torch.device("mps"), torch.device("cpu")
LIMIT = 2 ** 31
# the field that failed: K=30 on 334x334x678 (ZYX 678, 334, 334), restored to 512x512x311
REPORTED = ((30, 678, 334, 334), (311, 512, 512))
CHANNEL_AT_LIMIT = (2, 2048, 1024, 1024)             # one channel of exactly 2^31 voxels
CHANNEL_SHORT = (2, 1, 1, LIMIT - 1)
SMALL_OUT = (64, 64, 64)


class AutoOnCuda(unittest.TestCase):

    def setUp(self):
        p = mock.patch.object(triton_gpu, "available", lambda: True)
        p.start()
        self.addCleanup(p.stop)

    def test_the_reported_field_stays_on_triton(self):
        assert REPORTED[0][0] * np.prod(REPORTED[0][1:]) >= LIMIT        # the case really is past it
        self.assertEqual(backends.select("auto", CUDA, *REPORTED), ("triton", triton_gpu, None))

    def test_a_channel_of_2_31_voxels_goes_to_torch_and_says_why(self):
        c = backends.select("auto", CUDA, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertEqual((c.name, c.module), ("torch", torch_gather))
        self.assertIn("triton", c.fallback)
        self.assertIn("2,147,483,648", c.fallback)

    def test_a_channel_one_voxel_short_of_2_31_stays_on_triton(self):
        self.assertEqual(backends.select("auto", CUDA, CHANNEL_SHORT, SMALL_OUT).name, "triton")

    def test_an_output_of_2_31_voxels_goes_to_torch(self):
        """Triton's output offsets are 32-bit: past 2^31 ``pid * BLOCK`` wraps negative, passes
        the ``offs < n_out`` mask and writes before the buffer. It is refused, not launched."""
        c = backends.select("auto", CUDA, (30, 100, 100, 100), (2048, 1024, 1024))
        self.assertEqual(c.name, "torch")
        self.assertIn("output", c.fallback)
        self.assertEqual(backends.select("auto", CUDA, (30, 100, 100, 100), (1, 1, LIMIT - 1)).name, "triton")

    def test_asking_for_triton_by_name_refuses_what_it_cannot_take(self):
        with self.assertRaises(ValueError) as e:
            backends.select("triton", CUDA, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertIn("backend='torch'", str(e.exception))

    def test_without_triton_auto_is_torch_and_nothing_fell_back(self):
        with mock.patch.object(triton_gpu, "available", lambda: False):
            self.assertEqual(backends.select("auto", CUDA, *REPORTED), ("torch", torch_gather, None))


class AutoOnMps(unittest.TestCase):

    def setUp(self):
        p = mock.patch.object(metal, "available", lambda: True)
        p.start()
        self.addCleanup(p.stop)

    def test_the_reported_field_stays_on_metal(self):
        self.assertEqual(backends.select("auto", MPS, *REPORTED), ("metal", metal, None))

    def test_a_channel_of_2_31_voxels_goes_to_torch(self):
        c = backends.select("auto", MPS, CHANNEL_AT_LIMIT, SMALL_OUT)
        self.assertEqual(c.name, "torch")
        self.assertIn("metal", c.fallback)
        self.assertEqual(backends.select("auto", MPS, CHANNEL_SHORT, SMALL_OUT).name, "metal")

    def test_a_large_output_stays_on_metal(self):
        """Metal launches the output in z-slabs and indexes it with 64-bit offsets."""
        self.assertEqual(backends.select("auto", MPS, (30, 100, 100, 100), (2048, 1024, 1024)).name, "metal")


class AutoOnCpu(unittest.TestCase):

    def test_cpu_is_torch_whatever_the_field(self):
        for shape in (REPORTED, (CHANNEL_AT_LIMIT, SMALL_OUT)):
            self.assertEqual(backends.select("auto", CPU, *shape), ("torch", torch_gather, None))


class _PretendCpuHasTriton:
    """Make "auto" on the CPU prefer the Triton backend, with a limit small enough that a test
    field crosses it, and a ``run`` that fails if it is ever called - so a fallback that did not
    happen is a failure, not a CUDA error this machine cannot raise."""

    def setUp(self):
        self.triton_runs = []
        for p in (mock.patch.dict(backends.FUSED, {"cpu": "triton"}),
                  mock.patch.object(triton_gpu, "available", lambda: True),
                  mock.patch.object(triton_gpu, "run", lambda *a, **k: self.triton_runs.append(a))):
            p.start()
            self.addCleanup(p.stop)

    def shrink_the_limit(self, limit=100):
        p = mock.patch.object(triton_gpu, "OFFSET_LIMIT", limit)
        p.start()
        self.addCleanup(p.stop)


class ToLabelsFallsBack(_PretendCpuHasTriton, unittest.TestCase):

    SRC, OUT = (5, 6, 7), (9, 10, 11)

    def _logits(self):
        return torch.from_numpy(voronoi_logits(K=4, shape=self.SRC, seed=3))

    def test_auto_past_the_limit_runs_torch_and_warns(self):
        self.shrink_the_limit()
        mapping = Mapping.center(self.OUT, self.SRC)
        with pytest.warns(RuntimeWarning, match="triton"):
            got = lg.to_labels(self._logits(), self.OUT, mapping, backend="auto")
        self.assertEqual(self.triton_runs, [])
        want = lg.to_labels(self._logits(), self.OUT, mapping, backend="torch")
        np.testing.assert_array_equal(got.numpy(), want.numpy())
        self.assertTrue(len(np.unique(want.numpy())) > 1)                  # a field with something in it

    def test_auto_within_the_limit_still_takes_the_fused_kernel(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lg.to_labels(self._logits(), self.OUT, Mapping.center(self.OUT, self.SRC), backend="auto")
        self.assertEqual(len(self.triton_runs), 1)


if __name__ == "__main__":
    unittest.main()
