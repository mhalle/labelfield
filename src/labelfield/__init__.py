"""labelfield: a per-class logit field to labels on any grid, fused on the GPU.

For every voxel of a caller-chosen grid, interpolate the K logits from the model grid and
decide - argmax, or per-region threshold - in one pass, painting into a shared buffer so a
multi-model task composites without anything K-channel-sized ever existing at the output
resolution. Metal (Apple GPUs) and Triton (CUDA) kernels, and a torch backend that runs
anywhere, all consuming the same host-built per-axis tables.

The geometry (:class:`Grid`, :class:`Mapping`, :func:`build_tables`) and the float64
:mod:`reference` need only numpy; :func:`to_labels` and the backends need torch and are
imported on first use.
"""
from __future__ import annotations

from . import reference
from .grid import Grid
from .mapping import Mapping
from .tables import AxisTable, axis_table, build_tables

__version__ = "0.1.0"

_TORCH_NAMES = {"to_labels", "resample_argmax", "resample_paint", "available_backends", "transparency_mask"}


def __getattr__(name):
    if name in _TORCH_NAMES:
        from . import labels
        return getattr(labels, name)
    if name == "backends":
        import importlib
        return importlib.import_module(".backends", __name__)
    raise AttributeError(f"module 'labelfield' has no attribute {name!r}")


__all__ = ["Grid", "Mapping", "AxisTable", "axis_table", "build_tables", "reference",
           "to_labels", "resample_argmax", "resample_paint", "available_backends", "transparency_mask",
           "backends"]
