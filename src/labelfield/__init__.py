"""labelfield: convert a per-class logit field on a model grid into a label map on any grid.

For every voxel of a caller-chosen output grid, :func:`to_labels` interpolates the K logits
from the model grid and decides a label (argmax, or per-region threshold) in one pass,
optionally compositing into an existing label map so that several models can write into one
buffer. The K interpolated channels are never materialized at output resolution. Backends:
fused kernels for Apple GPUs (Metal) and CUDA (Triton), and a portable torch backend. All use
the same per-axis tables computed on the host.

The geometry (:class:`Grid`, :class:`Mapping`, :class:`Affine`, :func:`build_tables`,
:func:`axis_table`, :func:`axis_coords`) and the float64 :mod:`reference` need only numpy.
:func:`to_labels`, :func:`resample_argmax`, :func:`resample_paint`,
:func:`available_backends`, :func:`transparency_mask` and :mod:`backends` need torch, which is
imported on first access to one of them. Axis order is (Z, Y, X) throughout.
"""
from __future__ import annotations

from . import reference
from .grid import Grid
from .mapping import Affine, Mapping
from .tables import AxisTable, axis_coords, axis_table, build_tables

__version__ = "0.1.5"

_TORCH_NAMES = {"to_labels", "resample_argmax", "resample_paint", "available_backends", "transparency_mask"}


def __getattr__(name):
    if name in _TORCH_NAMES:
        from . import labels
        return getattr(labels, name)
    if name == "backends":
        import importlib
        return importlib.import_module(".backends", __name__)
    raise AttributeError(f"module 'labelfield' has no attribute {name!r}")


__all__ = ["Grid", "Mapping", "Affine", "AxisTable", "axis_coords", "axis_table", "build_tables", "reference",
           "to_labels", "resample_argmax", "resample_paint", "available_backends", "transparency_mask",
           "backends"]
