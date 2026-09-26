"""Logit field -> labels on the grid you ask for.

The inverse leg of a segmentation pipeline: for every voxel of a caller-chosen grid, interpolate the K
logits from the model grid and decide (argmax, or per-region threshold + paint) - fused, so
nothing K-channel-sized is ever materialized at the output resolution.
"""
from __future__ import annotations

import warnings

import numpy as np
import torch

from . import backends
from .grid import Grid
from .mapping import Mapping
from .tables import build_tables

MODES = ("argmax", "regions")
TRANSPARENT = ("background", "zero")
# torch.uint16 arrived in torch 2.3; below it a label map is uint8, labels 0-255
UINT16 = getattr(torch, "uint16", None)
LABEL_DTYPES = tuple(d for d in (torch.uint8, UINT16) if d is not None)


def _as_torch(logits) -> torch.Tensor:
    if isinstance(logits, torch.Tensor):
        return logits
    return torch.from_numpy(np.ascontiguousarray(logits))


def kernel_lut(lut, *, mode: str, paint: bool, transparent: str) -> np.ndarray:
    """The label table the backends read: ``lut`` with every transparent channel set to -1.

    Only an argmax paint has transparent channels; every other call gets ``lut`` unchanged, so
    no negative entry ever reaches a write. ``"background"`` makes channel 0 transparent (the
    decision is background); ``"zero"`` makes every channel whose label is 0 transparent, which
    is TotalSegmentator's compositor - ``np.copyto(out, lut[seg], where=lut[seg] != 0)``.
    """
    lut = np.asarray(lut, dtype=np.int64).reshape(-1)
    if not (paint and mode == "argmax"):
        return lut.astype(np.int32)
    k = lut.astype(np.int32)
    if transparent == "background":
        k[0] = -1
    else:
        k[lut == 0] = -1
    return k


def to_labels(logits, grid, mapping: Mapping, *, interp="linear", outside: str = "background",
              lut=None, mode: str = "argmax", paint: bool = False, transparent: str = "background",
              threshold: float = 0.0, background: int = 0, out: torch.Tensor | None = None,
              out_dtype=None, backend: str = "auto", coord_dtype=np.float64,
              slab_voxels: int = 1 << 26) -> torch.Tensor:
    """Labels on ``grid`` from a ``(K, Z, Y, X)`` logit field.

    Parameters
    ----------
    logits : torch.Tensor or ndarray, ``(K, Zs, Ys, Xs)`` float32 / float16
        The field at the model's grid. Stays where it is (device, dtype).
    grid : Grid or (Z, Y, X)
        The output grid (only its shape matters here).
    mapping : Mapping
        Output index -> model-grid coordinate, per axis. Build it from the
        output grid, the source image and the forward resampler's convention
        (``Mapping.between(...) >> Mapping.center/corner/spacing(...)``).
    interp : "linear" | "nearest" or a (Z, Y, X) tuple
        Per-axis interpolation. ``nearest`` along z reproduces nnU-Net's
        separate-z (``order_z=0``) export on anisotropic data.
    outside : "background" | "clamp"
        Output voxels beyond the model grid get ``background`` (or are skipped
        when painting), or sample the clamped edge.
    lut : sequence of K ints, optional
        Local channel -> output label. Defaults to the channel index.
    mode : "argmax" | "regions"
        ``argmax``: label of the first maximal channel. ``regions``: every
        channel above ``threshold`` paints ``lut[k]`` in channel order
        (later wins) - for sigmoid-head models.
    paint : bool
        Background-transparent write into ``out``: voxels whose decision is
        background (``argmax`` channel 0; no region above threshold) are left
        untouched. Compositing multi-model tasks = calling this once per part
        with that part's ``lut`` into one shared ``out``.
    transparent : "background" | "zero"
        Which argmax decisions a paint leaves untouched: ``background`` (the
        default) when the winning channel is 0; ``zero`` when the winning
        channel's label is 0 - TotalSegmentator's compositor, which writes
        ``lut[seg]`` only where it is non-zero. Argmax paint only.
    background : int
        Label written for outside / no-region voxels when not painting.
    out : torch.Tensor, optional
        ``(Z, Y, X)`` uint8 / uint16 on ``logits.device``; allocated if absent.
    backend : "auto" | "metal" | "torch" | "triton"
    coord_dtype : np.float64 | np.float32
        Coordinate arithmetic; float32 mimics the nnunet-inference-mlx kernel exactly.
    """
    lg = _as_torch(logits)
    if lg.ndim != 4:
        raise ValueError(f"logits must be (K, Z, Y, X); got shape {tuple(lg.shape)}")
    if not lg.dtype.is_floating_point:
        raise TypeError(f"logits must be floating point; got {lg.dtype}")
    K = int(lg.shape[0])
    src_shape = tuple(int(s) for s in lg.shape[1:])
    out_shape = grid.shape if isinstance(grid, Grid) else tuple(int(x) for x in grid)
    if len(out_shape) != 3:
        raise ValueError(f"grid must be a Grid or a (Z, Y, X) shape; got {grid!r}")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}; got {mode!r}")
    lut_arr = np.arange(K, dtype=np.int64) if lut is None else np.asarray(lut, dtype=np.int64).reshape(-1)
    if lut_arr.shape[0] != K:
        raise ValueError(f"lut must have K={K} entries; got {lut_arr.shape[0]}")
    if lut_arr.min() < 0 or int(background) < 0:
        raise ValueError("labels must be non-negative")
    if transparent not in TRANSPARENT:
        raise ValueError(f"transparent must be one of {TRANSPARENT}; got {transparent!r}")
    if transparent != "background" and not (paint and mode == "argmax"):
        raise ValueError(f"transparent={transparent!r} applies to an argmax paint only")
    max_label = max(int(lut_arr.max()), int(background))

    if out is None:
        if out_dtype is not None:
            dt = out_dtype
        elif max_label <= 255:
            dt = torch.uint8
        elif UINT16 is None:
            raise ValueError(f"label {max_label} needs a uint16 label map, which needs torch >= 2.3 "
                             f"(this is {torch.__version__})")
        else:
            dt = UINT16
        out = torch.zeros(out_shape, dtype=dt, device=lg.device)
    else:
        if tuple(out.shape) != tuple(out_shape):
            raise ValueError(f"out has shape {tuple(out.shape)}, grid has {out_shape}")
        if out.device != lg.device:
            raise ValueError(f"out is on {out.device}, logits on {lg.device}")
    if out.dtype not in LABEL_DTYPES:
        raise TypeError(f"out must be uint8 or uint16; got {out.dtype}")
    if max_label > (255 if out.dtype == torch.uint8 else 65535):
        raise ValueError(f"label {max_label} does not fit {out.dtype}")

    choice = backends.select(backend, lg.device, tuple(lg.shape), out_shape)
    if choice.fallback:
        # `segment` selects first and records this as a deviation, passing the name it chose;
        # a caller of this function would otherwise never learn it got the slower backend
        warnings.warn(f"labelfield.to_labels: {choice.fallback}; restoring with the torch backend, "
                      f"which has no such limit but is slower", RuntimeWarning, stacklevel=2)
    tables = build_tables(out_shape, src_shape, mapping, interp=interp, outside=outside, coord_dtype=coord_dtype)
    opts = {"slab_voxels": int(slab_voxels)} if choice.name == "metal" else {}
    klut = kernel_lut(lut_arr, mode=mode, paint=bool(paint), transparent=transparent)
    choice.module.run(lg, out, tables, klut, mode=mode, paint=bool(paint),
                      background=int(background), threshold=float(threshold), **opts)
    return out


def resample_argmax(logits, out_shape, mapping: Mapping, **kw) -> torch.Tensor:
    """Mechanism-level name: trilinear resample of the logits + argmax, fused."""
    return to_labels(logits, out_shape, mapping, mode="argmax", **kw)


def resample_paint(logits, out_shape, mapping: Mapping, labels, *, threshold: float = 0.0, **kw) -> torch.Tensor:
    """Mechanism-level name for region (sigmoid-head) models: trilinear resample
    of each region's logit, threshold, paint ``labels[k]`` in channel order."""
    return to_labels(logits, out_shape, mapping, mode="regions", lut=labels, threshold=threshold, **kw)


def available_backends() -> list[str]:
    return backends.available_backends()
