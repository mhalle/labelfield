"""Logit field -> label map on a caller-chosen output grid.

For every voxel of the output grid, interpolate the K logits from the model grid and decide a
label (argmax, or per-region threshold), optionally compositing into an existing label map, in
one pass: the K interpolated channels are never materialized at output resolution.
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


def transparency_mask(lut, *, transparent: str = "background") -> np.ndarray:
    """Per-channel transparency for an argmax paint: uint8 array of length K, 1 = transparent.

    ``lut`` is the label table (K ints). With ``transparent="background"`` only channel 0 is
    transparent; with ``"zero"`` every channel whose label is 0 is (TotalSegmentator's
    compositing rule). :func:`to_labels` computes this and passes it to the backend; call it
    directly only when driving a backend's ``run`` yourself. Any other value raises
    ``ValueError``.
    """
    if transparent not in TRANSPARENT:
        raise ValueError(f"transparent must be one of {TRANSPARENT}; got {transparent!r}")
    lut = np.asarray(lut, dtype=np.int64).reshape(-1)
    if transparent == "background":
        mask = np.zeros(lut.shape[0], dtype=np.uint8)
        mask[0] = 1
        return mask
    return (lut == 0).astype(np.uint8)


def to_labels(logits, grid, mapping: Mapping, *, interp="linear", outside: str = "background",
              lut=None, mode: str = "argmax", paint: bool = False, transparent: str = "background",
              threshold: float = 0.0, background: int = 0, out: torch.Tensor | None = None,
              out_dtype=None, backend: str = "auto", coord_dtype=np.float64,
              slab_voxels: int = 1 << 26, out_start=(0, 0, 0)) -> torch.Tensor:
    """Label map on an output grid from a ``(K, Z, Y, X)`` logit field on a model grid.

    For each output voxel, the K logits are interpolated from the model grid at the coordinate
    given by ``mapping`` and a label is decided, in one pass, without materializing the K
    interpolated channels at output resolution.

    Parameters
    ----------
    logits : torch.Tensor or numpy.ndarray, shape (K, Zs, Ys, Xs), floating point
        Per-class scores on the model grid, normally the network's raw outputs (logits,
        before softmax or sigmoid); they are interpolated as given. A numpy array is wrapped as a CPU tensor without copying when it is
        contiguous. The tensor is not moved or cast: the computation runs on its device.
        Supported dtypes depend on the backend: torch takes any floating dtype (computing in
        float32), Metal float32 and float16, Triton float32, float16 and bfloat16.
    grid : Grid or 3 ints (Z, Y, X)
        The output grid. Only its shape is used; its spacing and origin enter only through
        ``mapping``.
    mapping : Mapping
        Output voxel index -> continuous model-grid coordinate, per axis. For example
        ``Mapping.center(output_shape, model_shape)``, ``Mapping.corner(...)``,
        ``Mapping.spacing(...)`` or ``Mapping.between(output_grid, model_grid)``; see the
        documentation on choosing one. An :class:`~labelfield.mapping.Affine` is not accepted
        (use its ``separable`` Mapping when it has one).
    interp : "linear" | "nearest" or a (Z, Y, X) tuple of them, default "linear"
        Interpolation per axis. "linear" on all axes is trilinear interpolation. Nearest along
        Z only, ``("nearest", "linear", "linear")``, reproduces nnU-Net's separate-Z export
        (``order_z=0``) for anisotropic data.
    outside : "background" | "clamp", default "background"
        Handling of output voxels whose coordinate falls outside the model grid (more than half
        a voxel beyond the first or last sample center; see
        :func:`~labelfield.tables.axis_coords`). "background": they receive ``background``, or
        are left unchanged when painting. "clamp": every coordinate is clamped into the model
        grid, so the edge values extend indefinitely.
    lut : sequence of K non-negative ints, optional
        Label table: ``lut[k]`` is the label written when channel ``k`` is chosen. Defaults to
        ``range(K)``.
    mode : "argmax" | "regions", default "argmax"
        "argmax": write ``lut[k]`` for the channel with the largest interpolated value (the
        lowest ``k`` among exact ties). "regions": for sigmoid-output models whose channels may
        overlap; every channel whose interpolated value exceeds ``threshold`` writes ``lut[k]``,
        in channel order, so a later channel wins where several exceed it. A voxel where no
        channel exceeds it receives ``background``.
    paint : bool, default False
        Composite into ``out`` instead of overwriting it. Voxels whose decision is transparent
        (see ``transparent``; in "regions" mode, voxels where no channel exceeds ``threshold``)
        and voxels outside the model grid are left unchanged. Calling ``to_labels`` once per
        model with its own ``lut`` and the same ``out`` composites several models into one
        label map; later calls win where they write. Pass ``out``; without it a zero-filled
        buffer is allocated and painted.
    transparent : "background" | "zero", default "background"
        Which argmax decisions a paint leaves unchanged. "background": the winning channel is
        channel 0. "zero": the winning channel's label ``lut[k]`` is 0, which also covers
        auxiliary classes mapped to 0 (TotalSegmentator's compositor,
        ``np.copyto(out, lut[seg], where=lut[seg] != 0)``). Any value other than "background"
        requires ``paint=True`` and ``mode="argmax"``.
    threshold : float, default 0.0
        "regions" mode only: a channel is on where its interpolated value is strictly greater
        than this. For logits of a sigmoid output, 0.0 corresponds to probability 0.5. Ignored
        in "argmax" mode.
    background : int >= 0, default 0
        Label written, when not painting, to voxels outside the model grid and, in "regions"
        mode, to voxels where no channel is on.
    out : torch.Tensor, optional
        Destination, shape equal to the output grid, dtype uint8 or uint16, on the same device
        as ``logits``. Written in place and returned. The Metal and Triton backends require it
        to be contiguous. If omitted, a zero-filled tensor is allocated.
    out_dtype : torch.dtype, optional
        dtype of the allocated ``out`` (ignored when ``out`` is given): ``torch.uint8`` or
        ``torch.uint16``. By default uint8 if every label (``lut`` and ``background``) is at
        most 255, otherwise uint16, which requires torch >= 2.3.
    backend : "auto" | "torch" | "metal" | "triton", default "auto"
        "auto" uses the fused kernel for the device ("metal" on MPS, "triton" on CUDA) when it
        is available and can address the field, and otherwise "torch", emitting a
        ``RuntimeWarning`` if it had to fall back from an available kernel. A backend named
        explicitly is never replaced; if it cannot run, ``ValueError`` is raised. See
        :func:`labelfield.backends.select`.
    coord_dtype : np.float64 (default) or np.float32
        Precision of the coordinate computation; see :func:`~labelfield.tables.axis_table`.
        float32 reproduces the nnunet-inference-mlx kernel bit for bit.
    slab_voxels : int, default 2**26
        Metal only: approximate number of output voxels per kernel launch (rounded down to
        whole Z planes, at least one plane). Ignored by other backends.
    out_start : 3 ints (Z, Y, X), default (0, 0, 0)
        Position of this output's voxel (0, 0, 0) within the larger output grid that
        ``mapping`` was built for. Output voxel ``j`` samples at ``mapping(j + out_start)``,
        computed from that integer index. Restoring a large grid in slabs
        (``out=big[z0:z1]``, ``out_start=(z0, 0, 0)``) then gives exactly the labels of a
        single call.

    Returns
    -------
    torch.Tensor, shape of the output grid, uint8 or uint16, on ``logits.device``
        ``out`` if it was given, otherwise the new tensor.

    Raises
    ------
    ValueError
        ``logits`` not 4-D; ``grid`` not 3-D; unknown ``mode``, ``transparent``, ``interp`` or
        ``outside``; ``lut`` length not K; a negative label; ``transparent`` misused; ``out``
        of the wrong shape or device; a label that does not fit the output dtype (or needs
        uint16 on torch < 2.3); an explicitly named backend that is unavailable, on the wrong
        device, or cannot address the field; a non-contiguous ``out`` on an explicitly named
        fused backend.
    TypeError
        ``mapping`` not a :class:`Mapping` (an :class:`Affine` included: use its ``separable``);
        ``logits`` not floating point; ``out`` not uint8/uint16; a logits dtype an explicitly
        named fused backend does not support. ("auto" uses the torch backend instead for the
        dtype and contiguity cases, with a ``RuntimeWarning``.)
    """
    if not isinstance(mapping, Mapping):
        hint = (" (an Affine: use aff.separable, which is None when the grids do not line up)"
                if hasattr(mapping, "separable") else "")
        raise TypeError(f"mapping must be a labelfield.Mapping; got {type(mapping).__name__}{hint}")
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

    choice = backends.select(backend, lg.device, tuple(lg.shape), out_shape, logits_dtype=lg.dtype,
                             out_contiguous=out.is_contiguous())
    if choice.fallback:
        # without a warning a caller would not learn that it got the slower backend
        warnings.warn(f"labelfield.to_labels: {choice.fallback}; restoring with the torch backend, "
                      f"which has no such limit but is slower", RuntimeWarning, stacklevel=2)
    tables = build_tables(out_shape, src_shape, mapping, interp=interp, outside=outside, coord_dtype=coord_dtype,
                          out_start=out_start)
    opts = {"slab_voxels": int(slab_voxels)} if choice.name == "metal" else {}
    choice.module.run(lg, out, tables, lut_arr.astype(np.int32), mode=mode, paint=bool(paint),
                      skip=transparency_mask(lut_arr, transparent=transparent),
                      background=int(background), threshold=float(threshold), **opts)
    return out


def resample_argmax(logits, out_shape, mapping: Mapping, **kw) -> torch.Tensor:
    """``to_labels(logits, out_shape, mapping, mode="argmax", **kw)``: interpolate the logits
    (trilinear by default) and take the argmax. Keyword arguments are those of :func:`to_labels`."""
    return to_labels(logits, out_shape, mapping, mode="argmax", **kw)


def resample_paint(logits, out_shape, mapping: Mapping, labels, *, threshold: float = 0.0, **kw) -> torch.Tensor:
    """``to_labels(logits, out_shape, mapping, mode="regions", lut=labels, threshold=threshold,
    **kw)``, for models with one sigmoid output per region: interpolate each channel, and write
    ``labels[k]`` wherever channel ``k`` exceeds ``threshold``, later channels winning.
    Despite the name, this overwrites ``out`` unless ``paint=True`` is passed."""
    return to_labels(logits, out_shape, mapping, mode="regions", lut=labels, threshold=threshold, **kw)


def available_backends() -> list[str]:
    """Names of the backends that can run in this process: always "torch"; "metal" if an MPS
    device is available and torch >= 2.7; "triton" if Triton imports and a CUDA device is
    available. Availability does not mean a backend can address every field size."""
    return backends.available_backends()
