"""Per-axis index and weight tables, computed on the host (CPU, numpy).

This is the only place output coordinates are computed. Every backend consumes the same
tables, so backends can differ only in float32 rounding of the interpolation and hence in
near-tied decisions, never in which samples they read.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import Mapping

INTERP = ("linear", "nearest")
OUTSIDE = ("background", "clamp")


@dataclass(frozen=True)
class AxisTable:
    """Interpolation table for one axis: for each output index, the two source indices to
    blend and the weight of the second.

    Attributes
    ----------
    i0, i1 : int32 arrays of length ``n``
        Source indices. The interpolated value is ``(1 - f) * v[i0] + f * v[i1]``. For nearest
        interpolation ``i0 == i1``. ``i0 == -1`` marks an output index outside the source
        (possible only with ``outside="background"``); ``i1`` is then a valid but unused index.
    f : float32 array
        Weight of ``i1``, in ``[0, 1)``; 0 for nearest. The backends use this.
    f64 : float64 array
        The same weight before rounding to float32. The float64 reference uses this.
    """

    i0: np.ndarray   # int32
    i1: np.ndarray   # int32
    f: np.ndarray    # float32, weight of i1 (0 for nearest) - what the kernels use
    f64: np.ndarray  # float64, the same weight before rounding - what the reference uses

    @property
    def n(self) -> int:
        """Number of output indices along this axis."""
        return int(self.i0.shape[0])


def normalize_interp(interp) -> tuple[str, str, str]:
    """``interp`` as a (Z, Y, X) tuple of "linear" / "nearest"; raises ``ValueError`` otherwise."""
    if isinstance(interp, str):
        interp = (interp, interp, interp)
    t = tuple(str(x) for x in interp)
    if len(t) != 3 or any(x not in INTERP for x in t):
        raise ValueError(f"interp must be 'linear' / 'nearest' or a (Z, Y, X) tuple of them; got {interp!r}")
    return t


def axis_coords(c, n_src: int, *, interp: str = "linear", outside: str = "background"):
    """The per-axis sampling rule, applied to arbitrary coordinates.

    Parameters
    ----------
    c : array-like of float, any shape
        Continuous source coordinates along one axis, in voxel units (integers are voxel
        centers). Converted to float64.
    n_src : int
        Number of source samples along the axis.
    interp : "linear" | "nearest", default "linear"
    outside : "background" | "clamp", default "background"

    Returns
    -------
    valid : bool array, shape of ``c``
        Whether each coordinate is inside the source.
    i0, i1 : int64 arrays, shape of ``c``
        The two samples to blend. Computed for every coordinate, including invalid ones, where
        they are in range but meaningless; callers mask them with ``valid``.
    f : float64 array, shape of ``c``
        Weight of ``i1``; 0 for nearest.

    Rules. *Linear*: inside if ``-0.5 <= c <= n_src - 0.5``. *Nearest*: inside if
    ``-0.5 <= c < n_src - 0.5``, i.e. only if the sample it rounds to exists. With
    ``outside="clamp"`` every coordinate is inside. An inside coordinate is first clamped to
    ``[0, n_src - 1]``, so in the half voxel beyond the first or last sample center the edge
    value is repeated (as skimage ``mode="edge"`` and scipy ``mode="nearest"``). Then, for
    linear, ``i0 = floor(c)``, ``f = c - i0``, ``i1 = min(i0 + 1, n_src - 1)``; for nearest,
    ``i0 = i1 = floor(c + 0.5)`` (a coordinate exactly half-way between two samples rounds up,
    as in ``scipy.ndimage`` order 0) and ``f = 0``.

    :func:`axis_table` uses this function, so it gives the same decisions as the tables for any
    coordinate both can express; use it directly for coordinates from an
    :class:`~labelfield.mapping.Affine`. Raises ``ValueError`` for an unknown ``interp`` or
    ``outside``.
    """
    if interp not in INTERP:
        raise ValueError(f"interp must be one of {INTERP}; got {interp!r}")
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}; got {outside!r}")
    n_src = int(n_src)
    c = np.asarray(c, dtype=np.float64)
    if outside == "clamp":
        valid = np.ones(c.shape, dtype=bool)
    else:
        valid = (c >= -0.5) & ((c <= n_src - 0.5) if interp == "linear" else (c < n_src - 0.5))
    c = np.clip(c, 0.0, float(n_src - 1))
    if interp == "linear":
        i0 = np.floor(c)
        f = c - i0
        i1 = np.minimum(i0 + 1, n_src - 1)
    else:
        i0 = np.minimum(np.floor(c + 0.5), n_src - 1)
        i1 = i0
        f = np.zeros_like(c)
    return valid, i0.astype(np.int64), i1.astype(np.int64), f


def axis_table(n_out: int, n_src: int, a: float, b: float, *, interp: str = "linear",
               outside: str = "background", coord_dtype=np.float64, start: int = 0,
               centered: bool = False, pre: int = 0, post: int = 0) -> AxisTable:
    """The :class:`AxisTable` for one axis of the map ``x_src = a * j + b``, for output indices
    ``j = start, ..., start + n_out - 1``.

    Parameters
    ----------
    n_out, n_src : int >= 1
        Number of output indices and of source samples.
    a, b : float
        Scale and offset of the map (one axis of a :class:`~labelfield.mapping.Mapping`).
    interp, outside
        As in :func:`axis_coords`, which makes every decision.
    coord_dtype : np.float64 (default) or np.float32
        Precision in which the coordinate is evaluated. float64 matches scipy and
        scikit-image. float32 matches the nnunet-inference-mlx Metal kernel, which computes
        ``(float)j * s2t`` in float32. The inside test and rounding are then done in float64,
        which represents either result exactly.
    start : int, default 0
        Global index of the first output index. Used to build the table for a slab of a
        larger output grid: the coordinate is computed from the global integer index, so each
        decision equals the one the whole grid would make. Folding the offset into ``b``
        instead adds a second rounding, which can flip a nearest pick that lies exactly
        half-way between two samples.
    centered : bool, default False
        Evaluate the coordinate as ``(j + 0.5) * a - 0.5`` (the arithmetic of scipy and
        scikit-image for the voxel-center rule) instead of ``a * j + b``; ``b`` is then
        ignored. :func:`build_tables` passes ``Mapping.centered``.
    pre, post : int, default 0
        Integer shifts before and after the map: the coordinate is ``map(j + pre) - post``,
        with both steps exact. :func:`build_tables` passes them from ``Mapping.terms()``, and
        ``b`` is then the core offset, not the folded one.

    An output index outside the source gets ``i0 = -1`` (only with ``outside="background"``).
    Raises ``ValueError`` for invalid arguments.
    """
    if interp not in INTERP:
        raise ValueError(f"interp must be one of {INTERP}; got {interp!r}")
    if outside not in OUTSIDE:
        raise ValueError(f"outside must be one of {OUTSIDE}; got {outside!r}")
    dt = np.dtype(coord_dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"coord_dtype must be float32 or float64; got {coord_dtype!r}")
    n_out, n_src = int(n_out), int(n_src)
    if n_out < 1 or n_src < 1:
        raise ValueError("n_out and n_src must be >= 1")
    # pre / post: integer shifts before and after the map (Mapping.exact), applied as exact
    # integer steps so a crop offset rounds no coordinate differently than the uncropped map
    j = np.arange(int(start) + int(pre), int(start) + int(pre) + n_out, dtype=dt)
    if centered:
        c = ((j + dt.type(0.5)) * dt.type(a) - dt.type(0.5)).astype(np.float64)
    else:
        c = (j * dt.type(a) + dt.type(b)).astype(np.float64)
    if post:
        c = c - float(post)
    valid, i0, i1, f = axis_coords(c, n_src, interp=interp, outside=outside)
    i0 = i0.astype(np.int32)
    i0[~valid] = -1
    return AxisTable(i0, i1.astype(np.int32), f.astype(np.float32), f.astype(np.float64))


def build_tables(out_shape, src_shape, mapping: Mapping, *, interp="linear", outside: str = "background",
                 coord_dtype=np.float64, out_start=(0, 0, 0)) -> tuple[AxisTable, AxisTable, AxisTable]:
    """The three :class:`AxisTable` objects, (Z, Y, X), that :func:`~labelfield.labels.to_labels`
    passes to the backends.

    Parameters
    ----------
    out_shape, src_shape : 3 ints, (Z, Y, X)
        Shape of the output grid and of the source (model) grid.
    mapping : Mapping
        Output index -> source coordinate. Its ``centered`` flag is honored.
    interp : "linear" | "nearest" or a (Z, Y, X) tuple of them, default "linear"
    outside : "background" | "clamp", default "background"
    coord_dtype : np.float64 (default) or np.float32
        See :func:`axis_table`.
    out_start : 3 ints, default (0, 0, 0)
        Index, in the full output grid that ``mapping`` was built for, of this output's voxel
        (0, 0, 0). See ``start`` in :func:`axis_table`.

    Raises ``ValueError`` for shapes that are not 3 entries long or other invalid arguments.
    """
    interp3 = normalize_interp(interp)
    out_shape = tuple(int(x) for x in out_shape)
    src_shape = tuple(int(x) for x in src_shape)
    out_start = tuple(int(x) for x in out_start)
    if len(out_shape) != 3 or len(src_shape) != 3 or len(out_start) != 3:
        raise ValueError("out_shape, src_shape and out_start must be (Z, Y, X)")
    core_b, pre, post = (mapping.terms() if hasattr(mapping, "terms")
                         else (mapping.b, (0, 0, 0), (0, 0, 0)))
    return tuple(
        axis_table(out_shape[ax], src_shape[ax], mapping.a[ax], core_b[ax],
                   interp=interp3[ax], outside=outside, coord_dtype=coord_dtype, start=out_start[ax],
                   centered=bool(getattr(mapping, "centered", False)), pre=pre[ax], post=post[ax])
        for ax in range(3)
    )
