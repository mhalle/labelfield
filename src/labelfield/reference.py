"""Float64 numpy reference implementation of the interpolation and decision.

It defines what the backends compute and is used to test them. It needs only numpy, is not
optimized, and materializes all K channels at output resolution, so use it for small shapes
only. Its inputs are the tables from :func:`labelfield.build_tables`.
"""
from __future__ import annotations

import numpy as np

from .tables import AxisTable


def _gather_idx(t: AxisTable):
    i0 = np.where(t.i0 < 0, 0, t.i0).astype(np.int64)
    return i0, t.i1.astype(np.int64), t.f64


def interpolate(logits, tables) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate all K channels onto the output grid.

    ``logits`` is array-like (K, Zs, Ys, Xs); ``tables`` the (Z, Y, X) tables from
    :func:`~labelfield.tables.build_tables`. Returns ``(values, valid)``: ``values`` float64
    (K, Za, Ya, Xa), and ``valid`` bool (Za, Ya, Xa), False where the voxel is outside the
    model grid (``values`` there is computed from clamped indices and is meaningless).

    The blend order is the backends': X, then Y, then Z, each as ``(1 - f) * v0 + f * v1``, but
    in float64 with the float64 weights. The backends use float32, which changes values by
    about 1e-7 relative and can change decisions where two channels are nearly tied.
    """
    lg = np.asarray(logits, dtype=np.float64)
    tz, ty, tx = tables
    z0, z1, zf = _gather_idx(tz)
    y0, y1, yf = _gather_idx(ty)
    x0, x1, xf = _gather_idx(tx)

    def g(iz, iy, ix):
        return lg[:, iz[:, None, None], iy[None, :, None], ix[None, None, :]]

    wx = xf[None, None, None, :]
    wy = yf[None, None, :, None]
    wz = zf[None, :, None, None]
    c00 = g(z0, y0, x0) * (1 - wx) + g(z0, y0, x1) * wx
    c01 = g(z0, y1, x0) * (1 - wx) + g(z0, y1, x1) * wx
    c10 = g(z1, y0, x0) * (1 - wx) + g(z1, y0, x1) * wx
    c11 = g(z1, y1, x0) * (1 - wx) + g(z1, y1, x1) * wx
    c0 = c00 * (1 - wy) + c01 * wy
    c1 = c10 * (1 - wy) + c11 * wy
    values = c0 * (1 - wz) + c1 * wz
    valid = (tz.i0 >= 0)[:, None, None] & (ty.i0 >= 0)[None, :, None] & (tx.i0 >= 0)[None, None, :]
    return values, valid


def decide(values: np.ndarray, valid: np.ndarray, *, lut=None, mode: str = "argmax", threshold: float = 0.0,
           background: int = 0, paint: bool = False, transparent: str = "background",
           out: np.ndarray | None = None) -> np.ndarray:
    """Decide labels from interpolated ``values`` (K, Za, Ya, Xa) and ``valid`` (Za, Ya, Xa), with
    the same semantics as :func:`labelfield.to_labels` (see there for ``lut``, ``mode``,
    ``threshold``, ``background``, ``paint`` and ``transparent``).

    Returns an int64 array (Za, Ya, Xa), or ``out`` (a numpy array, written in place) if given.
    ``paint=True`` requires ``out``. Raises ``ValueError`` for an unknown ``mode``. Unlike
    ``to_labels``, it does not validate ``transparent`` or the labels."""
    K = values.shape[0]
    lut = np.arange(K, dtype=np.int64) if lut is None else np.asarray(lut, dtype=np.int64).reshape(-1)
    if mode == "argmax":
        best = values.argmax(0)            # first maximal channel
        lab = lut[best]
        # what a paint leaves untouched: channel 0, or (TotalSegmentator's rule) any label 0
        hit = best != 0 if transparent == "background" else lab != 0
    elif mode == "regions":
        lab = np.full(values.shape[1:], int(background), dtype=np.int64)
        hit = np.zeros(values.shape[1:], dtype=bool)
        for k in range(K):                 # channel order = paint priority; later wins
            m = values[k] > threshold
            lab = np.where(m, lut[k], lab)
            hit |= m
    else:
        raise ValueError(f"mode must be 'argmax' or 'regions'; got {mode!r}")
    if paint:
        if out is None:
            raise ValueError("paint=True needs an `out` buffer to paint into")
        mask = valid & hit
        out[mask] = lab[mask]
        return out
    res = np.where(valid, lab, int(background))
    if out is not None:
        out[...] = res
        return out
    return res


def labels(logits, tables, **kw) -> np.ndarray:
    """``decide(*interpolate(logits, tables), **kw)``."""
    values, valid = interpolate(logits, tables)
    return decide(values, valid, **kw)


def margins(values: np.ndarray) -> np.ndarray:
    """Per voxel, the largest channel value minus the second largest: array (Za, Ya, Xa).

    Where the margin is small (below about 1e-4 for typical logits), float32 backends may
    choose a different argmax than the float64 reference. Infinite when K < 2."""
    if values.shape[0] < 2:
        return np.full(values.shape[1:], np.inf)
    part = np.partition(values, -2, axis=0)
    return part[-1] - part[-2]
