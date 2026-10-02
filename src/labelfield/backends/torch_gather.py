"""Portable torch backend: index_select and linear blends per output Z plane. Runs on any torch
device, computing in float32; it is also what "auto" falls back to when a fused kernel cannot
address a field (and what the CPU runs).

Two paths with the same result. The dense path interpolates every channel at every output voxel
of a plane, from a cache of X/Y-interpolated source planes. The argmax path (``mode="argmax"``,
finite logits) first prunes channels per model cell: an interpolated value is a weighted average
of the cell's corners, so a channel whose largest corner value lies below another channel's
smallest corner value cannot win anywhere in that cell. A cell left with one candidate gets that
channel's label without interpolating; the remaining voxels are blended over their candidates
only, with the dense path's arithmetic in the dense path's order, so the labels are the dense
path's bit for bit. On a CPU most cells - organ interiors and background - have one candidate,
which is where the time goes otherwise (2026-10-02: a 118-channel field onto a 768x768x709 grid
took 134 s on 8 cores)."""
from __future__ import annotations

import numpy as np
import torch

from ..tables import AxisTable


def available() -> bool:
    return True


def _axis(t: AxisTable, device):
    i0 = torch.from_numpy(np.where(t.i0 < 0, 0, t.i0).astype(np.int64)).to(device)
    i1 = torch.from_numpy(t.i1.astype(np.int64)).to(device)
    f = torch.from_numpy(np.ascontiguousarray(t.f, dtype=np.float32)).to(device)
    return i0, i1, f


#: Slack on the pruning test, relative to the logits' scale: a channel is dropped only when its
#: largest corner value is below the best smallest corner value by more than this, so float32
#: rounding in the blend (a few ulp) cannot make a dropped channel the computed maximum.
PRUNE_MARGIN = 1e-4


@torch.no_grad()
def run(logits: torch.Tensor, out: torch.Tensor, tables, lut, *, mode: str, paint: bool,
        background: int, threshold: float, skip=None, prune: bool | None = None) -> None:
    """Restore ``logits`` into ``out``. ``prune`` (default: argmax mode with finite logits) takes
    the argmax path; ``False`` forces the dense path, which the tests compare it with."""
    if prune is None:
        prune = mode == "argmax" and bool(torch.isfinite(logits).all())
    if prune:
        if mode != "argmax":
            raise ValueError("prune=True needs mode='argmax'")
        return _run_argmax_pruned(logits, out, tables, lut, paint=paint, background=background, skip=skip)
    return _run_dense(logits, out, tables, lut, mode=mode, paint=paint, background=background,
                      threshold=threshold, skip=skip)


def _pairs(i0: torch.Tensor, i1: torch.Tensor, n: int):
    """The distinct (i0, i1) source pairs of an axis and, per output index, its pair's number."""
    code = i0 * (n + 1) + i1
    uniq, inv = torch.unique(code, return_inverse=True)
    return uniq // (n + 1), uniq % (n + 1), inv


def _run_argmax_pruned(logits, out, tables, lut, *, paint, background, skip):
    device = logits.device
    K, Zt, Yt, Xt = logits.shape
    Za, Ya, Xa = out.shape
    tz, ty, tx = tables
    y0, y1, yf = _axis(ty, device)
    x0, x1, xf = _axis(tx, device)
    valid_plane = torch.from_numpy((ty.i0 >= 0)[:, None] & (tx.i0 >= 0)[None, :]).to(device)
    lut_t = torch.from_numpy(np.ascontiguousarray(lut, dtype=np.int32)).to(device)
    if skip is None:
        from . import default_skip
        skip = default_skip(K)
    skip_t = torch.from_numpy(np.ascontiguousarray(skip, dtype=bool)).to(device)
    bg = torch.full((Ya, Xa), int(background), dtype=torch.int32, device=device)
    wx0 = 1.0 - xf
    wy0 = 1.0 - yf
    uy0, uy1, inv_y = _pairs(y0, y1, Yt)
    ux0, ux1, inv_x = _pairs(x0, x1, Xt)
    nx = int(ux0.numel())
    channel = torch.arange(K, device=device)
    planes: dict[int, torch.Tensor] = {}
    cells: dict[tuple[int, int], tuple] = {}

    def source(zi: int) -> torch.Tensor:
        """Source plane ``zi`` as float32, flattened (K * Yt * Xt), for gathers."""
        got = planes.get(zi)
        if got is None:
            got = logits[:, zi].float().reshape(-1)
            planes[zi] = got
        return got

    def cell_view(za: int, zb: int):
        """Per (Y pair, X pair) cell of source planes za/zb: the number of candidate channels
        and, padded with K, the candidates in increasing order (ncell, C)."""
        got = cells.get((za, zb))
        if got is None:
            a = logits[:, za].float()
            hi, lo = (a, a) if zb == za else (torch.maximum(a, logits[:, zb].float()),
                                              torch.minimum(a, logits[:, zb].float()))
            hi = torch.maximum(hi.index_select(2, ux0), hi.index_select(2, ux1))
            hi = torch.maximum(hi.index_select(1, uy0), hi.index_select(1, uy1))
            lo = torch.minimum(lo.index_select(2, ux0), lo.index_select(2, ux1))
            lo = torch.minimum(lo.index_select(1, uy0), lo.index_select(1, uy1))
            best_lo = lo.max(0).values
            cand = (hi >= best_lo - PRUNE_MARGIN * (1.0 + best_lo.abs())).reshape(K, -1)   # (K, ncell)
            count = cand.sum(0)
            C = int(count.max())
            order = torch.where(cand, channel[:, None], K).sort(0).values[:C].t().contiguous()  # (ncell, C)
            got = (count, order)
            cells.clear()                                  # z pairs only move forward
            cells[(za, zb)] = got
            for k in [k for k in planes if k < za]:
                del planes[k]
        return got

    def corner_blend(flat, index, wa, wb, va, vb):
        """The dense path's X-then-Y blend: ``index`` (M, C, 4) addresses the corners
        (y0 x0, y0 x1, y1 x0, y1 x1) of each pixel's candidates in ``flat``."""
        c = flat[index]
        top = c[..., 0] * wa + c[..., 1] * wb
        bot = c[..., 2] * wa + c[..., 3] * wb
        return top * va + bot * vb

    pixel_cell = (inv_y[:, None] * nx + inv_x[None, :])                  # (Ya, Xa)
    z0s, z1s, zfs = tz.i0.tolist(), tz.i1.tolist(), tz.f.tolist()
    for z in range(Za):
        z0 = z0s[z]
        if z0 < 0:
            if not paint:
                out[z].fill_(int(background))
            continue
        z1, w = z1s[z], zfs[z]
        single_plane = z1 == z0 or w == 0.0
        count, order = cell_view(z0, z0 if single_plane else z1)
        best = order[:, 0][pixel_cell]                                    # the only candidate, where one
        mixed = (count[pixel_cell] > 1) & valid_plane
        if bool(mixed.any()):
            ys, xs = mixed.nonzero(as_tuple=True)
            cell = pixel_cell[ys, xs]
            n = count[cell]
            corners = torch.stack([y0[ys] * Xt + x0[xs], y0[ys] * Xt + x1[xs],
                                   y1[ys] * Xt + x0[xs], y1[ys] * Xt + x1[xs]], 1)      # (M, 4)
            # pixels grouped by candidate count, each group padded only to its own width
            widest = order.shape[1]
            lo_n = 1
            for width in [w_ for w_ in (2, 4, 8) if w_ < widest] + [widest]:
                g = ((n > lo_n) & (n <= width)).nonzero().flatten()
                lo_n = width
                if g.numel() == 0:
                    continue
                cand = order[cell[g], :width]                             # (m, width), padded with K
                pad = cand == K
                index = cand.masked_fill(pad, 0)[:, :, None] * (Yt * Xt) + corners[g][:, None, :]
                gx, gy = xs[g], ys[g]
                wts = (wx0[gx, None], xf[gx, None], wy0[gy, None], yf[gy, None])
                v = corner_blend(source(z0), index, *wts)
                if not single_plane:
                    v = v * (1.0 - w) + corner_blend(source(z1), index, *wts) * w
                v = v.masked_fill(pad, float("-inf"))
                best[gy, gx] = cand.gather(1, v.argmax(1, keepdim=True))[:, 0]   # first maximal candidate
        lab = lut_t[best]
        hit = ~skip_t[best]
        if paint:
            out[z] = torch.where(valid_plane & hit, lab.to(out.dtype), out[z])
        else:
            out[z] = torch.where(valid_plane, lab, bg).to(out.dtype)


def _run_dense(logits, out, tables, lut, *, mode, paint, background, threshold, skip):
    device = logits.device
    K, Zt, Yt, Xt = logits.shape
    Za, Ya, Xa = out.shape
    tz, ty, tx = tables
    y0, y1, yf = _axis(ty, device)
    x0, x1, xf = _axis(tx, device)
    valid_plane = torch.from_numpy((ty.i0 >= 0)[:, None] & (tx.i0 >= 0)[None, :]).to(device)
    lut_t = torch.from_numpy(np.ascontiguousarray(lut, dtype=np.int32)).to(device)
    if skip is None:
        from . import default_skip
        skip = default_skip(K)
    skip_t = torch.from_numpy(np.ascontiguousarray(skip, dtype=bool)).to(device)
    bg = torch.full((Ya, Xa), int(background), dtype=torch.int32, device=device)
    wx0 = 1.0 - xf
    wy0 = (1.0 - yf)[:, None]
    wy1 = yf[:, None]
    cache: dict[int, torch.Tensor] = {}

    def plane(zi: int) -> torch.Tensor:
        t = cache.get(zi)
        if t is None:
            p = logits[:, zi].float()                                            # (K, Yt, Xt)
            px = p.index_select(2, x0) * wx0 + p.index_select(2, x1) * xf        # (K, Yt, Xa)
            t = px.index_select(1, y0) * wy0 + px.index_select(1, y1) * wy1      # (K, Ya, Xa)
            cache[zi] = t
        return t

    z0s, z1s, zfs = tz.i0.tolist(), tz.i1.tolist(), tz.f.tolist()
    for z in range(Za):
        z0 = z0s[z]
        if z0 < 0:
            if not paint:
                out[z].fill_(int(background))
            continue
        z1, w = z1s[z], zfs[z]
        a = plane(z0)
        v = a if (z1 == z0 or w == 0.0) else a * (1.0 - w) + plane(z1) * w
        if mode == "argmax":
            best = v.argmax(0)                       # first maximal channel
            lab = lut_t[best]
            hit = ~skip_t[best]                      # a transparent channel won: leave the voxel
        else:
            lab = bg.clone()
            hit = torch.zeros((Ya, Xa), dtype=torch.bool, device=device)
            for k in range(K):
                m = v[k] > threshold
                lab = torch.where(m, lut_t[k], lab)
                hit |= m
        if paint:
            out[z] = torch.where(valid_plane & hit, lab.to(out.dtype), out[z])
        else:
            out[z] = torch.where(valid_plane, lab, bg).to(out.dtype)
        for dead in [k for k in cache if k < z0]:   # z0 is monotonic (a >= 0): older planes are finished
            del cache[dead]
