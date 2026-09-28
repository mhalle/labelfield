"""Backend registry and selection.

Backends: "torch" (:mod:`.torch_gather`, any device), "metal" (:mod:`.metal`, Apple GPUs,
torch >= 2.7) and "triton" (:mod:`.triton_gpu`, CUDA, needs the ``triton`` package).

Each backend module exposes ``available()`` and
``run(logits, out, tables, lut, *, mode, paint, background, threshold, skip=None, **opts)``,
which writes labels into ``out`` in place. The fused backends also expose
``cannot_take(logits_shape, out_shape)``, which returns the reason the kernel cannot address a
field, or None. ``lut`` is the label table as given. ``skip`` is an optional uint8 per channel,
1 meaning that an argmax paint leaves the voxel unchanged when that channel wins
(:func:`labelfield.transparency_mask`); without it, channel 0 is transparent. Most callers use
:func:`labelfield.to_labels` instead of calling ``run``."""
from __future__ import annotations

from typing import NamedTuple

import torch

import numpy as np

from . import metal, torch_gather, triton_gpu


def default_skip(K: int) -> np.ndarray:
    """Channel 0 transparent: the paint rule when a caller passes no ``skip``."""
    skip = np.zeros(int(K), dtype=np.uint8)
    skip[0] = 1
    return skip

BACKENDS = {"torch": torch_gather, "metal": metal, "triton": triton_gpu}
# the fused kernel "auto" takes on each device type, if it is installed and can address the field
FUSED = {"mps": "metal", "cuda": "triton"}


class Choice(NamedTuple):
    name: str
    module: object
    # Why "auto" did not take the device's fused kernel, which is several times faster than
    # the torch backend: a deviation for the caller to report. None when it did, or when the
    # device has no fused kernel to take.
    fallback: str | None = None


def select(name: str, device: torch.device, logits_shape, out_shape) -> Choice:
    """Choose the backend for a ``(K, Zs, Ys, Xs)`` field on ``device`` and an output of
    ``out_shape`` (Z, Y, X). Returns a ``Choice(name, module, fallback)``.

    ``name="auto"``: on an MPS device "metal", on a CUDA device "triton", if that backend is
    available and its ``cannot_take`` accepts the shapes; otherwise "torch". ``fallback`` is
    None, or a message saying why an available fused kernel was not used (a model-grid channel
    of 2^31 voxels or more for either kernel; for Triton also an output of 2^31 voxels or
    more). "auto" decides from device and shapes only; it does not consider dtype or
    contiguity, which the chosen backend checks when it runs.

    A backend named explicitly is never replaced: ``ValueError`` if the name is unknown, if
    the device is not the backend's ("metal" needs MPS, "triton" needs CUDA), if Triton is
    unavailable, or if the kernel cannot address the field.
    """
    shapes = (tuple(int(v) for v in logits_shape), tuple(int(v) for v in out_shape))
    if name == "auto":
        fused = FUSED.get(device.type)
        if fused is None or not BACKENDS[fused].available():
            return Choice("torch", torch_gather)
        why = BACKENDS[fused].cannot_take(*shapes)
        if why is None:
            return Choice(fused, BACKENDS[fused])
        return Choice("torch", torch_gather, f"the {fused} kernel cannot take this field: {why}")
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose from {sorted(BACKENDS)} or 'auto'")
    if name != "torch":
        homes = sorted(d for d, n in FUSED.items() if n == name)
        if device.type not in homes:
            raise ValueError(f"backend={name!r} needs logits on a {' or '.join(map(repr, homes))} device")
        if name == "triton" and not triton_gpu.available():
            raise ValueError(f"backend='triton' unavailable: {triton_gpu.why_unavailable()}")
        why = BACKENDS[name].cannot_take(*shapes)
        if why is not None:
            raise ValueError(f"backend={name!r} cannot take this field: {why}; "
                             f"use backend='torch' (or 'auto', which does)")
    return Choice(name, BACKENDS[name])


def available_backends() -> list[str]:
    """Names of the backends whose ``available()`` is true in this process."""
    return [n for n, m in BACKENDS.items() if m.available()]
