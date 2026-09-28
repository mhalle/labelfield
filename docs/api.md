# API reference

All names are available from the top-level package (`import labelfield as lf`). Axis order
is `(Z, Y, X)` everywhere; a logit field is `(K, Z, Y, X)`. Names marked *torch* import torch
on first access; the others need only numpy. The docstrings in `src/labelfield/` are the
authoritative, more detailed version of this page.

Contents: [to_labels](#to_labels) ·
[resample_argmax, resample_paint](#resample_argmax-resample_paint) ·
[Grid](#grid) · [Mapping](#mapping) · [Affine](#affine) ·
[build_tables, axis_table, axis_coords, AxisTable](#tables) ·
[backends](#backends) · [transparency_mask](#transparency_mask) · [reference](#reference)

## to_labels

*torch*

```text
to_labels(logits, grid, mapping, *, interp="linear", outside="background", lut=None,
          mode="argmax", paint=False, transparent="background", threshold=0.0,
          background=0, out=None, out_dtype=None, backend="auto", coord_dtype=np.float64,
          slab_voxels=2**26, out_start=(0, 0, 0)) -> torch.Tensor
```

Label map on an output grid from a logit field on a model grid.

| Parameter | Type / values | Meaning |
|---|---|---|
| `logits` | `torch.Tensor` or `numpy.ndarray`, `(K, Zs, Ys, Xs)`, floating point | Per-class logits on the model grid. Not moved or cast; the computation runs on its device (numpy: CPU). |
| `grid` | `Grid` or 3 ints | Output grid. Only the shape is used. |
| `mapping` | `Mapping` | Output index to model coordinate. `Affine` is not accepted. |
| `interp` | `"linear"`, `"nearest"`, or a `(Z, Y, X)` tuple of them | Interpolation per axis. |
| `outside` | `"background"`, `"clamp"` | Voxels outside the model grid get `background` (unchanged when painting), or all coordinates are clamped into it. |
| `lut` | K non-negative ints, default `range(K)` | Label written for each channel. |
| `mode` | `"argmax"`, `"regions"` | Argmax over channels (lowest index wins exact ties), or per-channel threshold with later channels winning. |
| `paint` | bool | Leave transparent decisions and outside voxels unchanged in `out`. |
| `transparent` | `"background"`, `"zero"` | For an argmax paint: channel 0 is transparent, or every channel whose label is 0. `"zero"` requires `paint=True`, `mode="argmax"`. |
| `threshold` | float, default 0.0 | Regions mode: a channel is on where its value is strictly greater. |
| `background` | int >= 0, default 0 | Label for outside voxels and, in regions mode, voxels with no channel on (not painting). |
| `out` | `torch.Tensor`, uint8/uint16, output shape, same device | Written in place and returned. Must be contiguous for Metal and Triton. |
| `out_dtype` | `torch.uint8`, `torch.uint16` | dtype of an allocated `out`. Default: uint8 if all labels <= 255, else uint16 (torch >= 2.3). |
| `backend` | `"auto"`, `"torch"`, `"metal"`, `"triton"` | See [backends](#backends). |
| `coord_dtype` | `np.float64`, `np.float32` | Precision of coordinate evaluation. float32 reproduces the nnunet-inference-mlx kernel. |
| `slab_voxels` | int | Metal only: output voxels per kernel launch (whole Z planes). |
| `out_start` | 3 ints | Position of this output within the grid `mapping` was built for; voxel `j` samples `mapping(j + out_start)`. |

Returns `out`, or a new tensor, of the output shape on `logits.device`.

Raises `ValueError` for malformed shapes, unknown option values, a `lut` of the wrong length,
negative labels, labels that do not fit the output dtype, `out` of the wrong shape or device,
an explicitly named backend that cannot run, or a non-contiguous `out` on an explicitly named
fused backend. Raises `TypeError` for a `mapping` that is not a `Mapping` (an `Affine` included:
use `Affine.separable`), non-floating logits, an `out` that is not uint8/uint16, or a logits dtype
an explicitly named fused backend does not support. Emits `RuntimeWarning` when `backend="auto"`
falls back from an available fused kernel to torch (a field too large for it, a logits dtype it
does not read, or a non-contiguous `out`).

## resample_argmax, resample_paint

*torch*

```text
resample_argmax(logits, out_shape, mapping, **kw)
resample_paint(logits, out_shape, mapping, labels, *, threshold=0.0, **kw)
```

`resample_argmax` is `to_labels(..., mode="argmax", **kw)`. `resample_paint` is
`to_labels(..., mode="regions", lut=labels, threshold=threshold, **kw)`; it overwrites `out`
unless `paint=True` is passed.

## Grid

```text
Grid(shape, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0))
```

A regular, axis-aligned grid. `shape`: 3 positive ints. `spacing`: distance between voxel
centers (3 positive floats or one). `origin`: physical position of the center of voxel
`(0, 0, 0)`. No orientation. Frozen; compares by value. `ValueError` for invalid values.

| Member | Description |
|---|---|
| `n_voxels` | `Z * Y * X` |
| `index_to_mm(index)` | continuous index `(..., 3)` to physical position, float64 |
| `mm_to_index(mm)` | physical position `(..., 3)` to continuous index, float64 |
| `extent_mm` | `(lo, hi)`: outer faces of the first and last voxels |
| `center_extent_mm` | `(lo, hi)`: first and last voxel centers |
| `Grid.like(other)` | a Grid from any object with `shape` and optional `spacing`, `origin` |
| `resampled(spacing, *, align="edges")` | same field of view at a new spacing; `"edges"` keeps the outer boundary, `"centers"` keeps the first voxel center |
| `Grid.isotropic(spacing, *, like, align="edges")` | `Grid.like(like).resampled((spacing,) * 3, align=align)` |
| `roi(lo_mm, hi_mm)` | sub-grid on this lattice of voxels overlapping the box, clipped to the grid |

## Mapping

```text
Mapping(a, b=(0.0, 0.0, 0.0), centered=False, exact=None)
```

Per-axis map `x_to = a * x_from + b`, `a >= 0`. For `to_labels`, *from* is the output grid
and *to* is the model grid. Frozen; compares by value.

| Member | Description |
|---|---|
| `Mapping.center(shape_from, shape_to)` | voxel-center rule `(x + 0.5) * n_to / n_from - 0.5`; sets `centered` |
| `Mapping.corner(shape_from, shape_to)` | voxel-corner rule `x * (n_to - 1) / (n_from - 1)`; a length-1 axis maps to 0 |
| `Mapping.spacing(spacing_from, spacing_to, shift=(0, 0, 0))` | `x * s_from / s_to + shift` |
| `Mapping.between(grid_from, grid_to)` | through physical positions of two grids |
| `Mapping.identity()` | `x_to = x_from` |
| `apply(x)` | map coordinates `(..., 3)`, float64 |
| `then(other)`, `>>` | `m1 >> m2` applies `m1`, then `m2` |
| `inverse()` | inverse; `ValueError` if any scale is 0 |
| `is_identity` | exact identity test |
| `centered` | coordinates are evaluated as `(j + 0.5) * a - 0.5` (scipy's arithmetic); kept when composing with the identity or an integer shift |
| `exact` | set by composition: `(core_b, pre, post)`, the map as `core(x + pre) - post` with integer shifts `pre`, `post`; `b` is the folded offset |
| `terms()` | `(core_b, pre, post)`, the form the tables evaluate |
| `integer_shift` | the integer offsets if the mapping is a pure integer shift, else None |

See [concepts.md](concepts.md#choosing-a-mapping) for choosing a constructor.

## Affine

```text
Affine(m, b=(0.0, 0.0, 0.0))
```

General map `x_to = x_from @ m + b` on row vectors in `(Z, Y, X)` order; `m` is a finite 3x3
matrix.

| Member | Description |
|---|---|
| `Affine.between(geo_from, geo_to)` | from two objects with `directions` (3x3, row `i` = world step along array axis `i`) and `origin` (world position of voxel 0's center) |
| `apply(x)` | map coordinates `(..., 3)`, float64 |
| `inverse()` | inverse; `numpy.linalg.LinAlgError` if singular |
| `separable` | the equivalent `Mapping` if `m` is diagonal with non-negative entries (off-diagonal terms below `SEPARABLE_TOLERANCE` times the largest diagonal are ignored), else None |

`to_labels` does not accept an `Affine`; pass `separable` when it exists, or use
[`axis_coords`](#tables) on the output of `apply`.

## Tables

The per-axis tables every backend consumes. Most users do not call these directly.

```text
build_tables(out_shape, src_shape, mapping, *, interp="linear", outside="background",
             coord_dtype=np.float64, out_start=(0, 0, 0)) -> (AxisTable, AxisTable, AxisTable)
axis_table(n_out, n_src, a, b, *, interp="linear", outside="background",
           coord_dtype=np.float64, start=0, centered=False, pre=0, post=0) -> AxisTable
axis_coords(c, n_src, *, interp="linear", outside="background") -> (valid, i0, i1, f)
```

- `AxisTable(i0, i1, f, f64)`: for each output index, source indices `i0`, `i1` (int32), the
  float32 weight `f` of `i1` and the same weight in float64. `i0 == -1` marks an index outside
  the source. Property `n`: number of output indices.
- `axis_coords` applies the edge and rounding rules to any float coordinates `c` (any shape)
  and returns `valid` (bool) and `i0`, `i1` (int64), `f` (float64), all of `c`'s shape. The
  rules are in [concepts.md](concepts.md#edge-rules).

## backends

*torch*

```text
available_backends() -> list[str]
backends.select(name, device, logits_shape, out_shape) -> Choice(name, module, fallback)
```

`available_backends()` returns the backends usable in this process: always `"torch"`,
`"metal"` with an MPS device and torch >= 2.7, `"triton"` with Triton installed and a CUDA
device.

`backends.select` is the choice `to_labels` makes. With `name="auto"` it takes the device's
fused kernel (`mps` -> `metal`, `cuda` -> `triton`) if available and able to address the shapes,
otherwise `torch`; `fallback` is a message when an available kernel was passed over, else
None. It considers device and shapes only, not dtype or contiguity. An explicit name raises
`ValueError` if it is unknown, on the wrong device, unavailable, or unable to address the
field.

Backend modules: `labelfield.backends.torch_gather`, `labelfield.backends.metal`,
`labelfield.backends.triton_gpu`. Each has `available()` and
`run(logits, out, tables, lut, *, mode, paint, background, threshold, skip=None, ...)`.
`labelfield.backends.triton_gpu.warmup(mode="argmax", paint=False)` compiles the Triton kernel
ahead of the first call (about 1 s).

## transparency_mask

*torch*

```text
transparency_mask(lut, *, transparent="background") -> numpy.ndarray
```

uint8 per channel, 1 where an argmax paint leaves the voxel unchanged: channel 0 for
`"background"`, every channel whose label is 0 for `"zero"`. Needed only when calling a
backend's `run` directly.

## reference

`labelfield.reference` is a float64 numpy implementation of the same semantics, for tests on
small volumes. It materializes all K channels at output resolution.

| Function | Description |
|---|---|
| `interpolate(logits, tables)` | `(values (K, Za, Ya, Xa) float64, valid (Za, Ya, Xa) bool)` |
| `decide(values, valid, *, lut, mode, threshold, background, paint, transparent, out)` | labels with `to_labels` semantics; int64, or written into a numpy `out` |
| `labels(logits, tables, **kw)` | `decide(*interpolate(logits, tables), **kw)` |
| `margins(values)` | per voxel, largest minus second-largest channel value; small margins mark near-ties |

```python
import numpy as np
import labelfield as lf
from labelfield import reference

logits = np.random.default_rng(1).normal(size=(3, 4, 5, 6))
tables = lf.build_tables((8, 10, 12), (4, 5, 6), lf.Mapping.center((8, 10, 12), (4, 5, 6)))
values, valid = reference.interpolate(logits, tables)
labels = reference.decide(values, valid)
print(labels.shape, labels.dtype, bool(valid.all()))   # (8, 10, 12) int64 True
```
