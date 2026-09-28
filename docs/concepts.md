# Concepts and conventions

This page explains the coordinate conventions labelfield uses, how to choose a `Mapping`,
and the exact rules at grid edges and ties. The [README](../README.md) gives the overview;
[api.md](api.md) lists every public name.

## Axis order and units

- Volumes are indexed `(Z, Y, X)`. A logit field is `(K, Z, Y, X)`: K channels (classes)
  first.
- A *voxel coordinate* is a continuous position in voxel units along each axis. Integer
  coordinates are voxel centers; voxel `i` covers `[i - 0.5, i + 0.5]`.
- A `Grid` adds physical placement: `spacing` (distance between voxel centers) and `origin`
  (physical position of the center of voxel `(0, 0, 0)`). The unit is up to the caller;
  millimeters are conventional. A `Grid` has no orientation: grids that are compared are
  assumed to share world axes.

## Model grid and output grid

The **model grid** is the grid the network ran on, usually the input image resampled to the
model's training spacing. The **output grid** is the grid the labels should be on: usually
the original image grid, but it can be any grid, such as an isotropic grid or a region of
interest.

A `Mapping` states, for each axis, where each output voxel lies on the model grid:

    model_coordinate = a * output_index + b

`to_labels` evaluates this for every output voxel, interpolates the K logits at that
coordinate, and decides a label. The direction matters: a Mapping for `to_labels` always
goes **from output indices to model coordinates**, which is the inverse of the resampling
that produced the model grid.

## Choosing a Mapping

The correct Mapping is the one that inverts the resampling that produced the model grid.
Resamplers differ in how they align two grids of different sizes, and using the wrong
convention shifts the labels by up to half a model voxel.

The diagram shows one axis. The model grid has 2 voxels (`o` marks their centers, `|` their
boundaries). An output grid of 4 voxels (`x`) is placed on it by each of the two
shape-based conventions:

```text
model grid, n = 2          |         o         |         o         |
Mapping.center, n = 4      |    x    |    x    |    x    |    x    |
Mapping.corner, n = 4             |  x  |   x  |  x   |  x  |
model coordinate          -0.5       0        0.5        1        1.5
```

- `Mapping.center`: the outer boundaries of both grids coincide. The output centers land at
  model coordinates -0.25, 0.25, 0.75, 1.25.
- `Mapping.corner`: the first and last voxel centers coincide. The output centers land at
  0, 1/3, 2/3, 1.

| Constructor | Rule (per axis) | Use when the model grid was produced by |
|---|---|---|
| `Mapping.center(out_shape, model_shape)` | `x = (j + 0.5) * n_model / n_out - 0.5` | `skimage.transform.resize`; `scipy.ndimage.zoom(..., grid_mode=True)`; `torch.nn.functional.interpolate(..., align_corners=False)` in its linear modes; nnU-Net's resampler |
| `Mapping.corner(out_shape, model_shape)` | `x = j * (n_model - 1) / (n_out - 1)` | `scipy.ndimage.zoom` with its default `grid_mode=False`; `interpolate(..., align_corners=True)`; TotalSegmentator's `change_spacing` |
| `Mapping.spacing(out_spacing, model_spacing, shift=0)` | `x = j * s_out / s_model + shift` | a resampler that keeps voxel `(0, 0, 0)` fixed and scales by the exact spacing ratio, independent of the shapes (for example nnunet-inference-mlx) |
| `Mapping.between(out_grid, model_grid)` | `x = model_grid.mm_to_index(out_grid.index_to_mm(j))` | any case where both grids are known physically (spacing and origin), such as an output region of interest, or a model grid built with `Grid.resampled` |
| `Mapping(a, b)` | `x = a * j + b` | anything else that is per-axis linear |

`center` and `corner` depend only on the two shapes. `spacing` and `between` depend only on
spacings (and origins), not on shapes. When a resampler rounds the output shape, the
shape-based and spacing-based rules differ slightly, and only the one the resampler actually
used inverts it exactly.

`nearest` interpolation with `Mapping.center` does not match `torch.nn.functional.interpolate`
with `mode="nearest"`, which uses a different rounding rule; it matches
`scipy.ndimage.zoom(..., order=0, grid_mode=True)`.

Mappings compose with `>>` (`m1 >> m2` applies `m1` first), for example
`Mapping.between(roi, image) >> Mapping.center(image.shape, model.shape)` to go from a region
of interest through the image grid to the model grid. Composing with a pure integer shift
(`a == 1`, integer `b`, such as a crop offset on either side) keeps the mapping's exact form:
the shift is recorded (`Mapping.exact`) and applied as an exact integer step, and the
`centered` flag that `Mapping.center` sets survives (see [Ties](#ties-in-nearest-interpolation)).
Any other composition folds into `a * x + b` and drops the flag.

### Physical grids

```python
import numpy as np
import labelfield as lf

image = lf.Grid(shape=(40, 64, 64), spacing=(2.5, 0.8, 0.8))   # origin (0, 0, 0)
model = lf.Grid.isotropic(1.5, like=image)                      # same outer boundary
print(model)          # Grid(shape=(67, 34, 34), spacing=(1.5, 1.5, 1.5), origin=(-0.5, 0.35, 0.35))

by_space = lf.Mapping.between(image, model)
by_shape = lf.Mapping.center(image.shape, model.shape)
# 40 * 2.5 / 1.5 = 66.7 was rounded to 67, so the two rules differ slightly:
print(np.round(by_space.a, 4), np.round(by_shape.a, 4))   # [1.6667 0.5333 0.5333] [1.675  0.5312 0.5312]
```

`Grid.resampled(spacing, align="edges")` (used by `Grid.isotropic`) keeps the outer
boundary, the physical form of the center convention; `align="centers"` keeps the first voxel
center, the physical form of the corner convention.

## Restoring part of a grid: `out_start`

`out_start` places the output inside the larger grid the Mapping was built for: output voxel
`j` is sampled at `mapping(j + out_start)`, evaluated from that integer index. This gives
exactly the labels the full grid would get, which is useful for processing a large grid in
slabs, or for restoring only a region of interest:

```python
import numpy as np
import torch
import labelfield as lf

rng = np.random.default_rng(0)
logits = torch.from_numpy(rng.normal(size=(3, 12, 16, 16)).astype(np.float32))
image = lf.Grid((24, 48, 48), spacing=(2.0, 1.0, 1.0))
mapping = lf.Mapping.center(image.shape, tuple(logits.shape[1:]))
whole = lf.to_labels(logits, image, mapping)

# A region of interest on the image lattice, restored on its own.
roi = image.roi((10.0, 10.0, 10.0), (30.0, 30.0, 30.0))
z0, y0, x0 = (int(i) for i in np.rint(image.mm_to_index(roi.origin)))
part = lf.to_labels(logits, roi, mapping, out_start=(z0, y0, x0))
dz, dy, dx = roi.shape
assert torch.equal(part, whole[z0:z0 + dz, y0:y0 + dy, x0:x0 + dx])

# The full grid in slabs of 5 Z planes, written into one buffer.
out = torch.zeros(image.shape, dtype=torch.uint8)
for z in range(0, image.shape[0], 5):
    n = min(5, image.shape[0] - z)
    lf.to_labels(logits, (n, 48, 48), mapping, out=out[z:z + n], out_start=(z, 0, 0))
assert torch.equal(out, whole)
```

Folding the offset into the Mapping instead (`Mapping((1, 1, 1), (z0, y0, x0)) >> mapping`)
gives the same coordinates in exact arithmetic but adds a floating-point rounding step, which
can change nearest-neighbor picks at exact ties.

## Edge rules

For each axis with `n` model samples (centers at coordinates `0 .. n - 1`):

| | Inside when | Value inside |
|---|---|---|
| `interp="linear"` | `-0.5 <= c <= n - 0.5` | linear blend of samples `floor(c)` and `floor(c) + 1`, after clamping `c` to `[0, n - 1]` |
| `interp="nearest"` | `-0.5 <= c < n - 0.5` | sample `floor(c + 0.5)`, after clamping `c` to `[0, n - 1]` |

- **Half a voxel past the edge.** A coordinate up to half a voxel beyond the first or last
  sample center is inside (it lies within that voxel's extent). In that half voxel the
  coordinate is clamped, so the edge value is repeated rather than extrapolated (linear edge
  extension, as `skimage` `mode="edge"` and `scipy.ndimage` `mode="nearest"`).
- **The upper nearest bound is strict.** For nearest, `c = n - 0.5` would round to sample `n`,
  which does not exist, so it is outside. This matches a nearest resample of a larger label map
  that was cropped to the model grid: a voxel exactly half-way past the crop picks the voxel
  beyond it.
- **Outside.** With `outside="background"` a voxel outside on any axis receives `background`
  (or is left unchanged when painting). With `outside="clamp"` every coordinate is inside and
  clamped, so edge values extend indefinitely.

`labelfield.axis_coords` implements these rules for arbitrary coordinates and is what the
tables use.

## Ties in nearest interpolation

A coordinate exactly half-way between two samples (`c = k + 0.5`) rounds up to `k + 1`, as in
`scipy.ndimage` with `order=0`. Such coordinates occur for many pairs of shapes (for example
6 model voxels and 57 output voxels under `Mapping.center`). Whether a coordinate is exactly
half-way depends on floating-point evaluation, so `Mapping.center` marks itself `centered`
and its coordinates are computed as `(j + 0.5) * a - 0.5`, the same arithmetic scipy and
scikit-image use, rather than `a * j + b`. The two are equal mathematically but can round
differently. With `centered`, nearest picks match `scipy.ndimage.zoom(order=0, grid_mode=True)`
exactly.

## Label table, paint and transparency

- `lut[k]` is the label written when channel `k` is chosen; the default is `k`. Labels are
  non-negative. The output is `uint8` when every label (and `background`) is at most 255,
  otherwise `uint16` (which needs torch >= 2.3).
- `paint=True` composites into `out`: output voxels outside the model grid, and voxels whose
  decision is *transparent*, keep their current value. For argmax,
  `transparent="background"` (default) makes channel 0 transparent; `transparent="zero"` makes
  every channel whose label is 0 transparent, which also covers auxiliary classes a task maps
  to 0 (TotalSegmentator's compositing rule). In `mode="regions"`, a voxel where no channel
  exceeds `threshold` is transparent.
- Calling `to_labels` once per model into one `out` composites several models. Later calls
  overwrite earlier ones where they are not transparent, so call order is priority order.

## Precision

Tables (indices, weights, inside tests) are computed on the host in float64. The backends
interpolate and decide in float32 with float32 weights; the blend order is fixed (X, then Y,
then Z) and identical in every backend. Where two channels differ by less than about 1e-4
after interpolation, backends may choose different winners from each other or from the
float64 `labelfield.reference`. The Metal kernel tries to disable fused multiply-add
contraction so that it rounds like the torch backend (`labelfield.backends.metal.fp_contract()`
reports whether it succeeded).

## Grids that do not line up: `Affine`

When the model grid is rotated, flipped or has permuted axes relative to the output grid, no
per-axis Mapping relates them. `Affine.between(geo_from, geo_to)` builds the general map from
two geometry records (objects with `directions`, a 3x3 array whose row `i` is the world step
along array axis `i`, and `origin`, the world position of the center of voxel 0). If
`Affine.separable` returns a Mapping, the fused path applies; otherwise labelfield provides the
per-axis rule (`axis_coords`) but not the gather:

```python
from types import SimpleNamespace

import numpy as np
import labelfield as lf

def geometry(directions, origin):
    return SimpleNamespace(directions=np.asarray(directions, float), origin=np.asarray(origin, float))

image = geometry(np.diag([1.0, 0.8, 0.8]), (0.0, 0.0, 0.0))
model = geometry(np.diag([2.0, 1.6, 1.6]), (0.5, 0.4, 0.4))
print(lf.Affine.between(image, model).separable)   # Mapping(a=(0.5, 0.5, 0.5), b=(-0.25, -0.25, -0.25))

# A model grid rotated 90 degrees in the Y-X plane: no per-axis Mapping exists.
rotated = geometry([[2.0, 0, 0], [0, 0, 1.6], [0, -1.6, 0]], (0.0, 15.0, 0.0))
aff = lf.Affine.between(image, rotated)
print(aff.separable)                                # None

# Nearest-neighbor resample of a model-grid label map through the general map.
model_labels = np.random.default_rng(0).integers(0, 4, size=(8, 10, 10))
out_shape = (16, 20, 20)
index = np.stack(np.meshgrid(*[np.arange(n) for n in out_shape], indexing="ij"), axis=-1)
coords = aff.apply(index)                           # (16, 20, 20, 3) model coordinates
inside = np.ones(out_shape, dtype=bool)
picks = []
for axis in range(3):
    valid, i0, _, _ = lf.axis_coords(coords[..., axis], model_labels.shape[axis], interp="nearest")
    inside &= valid
    picks.append(i0)
labels = np.where(inside, model_labels[tuple(picks)], 0)
print(labels.shape)                                 # (16, 20, 20)
```
