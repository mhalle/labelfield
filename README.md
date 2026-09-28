# labelfield

labelfield converts the output of a volumetric segmentation network, a per-class logit field
on the network's grid, into a label map on any other grid: the original image grid, an
isotropic grid, or a region of interest. For each output voxel it interpolates the K logits
(trilinear, nearest, or chosen per axis) and decides a label (argmax, or a threshold per
region) in one pass. The K interpolated channels are not stored for the whole output grid:
the fused GPU kernels keep only a running decision per voxel, and the torch implementation
holds only a few output-resolution Z planes of K channels at a time.

A label table maps each channel to an output label, and a paint mode writes into an existing
label map while leaving selected decisions untouched, so that several models can be combined
into one label map.

Computation runs on the logits' device: a fused Metal kernel on Apple GPUs, a fused Triton
kernel on CUDA GPUs, and a portable torch implementation on any device, including CPU.

**Status:** alpha (0.1.x). The API may change in any release before 1.0. Pin a release tag.

## When to use it

- You have per-class logits on a model grid (for example from nnU-Net, TotalSegmentator or a
  similar network) and need labels on a different grid.
- You combine several models, each with its own classes, into one label map.
- You want smooth (linearly interpolated) label boundaries without allocating K channels at
  output resolution.

It is not suitable when:

- You need probabilities or interpolated logits at output resolution. labelfield produces
  labels only.
- You need decisions reproducible in float64. The backends interpolate and decide in float32;
  where two classes are nearly tied (differences below about 1e-4) the result may differ from a
  float64 computation or between backends. `labelfield.reference` is a float64 implementation
  intended for testing on small volumes.
- Your model grid is rotated or flipped relative to the output grid. The fused path handles
  per-axis (scale and offset) relations only; see [Affine](docs/concepts.md#grids-that-do-not-line-up-affine).
- You need to read or write image files or handle orientation metadata. labelfield works on
  arrays and leaves file formats and orientation to the caller.

## Installation

labelfield is distributed from GitHub only; it is not on PyPI. Install a tagged release:

    pip install "labelfield[torch] @ git+https://github.com/mhalle/labelfield.git@v0.1.2"

| Extra | Installs | Needed for |
|---|---|---|
| (none) | `numpy>=1.21` | geometry (`Grid`, `Mapping`, `Affine`, tables) and `labelfield.reference` |
| `torch` | `torch>=2.1` | `to_labels` and all backends |
| `cuda` | `triton>=3.0` | the Triton backend (also needs a CUDA build of torch, installed separately) |
| `test` | pytest, torch, scipy, scikit-image | running the test suite |

Requirements: Python 3.9 or newer. `import labelfield` does not import torch; torch is
imported when a torch-dependent name (`to_labels`, `resample_argmax`, `resample_paint`,
`available_backends`, `transparency_mask`, `backends`) is first used.

| Backend | Device | Requires |
|---|---|---|
| `torch` | any torch device | torch >= 2.1 |
| `metal` | Apple GPU (`mps`) | torch >= 2.7 (`torch.mps.compile_shader`); not offered on older torch |
| `triton` | CUDA GPU | `triton>=3.0` and a CUDA-enabled torch |

Labels above 255 require a `uint16` label map, which requires torch >= 2.3.

## Quick start

```python
import numpy as np
import torch
import labelfield as lf

# Synthetic logits for 3 classes on a 12 x 16 x 16 model grid, shape (K, Z, Y, X):
# class 1 is a ball, class 2 the slab z >= 8, class 0 (background) elsewhere.
z, y, x = np.meshgrid(np.arange(12), np.arange(16), np.arange(16), indexing="ij")
ball = 4.0 - np.sqrt((z - 5.5) ** 2 + (y - 7.5) ** 2 + (x - 7.5) ** 2)
logits = torch.from_numpy(np.stack([np.zeros_like(ball), ball, 2.0 * (z - 7.5)]).astype(np.float32))

# The model grid was made from a 24 x 48 x 48 image by a voxel-center resampler
# (such as skimage.transform.resize), so Mapping.center inverts it.
image_shape = (24, 48, 48)
mapping = lf.Mapping.center(image_shape, tuple(logits.shape[1:]))

labels = lf.to_labels(logits, image_shape, mapping)       # trilinear interpolation + argmax
print(tuple(labels.shape), labels.dtype)                  # (24, 48, 48) torch.uint8
print(torch.unique(labels).tolist())                      # [0, 1, 2]
```

The result is a torch tensor on the same device as `logits`. Moving `logits` to an `mps` or
`cuda` device selects the corresponding fused kernel automatically. numpy arrays are accepted
and processed on the CPU.

## Core concepts

[docs/concepts.md](docs/concepts.md) covers these in more detail, with diagrams and exact
edge rules.

**Logit field.** The network's raw per-class output on its grid, shape `(K, Z, Y, X)`, any
floating dtype (the fused kernels accept float32 and float16; Triton also bfloat16). Axis
order is `(Z, Y, X)` throughout labelfield.

**Model grid and output grid.** The model grid is where the logits live. The output grid is
where the labels are wanted; only its shape is passed to `to_labels`, as a `(Z, Y, X)` tuple
or a `Grid`. A `Grid` describes a grid physically by `shape`, `spacing` and `origin` (the
position of the center of voxel `(0, 0, 0)`).

**Mapping.** A `Mapping` gives, for each axis, the model-grid coordinate of each output voxel
index: `model_coordinate = a * output_index + b`, in voxel units where integers are voxel
centers. Pick the constructor that inverts the resampler that produced the model grid:

| Constructor | Aligns | Matches |
|---|---|---|
| `Mapping.center(out_shape, model_shape)` | outer boundaries of both grids | `skimage.transform.resize`, `scipy.ndimage.zoom(grid_mode=True)`, `interpolate(align_corners=False)`, nnU-Net |
| `Mapping.corner(out_shape, model_shape)` | first and last voxel centers | `scipy.ndimage.zoom` (default `grid_mode=False`), `interpolate(align_corners=True)`, TotalSegmentator |
| `Mapping.spacing(out_spacing, model_spacing)` | voxel `(0, 0, 0)`, scaled by the exact spacing ratio | resamplers that keep the origin and ignore rounding of the shape |
| `Mapping.between(out_grid, model_grid)` | physical positions of two `Grid`s | any pair of grids known by spacing and origin, e.g. a region of interest |

A mismatched convention shifts labels by up to half a model voxel. Mappings compose with `>>`
(`m1 >> m2` applies `m1` first).

**Label table.** `lut[k]` is the label written when channel `k` wins; the default is `k`. The
output dtype is `uint8` if all labels are at most 255 and `uint16` otherwise.

**Modes.** `mode="argmax"` (default) writes the label of the channel with the largest
interpolated logit. `mode="regions"` is for models with an independent sigmoid output per
region: every channel whose interpolated logit exceeds `threshold` (default 0.0, i.e.
probability 0.5) writes its label, later channels winning where they overlap.

**Paint and transparency.** With `paint=True`, `to_labels` writes into an existing `out`
tensor and leaves some voxels unchanged: voxels outside the model grid, and voxels whose
decision is transparent. With `transparent="background"` (default) a voxel won by channel 0
is transparent; with `transparent="zero"` a voxel whose winning label is 0 is transparent,
which also covers auxiliary classes mapped to 0. In regions mode, a voxel where no channel
exceeds the threshold is transparent.

**Outside the model grid.** An output voxel is inside if its model coordinate is within half
a voxel of the first or last sample center on every axis. With `outside="background"`
(default) other voxels receive `background` (default 0), or are left unchanged when painting.
With `outside="clamp"` all coordinates are clamped into the model grid.

**`out_start`.** Restores a part of a larger output grid with exactly the decisions the whole
grid would get: output voxel `j` is sampled at `mapping(j + out_start)`. Use it to process a
large grid in slabs, or a region of interest on the image lattice, without building a new
Mapping.

## Combining several models

Each model has its own classes. Give each a label table that assigns its classes their labels
in the combined map, and paint all of them into one buffer. Later calls take precedence where
they are not transparent.

```python
import numpy as np
import torch
import labelfield as lf

rng = np.random.default_rng(0)
model_shape, image_shape = (10, 12, 12), (20, 30, 30)
# Model A: background and three structures. Model B: background and two others.
logits_a = torch.from_numpy(rng.normal(size=(4, *model_shape)).astype(np.float32))
logits_b = torch.from_numpy(rng.normal(size=(3, *model_shape)).astype(np.float32))
parts = [(logits_a, [0, 1, 2, 3]), (logits_b, [0, 10, 11])]

mapping = lf.Mapping.corner(image_shape, model_shape)
out = torch.zeros(image_shape, dtype=torch.uint8)
for logits, lut in parts:
    lf.to_labels(logits, image_shape, mapping, lut=lut, paint=True, out=out)
print(sorted(torch.unique(out).tolist()))                 # [0, 1, 2, 3, 10, 11]
```

With `interp="nearest"`, `Mapping.corner`, `paint=True` and `transparent="zero"`, this
reproduces TotalSegmentator's multi-model output exactly (argmax per part on the model grid,
composite with `np.copyto(out, lut[seg], where=lut[seg] != 0)`, then
`scipy.ndimage.zoom(order=0)`); `tests/test_totalsegmentator.py` checks this. With
`interp="linear"` the same composite has smooth boundaries.

## Choosing a backend

`backend="auto"` (default) chooses from the device of `logits`:

- `mps`: `metal` if available (torch >= 2.7), otherwise `torch`.
- `cuda`: `triton` if Triton is installed, otherwise `torch`.
- anything else, including CPU: `torch`.

If a fused kernel is available but cannot address the field (a single model-grid channel of
2^31 voxels or more; for Triton, also an output of 2^31 voxels or more), `auto` uses `torch`
and issues a `RuntimeWarning`. Naming a backend explicitly (`backend="metal"`, `"triton"` or
`"torch"`) never substitutes another; if it cannot run, `ValueError` is raised.
`lf.available_backends()` lists the backends usable in the current process.

The fused backends need a contiguous `out` and support fewer logit dtypes (Metal: float32,
float16; Triton: float32, float16, bfloat16). For logits of another dtype, or a non-contiguous
`out` (a view such as `big[:, y0:y1]`), `auto` uses `torch` and issues a `RuntimeWarning`; a
backend named explicitly raises instead.

## Limits and edge rules

- **Float32 decisions.** Tables are computed in float64 on the host; interpolation and the
  decision run in float32. Near-tied classes can resolve differently across backends.
- **Nearest-neighbor ties.** A coordinate exactly half-way between two samples picks the
  higher one, as `scipy.ndimage` does. `Mapping.center` evaluates coordinates with scipy's
  arithmetic so such ties resolve as in `scipy.ndimage.zoom(order=0, grid_mode=True)`.
- **Half a voxel past the edge.** Coordinates up to half a voxel beyond the outermost sample
  centers are inside. For linear interpolation the bound is inclusive (`c <= n - 0.5`); for
  nearest it is exclusive (`c < n - 0.5`), because the sample it would round to does not
  exist.
- **Linear edge extension.** Within that half voxel, coordinates are clamped to the edge
  sample, so values are repeated, not extrapolated.
- **Per-axis geometry.** A `Mapping` scales and shifts each axis independently, with
  non-negative scale. Flips, rotations and axis permutations are not supported by the fused
  path; see `Affine` in [docs/concepts.md](docs/concepts.md#grids-that-do-not-line-up-affine).
- **Labels** are non-negative integers up to 255 (`uint8`) or 65535 (`uint16`, torch >= 2.3).

## API reference

[docs/api.md](docs/api.md) documents every public class and function. The docstrings in
`src/labelfield/` contain the same information.

## Development

    uv venv && uv pip install -e ".[test]" && pytest

The tests run on every available device (CPU, and MPS or CUDA when present) and compare each
backend against the float64 reference. `tools/cuda_check.py` runs the suite on a rented CUDA
GPU through [Modal](https://modal.com), for testing the Triton backend from a machine without
one. `tests/test_docs_examples.py` executes the Python examples in this README and in `docs/`.

## License

Apache-2.0.
