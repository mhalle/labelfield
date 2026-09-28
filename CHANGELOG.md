# Changelog

## Unreleased

Documentation only; no change to behavior or signatures.

- README rewritten for new users: purpose and limits, installation and extras, core concepts
  (logit field, model and output grids, choosing a `Mapping`, label table, paint and
  transparency, outside handling, `out_start`), examples, backend selection, edge rules.
- New `docs/concepts.md` (coordinate conventions, a diagram of the center and corner rules,
  exact edge and tie rules, precision, `Affine`) and `docs/api.md` (reference for every
  public name).
- Docstrings of all public classes and functions reviewed against the code and completed with
  shapes, dtypes, defaults and the errors raised. Corrections: the `Affine` and `axis_coords`
  docstrings implied that labelfield restores through a general `Affine`; it does not, and
  `to_labels` accepts only a `Mapping`. The `to_labels` docstring listed only float32 and
  float16 logits and omitted `threshold`, `out_dtype` and `slab_voxels`.
  `Mapping.center` no longer lists ITK among the matching resamplers.
- New `tests/test_docs_examples.py` runs every Python example in the README and `docs/` and
  checks the output shown in its comments.
- Package description in `pyproject.toml` reworded.

## 0.1.2 (2026-09-27)

- **Nearest-neighbor ties under the voxel-center rule now match scipy.** `Mapping.center` now
  sets a `centered` flag, and the tables compute its coordinates as `(j + 0.5) * a - 0.5`, the
  arithmetic of `scipy.ndimage.zoom(grid_mode=True)`, `skimage.transform.resize` and therefore
  nnU-Net's resampler, instead of `a * j + b`. The two are the same map and agree except at
  coordinates exactly half-way between two samples, where floating-point rounding can send
  them to different samples: previously 1453 of 577071 nearest picks in the test sweep
  differed from scipy's; now none do. `a` and `b` keep their meaning (`a * x + b`), so code
  that reads them is unaffected. Composing with the identity keeps `centered`; any other
  composition drops it.

## 0.1.1 (2026-09-27)

- `axis_coords(c, n_src, interp=, outside=)`: the per-axis decision (whether a coordinate is
  inside, the two samples, the weight) for any float64 coordinates, not only a grid's rows.
  `axis_table` now uses it, so the rule is implemented in one place. Callers that sample
  through a general `Affine` (outside labelfield) can use it to make the same decisions as
  the tables wherever both apply.
- `Affine` (and `SEPARABLE_TOLERANCE`): a general affine map between index spaces (rotation,
  flips, axis swaps), built with `Affine.between` from two geometry records (direction rows
  and origin), with `separable` giving the equivalent per-axis `Mapping` when the grids line
  up. Moved unchanged from rankfield, a related project.

## 0.1.0 (2026-09-27)

First version: the fused logit-to-label restore from haversack (the project this code was
extracted from), as a standalone package.

- From haversack: `Grid`, `Mapping`, `build_tables`, `to_labels` / `resample_argmax` /
  `resample_paint`, the float64 `reference`, and the Metal, Triton and torch backends, with
  their tests.
- New: `transparent="zero"` for an argmax paint, which leaves a voxel untouched when its label
  is 0 rather than when channel 0 wins (TotalSegmentator's compositor). The backends take a
  per-channel `skip` mask (`transparency_mask`) beside the label table, which holds the labels
  as given; without a mask, channel 0 is transparent, as in haversack.
- New: `import labelfield` needs only numpy; the torch-dependent names load on first use.
- New: Python 3.9 and torch 2.1 minimum versions, tested (Python 3.9.6 / torch 2.1.2 on
  macOS). Below torch 2.3 there is no `torch.uint16`, so labels above 255 raise; below torch
  2.7 there is no Metal kernel.
- New: `tests/test_totalsegmentator.py`: a nearest paint with `Mapping.corner` equals
  TotalSegmentator's composite-then-`zoom(order=0)` exactly, on CPU and MPS.
- New: `out_start` on `to_labels` / `build_tables`: restore a slab of a larger grid with the
  same decisions as a single call. Folding the offset into the mapping instead adds a second
  rounding, which can flip a nearest pick exactly half-way between samples.
- Changed from haversack: with `interp="nearest"` a coordinate is inside only if the sample it
  picks exists (`-0.5 <= c < n - 0.5`), so a coordinate exactly half a voxel past the last
  sample is outside, as in a nearest resample of the uncropped label map. Linear keeps
  `c <= n - 0.5`.
