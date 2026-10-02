# Changelog

## 0.1.4 - 2026-10-02

- **The torch backend's argmax is much faster where most of the field is one class** (the CPU,
  and CUDA without Triton, e.g. Windows). An interpolated value is a weighted average of its
  model cell's corners, so per cell the channels whose largest corner value lies below another
  channel's smallest cannot win; a cell left with one candidate gets that label without
  interpolating, and the other voxels are blended over their own candidates only. The labels
  are the dense path's bit for bit: the same blend arithmetic in the same order, and the same
  first-maximum rule at ties (tests compare the two, and 400 random configurations agreed).
  TotalSegmentator `--fast` on a 768x768x709 CT, 8 CPU cores: the restore went from 162 s to
  23 s, the run from 189 s to 58 s. Region mode and non-finite logits keep the dense path;
  `torch_gather.run(..., prune=False)` forces it.

## 0.1.3 - 2026-09-28

- **Importing the backends no longer keeps the caller alive.** On a machine without triton, the
  failed import was stored as an exception whose traceback held every frame live at the first
  import, so the first caller's arrays, tensors and models stayed referenced for the life of the
  process (found in TotalSegmentator: about 250 MB after a `--fast` call, GBs for `total` on a
  large CT). Only the message is kept now.
- **A crop offset no longer changes a nearest pick.** Composing a `Mapping` with a pure integer
  shift (`a == 1`, integer `b`: a crop offset on the output side or on the source side) records
  the shift (`Mapping.exact`, `terms()`) instead of folding it into `b`, and the tables apply it as
  an exact integer step; `centered` survives such a composition. Folding rounded differently at
  exact half-way coordinates, so a cropped or enveloped restore could pick the other sample from
  the uncropped one (found by the haversack audit: 7 % of center and 24 % of corner configurations
  differed somewhere). `a` and `b` read as before; any other composition folds as before.
- **Clearer failures, and `auto` falls back where a fused kernel cannot run.** `to_labels`
  raises `TypeError` for a mapping that is not a `Mapping`, pointing an `Affine` to
  `Affine.separable` (it raised `AttributeError`). `transparency_mask` raises `ValueError` for
  an unknown `transparent` value (it treated any value as `"zero"`). With `backend="auto"`,
  logits of a dtype the fused kernel does not read (float64 on either, bfloat16 on Metal) or a
  non-contiguous `out` now use the torch backend with a `RuntimeWarning`, as an oversized field
  already did; before, they raised. `backends.select` takes optional `logits_dtype` and
  `out_contiguous` for this. A backend named explicitly still raises.

Documentation (no change to behavior or signatures):

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
