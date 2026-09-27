# Changelog

## 0.1.0 (unreleased)

First version: haversack's fused logit -> label restore, extracted into its own package.

- From haversack: `Grid`, `Mapping`, `build_tables`, `to_labels` / `resample_argmax` /
  `resample_paint`, the float64 `reference`, and the Metal, Triton and torch backends, with
  their tests.
- New: `transparent="zero"` for an argmax paint, which leaves a voxel untouched when its label
  is 0 rather than when channel 0 wins - TotalSegmentator's compositor. The backends read a
  kernel label table with -1 for a transparent channel (`kernel_lut`); the default,
  `"background"`, is haversack's behavior.
- New: `import labelfield` needs only numpy; the torch names load on first use.
- New: Python 3.9 and torch 2.1 floors, tested (3.9.6 / torch 2.1.2 on macOS). Below torch
  2.3 there is no `torch.uint16`, so labels above 255 raise; below 2.7 there is no Metal kernel.
- New: `tests/test_totalsegmentator.py` - nearest paint with `Mapping.corner` equals
  TotalSegmentator's composite-then-`zoom(order=0)` exactly, on CPU and MPS.
- New: `out_start` on `to_labels` / `build_tables` - restore a slab of a larger grid with the
  decisions of one call. Folding the offset into the mapping instead rounds twice, which flips
  a nearest pick exactly half-way between samples.
- Changed from haversack: with `interp="nearest"` a coordinate is inside only if the sample it
  picks exists (`-0.5 <= c < n - 0.5`), so a coordinate half-way past a cropped edge is outside,
  as in a nearest resample of the uncropped label map. Linear keeps `c <= n - 0.5`.
