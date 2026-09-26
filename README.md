# labelfield

> **Alpha.** Extracted from haversack's restore (its kernels, geometry and tests); the package
> and its API are new and may still move. Pin a tag.

A segmentation network's last layer is a field: K logits per voxel, on the model's grid.
labelfield turns that field into labels on **any grid you ask for** - the input image's, an
isotropic one, a region of interest - by interpolating the logits and taking the decision in
one fused pass per output voxel. Nothing K-channel-sized ever exists at the output
resolution, so a smooth (trilinear) label map costs about what nearest-neighbor sampling of a
label map does.

- **Fused kernels:** Metal on Apple GPUs (torch >= 2.7), Triton on CUDA, and a torch backend
  that runs anywhere, CPU included. `backend="auto"` picks the fused kernel for the device and
  falls back to torch, with a warning, for a field the kernel cannot address.
- **Compositing:** `paint=True` writes into a shared buffer and leaves transparent decisions
  untouched, so a multi-model task composites by calling `to_labels` once per model with its
  label table. `transparent="background"` (default) skips voxels won by channel 0;
  `transparent="zero"` skips voxels whose label is 0 - TotalSegmentator's compositor
  (`np.copyto(out, lut[seg], where=lut[seg] != 0)`).
- **Geometry:** `Grid` and `Mapping` state how output indices map to model-grid coordinates.
  `Mapping.corner` is `scipy.ndimage.zoom(grid_mode=False)` - TotalSegmentator's
  `change_spacing` - and `Mapping.center` is `skimage.resize`, i.e. nnU-Net's resampler.
  All backends consume the same host-built per-axis tables, so they can differ only in float
  rounding of the blend, never in where they sample.
- **Reference:** `labelfield.reference` is a float64 numpy implementation of the same
  semantics, for tests.

```python
import torch, labelfield as lf

# one part of a multi-model task: logits (K, Z, Y, X) on the model grid, on the GPU
mapping = lf.Mapping.corner(input_shape, model_shape)        # output index -> model coordinate
out = torch.zeros(input_shape, dtype=torch.uint8, device=logits.device)
for logits, lut in parts:                                     # lut: channel -> output label
    lf.to_labels(logits, input_shape, mapping, lut=lut, paint=True, transparent="zero", out=out)
```

`interp="nearest"` with that mapping reproduces TotalSegmentator's default output exactly
(argmax on the model grid, composite, `zoom(order=0)`); `interp="linear"` is the smooth
version of the same labels. `tests/test_totalsegmentator.py` checks both.

## Install

    pip install "labelfield[torch] @ git+https://github.com/mhalle/labelfield.git@TAG"

numpy is all the geometry and the reference need; `import labelfield` does not import torch.
The `torch` extra brings `to_labels` and the backends, `cuda` brings Triton. Python 3.9 or
newer, torch 2.1 or newer; labels above 255 need a uint16 label map, which needs torch 2.3.
The Metal kernel needs torch 2.7 (`torch.mps.compile_shader`) and is not offered below it.

    uv venv && uv pip install -e ".[test]" && pytest

`tools/cuda_check.py` runs the suite on a CUDA box through Modal: the Triton kernel is the one
path a Mac cannot run.

Apache-2.0.
