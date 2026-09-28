"""Regular, axis-aligned sampling grids described by shape, spacing and origin."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Vec3 = tuple[float, float, float]
Shape3 = tuple[int, int, int]


def _vec3(v, name: str) -> Vec3:
    if np.isscalar(v):
        v = (v, v, v)
    t = tuple(float(x) for x in v)
    if len(t) != 3:
        raise ValueError(f"{name} must have 3 entries (Z, Y, X); got {v!r}")
    return t


def _shape3(v, name: str) -> Shape3:
    t = tuple(int(x) for x in v)
    if len(t) != 3 or any(x < 1 for x in t):
        raise ValueError(f"{name} must be 3 positive ints (Z, Y, X); got {v!r}")
    return t


@dataclass(frozen=True)
class Grid:
    """A regular, axis-aligned sampling grid.

    Parameters
    ----------
    shape : 3 positive ints, (Z, Y, X)
        Number of voxels along each axis.
    spacing : 3 positive floats or one float, default (1, 1, 1)
        Distance between neighboring voxel centers along each axis. The unit is the
        caller's choice (millimeters by convention); it only has to be the same for every
        grid that is compared. A scalar is used for all three axes.
    origin : 3 floats or one float, default (0, 0, 0)
        Physical position of the *center* of voxel (0, 0, 0).

    Voxel ``i`` along an axis is centered at ``origin + i * spacing`` and covers
    ``[origin + (i - 0.5) * spacing, origin + (i + 0.5) * spacing]``. Axis order is (Z, Y, X)
    throughout. A Grid has no orientation (no direction cosines): all grids that are compared
    are assumed to share one set of world axes. Use :class:`~labelfield.mapping.Affine` for
    grids that are rotated or flipped relative to each other.

    Instances are frozen (immutable) and compare by value. Raises ``ValueError`` if a shape
    entry is < 1, a spacing entry is <= 0, or a field does not have 3 entries.
    """

    shape: Shape3
    spacing: Vec3 = (1.0, 1.0, 1.0)
    origin: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        object.__setattr__(self, "shape", _shape3(self.shape, "shape"))
        object.__setattr__(self, "spacing", _vec3(self.spacing, "spacing"))
        object.__setattr__(self, "origin", _vec3(self.origin, "origin"))
        if any(s <= 0 for s in self.spacing):
            raise ValueError(f"spacing must be positive; got {self.spacing}")

    # -- geometry ---------------------------------------------------------
    @property
    def n_voxels(self) -> int:
        """Total number of voxels, ``Z * Y * X``."""
        return int(np.prod(self.shape))

    def index_to_mm(self, index) -> np.ndarray:
        """Continuous voxel index, array of shape (..., 3) in (Z, Y, X) order -> physical
        position (..., 3), float64."""
        return np.asarray(self.origin) + np.asarray(index, dtype=np.float64) * np.asarray(self.spacing)

    def mm_to_index(self, mm) -> np.ndarray:
        """Physical position, array of shape (..., 3) in (Z, Y, X) order -> continuous voxel
        index (..., 3), float64. Voxel centers have integer indices."""
        return (np.asarray(mm, dtype=np.float64) - np.asarray(self.origin)) / np.asarray(self.spacing)

    @property
    def extent_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Outer boundary of the grid, ``(lo, hi)``: the outer faces of the first and last
        voxels, i.e. the first and last voxel centers moved out by half a voxel."""
        sp = np.asarray(self.spacing)
        lo = np.asarray(self.origin) - sp / 2
        hi = np.asarray(self.origin) + (np.asarray(self.shape) - 1) * sp + sp / 2
        return lo, hi

    @property
    def center_extent_mm(self) -> tuple[np.ndarray, np.ndarray]:
        """Positions of the first and last voxel centers, ``(lo, hi)``."""
        lo = np.asarray(self.origin, dtype=np.float64)
        return lo, lo + (np.asarray(self.shape) - 1) * np.asarray(self.spacing)

    # -- constructors -----------------------------------------------------
    @classmethod
    def like(cls, other) -> "Grid":
        """A Grid with the geometry of ``other``: a Grid (returned as is) or any object with a
        ``shape`` attribute and optional ``spacing`` and ``origin`` attributes, in (Z, Y, X)
        order. Missing ``spacing`` / ``origin`` default to 1 and 0."""
        if isinstance(other, Grid):
            return other
        return cls(other.shape, getattr(other, "spacing", (1.0, 1.0, 1.0)), getattr(other, "origin", (0.0, 0.0, 0.0)))

    def resampled(self, spacing, *, align: str = "edges") -> "Grid":
        """A grid covering the same field of view at spacing ``spacing`` (3 floats or one).

        ``align="edges"`` (default) keeps the outer boundary (:attr:`extent_mm`) in place:
        ``n_out = round(n * s_in / s_out)`` and the origin moves by ``(s_out - s_in) / 2``. This
        is the physical form of the voxel-center convention (:meth:`Mapping.center`).
        ``align="centers"`` keeps the first voxel center in place and the last one as close
        as the new spacing allows: ``n_out = round((n - 1) * s_in / s_out) + 1``, origin
        unchanged. This is the physical form of the voxel-corner convention
        (:meth:`Mapping.corner`). ``n_out`` is at least 1. Because ``n_out`` is rounded, the
        boundary (or last center) is kept exactly only when the ratio is exact. Raises
        ``ValueError`` for any other ``align``.
        """
        s_out = _vec3(spacing, "spacing")
        n = np.asarray(self.shape)
        s_in = np.asarray(self.spacing)
        s_o = np.asarray(s_out)
        if align == "edges":
            n_out = np.maximum(1, np.rint(n * s_in / s_o)).astype(int)
            origin = np.asarray(self.origin) - s_in / 2 + s_o / 2
        elif align == "centers":
            n_out = np.maximum(1, np.rint((n - 1) * s_in / s_o) + 1).astype(int)
            origin = np.asarray(self.origin)
        else:
            raise ValueError(f"align must be 'edges' or 'centers'; got {align!r}")
        return Grid(tuple(int(x) for x in n_out), s_out, tuple(float(x) for x in origin))

    @classmethod
    def isotropic(cls, spacing: float, *, like: "Grid", align: str = "edges") -> "Grid":
        """An isotropic grid (``spacing`` on every axis) covering the field of view of
        ``like``: ``Grid.like(like).resampled((spacing,) * 3, align=align)``."""
        return Grid.like(like).resampled((float(spacing),) * 3, align=align)

    def roi(self, lo_mm, hi_mm) -> "Grid":
        """The sub-grid of this grid's voxels whose volumes overlap the box
        ``[lo_mm, hi_mm]`` (physical positions, (Z, Y, X)). The result keeps this grid's
        spacing and lies on its lattice. A voxel that only touches the box at a face is not
        included. The index range is clipped to this grid, so a box partly outside is
        trimmed and a box entirely outside yields a slab of edge voxels rather than an
        error. Raises ``ValueError`` if ``hi_mm < lo_mm`` on any axis."""
        lo = np.asarray(_vec3(lo_mm, "lo_mm"))
        hi = np.asarray(_vec3(hi_mm, "hi_mm"))
        if np.any(hi < lo):
            raise ValueError("hi_mm must be >= lo_mm on every axis")
        n = np.asarray(self.shape)
        i0 = np.clip(np.floor(self.mm_to_index(lo) + 0.5), 0, n - 1).astype(int)
        i1 = np.clip(np.ceil(self.mm_to_index(hi) - 0.5), i0, n - 1).astype(int)
        shape = tuple(int(x) for x in (i1 - i0 + 1))
        origin = tuple(float(x) for x in self.index_to_mm(i0))
        return Grid(shape, self.spacing, origin)

    def __repr__(self) -> str:
        sp = ", ".join(f"{s:g}" for s in self.spacing)
        og = ", ".join(f"{o:g}" for o in self.origin)
        return f"Grid(shape={self.shape}, spacing=({sp}), origin=({og}))"
