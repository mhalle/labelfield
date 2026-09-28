"""Maps between voxel index spaces: per-axis (:class:`Mapping`) and general (:class:`Affine`)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import Grid, Vec3, _vec3


@dataclass(frozen=True)
class Mapping:
    """A per-axis linear map ``x_to = a * x_from + b``, applied independently on Z, Y and X.

    In labelfield a Mapping takes a voxel index of a *from* grid (for :func:`to_labels`, the
    output grid) to a continuous voxel coordinate of a *to* grid (the model grid). Coordinates
    are in voxel units: integer values are voxel centers.

    Parameters
    ----------
    a : 3 floats or one float
        Scale per axis. Must be >= 0: a Mapping cannot flip, rotate or permute axes (use
        :class:`Affine` for that). A scale of 0 maps the whole axis to ``b``.
    b : 3 floats or one float, default (0, 0, 0)
        Offset per axis, in *to*-grid voxels.
    centered : bool, default False
        Set by :meth:`center`; normally not passed by hand. It does not change the map. It
        makes :func:`~labelfield.tables.axis_table` evaluate the coordinate as
        ``(j + 0.5) * a - 0.5`` instead of ``a * j + b``, which is how scipy and
        scikit-image compute it. The two forms are equal in exact arithmetic but can round
        differently in floating point when the coordinate lies exactly half-way between two
        samples, where a nearest-neighbor pick would then choose the other sample. Requires
        ``b == a / 2 - 1/2`` (``ValueError`` otherwise). Composing with the identity or with
        an integer shift (below) keeps the flag; any other composition, and :meth:`inverse`,
        drop it.
    exact : tuple or None, default None
        Set by composition; normally not passed by hand. ``(core_b, pre, post)``: the map is
        ``core(x + pre) - post``, where ``core`` is ``a * x + core_b`` (or the centered form)
        and ``pre`` and ``post`` are integer shifts, one per axis. :func:`~labelfield.tables.axis_table`
        evaluates it in that form, so the integer steps are exact. Composing a mapping with a
        pure integer shift (``a == 1``, integer ``b``) - a crop offset on either side - records
        the shift here instead of folding it into ``b``, which would round differently at an
        exact half-way coordinate. ``b`` is always the folded offset (``a * pre + core_b -
        post``), so ``apply`` and readers of ``b`` see the same map.

    Mappings compose with ``>>``: ``m1 >> m2`` applies ``m1`` first, then ``m2``. Instances
    are frozen and compare by value. Raises ``ValueError`` if ``a`` has a negative entry.
    """

    a: Vec3
    b: Vec3 = (0.0, 0.0, 0.0)
    centered: bool = False
    exact: tuple = None

    def __post_init__(self):
        object.__setattr__(self, "a", _vec3(self.a, "a"))
        object.__setattr__(self, "b", _vec3(self.b, "b"))
        object.__setattr__(self, "centered", bool(self.centered))
        if any(x < 0 for x in self.a):
            raise ValueError(f"a must be >= 0 on every axis (flips belong to the frame); got {self.a}")
        if self.exact is not None:
            core_b, pre, post = self.exact
            core_b = _vec3(core_b, "exact core_b")
            pre = tuple(int(v) for v in pre)
            post = tuple(int(v) for v in post)
            if len(pre) != 3 or len(post) != 3:
                raise ValueError("exact shifts must have 3 entries (Z, Y, X)")
            object.__setattr__(self, "exact", (core_b, pre, post))
        core_b = self.terms()[0]
        if self.centered and not np.allclose(core_b, 0.5 * np.asarray(self.a) - 0.5, rtol=0, atol=1e-9):
            raise ValueError(f"a centered mapping has b = a / 2 - 1/2; got a={self.a}, b={core_b}")

    def terms(self):
        """``(core_b, pre, post)`` - the form :func:`~labelfield.tables.axis_table` evaluates:
        ``core(x + pre) - post``. Without integer shifts recorded, ``(b, (0, 0, 0), (0, 0, 0))``."""
        if self.exact is None:
            return self.b, (0, 0, 0), (0, 0, 0)
        return self.exact

    @property
    def integer_shift(self):
        """The integer offsets if this mapping is a pure integer shift (``a == 1``, integer
        ``b``, nothing else recorded), else None."""
        if self.a != (1.0, 1.0, 1.0) or self.centered or self.exact is not None:
            return None
        if not all(float(v).is_integer() for v in self.b):
            return None
        return tuple(int(v) for v in self.b)

    @property
    def is_identity(self) -> bool:
        """True if ``a == (1, 1, 1)`` and ``b == (0, 0, 0)`` exactly."""
        return self.a == (1.0, 1.0, 1.0) and self.b == (0.0, 0.0, 0.0)

    def apply(self, x_from) -> np.ndarray:
        """Map coordinates of shape (..., 3), (Z, Y, X) order; returns float64 of the same shape.

        Always uses ``a * x + b``, also for a ``centered`` mapping."""
        return np.asarray(x_from, dtype=np.float64) * np.asarray(self.a) + np.asarray(self.b)

    def then(self, other: "Mapping") -> "Mapping":
        """The composition that applies ``self`` first, then ``other`` (same as ``self >> other``)."""
        if self.is_identity:
            return other
        if other.is_identity:
            return self
        shift = other.integer_shift
        if shift is not None:                                # a shift after: record it exactly
            core_b, pre, post = self.terms()
            new_post = tuple(p - s for p, s in zip(post, shift))
            return Mapping(self.a, tuple(np.asarray(self.b) + np.asarray(shift, dtype=np.float64)),
                           centered=self.centered, exact=(core_b, pre, new_post))
        shift = self.integer_shift
        if shift is not None:                                # a shift before: likewise
            core_b, pre, post = other.terms()
            new_pre = tuple(p + s for p, s in zip(pre, shift))
            b = np.asarray(other.a) * np.asarray(shift, dtype=np.float64) + np.asarray(other.b)
            return Mapping(other.a, tuple(b), centered=other.centered, exact=(core_b, new_pre, post))
        a1, b1 = np.asarray(self.a), np.asarray(self.b)
        a2, b2 = np.asarray(other.a), np.asarray(other.b)
        return Mapping(tuple(a2 * a1), tuple(a2 * b1 + b2))

    __rshift__ = then

    def inverse(self) -> "Mapping":
        """The inverse map. Raises ``ValueError`` if any scale is 0. The result is not ``centered``."""
        if any(x == 0 for x in self.a):
            raise ValueError("mapping with a zero factor has no inverse")
        a = 1.0 / np.asarray(self.a)
        return Mapping(tuple(a), tuple(-np.asarray(self.b) * a))

    # -- constructors -----------------------------------------------------
    @classmethod
    def identity(cls) -> "Mapping":
        """The map ``x_to = x_from``."""
        return cls((1.0, 1.0, 1.0), (0.0, 0.0, 0.0))

    @classmethod
    def center(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-center (half-pixel) rule: ``x_to = (x_from + 0.5) * n_to / n_from - 0.5``.

        ``shape_from`` and ``shape_to`` are (Z, Y, X) voxel counts. The outer boundaries of the
        two grids coincide: the outer face of the first voxel of each grid lies at
        coordinate -0.5 of the other. This is the convention of
        ``skimage.transform.resize``, ``scipy.ndimage.zoom(..., grid_mode=True)``,
        ``torch.nn.functional.interpolate(..., align_corners=False)`` in its linear modes, and
        hence of nnU-Net's resampler. Use ``Mapping.center(output_shape, model_shape)`` when
        the model grid was produced from the output grid by one of these. With
        ``interp="nearest"``, picks match ``scipy.ndimage.zoom(order=0, grid_mode=True)``
        exactly, including at exact ties (the result is ``centered``).
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = n_to / n_from
        return cls(tuple(a), tuple(0.5 * a - 0.5), centered=True)

    @classmethod
    def corner(cls, shape_from, shape_to) -> "Mapping":
        """Voxel-corner rule: ``x_to = x_from * (n_to - 1) / (n_from - 1)``.

        ``shape_from`` and ``shape_to`` are (Z, Y, X) voxel counts. The first voxel centers
        of the two grids coincide, and so do the last ones. This is the convention of
        ``scipy.ndimage.zoom`` with its default ``grid_mode=False`` and of
        ``torch.nn.functional.interpolate(..., align_corners=True)``; TotalSegmentator's
        ``change_spacing`` uses it. Use ``Mapping.corner(output_shape, model_shape)`` when the
        model grid was produced from the output grid by one of these. An axis where
        ``shape_from`` is 1 maps to coordinate 0.
        """
        n_from = np.asarray(shape_from, dtype=np.float64)
        n_to = np.asarray(shape_to, dtype=np.float64)
        a = np.where(n_from > 1, (n_to - 1) / np.maximum(n_from - 1, 1), 0.0)
        return cls(tuple(a), (0.0, 0.0, 0.0))

    @classmethod
    def spacing(cls, spacing_from, spacing_to, shift=(0.0, 0.0, 0.0)) -> "Mapping":
        """Origin-aligned rule from voxel spacings: ``x_to = x_from * s_from / s_to + shift``.

        ``spacing_from`` and ``spacing_to`` are the voxel spacings (3 floats or one) of the two
        grids, ``shift`` an offset in *to*-grid voxels. With ``shift = 0`` the centers of voxel
        (0, 0, 0) of both grids coincide and the scale is the exact spacing ratio, independent
        of the grid shapes. Use it when the model grid was sampled that way (for example by the
        nnunet-inference-mlx pipeline, whose scale factor is ``s2t = acq / target``).
        """
        s_from = np.asarray(_vec3(spacing_from, "spacing_from"))
        s_to = np.asarray(_vec3(spacing_to, "spacing_to"))
        return cls(tuple(s_from / s_to), _vec3(shift, "shift"))

    @classmethod
    def between(cls, grid_from: Grid, grid_to: Grid) -> "Mapping":
        """Physical rule: index on ``grid_from`` -> coordinate on ``grid_to``, through physical
        positions: ``grid_to.mm_to_index(grid_from.index_to_mm(x))``.

        Both arguments are :class:`~labelfield.grid.Grid` or anything :meth:`Grid.like`
        accepts. Use it when both grids are known in physical space (spacing and origin), for
        example an output region of interest and a model grid; the shapes do not enter.
        The identity when the grids coincide. The result is not ``centered``.
        """
        gf, gt = Grid.like(grid_from), Grid.like(grid_to)
        s_from, s_to = np.asarray(gf.spacing), np.asarray(gt.spacing)
        b = (np.asarray(gf.origin) - np.asarray(gt.origin)) / s_to
        return cls(tuple(s_from / s_to), tuple(b))

    def __repr__(self) -> str:
        a = ", ".join(f"{x:g}" for x in self.a)
        b = ", ".join(f"{x:g}" for x in self.b)
        shifts = ""
        if self.exact is not None and (any(self.exact[1]) or any(self.exact[2])):
            shifts = f", pre={self.exact[1]}, post={self.exact[2]}"
        return f"Mapping(a=({a}), b=({b}){', centered' if self.centered else ''}{shifts})"


#: Off-diagonal terms of an :class:`Affine` smaller than this fraction of its largest diagonal
#: term are treated as floating-point noise, not rotation, by ``Affine.separable``. It is far
#: below any real obliquity (a 0.001 degree tilt gives 1.7e-5) and far above the noise left by
#: composing two geometries (~1e-16).
SEPARABLE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Affine:
    """A general affine map between voxel index spaces: ``x_to = x_from @ m + b``.

    Coordinates are row vectors in array order (Z, Y, X). Rotation, shear, flips and axis
    swaps are all allowed.

    Parameters
    ----------
    m : 3x3 array-like of finite floats
    b : 3 floats or one float, default (0, 0, 0)

    Use it for two grids whose axes do not line up in world space, where no per-axis
    :class:`Mapping` relates them (for example a model's conformed grid against an oblique
    acquisition). :meth:`between` builds one from two geometry records.

    :func:`~labelfield.labels.to_labels` and :func:`~labelfield.tables.build_tables` accept
    only a :class:`Mapping`. If :attr:`separable` is not None, pass it instead. Otherwise
    labelfield has no fused path for the map: compute each output voxel's coordinates with
    :meth:`apply`, and use :func:`~labelfield.tables.axis_coords` per axis to get the same
    inside test, samples and weights that the per-axis tables would give.

    Instances are frozen. Raises ``ValueError`` if ``m`` is not a finite 3x3 matrix.
    """

    m: tuple
    b: Vec3 = (0.0, 0.0, 0.0)

    def __post_init__(self):
        m = np.asarray(self.m, dtype=np.float64)
        if m.shape != (3, 3) or not np.isfinite(m).all():
            raise ValueError(f"m must be a finite 3x3 matrix; got {self.m!r}")
        object.__setattr__(self, "m", tuple(tuple(float(v) for v in r) for r in m))
        object.__setattr__(self, "b", _vec3(self.b, "b"))

    def apply(self, x_from) -> np.ndarray:
        """Map coordinates of shape (..., 3), (Z, Y, X) order; returns float64 of the same shape."""
        return np.asarray(x_from, dtype=np.float64) @ np.asarray(self.m) + np.asarray(self.b)

    def inverse(self) -> "Affine":
        """The inverse map. Raises ``numpy.linalg.LinAlgError`` if ``m`` is singular."""
        mi = np.linalg.inv(np.asarray(self.m))
        return Affine(mi, tuple(-np.asarray(self.b) @ mi))

    @property
    def separable(self) -> Mapping | None:
        """The same map as a per-axis :class:`Mapping`, or None if it has none.

        A Mapping exists when ``m`` is diagonal with non-negative entries (no rotation,
        shear, flip or axis swap). Off-diagonal terms smaller than ``SEPARABLE_TOLERANCE``
        times the largest diagonal magnitude are treated as zero, because :meth:`between`
        applied to two grids of the same oblique orientation leaves floating-point residue
        of about 1e-17 there. The returned Mapping uses the diagonal of ``m`` and ``b``, and is
        not ``centered``."""
        m = np.asarray(self.m)
        d = np.diag(m)
        scale = float(np.abs(d).max()) or 1.0
        if np.any(np.abs(m - np.diag(d)) > SEPARABLE_TOLERANCE * scale) or np.any(d < 0):
            return None
        return Mapping(tuple(d), self.b)

    @classmethod
    def between(cls, geo_from, geo_to) -> "Affine":
        """Index on ``geo_from`` -> continuous index on ``geo_to``, through world space.

        Each argument is any object with two attributes: ``directions``, a 3x3 array whose
        row ``i`` is the world-space displacement of one step along array axis ``i`` (so it
        includes the spacing), and ``origin``, the world position of the center of voxel
        (0, 0, 0). This is NRRD's "space directions" / "space origin" convention, with rows in
        array order. Raises ``numpy.linalg.LinAlgError`` if ``geo_to.directions`` is singular.
        """
        d_from = np.asarray(geo_from.directions, dtype=np.float64)
        d_to_inv = np.linalg.inv(np.asarray(geo_to.directions, dtype=np.float64))
        b = (np.asarray(geo_from.origin, dtype=np.float64) - np.asarray(geo_to.origin, dtype=np.float64)) @ d_to_inv
        return cls(d_from @ d_to_inv, tuple(b))

    def __repr__(self) -> str:
        return f"Affine(m={np.round(np.asarray(self.m), 6).tolist()}, b={tuple(round(v, 6) for v in self.b)})"
