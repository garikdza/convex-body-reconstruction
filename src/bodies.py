"""Convex bodies described by their support functions.

The support function of a convex body ``K`` is ``h_K(u) = max_{x in K} <x, u>``. Every
body implements the :class:`ConvexBody` interface:

* ``support(U)``         evaluates ``h_K`` at each row of ``U`` (shape ``(k, d)``),
* ``support_points(U)``  returns a point of ``K`` where that maximum is attained,
* ``K + L``              is the Minkowski sum, whose support function is ``h_K + h_L``
                         (e.g. polygon + disk = polygon with rounded corners).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from .directions import evenly_spread
from .geometry import halfspace_vertices


class ConvexBody(ABC):
    """A convex body in ``R^2`` or ``R^3``, known through its support function."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def support(self, U: np.ndarray) -> np.ndarray:
        """``h_K(u)`` for each row ``u`` of ``U``, shape ``(k,)``."""

    @abstractmethod
    def support_points(self, U: np.ndarray) -> np.ndarray:
        """A point of ``K`` attaining ``h_K(u)`` for each row ``u`` of ``U``, shape ``(k, d)``."""

    def __add__(self, other: ConvexBody) -> MinkowskiSum:
        return MinkowskiSum((self, other))


@dataclass
class Polytope(ConvexBody):
    """A convex polytope given by its vertices; points that are not extreme are dropped.

    In 2D the vertices are stored in counter-clockwise order.
    """

    vertices: np.ndarray
    hull: ConvexHull = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError('vertices must have shape (m, 2) or (m, 3)')
        self.vertices = pts[ConvexHull(pts).vertices]
        self.hull = ConvexHull(self.vertices)

    @classmethod
    def from_halfspaces(cls, U: np.ndarray, h: np.ndarray) -> Polytope:
        """The polytope ``{x : <x, u_i> <= h_i for all i}``."""
        return cls(halfspace_vertices(U, h))

    @property
    def dim(self) -> int:
        return self.vertices.shape[1]

    @property
    def volume(self) -> float:
        """Area in 2D, volume in 3D."""
        return float(self.hull.volume)

    def support(self, U: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(U) @ self.vertices.T).max(axis=1)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        return self.vertices[(np.atleast_2d(U) @ self.vertices.T).argmax(axis=1)]


@dataclass
class Ellipsoid(ConvexBody):
    """Ellipse (2D) or ellipsoid (3D) ``{c + A z : |z| <= 1}`` with ``A`` invertible."""

    A: np.ndarray
    center: np.ndarray

    def __post_init__(self) -> None:
        self.A = np.asarray(self.A, dtype=float)
        self.center = np.asarray(self.center, dtype=float)

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    def support(self, U: np.ndarray) -> np.ndarray:
        U = np.atleast_2d(U)
        return U @ self.center + np.linalg.norm(U @ self.A, axis=1)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        W = np.atleast_2d(U) @ self.A
        return self.center + (W / np.linalg.norm(W, axis=1, keepdims=True)) @ self.A.T


@dataclass
class MinkowskiSum(ConvexBody):
    """``K_1 + ... + K_m = {x_1 + ... + x_m : x_j in K_j}``."""

    parts: tuple[ConvexBody, ...]

    @property
    def dim(self) -> int:
        return self.parts[0].dim

    def support(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support(U) for K in self.parts)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support_points(U) for K in self.parts)

    def __add__(self, other: ConvexBody) -> MinkowskiSum:
        return MinkowskiSum((*self.parts, other))


def to_polytope(body: ConvexBody, n: int | None = None) -> Polytope:
    """A fine inscribed polytope approximation of ``body`` (for plotting and volumes),
    built from its support points in ``n`` evenly spread directions."""
    if isinstance(body, Polytope):
        return body
    n = n or (2000 if body.dim == 2 else 4000)
    return Polytope(body.support_points(evenly_spread(n, body.dim)))
