"""Convex bodies described by their support functions.

The support function of a convex body ``K`` is ``h_K(u) = max_{x in K} <x, u>``.
Every body here exposes ``support(U)``, which evaluates ``h_K`` at each row of
``U`` (shape ``(k, d)``), and ``support_points(U)``, which returns a point of ``K`` where
the maximum is attained. Bodies can be added (Minkowski sum): ``polygon + ellipse`` is a
rounded polygon whose support function is the sum of the two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection


@dataclass
class Polytope:
    """A convex polytope given by its vertices (convex position is enforced)."""

    vertices: np.ndarray
    hull: ConvexHull = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError("vertices must have shape (m, 2) or (m, 3)")
        # keep only extreme points (counter-clockwise in 2D), then index the hull into them
        self.vertices = pts[ConvexHull(pts).vertices]
        self.hull = ConvexHull(self.vertices)

    @property
    def dim(self) -> int:
        return self.vertices.shape[1]

    def support(self, U: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(U) @ self.vertices.T).max(axis=1)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        """A point of the body attaining ``h(u)`` for each direction ``u``."""
        return self.vertices[(np.atleast_2d(U) @ self.vertices.T).argmax(axis=1)]

    def __add__(self, other) -> MinkowskiSum:
        return MinkowskiSum((self, other))

    def boundary(self) -> np.ndarray:
        """2D only: vertices in counter-clockwise order, first vertex repeated at the end."""
        if self.dim != 2:
            raise ValueError("boundary() is only defined for polygons")
        return np.vstack([self.vertices, self.vertices[:1]])

    def triangles(self) -> np.ndarray:
        """3D only: boundary triangles, shape ``(t, 3, 3)``."""
        if self.dim != 3:
            raise ValueError("triangles() is only defined for 3D polytopes")
        return self.vertices[self.hull.simplices]

    @property
    def volume(self) -> float:
        """Area in 2D, volume in 3D."""
        return float(self.hull.volume)

    @classmethod
    def from_halfspaces(cls, U: np.ndarray, h: np.ndarray) -> Polytope:
        """The polytope ``{x : <x, u_i> <= h_i for all i}``.

        Raises ``ValueError`` if the intersection is unbounded or has empty interior.
        """
        U = np.asarray(U, dtype=float)
        h = np.asarray(h, dtype=float)
        center, radius = chebyshev_center(U, h)
        scale = max(1.0, float(np.abs(h).max()))
        if radius <= 1e-9 * scale:
            raise ValueError("halfspace intersection has (numerically) empty interior")
        hs = HalfspaceIntersection(np.column_stack([U, -h]), center)
        return cls(hs.intersections)


@dataclass
class Ellipsoid:
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
        U = np.atleast_2d(U)
        W = U @ self.A
        return self.center + (W / np.linalg.norm(W, axis=1, keepdims=True)) @ self.A.T

    def __add__(self, other) -> MinkowskiSum:
        return MinkowskiSum((self, other))


@dataclass
class MinkowskiSum:
    """``K_1 + ... + K_m = {x_1 + ... + x_m : x_j in K_j}``; support functions simply add."""

    parts: tuple

    @property
    def dim(self) -> int:
        return self.parts[0].dim

    def support(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support(U) for K in self.parts)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support_points(U) for K in self.parts)

    def __add__(self, other) -> MinkowskiSum:
        return MinkowskiSum((*self.parts, other))


def to_polytope(body, n: int | None = None) -> Polytope:
    """A fine inscribed polytope approximation of ``body`` (for plotting and volumes).

    Built from the body's support points in ``n`` evenly spread directions.
    """
    from .directions import evenly_spread

    if isinstance(body, Polytope):
        return body
    d = body.dim
    n = n or (2000 if d == 2 else 4000)
    return Polytope(body.support_points(evenly_spread(n, d)))


def chebyshev_center(U: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    """Center and radius of the largest ball inside ``{x : <x, u_i> <= h_i}``."""
    k, d = U.shape
    norms = np.linalg.norm(U, axis=1)
    c = np.zeros(d + 1)
    c[-1] = -1.0  # maximise the radius
    res = linprog(
        c,
        A_ub=np.column_stack([U, norms]),
        b_ub=h,
        bounds=[(None, None)] * d + [(0, None)],
        method="highs",
    )
    if res.status == 3:
        raise ValueError(
            "halfspace intersection is unbounded; directions must not lie in a closed hemisphere"
        )
    if not res.success:
        raise ValueError(f"could not find an interior point: {res.message}")
    return res.x[:d], float(res.x[-1])


def random_polygon(n: int, rng: np.random.Generator | int | None = None) -> Polytope:
    """A random convex polygon with ``n`` vertices near the unit circle."""
    rng = np.random.default_rng(rng)
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.7, 1.0, n)
    pts = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    return Polytope(pts)


def random_polytope(n: int, rng: np.random.Generator | int | None = None) -> Polytope:
    """The convex hull of ``n`` random points near the unit sphere in ``R^3``."""
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.7, 1.0, (n, 1)) * np.array([1.0, 0.8, 0.6])
    return Polytope(v)


def measure(
    body, directions: np.ndarray, sigma: float = 0.0, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """Noisy support-function measurements ``y_i = h_K(u_i) + N(0, sigma^2)``."""
    rng = np.random.default_rng(rng)
    h = body.support(directions)
    return h + sigma * rng.standard_normal(h.shape)
