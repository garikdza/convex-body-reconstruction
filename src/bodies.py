from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial import ConvexHull

from src.directions import evenly_spread
from src.geometry import halfspace_vertices

# directions whose support points span the polytope that to_polytope returns, by dimension
POLYTOPE_DIRECTIONS = {2: 2000, 3: 4000}


class ConvexBody(ABC):
    """
    A convex body ``K`` in ``R^2`` or ``R^3``, known through its support function
    ``h_K(u) = max_{x in K} <x, u>``.

    ``K + L`` is the Minkowski sum ``{x + y : x in K, y in L}``, whose support function is ``h_K + h_L``; a polygon
    plus a disk is the polygon with rounded corners.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimension of the space, 2 or 3."""

    @abstractmethod
    def support(self, directions: np.ndarray) -> np.ndarray:
        """
        The support function at each direction.

        Parameters
        ----------
        directions : np.ndarray
            directions, shape ``(k, d)``

        Returns
        -------
        h : np.ndarray
            ``h_K(u)`` for each row ``u`` of ``directions``, shape ``(k,)``
        """

    @abstractmethod
    def support_points(self, directions: np.ndarray) -> np.ndarray:
        """
        A point of the body attaining the support function at each direction.

        Parameters
        ----------
        directions : np.ndarray
            directions, shape ``(k, d)``

        Returns
        -------
        X : np.ndarray
            for each row ``u`` of ``directions``, a point ``x`` of the body with ``<x, u> = h_K(u)``, shape ``(k, d)``
        """

    def to_polytope(self, n: int | None = None) -> 'Polytope':
        """
        A fine inscribed polytope approximating the body, for drawing it and measuring its volume.

        Parameters
        ----------
        n : int | None
            number of evenly spread directions whose support points span the polytope; ``None`` uses
            ``POLYTOPE_DIRECTIONS``

        Returns
        -------
        polytope : Polytope
            convex hull of the support points
        """
        n = n or POLYTOPE_DIRECTIONS[self.dim]
        return Polytope(self.support_points(evenly_spread(n, self.dim)))

    def __add__(self, other: 'ConvexBody') -> 'MinkowskiSum':
        return MinkowskiSum((self, other))


class Polytope(ConvexBody):
    """
    A convex polytope, the convex hull of its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        points whose convex hull is the polytope, shape ``(m, 2)`` or ``(m, 3)``; only the extreme points are kept,
        counter-clockwise in 2D
    """

    def __init__(self, vertices: np.ndarray) -> None:
        pts = np.asarray(vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(f'vertices must have shape (m, 2) or (m, 3), got {pts.shape}')
        self._vertices = pts[ConvexHull(pts).vertices]
        # the hull of the extreme points alone, so its simplices index into ``vertices``
        self._hull = ConvexHull(self._vertices)

    @classmethod
    def from_halfspaces(cls, directions: np.ndarray, h: np.ndarray) -> 'Polytope':
        """The polytope ``{x : <x, u_i> <= h_i for all i}`` for normals ``directions`` and offsets ``h``."""
        return cls(halfspace_vertices(directions, h))

    @property
    def vertices(self) -> np.ndarray:
        """The extreme points, shape ``(m, d)``, counter-clockwise in 2D."""
        return self._vertices

    @property
    def hull(self) -> ConvexHull:
        """The convex hull of ``vertices``."""
        return self._hull

    @property
    def dim(self) -> int:
        return self._vertices.shape[1]

    @property
    def volume(self) -> float:
        """Area in 2D, volume in 3D."""
        return float(self._hull.volume)

    def support(self, directions: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(directions) @ self._vertices.T).max(axis=1)

    def support_points(self, directions: np.ndarray) -> np.ndarray:
        return self._vertices[(np.atleast_2d(directions) @ self._vertices.T).argmax(axis=1)]

    def to_polytope(self, n: int | None = None) -> 'Polytope':
        return self

    def __repr__(self) -> str:
        return f'Polytope({len(self._vertices)} vertices in {self.dim}D)'


class Ellipsoid(ConvexBody):
    """
    The ellipse or ellipsoid ``{c + M z : |z| <= 1}`` for the matrix ``M``.

    Parameters
    ----------
    matrix : np.ndarray
        invertible shape matrix, shape ``(d, d)``
    center : np.ndarray
        centre ``c``, shape ``(d,)``
    """

    def __init__(self, matrix: np.ndarray, center: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)
        self.center = np.asarray(center, dtype=float)
        d = self.center.size
        if self.center.shape != (d,) or self.matrix.shape != (d, d):
            raise ValueError(
                f'matrix must be (d, d) for a center of shape (d,), got {self.matrix.shape} and {self.center.shape}'
            )
        if np.linalg.matrix_rank(self.matrix) < d:
            raise ValueError('matrix must be invertible')

    @property
    def dim(self) -> int:
        return self.center.shape[0]

    def support(self, directions: np.ndarray) -> np.ndarray:
        directions = np.atleast_2d(directions)
        return directions @ self.center + np.linalg.norm(directions @ self.matrix, axis=1)

    def support_points(self, directions: np.ndarray) -> np.ndarray:
        scaled = np.atleast_2d(directions) @ self.matrix
        return self.center + (scaled / np.linalg.norm(scaled, axis=1, keepdims=True)) @ self.matrix.T

    def __repr__(self) -> str:
        return f'Ellipsoid(matrix={self.matrix.tolist()}, center={self.center.tolist()})'


class MinkowskiSum(ConvexBody):
    """
    The Minkowski sum ``K_1 + ... + K_m``, usually built with ``+``.

    Parameters
    ----------
    parts : tuple[ConvexBody, ...]
        the summands, all of the same dimension; summands that are sums themselves are flattened into their parts
    """

    def __init__(self, parts: tuple[ConvexBody, ...]) -> None:
        self.parts = tuple(q for body in parts for q in (body.parts if isinstance(body, MinkowskiSum) else (body,)))
        dims = [body.dim for body in self.parts]
        if len(set(dims)) != 1:
            raise ValueError(f'summands must have the same dimension, got {dims}')

    @property
    def dim(self) -> int:
        return self.parts[0].dim

    def support(self, directions: np.ndarray) -> np.ndarray:
        return sum(body.support(directions) for body in self.parts)

    def support_points(self, directions: np.ndarray) -> np.ndarray:
        return sum(body.support_points(directions) for body in self.parts)

    def __repr__(self) -> str:
        return ' + '.join(map(repr, self.parts))
