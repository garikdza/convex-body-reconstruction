from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull

from src.directions import evenly_spread
from src.geometry import halfspace_vertices


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
    def support(self, U: np.ndarray) -> np.ndarray:
        """
        The support function at each direction.

        Parameters
        ----------
        U : np.ndarray
            directions, shape ``(k, d)``

        Returns
        -------
        h : np.ndarray
            ``h_K(u)`` for each row ``u`` of ``U``, shape ``(k,)``
        """

    @abstractmethod
    def support_points(self, U: np.ndarray) -> np.ndarray:
        """
        A point of the body attaining the support function at each direction.

        Parameters
        ----------
        U : np.ndarray
            directions, shape ``(k, d)``

        Returns
        -------
        X : np.ndarray
            for each row ``u`` of ``U``, a point ``x`` of ``K`` with ``<x, u> = h_K(u)``, shape ``(k, d)``
        """

    def __add__(self, other: 'ConvexBody') -> 'MinkowskiSum':
        return MinkowskiSum((self, other))


@dataclass
class Polytope(ConvexBody):
    """
    A convex polytope, the convex hull of its vertices.

    Parameters
    ----------
    vertices : np.ndarray
        points whose convex hull is the polytope, shape ``(m, 2)`` or ``(m, 3)``; only the extreme points are kept,
        counter-clockwise in 2D

    Attributes
    ----------
    hull : scipy.spatial.ConvexHull
        convex hull of ``vertices``
    """

    vertices: np.ndarray
    hull: ConvexHull = field(init=False, repr=False)

    def __post_init__(self) -> None:
        pts = np.asarray(self.vertices, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3):
            raise ValueError(f'vertices must have shape (m, 2) or (m, 3), got {pts.shape}')
        self.vertices = pts[ConvexHull(pts).vertices]
        self.hull = ConvexHull(self.vertices)

    @classmethod
    def from_halfspaces(cls, U: np.ndarray, h: np.ndarray) -> 'Polytope':
        """The polytope ``{x : <x, u_i> <= h_i for all i}`` for normals ``U`` and offsets ``h``."""
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
    """
    The ellipse or ellipsoid ``{c + A z : |z| <= 1}``.

    Parameters
    ----------
    A : np.ndarray
        invertible shape matrix, shape ``(d, d)``
    center : np.ndarray
        centre ``c``, shape ``(d,)``
    """

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
    """
    The Minkowski sum ``K_1 + ... + K_m``, usually built with ``+``.

    Parameters
    ----------
    parts : tuple[ConvexBody, ...]
        the summands, all of the same dimension
    """

    parts: tuple[ConvexBody, ...]

    @property
    def dim(self) -> int:
        return self.parts[0].dim

    def support(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support(U) for K in self.parts)

    def support_points(self, U: np.ndarray) -> np.ndarray:
        return sum(K.support_points(U) for K in self.parts)

    def __add__(self, other: ConvexBody) -> 'MinkowskiSum':
        return MinkowskiSum((*self.parts, other))


def to_polytope(body: ConvexBody, n: int | None = None) -> Polytope:
    """
    A fine inscribed polytope approximating ``body``, for drawing it and measuring its volume.

    Parameters
    ----------
    body : ConvexBody
        body to approximate; a ``Polytope`` is returned as it is
    n : int | None
        number of evenly spread directions whose support points span the polytope; ``None`` uses 2000 in 2D and
        4000 in 3D

    Returns
    -------
    P : Polytope
        convex hull of the support points
    """
    if isinstance(body, Polytope):
        return body
    n = n or (2000 if body.dim == 2 else 4000)
    return Polytope(body.support_points(evenly_spread(n, body.dim)))
