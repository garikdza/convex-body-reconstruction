import numpy as np

from src.bodies import ConvexBody
from src.directions import evenly_spread


def hausdorff_distance(K: ConvexBody, L: ConvexBody, n_directions: int | None = None) -> float:
    """
    Hausdorff distance between two convex bodies, ``max_u |h_K(u) - h_L(u)|`` over unit ``u``.

    Parameters
    ----------
    K, L : ConvexBody
        bodies of the same dimension
    n_directions : int | None
        number of evenly spread directions the maximum is taken over; ``None`` uses 20000 in 2D and 50000 in 3D

    Returns
    -------
    distance : float
        the distance; a sampled maximum, so a lower bound that is tight for this many directions
    """
    return float(np.abs(_support_gap(K, L, n_directions)).max())


def l2_distance(K: ConvexBody, L: ConvexBody, n_directions: int | None = None) -> float:
    """
    L2 distance between the support functions of two convex bodies, ``(mean_u (h_K(u) - h_L(u))^2)^(1/2)``.

    The mean is over the uniform probability measure on the unit sphere, so this is the L2 norm used by Gardner,
    Kiderlen & Milanfar (2006) divided by the square root of the sphere's surface area.

    Parameters
    ----------
    K, L : ConvexBody
        bodies of the same dimension
    n_directions : int | None
        number of evenly spread directions the mean is taken over; ``None`` uses 20000 in 2D and 50000 in 3D

    Returns
    -------
    distance : float
        the distance
    """
    return float(np.sqrt(np.mean(_support_gap(K, L, n_directions) ** 2)))


def _support_gap(K: ConvexBody, L: ConvexBody, n_directions: int | None) -> np.ndarray:
    if K.dim != L.dim:
        raise ValueError(f'bodies must have the same dimension, got {K.dim} and {L.dim}')
    if n_directions is None:
        n_directions = 20_000 if K.dim == 2 else 50_000
    U = evenly_spread(n_directions, K.dim)
    return K.support(U) - L.support(U)
