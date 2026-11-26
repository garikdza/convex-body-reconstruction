import numpy as np

from src.bodies import ConvexBody, Polytope
from src.directions import random_directions


def random_polygon(n: int, rng: np.random.Generator | int | None = None) -> Polytope:
    """
    A random convex polygon near the unit circle.

    Parameters
    ----------
    n : int
        number of points drawn at random angles and radii in [0.7, 1]; the polygon is their convex hull
    rng : np.random.Generator | int | None
        random generator or seed

    Returns
    -------
    P : Polytope
        the polygon
    """
    rng = np.random.default_rng(rng)
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.7, 1.0, n)
    return Polytope(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))


def random_polytope(n: int, rng: np.random.Generator | int | None = None) -> Polytope:
    """
    A random convex polytope near the ellipsoid with semi-axes 1, 0.8 and 0.6.

    Parameters
    ----------
    n : int
        number of points drawn in random directions at radii in [0.7, 1], then scaled to the semi-axes; the polytope
        is their convex hull
    rng : np.random.Generator | int | None
        random generator or seed

    Returns
    -------
    P : Polytope
        the polytope
    """
    rng = np.random.default_rng(rng)
    v = random_directions(n, 3, rng)
    v *= rng.uniform(0.7, 1.0, (n, 1)) * np.array([1.0, 0.8, 0.6])
    return Polytope(v)


def measure(
    body: ConvexBody, directions: np.ndarray, sigma: float = 0.0, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    """
    Noisy support-function measurements ``y_i = h_K(u_i) + e_i``, with ``e_i`` independent ``N(0, sigma^2)``.

    Parameters
    ----------
    body : ConvexBody
        the body ``K``
    directions : np.ndarray
        unit directions ``u_i``, shape ``(k, d)``
    sigma : float
        noise standard deviation
    rng : np.random.Generator | int | None
        random generator or seed

    Returns
    -------
    y : np.ndarray
        measurements, shape ``(k,)``
    """
    rng = np.random.default_rng(rng)
    h = body.support(directions)
    return h + sigma * rng.standard_normal(h.shape)
