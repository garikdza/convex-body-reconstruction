import numpy as np


def circle_directions(k: int, offset: float = 0.0) -> np.ndarray:
    """
    Equally spaced unit vectors in the plane.

    Parameters
    ----------
    k : int
        number of directions, at least 3
    offset : float
        angle of the first direction, in radians

    Returns
    -------
    directions : np.ndarray
        directions at angles ``offset + 2πi/k``, shape ``(k, 2)``
    """
    if k < 3:
        raise ValueError(f'need at least 3 directions in 2D, got k={k!r}')
    theta = offset + 2.0 * np.pi * np.arange(k) / k
    return np.column_stack([np.cos(theta), np.sin(theta)])


def sphere_directions(k: int) -> np.ndarray:
    """
    Nearly evenly spread unit vectors on the sphere, from the Fibonacci lattice.

    Parameters
    ----------
    k : int
        number of directions, at least 4

    Returns
    -------
    directions : np.ndarray
        directions, shape ``(k, 3)``
    """
    if k < 4:
        raise ValueError(f'need at least 4 directions in 3D, got k={k!r}')
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(1.0 - z**2)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i  # golden angle increments
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def evenly_spread(k: int, dim: int) -> np.ndarray:
    """``circle_directions(k)`` for ``dim=2`` and ``sphere_directions(k)`` for ``dim=3``."""
    if dim == 2:
        return circle_directions(k)
    if dim == 3:
        return sphere_directions(k)
    raise ValueError(f'dim must be 2 or 3, got {dim!r}')


def random_directions(k: int, dim: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """
    Directions drawn uniformly from the unit sphere.

    Parameters
    ----------
    k : int
        number of directions
    dim : int
        dimension of the space
    rng : np.random.Generator | int | None
        random generator or seed

    Returns
    -------
    directions : np.ndarray
        directions, shape ``(k, dim)``
    """
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((k, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
