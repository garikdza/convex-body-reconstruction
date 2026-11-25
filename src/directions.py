"""Measurement directions on the unit circle / sphere."""

from __future__ import annotations

import numpy as np


def circle_directions(k: int, offset: float = 0.0) -> np.ndarray:
    """``k`` equally spaced unit vectors in the plane, shape ``(k, 2)``."""
    if k < 3:
        raise ValueError("need at least 3 directions in 2D")
    theta = offset + 2.0 * np.pi * np.arange(k) / k
    return np.column_stack([np.cos(theta), np.sin(theta)])


def sphere_directions(k: int) -> np.ndarray:
    """``k`` evenly spread unit vectors on the sphere (Fibonacci lattice), shape ``(k, 3)``."""
    if k < 4:
        raise ValueError("need at least 4 directions in 3D")
    i = np.arange(k) + 0.5
    z = 1.0 - 2.0 * i / k
    r = np.sqrt(1.0 - z**2)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i  # golden angle increments
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def random_directions(k: int, dim: int, rng: np.random.Generator | int | None = None) -> np.ndarray:
    """``k`` directions drawn uniformly from the unit sphere in ``R^dim``."""
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((k, dim))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def evenly_spread(k: int, dim: int) -> np.ndarray:
    """Deterministic, evenly spread directions in ``R^dim`` (``dim`` is 2 or 3)."""
    if dim == 2:
        return circle_directions(k)
    if dim == 3:
        return sphere_directions(k)
    raise ValueError("dim must be 2 or 3")
