"""Synthetic data: random test bodies and the noisy measurement model."""

from __future__ import annotations

import numpy as np

from .bodies import ConvexBody, Polytope

Seed = np.random.Generator | int | None


def random_polygon(n: int, rng: Seed = None) -> Polytope:
    """A random convex polygon with ``n`` vertices near the unit circle."""
    rng = np.random.default_rng(rng)
    theta = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = rng.uniform(0.7, 1.0, n)
    return Polytope(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))


def random_polytope(n: int, rng: Seed = None) -> Polytope:
    """The convex hull of ``n`` random points near a flattened unit sphere in ``R^3``."""
    rng = np.random.default_rng(rng)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v *= rng.uniform(0.7, 1.0, (n, 1)) * np.array([1.0, 0.8, 0.6])
    return Polytope(v)


def measure(
    body: ConvexBody, directions: np.ndarray, sigma: float = 0.0, rng: Seed = None
) -> np.ndarray:
    """Noisy support-function measurements ``y_i = h_K(u_i) + N(0, sigma^2)``."""
    rng = np.random.default_rng(rng)
    h = body.support(directions)
    return h + sigma * rng.standard_normal(h.shape)
