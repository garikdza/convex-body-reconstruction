"""The convex bodies the example scripts reconstruct. No matplotlib."""

import numpy as np

from src import Ellipsoid, Polytope, random_polytope
from src.bodies import ConvexBody

HEPTAGON = np.array(
    [[-1.0, -0.3], [-0.5, -0.8], [0.6, -0.72], [1.02, -0.05], [0.7, 0.62], [-0.05, 0.86], [-0.78, 0.42]]
)


def rounded_heptagon() -> ConvexBody:
    """An irregular heptagon with corners rounded by a disk of radius 0.1 (a Minkowski sum)."""
    return Polytope(HEPTAGON) + Ellipsoid(0.1 * np.eye(2), np.zeros(2))


def rounded_polytope() -> ConvexBody:
    """A random 3D polytope with edges rounded by a ball of radius 0.12 (a Minkowski sum)."""
    return random_polytope(14, rng=5) + Ellipsoid(0.12 * np.eye(3), np.zeros(3))
