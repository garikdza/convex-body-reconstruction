"""The convex bodies the example scripts reconstruct. No matplotlib."""

import numpy as np

from src import Ellipsoid, Polytope, random_polytope
from src.bodies import ConvexBody

# radius of the disk and the ball that round the corners and edges
ROUNDING_2D = 0.1
ROUNDING_3D = 0.12
# the random polytope: number of points and seed
POLYTOPE_POINTS = 14
POLYTOPE_SEED = 5
HEPTAGON = np.array(
    [[-1.0, -0.3], [-0.5, -0.8], [0.6, -0.72], [1.02, -0.05], [0.7, 0.62], [-0.05, 0.86], [-0.78, 0.42]]
)


def rounded_heptagon() -> ConvexBody:
    """An irregular heptagon with corners rounded by a disk of radius ``ROUNDING_2D``."""
    return Polytope(HEPTAGON) + Ellipsoid(ROUNDING_2D * np.eye(2), np.zeros(2))


def rounded_polytope() -> ConvexBody:
    """A random 3D polytope with edges rounded by a ball of radius ``ROUNDING_3D``."""
    return random_polytope(POLYTOPE_POINTS, rng=POLYTOPE_SEED) + Ellipsoid(ROUNDING_3D * np.eye(3), np.zeros(3))
