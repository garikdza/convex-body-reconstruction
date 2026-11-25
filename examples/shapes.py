"""The example bodies used in the README figures and the benchmark."""

import numpy as np

import src as cr


def rounded_heptagon() -> cr.ConvexBody:
    """An irregular heptagon with rounded corners: polygon + small disk."""
    P = cr.Polytope(
        np.array(
            [
                [-1.0, -0.3],
                [-0.5, -0.8],
                [0.6, -0.72],
                [1.02, -0.05],
                [0.7, 0.62],
                [-0.05, 0.86],
                [-0.78, 0.42],
            ]
        )
    )
    return P + cr.Ellipsoid(0.1 * np.eye(2), np.zeros(2))


def rounded_polytope() -> cr.ConvexBody:
    """A random 3D polytope with rounded edges: polytope + small ball."""
    return cr.random_polytope(14, rng=5) + cr.Ellipsoid(0.12 * np.eye(3), np.zeros(3))
