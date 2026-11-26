import numpy as np
import pytest

from src.bodies import Polytope
from src.directions import circle_directions
from src.simulate import noisy_support, random_polygon, random_polytope


def test_noisy_support_noise_standard_deviation(square: Polytope):
    U = circle_directions(20_000)
    noise = noisy_support(square, U, sigma=0.1, rng=0) - square.support(U)
    # the sample standard deviation of n normal draws has standard error sigma / sqrt(2n) = 0.0005
    assert np.std(noise) == pytest.approx(0.1, abs=0.002)


def test_random_polygon_vertices_in_annulus():
    radii = np.linalg.norm(random_polygon(30, rng=0).vertices, axis=1)
    assert radii.min() >= 0.7
    assert radii.max() <= 1.0


def test_random_polytope_vertices_in_scaled_shell():
    # dividing by the semi-axes undoes the scaling, leaving radii in [0.7, 1]
    radii = np.linalg.norm(random_polytope(50, rng=0).vertices / [1.0, 0.8, 0.6], axis=1)
    assert radii.min() >= 0.7
    assert radii.max() <= 1.0
