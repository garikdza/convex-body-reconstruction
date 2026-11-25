import numpy as np
import pytest

from src.bodies import Ellipsoid, Polytope
from src.metrics import hausdorff_distance, l2_distance


def test_distances_between_nested_squares(square: Polytope):
    # the support functions differ by h(u) = |cos t| + |sin t|: largest, sqrt(2), at the diagonals, and its mean
    # square over the circle is 1 + mean |sin 2t| = 1 + 2 / pi
    big = Polytope(2 * square.vertices)
    assert hausdorff_distance(square, big) == pytest.approx(np.sqrt(2), rel=1e-6)
    assert l2_distance(square, big) == pytest.approx(np.sqrt(1 + 2 / np.pi), rel=1e-4)


def test_distances_between_concentric_balls():
    # the support functions are the constants 1 and 1.5
    small, big = Ellipsoid(np.eye(3), np.zeros(3)), Ellipsoid(1.5 * np.eye(3), np.zeros(3))
    assert hausdorff_distance(small, big) == pytest.approx(0.5)
    assert l2_distance(small, big) == pytest.approx(0.5)


def test_different_dimensions_raise(square: Polytope):
    with pytest.raises(ValueError, match='same dimension, got 2 and 3'):
        hausdorff_distance(square, Ellipsoid(np.eye(3), np.zeros(3)))
