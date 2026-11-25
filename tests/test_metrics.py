import numpy as np
import pytest

import src as cr


def test_distances_between_nested_squares(square):
    big = cr.Polytope(2 * square.vertices)
    assert cr.hausdorff_distance(square, big) == pytest.approx(np.sqrt(2), rel=1e-6)
    assert cr.hausdorff_distance(square, square) == 0
    assert 0 < cr.l2_distance(square, big) < cr.hausdorff_distance(square, big)


def test_distance_between_concentric_balls_3d():
    small = cr.Ellipsoid(np.eye(3), np.zeros(3))
    big = cr.Ellipsoid(1.5 * np.eye(3), np.zeros(3))
    assert cr.hausdorff_distance(small, big) == pytest.approx(0.5)
    assert cr.l2_distance(small, big) == pytest.approx(0.5)
