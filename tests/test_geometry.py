import numpy as np
import pytest

from src.geometry import chebyshev_center, halfspace_vertices

AXES_2D = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)


def test_chebyshev_center_of_rectangle():
    # the rectangle [0, 4] x [0, 2]: the largest disk has radius 1, half the short side, centred at height 1
    center, radius = chebyshev_center(AXES_2D, np.array([4.0, 2.0, 0.0, 0.0]))
    assert radius == pytest.approx(1.0)
    assert center[1] == pytest.approx(1.0)


def test_halfspace_vertices_of_square():
    vertices = halfspace_vertices(AXES_2D, np.ones(4))
    assert sorted(map(tuple, np.round(vertices, 12))) == [(-1, -1), (-1, 1), (1, -1), (1, 1)]


@pytest.mark.parametrize(
    ('directions', 'h', 'match'),
    [
        # three directions in the upper-right quarter leave the lower-left side open
        pytest.param(np.array([[1, 0], [0, 1], [np.sqrt(0.5), np.sqrt(0.5)]]), np.ones(3), 'unbounded', id='unbounded'),
        # x <= -1 and x >= 1
        pytest.param(AXES_2D, np.array([-1.0, 1.0, -1.0, 1.0]), 'interior point', id='empty'),
        # x = 0, |y| <= 1: a segment
        pytest.param(AXES_2D, np.array([0.0, 1.0, 0.0, 1.0]), 'empty interior', id='flat'),
    ],
)
def test_bad_halfspaces_raise(directions: np.ndarray, h: np.ndarray, match: str):
    with pytest.raises(ValueError, match=match):
        halfspace_vertices(directions, h)
