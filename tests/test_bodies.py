import numpy as np
import pytest

from src.bodies import Ellipsoid, Polytope
from src.directions import evenly_spread
from src.simulate import random_polygon, random_polytope


def test_polytope_support_of_square(square: Polytope):
    # h(u) = |u_1| + |u_2| for the square [-1, 1]^2
    directions = np.array([[1, 0], [0, 1], [np.sqrt(0.5), np.sqrt(0.5)]])
    np.testing.assert_allclose(square.support(directions), [1, 1, np.sqrt(2)])


def test_polytope_drops_interior_points():
    polytope = Polytope(np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1]], dtype=float))
    assert len(polytope.vertices) == 4


def test_ellipsoid_support_hand_values():
    # h(u) = <c, u> + |A^T u| with A = diag(2, 1), c = (1, 0): 1 + 2 along +x, -1 + 2 along -x, 0 + 1 along +y
    ellipsoid = Ellipsoid(np.diag([2.0, 1.0]), np.array([1.0, 0.0]))
    np.testing.assert_allclose(ellipsoid.support(np.array([[1, 0], [-1, 0], [0, 1]])), [3, 1, 1])


@pytest.mark.parametrize('dim', [2, 3])
def test_support_points_attain_support(dim: int):
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((dim, dim)) + 2 * np.eye(dim)
    polytope = random_polygon(6, rng) if dim == 2 else random_polytope(10, rng)
    directions = evenly_spread(50, dim)
    for body in [Ellipsoid(matrix, rng.standard_normal(dim)), polytope, polytope + Ellipsoid(matrix, np.zeros(dim))]:
        np.testing.assert_allclose(
            np.einsum('ij,ij->i', body.support_points(directions), directions), body.support(directions)
        )


def test_minkowski_sum_rounded_square_area(square: Polytope):
    # the square of side 2 rounded by radius 0.5: 4 for the square, 4 * 2 * 0.5 for the side strips and pi / 4 for
    # the corners
    body = square + Ellipsoid(0.5 * np.eye(2), np.zeros(2))
    assert body.to_polytope(4000).volume == pytest.approx(8 + np.pi / 4, rel=1e-4)


@pytest.mark.parametrize(
    ('build', 'match'),
    [
        pytest.param(lambda: Polytope(np.zeros((5, 4))), r'shape \(m, 2\) or \(m, 3\), got \(5, 4\)', id='vertices'),
        pytest.param(lambda: Ellipsoid(np.eye(3), np.zeros(2)), r'got \(3, 3\) and \(2,\)', id='ellipsoid'),
        pytest.param(lambda: Ellipsoid(np.diag([1.0, 0.0]), np.zeros(2)), 'matrix must be invertible', id='singular'),
        pytest.param(lambda: random_polygon(5) + Ellipsoid(np.eye(3), np.zeros(3)), r'got \[2, 3\]', id='sum'),
    ],
)
def test_bad_bodies_raise(build, match: str):
    with pytest.raises(ValueError, match=match):
        build()
