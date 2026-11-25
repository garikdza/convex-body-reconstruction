import numpy as np
import pytest

import src as cr


def test_polytope_support(square):
    U = np.array([[1, 0], [0, 1], [np.sqrt(0.5), np.sqrt(0.5)]])
    np.testing.assert_allclose(square.support(U), [1, 1, np.sqrt(2)])


def test_polytope_drops_interior_points():
    P = cr.Polytope(np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1]], float))
    assert len(P.vertices) == 4


@pytest.mark.parametrize('dim', [2, 3])
def test_support_points_attain_support(dim):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim)) + 2 * np.eye(dim)
    P = cr.random_polygon(6, rng) if dim == 2 else cr.random_polytope(10, rng)
    for K in [cr.Ellipsoid(A, rng.standard_normal(dim)), P, P + cr.Ellipsoid(A, np.zeros(dim))]:
        U = cr.evenly_spread(50, dim)
        np.testing.assert_allclose(np.einsum('ij,ij->i', K.support_points(U), U), K.support(U))


def test_bodies_share_the_interface(square):
    for K in [square, cr.Ellipsoid(np.eye(2), np.zeros(2)), square + square]:
        assert isinstance(K, cr.ConvexBody)
        assert K.dim == 2


def test_minkowski_sum_of_square_and_disk(square):
    K = square + cr.Ellipsoid(0.5 * np.eye(2), np.zeros(2))
    # area of a square of side 2 rounded by radius 0.5: 4 + 4 * 2 * 0.5 + pi * 0.25
    assert cr.to_polytope(K, 4000).volume == pytest.approx(8 + np.pi / 4, rel=1e-4)


def test_from_halfspaces_roundtrip(square):
    P = cr.Polytope.from_halfspaces(cr.circle_directions(4), np.ones(4))
    assert cr.hausdorff_distance(P, square) < 1e-9


def test_from_halfspaces_unbounded():
    U = np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]])
    with pytest.raises(ValueError, match='unbounded'):
        cr.Polytope.from_halfspaces(U, np.ones(3))


def test_sphere_directions_are_unit_and_balanced():
    U = cr.sphere_directions(500)
    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1)
    assert np.linalg.norm(U.mean(axis=0)) < 1e-2


def test_measure_noise_level(square):
    U = cr.circle_directions(20_000)
    noise = cr.measure(square, U, sigma=0.1, rng=0) - square.support(U)
    assert np.std(noise) == pytest.approx(0.1, rel=0.03)
