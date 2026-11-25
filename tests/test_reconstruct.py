import numpy as np
import pytest

import convex_recon as cr


def square():
    return cr.Polytope(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float))


def cube():
    return cr.Polytope(np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * 1.0)


# --- support functions and metrics -------------------------------------------------------


def test_polytope_support():
    U = np.array([[1, 0], [0, 1], [np.sqrt(0.5), np.sqrt(0.5)]])
    np.testing.assert_allclose(square().support(U), [1, 1, np.sqrt(2)])


def test_polytope_drops_interior_points():
    P = cr.Polytope(np.array([[0, 0], [2, 0], [0, 2], [2, 2], [1, 1]], float))
    assert len(P.vertices) == 4


@pytest.mark.parametrize("dim", [2, 3])
def test_support_points_attain_support(dim):
    rng = np.random.default_rng(0)
    A = rng.standard_normal((dim, dim)) + 2 * np.eye(dim)
    P = cr.random_polygon(6, rng) if dim == 2 else cr.random_polytope(10, rng)
    for K in [cr.Ellipsoid(A, rng.standard_normal(dim)), P, P + cr.Ellipsoid(A, np.zeros(dim))]:
        U = cr.evenly_spread(50, dim)
        np.testing.assert_allclose(np.einsum("ij,ij->i", K.support_points(U), U), K.support(U))


def test_minkowski_sum_of_square_and_disk():
    K = square() + cr.Ellipsoid(0.5 * np.eye(2), np.zeros(2))
    # area of a square of side 2 rounded by radius 0.5: 4 + 4 * 2 * 0.5 + pi * 0.25
    assert cr.to_polytope(K, 4000).volume == pytest.approx(8 + np.pi / 4, rel=1e-4)


def test_from_halfspaces_roundtrip():
    U = cr.circle_directions(4)
    P = cr.Polytope.from_halfspaces(U, np.ones(4))
    assert cr.hausdorff_distance(P, square()) < 1e-9


def test_from_halfspaces_unbounded():
    U = np.array([[1.0, 0.0], [0.0, 1.0], [np.sqrt(0.5), np.sqrt(0.5)]])
    with pytest.raises(ValueError, match="unbounded"):
        cr.Polytope.from_halfspaces(U, np.ones(3))


def test_metrics():
    big = cr.Polytope(2 * square().vertices)
    assert cr.hausdorff_distance(square(), big) == pytest.approx(np.sqrt(2), rel=1e-6)
    assert cr.hausdorff_distance(square(), square()) == 0
    assert 0 < cr.l2_distance(square(), big) < cr.hausdorff_distance(square(), big)


def test_sphere_directions_are_unit_and_balanced():
    U = cr.sphere_directions(500)
    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1)
    assert np.linalg.norm(U.mean(axis=0)) < 1e-2


# --- reconstruction ----------------------------------------------------------------------
# Interior-point solvers return the optimum to ~1e-5 here (bodies have unit size), and
# noise-free data are the hardest case because every constraint is active.
TOL = 1e-4


@pytest.mark.parametrize("method", ["lsq", "lad"])
@pytest.mark.parametrize("formulation", ["local", "pairwise"])
def test_exact_measurements_are_reproduced_2d(method, formulation):
    """Noise-free data are consistent, so they must be returned unchanged."""
    K = cr.random_polygon(7, rng=3)
    U = cr.circle_directions(40)
    rec = cr.reconstruct(U, K.support(U), method=method, formulation=formulation)
    np.testing.assert_allclose(rec.support_values, K.support(U), atol=TOL)


def test_square_recovered_exactly_from_its_facet_normals():
    U = cr.circle_directions(8)  # contains the 4 facet normals of the square
    rec = cr.reconstruct(U, square().support(U))
    assert cr.hausdorff_distance(rec.polytope, square()) < TOL


def test_cube_recovered_exactly_in_3d():
    U = np.vstack([np.eye(3), -np.eye(3), cr.sphere_directions(30)])
    rec = cr.reconstruct(U, cube().support(U))
    assert cr.hausdorff_distance(rec.polytope, cube()) < TOL
    assert rec.polytope.volume == pytest.approx(8, rel=TOL)


@pytest.mark.parametrize("method", ["lsq", "lad"])
def test_local_and_pairwise_formulations_agree(method):
    """The O(k) Rademacher-type constraints in 2D define the same feasible set."""
    rng = np.random.default_rng(0)
    U = cr.random_directions(60, 2, rng)
    y = cr.measure(cr.random_polygon(6, rng), U, sigma=0.2, rng=rng)
    a = cr.reconstruct(U, y, method=method, formulation="local").support_values
    b = cr.reconstruct(U, y, method=method, formulation="pairwise").support_values
    if method == "lsq":  # strictly convex objective: unique optimum
        np.testing.assert_allclose(a, b, atol=TOL)
    else:  # the LP optimum need not be unique; compare objective values
        assert np.abs(y - a).sum() == pytest.approx(np.abs(y - b).sum(), abs=TOL)


@pytest.mark.parametrize("dim", [2, 3])
def test_output_is_consistent(dim):
    """The fitted values are exactly the support function of the output polytope."""
    U = cr.evenly_spread(80, dim)
    K = cr.random_polygon(6, rng=1) if dim == 2 else cr.random_polytope(15, rng=1)
    rec = cr.reconstruct(U, cr.measure(K, U, sigma=0.1, rng=2))
    np.testing.assert_allclose(rec.polytope.support(U), rec.support_values, atol=TOL)


def test_lsq_is_a_projection():
    """The LSQ fit is at least as close to y as the true support values (also feasible)."""
    K = cr.random_polytope(12, rng=4)
    U = cr.sphere_directions(100)
    y = cr.measure(K, U, sigma=0.1, rng=5)
    rec = cr.reconstruct(U, y)
    assert np.sum(rec.residuals**2) <= np.sum((y - K.support(U)) ** 2) + 1e-8


def test_error_decreases_with_more_measurements():
    K = cr.Ellipsoid(np.diag([1.0, 0.6]), np.zeros(2))
    errs = []
    for k in [20, 200, 2000]:
        U = cr.circle_directions(k)
        rec = cr.reconstruct(U, cr.measure(K, U, sigma=0.1, rng=k))
        errs.append(cr.l2_distance(K, rec.polytope))
    assert errs[0] > errs[1] > errs[2]


def test_lad_is_robust_to_outliers():
    K = cr.random_polygon(6, rng=7)
    U = cr.circle_directions(200)
    y = cr.measure(K, U, sigma=0.01, rng=8)
    y[::10] += 1.5  # 10% gross outliers, all outward
    lsq = cr.reconstruct(U, y, method="lsq")
    lad = cr.reconstruct(U, y, method="lad")
    assert cr.hausdorff_distance(K, lad.polytope) < 0.5 * cr.hausdorff_distance(K, lsq.polytope)


def test_input_validation():
    U = cr.circle_directions(10)
    with pytest.raises(ValueError, match="measurements"):
        cr.reconstruct(U, np.ones(9))
    with pytest.raises(ValueError, match="unit"):
        cr.reconstruct(2 * U, np.ones(10))
    with pytest.raises(ValueError, match="half-plane"):
        cr.reconstruct(cr.circle_directions(20)[:8], np.ones(8))
    with pytest.raises(ValueError, match="method"):
        cr.reconstruct(U, np.ones(10), method="foo")
