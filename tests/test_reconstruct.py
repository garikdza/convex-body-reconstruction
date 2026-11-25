import numpy as np
import pytest

import src as cr

# Interior-point solvers return the optimum to ~1e-5 here (bodies have unit size), and
# noise-free data are the hardest case because every constraint is active.
TOL = 1e-4


@pytest.mark.parametrize('method', ['lsq', 'lad'])
@pytest.mark.parametrize('formulation', ['local', 'pairwise'])
def test_exact_measurements_are_reproduced_2d(method, formulation):
    """Noise-free data are consistent, so they must be returned unchanged."""
    K = cr.random_polygon(7, rng=3)
    U = cr.circle_directions(40)
    rec = cr.reconstruct(U, K.support(U), method=method, formulation=formulation)
    np.testing.assert_allclose(rec.support_values, K.support(U), atol=TOL)


def test_square_recovered_exactly_from_its_facet_normals(square):
    U = cr.circle_directions(8)  # contains the 4 facet normals of the square
    rec = cr.reconstruct(U, square.support(U))
    assert cr.hausdorff_distance(rec.polytope, square) < TOL


def test_cube_recovered_exactly_in_3d(cube):
    U = np.vstack([np.eye(3), -np.eye(3), cr.sphere_directions(30)])
    rec = cr.reconstruct(U, cube.support(U))
    assert cr.hausdorff_distance(rec.polytope, cube) < TOL
    assert rec.polytope.volume == pytest.approx(8, rel=TOL)


@pytest.mark.parametrize('method', ['lsq', 'lad'])
def test_local_and_pairwise_formulations_agree(method):
    """The O(k) Rademacher-type constraints in 2D define the same feasible set."""
    rng = np.random.default_rng(0)
    U = cr.random_directions(60, 2, rng)
    y = cr.measure(cr.random_polygon(6, rng), U, sigma=0.2, rng=rng)
    a = cr.reconstruct(U, y, method=method, formulation='local').support_values
    b = cr.reconstruct(U, y, method=method, formulation='pairwise').support_values
    if method == 'lsq':  # strictly convex objective: unique optimum
        np.testing.assert_allclose(a, b, atol=TOL)
    else:  # the LP optimum need not be unique; compare objective values
        assert np.abs(y - a).sum() == pytest.approx(np.abs(y - b).sum(), abs=TOL)


@pytest.mark.parametrize('dim', [2, 3])
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
    lsq = cr.reconstruct(U, y, method='lsq')
    lad = cr.reconstruct(U, y, method='lad')
    assert cr.hausdorff_distance(K, lad.polytope) < 0.5 * cr.hausdorff_distance(K, lsq.polytope)


def test_input_validation():
    U = cr.circle_directions(10)
    with pytest.raises(ValueError, match='measurements'):
        cr.reconstruct(U, np.ones(9))
    with pytest.raises(ValueError, match='unit'):
        cr.reconstruct(2 * U, np.ones(10))
    with pytest.raises(ValueError, match='half-plane'):
        cr.reconstruct(cr.circle_directions(20)[:8], np.ones(8))
    with pytest.raises(ValueError, match='method'):
        cr.reconstruct(U, np.ones(10), method='foo')
