import numpy as np
import pytest

from src.bodies import Ellipsoid, Polytope
from src.directions import circle_directions, evenly_spread, sphere_directions
from src.metrics import hausdorff_distance, l2_distance
from src.reconstruct import reconstruct
from src.simulate import measure, random_polygon, random_polytope

# interior-point solvers return the optimum to about 1e-5 here (bodies of unit size)
TOL = 1e-4


def test_square_recovered_exactly_from_its_facet_normals(square: Polytope):
    # 8 directions include the square's 4 facet normals, so the halfspaces cut out the square itself
    U = circle_directions(8)
    assert hausdorff_distance(reconstruct(U, square.support(U)).polytope, square) < TOL


def test_cube_recovered_exactly_in_3d(cube: Polytope):
    U = np.vstack([np.eye(3), -np.eye(3), sphere_directions(30)])
    rec = reconstruct(U, cube.support(U))
    assert hausdorff_distance(rec.polytope, cube) < TOL
    assert rec.polytope.volume == pytest.approx(8, rel=TOL)


@pytest.mark.parametrize('dim', [2, 3])
def test_polytope_support_equals_fitted_values(dim: int):
    U = evenly_spread(80, dim)
    K = random_polygon(6, rng=1) if dim == 2 else random_polytope(15, rng=1)
    rec = reconstruct(U, measure(K, U, sigma=0.1, rng=2))
    np.testing.assert_allclose(rec.polytope.support(U), rec.support_values, atol=TOL)


def test_lsq_fit_no_farther_from_data_than_truth():
    # the true support values are consistent too, so the least-squares fit is at least as close to y
    K = random_polytope(12, rng=4)
    U = sphere_directions(100)
    y = measure(K, U, sigma=0.1, rng=5)
    assert np.sum(reconstruct(U, y).residuals ** 2) <= np.sum((y - K.support(U)) ** 2) + 1e-8


def test_error_decreases_with_more_measurements():
    K = Ellipsoid(np.diag([1.0, 0.6]), np.zeros(2))
    errors = []
    for k in [20, 200, 2000]:
        U = circle_directions(k)
        errors.append(l2_distance(K, reconstruct(U, measure(K, U, sigma=0.1, rng=k)).polytope))
    assert errors[0] > errors[1] > errors[2]


def test_lad_less_affected_by_outliers_than_lsq():
    K = random_polygon(6, rng=7)
    U = circle_directions(200)
    y = measure(K, U, sigma=0.01, rng=8)
    y[::10] += 1.5  # every tenth measurement pushed far outward
    lsq_error = hausdorff_distance(K, reconstruct(U, y, method='lsq').polytope)
    lad_error = hausdorff_distance(K, reconstruct(U, y, method='lad').polytope)
    assert lad_error < 0.5 * lsq_error


U10 = circle_directions(10)


@pytest.mark.parametrize(
    ('directions', 'measurements', 'kwargs', 'match'),
    [
        pytest.param(np.ones((10, 4)), np.ones(10), {}, r'shape \(k, 2\) or \(k, 3\)', id='4d'),
        pytest.param(U10, np.ones(9), {}, 'expected 10 measurements', id='count'),
        pytest.param(U10[:2], np.ones(2), {}, 'at least 3 directions', id='too-few'),
        pytest.param(2 * U10, np.ones(10), {}, 'unit vectors', id='not-unit'),
        pytest.param(U10, np.ones(10), {'method': 'l2'}, "method must be 'lsq' or 'lad'", id='method'),
        pytest.param(U10, np.ones(10), {'formulation': 'dense'}, 'formulation must be', id='formulation'),
    ],
)
def test_bad_input_raises(directions: np.ndarray, measurements: np.ndarray, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        reconstruct(directions, measurements, **kwargs)
