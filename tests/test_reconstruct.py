import numpy as np
import pytest

from src.bodies import Ellipsoid, Polytope
from src.directions import circle_directions, evenly_spread, sphere_directions
from src.metrics import hausdorff_distance, l2_distance
from src.reconstruct import reconstruct
from src.simulate import noisy_support, random_polygon, random_polytope
from src.solvers import fit_local_2d, fit_pairwise

# interior-point solvers return the optimum to about 1e-5 here (bodies of unit size)
TOL = 1e-4


def test_square_recovered_exactly_from_its_facet_normals(square: Polytope):
    # 8 directions include the square's 4 facet normals, so the halfspaces cut out the square itself
    directions = circle_directions(8)
    assert hausdorff_distance(reconstruct(directions, square.support(directions)), square) < TOL


def test_cube_recovered_exactly_in_3d(cube: Polytope):
    directions = np.vstack([np.eye(3), -np.eye(3), sphere_directions(30)])
    polytope = reconstruct(directions, cube.support(directions))
    assert hausdorff_distance(polytope, cube) < TOL
    assert polytope.volume == pytest.approx(8, rel=TOL)


@pytest.mark.parametrize(('dim', 'fit'), [(2, fit_local_2d), (3, fit_pairwise)])
def test_polytope_support_equals_fitted_values(dim: int, fit):
    # the returned polytope must carry the fitted values as its support function, so callers can read them from it
    directions = evenly_spread(80, dim)
    body = random_polygon(6, rng=1) if dim == 2 else random_polytope(15, rng=1)
    y = noisy_support(body, directions, sigma=0.1, rng=2)
    np.testing.assert_allclose(reconstruct(directions, y).support(directions), fit(directions, y), atol=TOL)


def test_lsq_fit_no_farther_from_data_than_truth():
    # the true support values are consistent too, so the least-squares fit is at least as close to y
    body = random_polytope(12, rng=4)
    directions = sphere_directions(100)
    y = noisy_support(body, directions, sigma=0.1, rng=5)
    assert (
        np.sum((y - reconstruct(directions, y).support(directions)) ** 2)
        <= np.sum((y - body.support(directions)) ** 2) + 1e-8
    )


def test_error_decreases_with_more_measurements():
    body = Ellipsoid(np.diag([1.0, 0.6]), np.zeros(2))
    errors = []
    for k in [20, 200, 2000]:
        directions = circle_directions(k)
        errors.append(l2_distance(body, reconstruct(directions, noisy_support(body, directions, sigma=0.1, rng=k))))
    assert errors[0] > errors[1] > errors[2]


def test_lad_less_affected_by_outliers_than_lsq():
    body = random_polygon(6, rng=7)
    directions = circle_directions(200)
    y = noisy_support(body, directions, sigma=0.01, rng=8)
    y[::10] += 1.5  # every tenth measurement pushed far outward
    lsq_error = hausdorff_distance(body, reconstruct(directions, y, method='lsq'))
    lad_error = hausdorff_distance(body, reconstruct(directions, y, method='lad'))
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
        # all ten directions have z >= 0, so nothing bounds the estimate from below
        pytest.param(np.abs(sphere_directions(10)), np.ones(10), {}, 'closed hemisphere', id='hemisphere'),
    ],
)
def test_bad_input_raises(directions: np.ndarray, measurements: np.ndarray, kwargs: dict, match: str):
    with pytest.raises(ValueError, match=match):
        reconstruct(directions, measurements, **kwargs)
