import numpy as np
import pytest

from src.directions import circle_directions, random_directions, sphere_directions
from src.simulate import noisy_support, random_polygon, random_polytope
from src.solvers import fit_local_2d, fit_pairwise

# interior-point solvers return the optimum to about 1e-5 here (bodies of unit size)
TOL = 1e-4
# three directions 120 degrees apart sum to zero, so <x, u_1 + u_2 + u_3> = 0 <= h_1 + h_2 + h_3 for any x: the values
# are consistent iff their sum is >= 0
TRIANGLE = circle_directions(3)
FITS = [pytest.param(fit_local_2d, id='local'), pytest.param(fit_pairwise, id='pairwise')]


@pytest.mark.parametrize('fit', FITS)
def test_lsq_projects_inconsistent_values_onto_sum_constraint(fit):
    # y = (-1, -1, -1) has sum -3 < 0; its projection onto {h : sum h >= 0} adds 1 to each value
    np.testing.assert_allclose(fit(TRIANGLE, -np.ones(3), 'lsq'), np.zeros(3), atol=TOL)


@pytest.mark.parametrize('fit', FITS)
def test_lad_inconsistent_values_cost_their_deficit(fit):
    # raising the sum from -3 to 0 costs at least 3 in absolute changes, and any h >= -1 with sum 0 achieves it
    h = fit(TRIANGLE, -np.ones(3), 'lad')
    assert np.abs(h + 1).sum() == pytest.approx(3, abs=TOL)
    assert h.sum() >= -TOL


@pytest.mark.parametrize('loss', ['lsq', 'lad'])
@pytest.mark.parametrize('fit', FITS)
def test_consistent_values_returned_unchanged(fit, loss: str):
    body = random_polygon(7, rng=3)
    directions = circle_directions(40)
    np.testing.assert_allclose(fit(directions, body.support(directions), loss), body.support(directions), atol=TOL)


@pytest.mark.parametrize('loss', ['lsq', 'lad'])
def test_local_and_pairwise_same_optimum(loss: str):
    rng = np.random.default_rng(0)
    directions = random_directions(60, 2, rng)
    y = noisy_support(random_polygon(6, rng), directions, sigma=0.2, rng=rng)
    local, pairwise = fit_local_2d(directions, y, loss), fit_pairwise(directions, y, loss)
    if loss == 'lsq':  # strictly convex objective, so the optimum is unique
        np.testing.assert_allclose(local, pairwise, atol=TOL)
    else:  # the LP optimum need not be unique; its value is
        assert np.abs(y - local).sum() == pytest.approx(np.abs(y - pairwise).sum(), abs=TOL)


def test_pairwise_adds_constraints_beyond_neighbours_3d():
    # with one neighbour per direction the first relaxation is too loose; the generated constraints must bring the
    # result back to the noise-free support values
    body = random_polytope(12, rng=1)
    directions = sphere_directions(80)
    np.testing.assert_allclose(
        fit_pairwise(directions, body.support(directions), n_neighbors=1), body.support(directions), atol=TOL
    )


def test_pairwise_gives_up_after_max_rounds():
    directions = sphere_directions(80)
    y = noisy_support(random_polytope(12, rng=1), directions, sigma=0.1, rng=2)
    with pytest.raises(RuntimeError, match='max_rounds=1'):
        fit_pairwise(directions, y, n_neighbors=1, max_rounds=1)


@pytest.mark.parametrize(
    ('directions', 'match'),
    [
        pytest.param(sphere_directions(10), 'only available in 2D', id='3d'),
        pytest.param(np.vstack([circle_directions(4), [[1.0, 0.0]]]), 'distinct', id='repeated'),
        pytest.param(circle_directions(20)[:8], 'closed half-plane', id='half-plane'),
    ],
)
def test_local_bad_directions_raise(directions: np.ndarray, match: str):
    with pytest.raises(ValueError, match=match):
        fit_local_2d(directions, np.ones(len(directions)))


def test_unknown_loss_raises():
    # both fits build their objective with the same helper, so one of them covers it
    with pytest.raises(ValueError, match="loss must be 'lsq' or 'lad', got 'l2'"):
        fit_local_2d(TRIANGLE, np.ones(3), 'l2')
