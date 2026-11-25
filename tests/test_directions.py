import numpy as np
import pytest

from src.directions import circle_directions, evenly_spread, random_directions, sphere_directions


def test_circle_directions_quarter_turns():
    # four directions start at angle 0 and step by pi/2
    np.testing.assert_allclose(circle_directions(4), [[1, 0], [0, 1], [-1, 0], [0, -1]], atol=1e-15)


def test_sphere_directions_unit_and_balanced():
    U = sphere_directions(500)
    np.testing.assert_allclose(np.linalg.norm(U, axis=1), 1)
    # an evenly spread set has its centre of mass near the origin
    assert np.linalg.norm(U.mean(axis=0)) < 1e-2


def test_random_directions_unit():
    np.testing.assert_allclose(np.linalg.norm(random_directions(100, 3, rng=0), axis=1), 1)


@pytest.mark.parametrize(
    ('make', 'match'),
    [
        pytest.param(lambda: circle_directions(2), 'at least 3 directions in 2D', id='circle-k=2'),
        pytest.param(lambda: sphere_directions(3), 'at least 4 directions in 3D', id='sphere-k=3'),
        pytest.param(lambda: evenly_spread(10, 4), 'dim must be 2 or 3', id='dim=4'),
    ],
)
def test_bad_params_raise(make, match: str):
    with pytest.raises(ValueError, match=match):
        make()
