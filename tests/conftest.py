import numpy as np
import pytest

from src import Polytope


@pytest.fixture
def square() -> Polytope:
    """The square [-1, 1]^2."""
    return Polytope(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float))


@pytest.fixture
def cube() -> Polytope:
    """The cube [-1, 1]^3."""
    return Polytope(np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T.astype(float))
