import numpy as np
import pytest

import src as cr


@pytest.fixture
def square():
    return cr.Polytope(np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float))


@pytest.fixture
def cube():
    return cr.Polytope(np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).reshape(3, -1).T * 1.0)
