import numpy as np

from src.bodies import Polytope
from src.geometry import chebyshev_center
from src.solvers import Loss, fit_local_2d, fit_pairwise

# how far the norm of a direction may be from 1
UNIT_TOL = 1e-8


def reconstruct(directions: np.ndarray, measurements: np.ndarray, method: Loss = 'lsq') -> Polytope:
    """
    Estimate a convex body from noisy measurements ``y_i = h_K(u_i) + e_i`` of its support function.

    Fits the consistent support values ``h`` closest to the measurements and returns the polytope they define,
    Algorithm NoisySupportLSQ of Gardner, Kiderlen & Milanfar (2006), due to Prince & Willsky (1990). Consistency is
    imposed with ``fit_local_2d`` in 2D and ``fit_pairwise`` in 3D.

    Parameters
    ----------
    directions : np.ndarray
        unit directions ``u_i``, not all in a closed hemisphere, shape ``(k, 2)`` or ``(k, 3)``
    measurements : np.ndarray
        measurements ``y_i``, shape ``(k,)``
    method : {'lsq', 'lad'}
        least squares (a quadratic program) or least absolute deviations (a linear program, robust to outliers)

    Returns
    -------
    P : Polytope
        the estimate ``{x : <x, u_i> <= h_i for all i}``; its support function at ``u_i`` is the fitted ``h_i``, so
        ``P.support(directions)`` gives the fitted values and ``measurements - P.support(directions)`` the residuals
    """
    U, y = _validate(directions, measurements, method)
    fit = fit_local_2d if U.shape[1] == 2 else fit_pairwise
    return Polytope.from_halfspaces(U, fit(U, y, method))


def _validate(directions: np.ndarray, measurements: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(directions, dtype=float)
    y = np.asarray(measurements, dtype=float).ravel()
    if U.ndim != 2 or U.shape[1] not in (2, 3):
        raise ValueError(f'directions must have shape (k, 2) or (k, 3), got {U.shape}')
    k, d = U.shape
    if y.shape != (k,):
        raise ValueError(f'expected {k} measurements, one per direction, got {y.size}')
    if k < d + 1:
        raise ValueError(f'need at least {d + 1} directions in {d}D, got {k}')
    if not np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=UNIT_TOL):
        raise ValueError('directions must be unit vectors')
    if method not in ('lsq', 'lad'):
        raise ValueError(f"method must be 'lsq' or 'lad', got {method!r}")
    # the directions lie in a closed hemisphere iff {x : <x, u_i> <= 1} is unbounded, which chebyshev_center reports;
    # checking before the fit spares a solve whose estimate would be unbounded
    chebyshev_center(U, np.ones(k))
    return U, y
