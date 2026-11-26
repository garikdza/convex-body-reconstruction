from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.bodies import Polytope
from src.geometry import chebyshev_center
from src.solvers import Loss, fit_local_2d, fit_pairwise

Formulation = Literal['auto', 'pairwise', 'local']


@dataclass
class Reconstruction:
    """
    Result of ``reconstruct``.

    Attributes
    ----------
    directions : np.ndarray
        measurement directions, shape ``(k, d)``
    measurements : np.ndarray
        noisy measurements ``y``, shape ``(k,)``
    support_values : np.ndarray
        fitted consistent support values ``h``, shape ``(k,)``
    polytope : Polytope
        the estimate ``{x : <x, u_i> <= h_i for all i}``, whose support function at ``u_i`` is ``h_i``
    """

    directions: np.ndarray
    measurements: np.ndarray
    support_values: np.ndarray
    polytope: Polytope

    @property
    def residuals(self) -> np.ndarray:
        """Measurements minus fitted support values, shape ``(k,)``."""
        return self.measurements - self.support_values


def reconstruct(
    directions: np.ndarray,
    measurements: np.ndarray,
    method: Loss = 'lsq',
    formulation: Formulation = 'auto',
    solver: str | None = None,
) -> Reconstruction:
    """
    Estimate a convex body from noisy measurements ``y_i = h_K(u_i) + e_i`` of its support function.

    Fits the consistent support values closest to the measurements and returns the polytope they define, Algorithm
    NoisySupportLSQ of Gardner, Kiderlen & Milanfar (2006), due to Prince & Willsky (1990).

    Parameters
    ----------
    directions : np.ndarray
        unit directions ``u_i``, not all in a closed hemisphere, shape ``(k, 2)`` or ``(k, 3)``
    measurements : np.ndarray
        measurements ``y_i``, shape ``(k,)``
    method : {'lsq', 'lad'}
        least squares (a quadratic program) or least absolute deviations (a linear program, robust to outliers)
    formulation : {'auto', 'local', 'pairwise'}
        consistency constraints: ``'local'`` (2D only, ``fit_local_2d``) or ``'pairwise'`` (``fit_pairwise``);
        ``'auto'`` uses ``'local'`` in 2D and ``'pairwise'`` in 3D
    solver : str | None
        cvxpy solver name; ``None`` uses Clarabel

    Returns
    -------
    reconstruction : Reconstruction
        the estimate and the fitted support values
    """
    U, y = _validate(directions, measurements, method)
    if formulation == 'auto':
        formulation = 'local' if U.shape[1] == 2 else 'pairwise'
    if formulation == 'local':
        h = fit_local_2d(U, y, method, solver)
    elif formulation == 'pairwise':
        h = fit_pairwise(U, y, method, solver)
    else:
        raise ValueError(f"formulation must be 'auto', 'local' or 'pairwise', got {formulation!r}")
    return Reconstruction(U, y, h, Polytope.from_halfspaces(U, h))


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
    if not np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-8):
        raise ValueError('directions must be unit vectors')
    if method not in ('lsq', 'lad'):
        raise ValueError(f"method must be 'lsq' or 'lad', got {method!r}")
    # the directions lie in a closed hemisphere iff {x : <x, u_i> <= 1} is unbounded, which chebyshev_center reports;
    # checking before the fit spares a solve whose estimate would be unbounded
    chebyshev_center(U, np.ones(k))
    return U, y
