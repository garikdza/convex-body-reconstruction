"""Reconstruct a convex body from noisy support-function measurements.

Given unit directions ``u_1, ..., u_k`` and measurements ``y_i = h_K(u_i) + noise``, fit
support values ``h_i`` that are consistent (see :mod:`src.solvers`) and return the
polytope ``P = {x : <x, u_i> <= h_i for all i}``. This is Algorithm NoisySupportLSQ of
Gardner, Kiderlen & Milanfar (Ann. Statist. 2006), due to Prince & Willsky (1990).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .bodies import Polytope
from .solvers import Loss, fit_local_2d, fit_pairwise

Formulation = Literal["auto", "pairwise", "local"]


@dataclass
class Reconstruction:
    """Result of :func:`reconstruct`."""

    directions: np.ndarray
    """The measurement directions, shape ``(k, d)``."""
    measurements: np.ndarray
    """The noisy measurements ``y``, shape ``(k,)``."""
    support_values: np.ndarray
    """The fitted, consistent support values ``h``, shape ``(k,)``."""
    polytope: Polytope
    """The estimate ``{x : <x, u_i> <= h_i}``."""
    method: Loss
    formulation: Formulation

    @property
    def residuals(self) -> np.ndarray:
        return self.measurements - self.support_values


def reconstruct(
    directions: np.ndarray,
    measurements: np.ndarray,
    method: Loss = "lsq",
    formulation: Formulation = "auto",
    solver: str | None = None,
) -> Reconstruction:
    """Estimate a convex body from noisy support-function measurements.

    Parameters
    ----------
    directions
        Unit vectors ``u_i``, shape ``(k, d)`` with ``d`` 2 or 3. They must not all lie in a
        closed hemisphere, otherwise the estimate is unbounded.
    measurements
        Noisy support values ``y_i``, shape ``(k,)``.
    method
        ``"lsq"`` for least squares (QP), ``"lad"`` for least absolute deviations (LP,
        more robust to outliers).
    formulation
        How consistency is imposed; ``"auto"`` uses ``"local"`` in 2D and ``"pairwise"``
        otherwise. See :mod:`src.solvers`.
    solver
        Optional cvxpy solver name. Defaults to Clarabel (interior point), which is much
        more accurate here than cvxpy's default first-order QP solver.
    """
    U, y = _validate(directions, measurements, method)
    if formulation == "auto":
        formulation = "local" if U.shape[1] == 2 else "pairwise"
    if formulation == "local":
        h = fit_local_2d(U, y, method, solver)
    elif formulation == "pairwise":
        h = fit_pairwise(U, y, method, solver)
    else:
        raise ValueError("formulation must be 'auto', 'pairwise' or 'local'")

    return Reconstruction(
        directions=U,
        measurements=y,
        support_values=h,
        polytope=Polytope.from_halfspaces(U, h),
        method=method,
        formulation=formulation,
    )


def _validate(directions, measurements, method) -> tuple[np.ndarray, np.ndarray]:
    U = np.asarray(directions, dtype=float)
    y = np.asarray(measurements, dtype=float).ravel()
    if U.ndim != 2 or U.shape[1] not in (2, 3):
        raise ValueError("directions must have shape (k, 2) or (k, 3)")
    k, d = U.shape
    if y.shape != (k,):
        raise ValueError(f"expected {k} measurements, got {y.shape[0]}")
    if k < d + 1:
        raise ValueError(f"need at least {d + 1} directions in {d}D")
    if not np.allclose(np.linalg.norm(U, axis=1), 1.0, atol=1e-8):
        raise ValueError("directions must be unit vectors")
    if method not in ("lsq", "lad"):
        raise ValueError("method must be 'lsq' or 'lad'")
    return U, y
