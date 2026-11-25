"""Reconstruction of a convex body from noisy support-function measurements.

Given unit directions ``u_1, ..., u_k`` and measurements ``y_i = h_K(u_i) + noise``,
find support values ``h_1, ..., h_k`` that are *consistent* (they are the support
values of some convex body at the ``u_i``) and fit the data as well as possible:

    minimise   sum_i loss(y_i - h_i)    subject to   h is consistent.

The estimate is the polytope ``P = {x : <x, u_i> <= h_i for all i}``
(Algorithm NoisySupportLSQ in Gardner, Kiderlen & Milanfar, Ann. Statist. 2006,
due to Prince & Willsky 1990). ``loss`` is the squared error (``method="lsq"``, a
quadratic program) or the absolute error (``method="lad"``, a linear program).

Two equivalent ways of writing the consistency constraint are implemented:

``"pairwise"`` (any dimension)
    ``h`` is consistent iff there are points ``x_i`` with ``<x_i, u_i> = h_i`` and
    ``<x_j, u_i> <= h_i`` for all ``i, j``: the ``x_i`` are contact points of the body in
    direction ``u_i``. This needs ``d*k`` variables and ``k^2`` linear constraints.

``"local"`` (2D only)
    With the directions sorted by angle ``theta_i``, ``h`` is consistent iff for every
    ``i`` (cyclically)
    ``h_{i-1} sin(t_{i+1} - t_i) - h_i sin(t_{i+1} - t_{i-1}) + h_{i+1} sin(t_i - t_{i-1}) >= 0``,
    i.e. every halfspace touches the polygon cut out by its two neighbours
    (Lele, Kulkarni & Willsky 1992). Only ``k`` constraints, so it scales to large ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from .bodies import Polytope

Method = Literal["lsq", "lad"]
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
    method: str
    formulation: str

    def support(self, U: np.ndarray) -> np.ndarray:
        return self.polytope.support(U)

    @property
    def residuals(self) -> np.ndarray:
        return self.measurements - self.support_values


def reconstruct(
    directions: np.ndarray,
    measurements: np.ndarray,
    method: Method = "lsq",
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
        otherwise. See the module docstring.
    solver
        Optional cvxpy solver name. Defaults to Clarabel (interior point), which is much
        more accurate here than cvxpy's default first-order QP solver.
    """
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

    if formulation == "auto":
        formulation = "local" if d == 2 else "pairwise"
    if formulation == "local":
        h = _solve_local_2d(U, y, method, solver)
    elif formulation == "pairwise":
        h = _solve_pairwise(U, y, method, solver)
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


def _objective(y: np.ndarray, h: cp.Expression, method: str) -> cp.Minimize:
    if method == "lsq":
        return cp.Minimize(cp.sum_squares(y - h))
    return cp.Minimize(cp.norm1(y - h))


def _solve(problem: cp.Problem, solver: str | None) -> None:
    if solver is None:
        # cvxpy's default QP solver (OSQP) is first-order and stalls on these problems;
        # the interior-point solver Clarabel is fast and accurate.
        solver = cp.CLARABEL if cp.CLARABEL in cp.installed_solvers() else None
    problem.solve(solver=solver)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"solver failed with status {problem.status!r}")


def _pair_matrix(U: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    """Sparse matrix whose row for pair ``(i, j)`` maps ``vec(X)`` to ``<u_i, x_j - x_i>``."""
    k, d = U.shape
    m = rows.size
    r = np.repeat(np.arange(m), 2 * d)
    c = np.concatenate([cols[:, None] * d + np.arange(d), rows[:, None] * d + np.arange(d)], axis=1)
    v = np.concatenate([U[rows], -U[rows]], axis=1)
    return sp.csr_matrix((v.ravel(), (r, c.ravel())), shape=(m, k * d))


def _solve_pairwise(
    U: np.ndarray,
    y: np.ndarray,
    method: str,
    solver: str | None,
    n_neighbors: int = 12,
    max_rounds: int = 50,
) -> np.ndarray:
    """Pairwise formulation, solved by constraint generation.

    Most of the ``k^2`` constraints ``<u_i, x_j> <= <u_i, x_i>`` are inactive: only
    directions that are close to each other compete. We start from the constraints between
    each direction and its nearest neighbours, solve, add every violated constraint and
    repeat. A solution of a relaxation that satisfies all constraints is optimal for the
    full problem, so the result is exact (up to solver tolerance).
    """
    k, d = U.shape
    X = cp.Variable(k * d)  # contact points x_i, stacked row by row
    S = sp.csr_matrix((U.ravel(), (np.repeat(np.arange(k), d), np.arange(k * d))), shape=(k, k * d))
    h = S @ X  # h_i = <u_i, x_i>
    objective = _objective(y, h, method)

    # initial working set: nearest neighbours of every direction (by angle)
    m = min(n_neighbors, k - 1)
    nbrs = np.argsort(-(U @ U.T), axis=1)[:, 1 : m + 1]
    rows, cols = np.repeat(np.arange(k), m), nbrs.ravel()
    active = np.zeros((k, k), dtype=bool)
    active[rows, cols] = True

    scale = max(1.0, float(np.abs(y).max()))
    for _ in range(max_rounds):
        A = _pair_matrix(U, rows, cols)
        _solve(cp.Problem(objective, [A @ X <= 0]), solver)
        Xv = X.value.reshape(k, d)
        hv = np.einsum("ij,ij->i", U, Xv)
        viol = (U @ Xv.T) - hv[:, None] > 1e-7 * scale
        viol &= ~active
        if not viol.any():
            return hv
        active |= viol
        rows, cols = np.nonzero(active)
    raise RuntimeError("constraint generation did not converge")


def _solve_local_2d(U: np.ndarray, y: np.ndarray, method: str, solver: str | None) -> np.ndarray:
    if U.shape[1] != 2:
        raise ValueError("the 'local' formulation is only available in 2D")
    k = U.shape[0]
    theta = np.arctan2(U[:, 1], U[:, 0])
    order = np.argsort(theta)
    t = theta[order]
    gaps = np.diff(np.append(t, t[0] + 2 * np.pi))
    if np.any(gaps < 1e-12):
        raise ValueError("directions must be distinct for the 'local' formulation")
    if np.any(gaps >= np.pi):
        raise ValueError("directions must not lie in a closed half-plane")

    prev = np.roll(np.arange(k), 1)
    nxt = np.roll(np.arange(k), -1)
    # angle differences, unwrapped through the gaps so they stay in (0, 2*pi)
    a = gaps  # t_{i+1} - t_i
    b = np.roll(gaps, 1)  # t_i - t_{i-1}
    coef = np.column_stack([np.sin(a), -np.sin(a + b), np.sin(b)])
    # rows are O(gap) in size; normalise them so the solver stays well conditioned for large k
    coef /= np.linalg.norm(coef, axis=1, keepdims=True)
    idx = np.arange(k)
    A = sp.csr_matrix(
        (coef.ravel(), (np.repeat(idx, 3), np.column_stack([prev, idx, nxt]).ravel())),
        shape=(k, k),
    )

    hs = cp.Variable(k)  # support values in sorted order
    problem = cp.Problem(_objective(y[order], hs, method), [A @ hs >= 0])
    _solve(problem, solver)
    h = np.empty(k)
    h[order] = hs.value
    return h
