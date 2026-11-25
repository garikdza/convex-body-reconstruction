"""Convex programs that fit consistent support values to measurements.

Both functions solve

    minimise   sum_i loss(y_i - h_i)    subject to   h is consistent,

where *consistent* means ``h_1 ... h_k`` are the support values of some convex body at
the directions ``u_i``, and ``loss`` is squared (``"lsq"``, a QP) or absolute (``"lad"``,
an LP). They differ in how consistency is written as linear constraints:

``fit_local_2d`` (2D only)
    With the directions sorted by angle ``t_i``, ``h`` is consistent iff for every ``i``
    (cyclically)
    ``h_{i-1} sin(t_{i+1} - t_i) - h_i sin(t_{i+1} - t_{i-1}) + h_{i+1} sin(t_i - t_{i-1}) >= 0``,
    i.e. every halfspace touches the polygon cut out by its two neighbours
    (Lele, Kulkarni & Willsky 1992). Only ``k`` constraints, so it scales to large ``k``.

``fit_pairwise`` (any dimension)
    ``h`` is consistent iff there are points ``x_i`` with ``<x_i, u_i> = h_i`` and
    ``<x_j, u_i> <= h_i`` for all ``i, j``: the ``x_i`` are contact points of the body in
    direction ``u_i``. This needs ``d*k`` variables and ``k^2`` constraints, handled by
    constraint generation.
"""

from __future__ import annotations

from typing import Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

Loss = Literal['lsq', 'lad']


def fit_local_2d(U: np.ndarray, y: np.ndarray, loss: Loss, solver: str | None) -> np.ndarray:
    """Fitted support values using the ``k`` local constraints (2D only)."""
    if U.shape[1] != 2:
        raise ValueError("the 'local' formulation is only available in 2D")
    k = U.shape[0]
    theta = np.arctan2(U[:, 1], U[:, 0])
    order = np.argsort(theta)
    t = theta[order]
    gaps = np.diff(np.append(t, t[0] + 2 * np.pi))  # t_{i+1} - t_i, wrapping around
    if np.any(gaps < 1e-12):
        raise ValueError("directions must be distinct for the 'local' formulation")
    if np.any(gaps >= np.pi):
        raise ValueError('directions must not lie in a closed half-plane')

    a = gaps  # t_{i+1} - t_i
    b = np.roll(gaps, 1)  # t_i - t_{i-1}
    coef = np.column_stack([np.sin(a), -np.sin(a + b), np.sin(b)])
    # rows are O(gap) in size; normalise them so the solver stays well conditioned for large k
    coef /= np.linalg.norm(coef, axis=1, keepdims=True)
    idx = np.arange(k)
    cols = np.column_stack([np.roll(idx, 1), idx, np.roll(idx, -1)])
    A = sp.csr_matrix((coef.ravel(), (np.repeat(idx, 3), cols.ravel())), shape=(k, k))

    hs = cp.Variable(k)  # support values in sorted order
    _solve(cp.Problem(_objective(y[order], hs, loss), [A @ hs >= 0]), solver)
    h = np.empty(k)
    h[order] = hs.value
    return h


def fit_pairwise(
    U: np.ndarray,
    y: np.ndarray,
    loss: Loss,
    solver: str | None,
    n_neighbors: int = 12,
    max_rounds: int = 50,
) -> np.ndarray:
    """Fitted support values using the pairwise constraints, by constraint generation.

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
    objective = _objective(y, h, loss)

    # initial working set: nearest neighbours of every direction (by angle)
    m = min(n_neighbors, k - 1)
    nbrs = np.argsort(-(U @ U.T), axis=1)[:, 1 : m + 1]
    active = np.zeros((k, k), dtype=bool)
    active[np.repeat(np.arange(k), m), nbrs.ravel()] = True

    tol = 1e-7 * max(1.0, float(np.abs(y).max()))
    for _ in range(max_rounds):
        rows, cols = np.nonzero(active)
        _solve(cp.Problem(objective, [_pair_matrix(U, rows, cols) @ X <= 0]), solver)
        Xv = X.value.reshape(k, d)
        hv = np.einsum('ij,ij->i', U, Xv)
        violated = ((U @ Xv.T) - hv[:, None] > tol) & ~active
        if not violated.any():
            return hv
        active |= violated
    raise RuntimeError('constraint generation did not converge')


def _pair_matrix(U: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    """Sparse matrix whose row for pair ``(i, j)`` maps ``vec(X)`` to ``<u_i, x_j - x_i>``."""
    k, d = U.shape
    m = rows.size
    r = np.repeat(np.arange(m), 2 * d)
    c = np.concatenate([cols[:, None] * d + np.arange(d), rows[:, None] * d + np.arange(d)], axis=1)
    v = np.concatenate([U[rows], -U[rows]], axis=1)
    return sp.csr_matrix((v.ravel(), (r, c.ravel())), shape=(m, k * d))


def _objective(y: np.ndarray, h: cp.Expression, loss: Loss) -> cp.Minimize:
    if loss == 'lsq':
        return cp.Minimize(cp.sum_squares(y - h))
    return cp.Minimize(cp.norm1(y - h))


def _solve(problem: cp.Problem, solver: str | None) -> None:
    if solver is None:
        # cvxpy's default QP solver (OSQP) is first-order and stalls on these problems;
        # the interior-point solver Clarabel is fast and accurate.
        solver = cp.CLARABEL if cp.CLARABEL in cp.installed_solvers() else None
    problem.solve(solver=solver)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'solver failed with status {problem.status!r}')
