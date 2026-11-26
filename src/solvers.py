from typing import Literal

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

Loss = Literal['lsq', 'lad']


def fit_local_2d(U: np.ndarray, y: np.ndarray, loss: Loss = 'lsq', solver: str | None = None) -> np.ndarray:
    """
    Consistent support values closest to ``y``, written with one local constraint per direction (2D only).

    Solves ``min sum_i loss(y_i - h_i)`` subject to ``h`` being consistent: the support values of some convex body at
    the directions ``u_i``. With the directions sorted by angle ``t_i``, ``h`` is consistent iff for every ``i``
    (cyclically) ``h_{i-1} sin(t_{i+1} - t_i) - h_i sin(t_{i+1} - t_{i-1}) + h_{i+1} sin(t_i - t_{i-1}) >= 0``, that
    is, every line ``<x, u_i> = h_i`` touches the polygon cut out by its two neighbours (Lele, Kulkarni & Willsky,
    1992). These are ``k`` constraints, so the problem stays small for large ``k``.

    Parameters
    ----------
    U : np.ndarray
        distinct unit directions, not all in a closed half-plane, shape ``(k, 2)``
    y : np.ndarray
        measurements, shape ``(k,)``
    loss : {'lsq', 'lad'}
        squared error (a quadratic program) or absolute error (a linear program)
    solver : str | None
        cvxpy solver name; ``None`` uses Clarabel

    Returns
    -------
    h : np.ndarray
        fitted support values, in the order of ``U``, shape ``(k,)``
    """
    if U.shape[1] != 2:
        raise ValueError(f'the local formulation is only available in 2D, got directions of shape {U.shape}')
    k = U.shape[0]
    theta = np.arctan2(U[:, 1], U[:, 0])
    order = np.argsort(theta)
    t = theta[order]
    gaps = np.diff(np.append(t, t[0] + 2 * np.pi))  # t_{i+1} - t_i, wrapping around
    if np.any(gaps < 1e-12):
        raise ValueError('directions must be distinct for the local formulation')
    if np.any(gaps >= np.pi):
        raise ValueError('directions must not lie in a closed half-plane')

    a = gaps  # t_{i+1} - t_i
    b = np.roll(gaps, 1)  # t_i - t_{i-1}
    coef = np.column_stack([np.sin(a), -np.sin(a + b), np.sin(b)])
    # rows are O(gap) in size; normalising them keeps the solver well conditioned for large k
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
    loss: Loss = 'lsq',
    solver: str | None = None,
    n_neighbors: int = 12,
    max_rounds: int = 50,
) -> np.ndarray:
    """
    Consistent support values closest to ``y``, written with contact points (any dimension).

    Solves ``min sum_i loss(y_i - h_i)`` subject to ``h`` being consistent. ``h`` is consistent iff there are points
    ``x_i`` with ``<x_i, u_i> = h_i`` and ``<x_j, u_i> <= h_i`` for all ``i, j``: the ``x_i`` are contact points of
    the body in direction ``u_i``. That is ``d k`` variables and ``k^2`` constraints, but only directions close to each
    other compete, so most constraints are inactive. Constraint generation starts from each direction's
    ``n_neighbors`` nearest directions, solves, adds every violated constraint and repeats; a solution that satisfies
    all constraints is optimal for the full problem, so the result is exact up to the solver tolerance.

    Parameters
    ----------
    U : np.ndarray
        unit directions, not all in a closed hemisphere, shape ``(k, d)``
    y : np.ndarray
        measurements, shape ``(k,)``
    loss : {'lsq', 'lad'}
        squared error (a quadratic program) or absolute error (a linear program)
    solver : str | None
        cvxpy solver name; ``None`` uses Clarabel
    n_neighbors : int
        constraints per direction in the first round
    max_rounds : int
        rounds of constraint generation before giving up

    Returns
    -------
    h : np.ndarray
        fitted support values, shape ``(k,)``
    """
    k, d = U.shape
    X = cp.Variable(k * d)  # contact points x_i, stacked row by row
    S = sp.csr_matrix((U.ravel(), (np.repeat(np.arange(k), d), np.arange(k * d))), shape=(k, k * d))
    h = S @ X  # h_i = <u_i, x_i>
    objective = _objective(y, h, loss)

    m = min(n_neighbors, k - 1)
    nbrs = np.argsort(-(U @ U.T), axis=1)[:, 1 : m + 1]
    active = np.zeros((k, k), dtype=bool)
    active[np.repeat(np.arange(k), m), nbrs.ravel()] = True

    tol = 1e-7 * max(1.0, float(np.abs(y).max()))
    for _ in range(max_rounds):
        rows, cols = np.nonzero(active)
        _solve(cp.Problem(objective, [_pair_matrix(U, rows, cols) @ X <= 0]), solver)
        Xv = X.value.reshape(k, d)
        hv = h.value
        violated = ((U @ Xv.T) - hv[:, None] > tol) & ~active
        if not violated.any():
            return hv
        active |= violated
    raise RuntimeError(f'constraint generation did not converge in max_rounds={max_rounds} rounds')


def _pair_matrix(U: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> sp.csr_matrix:
    # the row for pair (i, j) maps the stacked contact points to <u_i, x_j - x_i>
    k, d = U.shape
    m = rows.size
    r = np.repeat(np.arange(m), 2 * d)
    c = np.concatenate([cols[:, None] * d + np.arange(d), rows[:, None] * d + np.arange(d)], axis=1)
    v = np.concatenate([U[rows], -U[rows]], axis=1)
    return sp.csr_matrix((v.ravel(), (r, c.ravel())), shape=(m, k * d))


def _objective(y: np.ndarray, h: cp.Expression, loss: Loss) -> cp.Minimize:
    if loss == 'lsq':
        return cp.Minimize(cp.sum_squares(y - h))
    if loss == 'lad':
        return cp.Minimize(cp.norm1(y - h))
    raise ValueError(f"loss must be 'lsq' or 'lad', got {loss!r}")


def _solve(problem: cp.Problem, solver: str | None) -> None:
    if solver is None:
        # cvxpy's default QP solver, OSQP, is first-order and stalls on these problems; Clarabel does not
        solver = cp.CLARABEL
    if solver == cp.CLARABEL:
        # Clarabel's default factoriser, faer, runs on every core and its threads slow each other down on these
        # problems; the single-threaded QDLDL solves a 3D fit with k = 600 about 6 times faster
        problem.solve(solver=solver, direct_solve_method='qdldl')
    else:
        problem.solve(solver=solver)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f'solver {solver} failed with status {problem.status!r}')
