"""Halfspace intersections: turning support values back into a polytope.

A set of directions ``u_i`` and values ``h_i`` describes the polytope
``P = {x : <x, u_i> <= h_i for all i}``. To list its vertices we need a point strictly
inside it, which the Chebyshev centre provides.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


def chebyshev_center(U: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    """Centre and radius of the largest ball inside ``{x : <x, u_i> <= h_i}``.

    Solved as a linear program: maximise ``r`` subject to ``<c, u_i> + r |u_i| <= h_i``.
    """
    k, d = U.shape
    norms = np.linalg.norm(U, axis=1)
    cost = np.zeros(d + 1)
    cost[-1] = -1.0  # maximise the radius
    res = linprog(
        cost,
        A_ub=np.column_stack([U, norms]),
        b_ub=h,
        bounds=[(None, None)] * d + [(0, None)],
        method='highs',
    )
    if res.status == 3:
        raise ValueError('halfspace intersection is unbounded; directions must not lie in a closed hemisphere')
    if not res.success:
        raise ValueError(f'could not find an interior point: {res.message}')
    return res.x[:d], float(res.x[-1])


def halfspace_vertices(U: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Vertices of ``{x : <x, u_i> <= h_i for all i}`` (may include duplicates).

    Raises ``ValueError`` if the intersection is unbounded or has empty interior.
    """
    U = np.asarray(U, dtype=float)
    h = np.asarray(h, dtype=float)
    center, radius = chebyshev_center(U, h)
    scale = max(1.0, float(np.abs(h).max()))
    if radius <= 1e-9 * scale:
        raise ValueError('halfspace intersection has (numerically) empty interior')
    return HalfspaceIntersection(np.column_stack([U, -h]), center).intersections
