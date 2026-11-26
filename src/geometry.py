import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection


def chebyshev_center(U: np.ndarray, h: np.ndarray) -> tuple[np.ndarray, float]:
    """
    The largest ball inside the polyhedron ``{x : <x, u_i> <= h_i for all i}``.

    Solved as the linear program: maximise ``r`` subject to ``<c, u_i> + r |u_i| <= h_i``.

    Parameters
    ----------
    U : np.ndarray
        halfspace normals, shape ``(k, d)``
    h : np.ndarray
        halfspace offsets, shape ``(k,)``

    Returns
    -------
    center : np.ndarray
        centre of the ball, shape ``(d,)``
    radius : float
        radius of the ball, 0 when the polyhedron has empty interior
    """
    d = U.shape[1]
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
    """
    Vertices of the polytope ``{x : <x, u_i> <= h_i for all i}``.

    Parameters
    ----------
    U : np.ndarray
        halfspace normals, shape ``(k, d)``
    h : np.ndarray
        halfspace offsets, shape ``(k,)``

    Returns
    -------
    vertices : np.ndarray
        vertices, shape ``(m, d)``; a vertex where more than ``d`` halfspaces meet can appear more than once
    """
    U = np.asarray(U, dtype=float)
    h = np.asarray(h, dtype=float)
    center, radius = chebyshev_center(U, h)
    if radius <= 1e-9 * max(1.0, float(np.abs(h).max())):
        raise ValueError('halfspace intersection has empty interior')
    return HalfspaceIntersection(np.column_stack([U, -h]), center).intersections
