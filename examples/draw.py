"""
Drawing helpers shared by the example scripts. They draw on the axes they are given, so this module imports no
matplotlib.
"""

import numpy as np

from src.bodies import ConvexBody, Polytope
from style import MEASUREMENT, TRUTH

# the 3D view and the light direction relative to it
ELEV, AZIM = 24, -58
LIGHT_ELEV, LIGHT_AZIM = ELEV + 40, AZIM - 50
# half the side of the cube the 3D axes show
LIM_3D = 0.82
# half the side of the square the 2D axes show
LIM_2D = 1.3
# half the length of each drawn piece of a supporting line
HALF_LENGTH = 0.22
# line widths
BODY_WIDTH = 1.8
ESTIMATE_WIDTH = 1.6
MEASUREMENT_WIDTH = 1.0
# opacity of an estimate's fill
ESTIMATE_FILL = 0.15
# drawing order, bottom to top; the true body goes on top of the estimates, or between their fill and outline
Z_MEASUREMENT, Z_FILL, Z_BODY_UNDER, Z_ESTIMATE, Z_BODY, Z_POINT = 1, 2, 2.5, 3, 4, 5
# measurement lines get fainter as k grows: opacity MEASUREMENT_FADE / sqrt(k), clipped to MEASUREMENT_ALPHA
MEASUREMENT_FADE = 3
MEASUREMENT_ALPHA = (0.15, 0.9)
# measurement points are smaller when there are more than MANY_POINTS of them
POINT_SIZE, POINT_SIZE_MANY, MANY_POINTS = 12, 6, 100
# 3D shading: a face's light intensity is AMBIENT + (1 - AMBIENT) cos(angle to the light), and its colour is the base
# colour scaled by SHADE_BASE + SHADE_GAIN * intensity
AMBIENT, SHADE_BASE, SHADE_GAIN = 0.25, 0.3, 0.95
# width of the face outlines that hide the seams between triangles, and the facet edges' width and opacity
SEAM_WIDTH = 0.4
EDGE_WIDTH, EDGE_ALPHA = 0.6, 0.7
# a face is drawn when its normal's cosine with the view exceeds FRONT_TOL; two triangles lie on one facet when
# their normals' cosine exceeds 1 - COPLANAR_TOL
FRONT_TOL, COPLANAR_TOL = 1e-9, 1e-7


def outline(body: ConvexBody) -> np.ndarray:
    """Closed boundary curve of a 2D body, its first point repeated at the end."""
    v = body.to_polytope().vertices
    return np.vstack([v, v[:1]])


def setup_2d(ax, lim: float = LIM_2D) -> None:
    """Square, equal-aspect axes on ``[-lim, lim]^2`` with no ticks, frame or grid."""
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_axis_off()


def draw_body(ax, body: ConvexBody, label: str | None = 'true body', fill: float = 0.0, on_top: bool = True) -> None:
    """The outline of a 2D body, filled with opacity ``fill``, drawn over the estimates unless ``on_top`` is false."""
    b = outline(body)
    if fill:
        ax.fill(b[:, 0], b[:, 1], color=TRUTH, alpha=fill, linewidth=0)
    ax.plot(b[:, 0], b[:, 1], color=TRUTH, linewidth=BODY_WIDTH, label=label, zorder=Z_BODY if on_top else Z_BODY_UNDER)


def draw_estimate(
    ax, P: Polytope, color: str, label: str | None = 'reconstruction', fill: float = ESTIMATE_FILL
) -> None:
    """A reconstructed polygon, filled with opacity ``fill``."""
    b = outline(P)
    if fill:
        ax.fill(b[:, 0], b[:, 1], color=color, alpha=fill, linewidth=0, zorder=Z_FILL)
    ax.plot(b[:, 0], b[:, 1], color=color, linewidth=ESTIMATE_WIDTH, label=label, zorder=Z_ESTIMATE)


def draw_measurements(
    ax,
    U: np.ndarray,
    y: np.ndarray,
    color: str = MEASUREMENT,
    alpha: float | None = None,
    label: str | None = 'measurement',
    fade: float = 1.0,
) -> None:
    """
    Measurements as short pieces of their supporting lines ``<x, u_i> = y_i``, each marked at ``y_i u_i``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        2D axes to draw on
    U : np.ndarray
        directions, shape ``(k, 2)``
    y : np.ndarray
        measurements, shape ``(k,)``
    color : str
        colour of the lines and points
    alpha : float | None
        opacity of the lines; ``None`` fades them as ``k`` grows
    label : str | None
        legend label
    fade : float
        factor in [0, 1] applied to the opacity of both the lines and the points, to fade them out
    """
    if alpha is None:
        alpha = float(np.clip(MEASUREMENT_FADE / np.sqrt(len(y)), *MEASUREMENT_ALPHA))
    alpha *= fade
    p = U * y[:, None]
    t = np.column_stack([-U[:, 1], U[:, 0]])
    # one polyline with NaN breaks draws all the pieces in one call
    segments = np.stack([p - HALF_LENGTH * t, p + HALF_LENGTH * t, np.full_like(p, np.nan)], axis=1).reshape(-1, 2)
    ax.plot(
        segments[:, 0],
        segments[:, 1],
        color=color,
        linewidth=MEASUREMENT_WIDTH,
        alpha=alpha,
        label=label,
        zorder=Z_MEASUREMENT,
    )
    size = POINT_SIZE_MANY if len(y) > MANY_POINTS else POINT_SIZE
    ax.scatter(p[:, 0], p[:, 1], s=size, color=color, alpha=fade, linewidth=0, zorder=Z_POINT)


def _unit(elev: float, azim: float) -> np.ndarray:
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def _rgb(hex_colour: str) -> np.ndarray:
    return np.array([int(hex_colour[i : i + 2], 16) for i in (1, 3, 5)]) / 255


def draw_polytope_3d(ax, P: Polytope, color: str, edge_color: str | None = None) -> None:
    """
    A 3D polytope with flat shading and, optionally, its facet edges, seen from ``ELEV`` and ``AZIM``.

    Only the triangles facing the viewer are drawn: for a convex body the others are hidden, and leaving them out
    avoids matplotlib's depth-sorting artefacts.

    Parameters
    ----------
    ax : mpl_toolkits.mplot3d.axes3d.Axes3D
        3D axes to draw on
    P : Polytope
        3D polytope
    color : str
        base colour as ``'#rrggbb'``, darkened on faces turned away from the light
    edge_color : str | None
        colour of the facet edges; ``None`` draws none
    """
    hull = P.hull
    normals = hull.equations[:, :3]  # outward unit normals
    front = normals @ _unit(ELEV, AZIM) > FRONT_TOL
    intensity = AMBIENT + (1 - AMBIENT) * np.clip(normals[front] @ _unit(LIGHT_ELEV, LIGHT_AZIM), 0, 1)
    face_colours = np.clip(_rgb(color) * (SHADE_BASE + SHADE_GAIN * intensity[:, None]), 0, 1)
    x, y, z = P.vertices.T
    surface = ax.plot_trisurf(x, y, z, triangles=hull.simplices[front], shade=False)
    surface.set_facecolor(face_colours)
    surface.set_edgecolor(face_colours)  # hides the seams between triangles of one facet
    surface.set_linewidth(SEAM_WIDTH)

    if edge_color is not None:
        # an edge between two triangles is a facet edge when their normals differ
        points = []
        for t in np.flatnonzero(front):
            for j, nb in enumerate(hull.neighbors[t]):
                if (nb > t or not front[nb]) and normals[t] @ normals[nb] < 1 - COPLANAR_TOL:
                    points += [*P.vertices[np.delete(hull.simplices[t], j)], np.full(3, np.nan)]
        ax.plot(*np.array(points).T, color=edge_color, linewidth=EDGE_WIDTH, alpha=EDGE_ALPHA)

    ax.set_xlim(-LIM_3D, LIM_3D)
    ax.set_ylim(-LIM_3D, LIM_3D)
    ax.set_zlim(-LIM_3D, LIM_3D)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
