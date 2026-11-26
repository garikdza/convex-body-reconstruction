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
# half the length of each drawn piece of a supporting line
HALF_LENGTH = 0.22


def outline(body: ConvexBody) -> np.ndarray:
    """Closed boundary curve of a 2D body, its first point repeated at the end."""
    v = body.to_polytope().vertices
    return np.vstack([v, v[:1]])


def setup_2d(ax, lim: float = 1.3) -> None:
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
    ax.plot(b[:, 0], b[:, 1], color=TRUTH, linewidth=1.8, label=label, zorder=4 if on_top else 2.5)


def draw_estimate(ax, P: Polytope, color: str, label: str | None = 'reconstruction', fill: float = 0.15) -> None:
    """A reconstructed polygon, filled with opacity ``fill``."""
    b = outline(P)
    if fill:
        ax.fill(b[:, 0], b[:, 1], color=color, alpha=fill, linewidth=0, zorder=2)
    ax.plot(b[:, 0], b[:, 1], color=color, linewidth=1.6, label=label, zorder=3)


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
        alpha = float(np.clip(3 / np.sqrt(len(y)), 0.15, 0.9))
    alpha *= fade
    p = U * y[:, None]
    t = np.column_stack([-U[:, 1], U[:, 0]])
    # one polyline with NaN breaks draws all the pieces in one call
    segments = np.stack([p - HALF_LENGTH * t, p + HALF_LENGTH * t, np.full_like(p, np.nan)], axis=1).reshape(-1, 2)
    ax.plot(segments[:, 0], segments[:, 1], color=color, linewidth=1.0, alpha=alpha, label=label, zorder=1)
    ax.scatter(p[:, 0], p[:, 1], s=6 if len(y) > 100 else 12, color=color, alpha=fade, linewidth=0, zorder=5)


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
    front = normals @ _unit(ELEV, AZIM) > 1e-9
    intensity = 0.25 + 0.75 * np.clip(normals[front] @ _unit(LIGHT_ELEV, LIGHT_AZIM), 0, 1)
    face_colours = np.clip(_rgb(color) * (0.3 + 0.95 * intensity[:, None]), 0, 1)
    x, y, z = P.vertices.T
    surface = ax.plot_trisurf(x, y, z, triangles=hull.simplices[front], shade=False)
    surface.set_facecolor(face_colours)
    surface.set_edgecolor(face_colours)  # hides the seams between triangles of one facet
    surface.set_linewidth(0.4)

    if edge_color is not None:
        # an edge between two triangles is a facet edge when their normals differ
        points = []
        for t in np.flatnonzero(front):
            for j, nb in enumerate(hull.neighbors[t]):
                if (nb > t or not front[nb]) and normals[t] @ normals[nb] < 1 - 1e-7:
                    points += [*P.vertices[np.delete(hull.simplices[t], j)], np.full(3, np.nan)]
        ax.plot(*np.array(points).T, color=edge_color, linewidth=0.6, alpha=0.7)

    ax.set_xlim(-LIM_3D, LIM_3D)
    ax.set_ylim(-LIM_3D, LIM_3D)
    ax.set_zlim(-LIM_3D, LIM_3D)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
