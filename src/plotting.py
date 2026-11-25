"""Matplotlib drawing helpers for bodies, measurements and reconstructions."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from .bodies import ConvexBody, Polytope, to_polytope

# palette
INK = '#1e293b'
MUTED = '#94a3b8'
SUBTLE = '#475569'
MEAS = '#f97316'
RECON = '#0d9488'
LAD = '#7c3aed'
BAD = '#dc2626'
TRUTH_3D = '#64748b'


def use_style() -> None:
    """Apply the project's figure style."""
    plt.rcParams.update(
        {
            'font.family': 'DejaVu Sans',
            'font.size': 11,
            'axes.titlesize': 12,
            'axes.titleweight': 'bold',
            'axes.edgecolor': MUTED,
            'axes.labelcolor': INK,
            'text.color': INK,
            'xtick.color': SUBTLE,
            'ytick.color': SUBTLE,
            'figure.dpi': 110,
            'savefig.dpi': 160,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
        }
    )


# --- 2D ----------------------------------------------------------------------------------


def outline(body: ConvexBody) -> np.ndarray:
    """Closed boundary curve of a 2D body (first point repeated at the end)."""
    v = to_polytope(body).vertices
    return np.vstack([v, v[:1]])


def setup_2d(ax, lim: float = 1.3) -> None:
    """Equal-aspect square axes without ticks or frame."""
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_body(ax, body: ConvexBody, color: str = INK, label: str | None = 'true body', fill=0.0):
    """Outline of a 2D body, optionally lightly filled."""
    b = outline(body)
    if fill:
        ax.fill(b[:, 0], b[:, 1], color=color, alpha=fill, lw=0)
    ax.plot(b[:, 0], b[:, 1], color=color, lw=2.0, label=label, zorder=4)


def draw_estimate(ax, P: Polytope, color: str = RECON, label: str | None = 'reconstruction'):
    """Filled outline of a reconstructed polygon."""
    b = outline(P)
    ax.fill(b[:, 0], b[:, 1], color=color, alpha=0.18, zorder=2, lw=0)
    ax.plot(b[:, 0], b[:, 1], color=color, lw=1.6, label=label, zorder=3)


def draw_measurements(ax, U, y, half_len=0.22, color=MEAS, alpha: float | None = None, label='measurement'):
    """Each measurement is a supporting line ``<x, u_i> = y_i``; draw a short piece of it
    and the point ``y_i u_i``."""
    k = len(y)
    if alpha is None:
        alpha = float(np.clip(3 / np.sqrt(k), 0.15, 0.9))
    p = U * y[:, None]
    t = np.column_stack([-U[:, 1], U[:, 0]])
    for i, (a, b) in enumerate(zip(p - half_len * t, p + half_len * t, strict=True)):
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color=color,
            lw=1.0,
            alpha=alpha,
            zorder=1,
            label=label if i == 0 else None,
        )
    ax.scatter(p[:, 0], p[:, 1], s=6 if k > 100 else 12, color=color, zorder=5, lw=0)


def caption(ax, text: str) -> None:
    """Small grey text centred under the axes content."""
    ax.text(0.5, 0.02, text, transform=ax.transAxes, ha='center', color=SUBTLE, fontsize=10)


# --- 3D ----------------------------------------------------------------------------------

ELEV, AZIM = 24, -58


def _view_dir(elev: float = ELEV, azim: float = AZIM) -> np.ndarray:
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_polytope_3d(ax, P: Polytope, color: str, title: str = '', edges: bool = True):
    """Render a 3D polytope with flat shading and (optionally) its facet edges.

    For a convex body only front-facing triangles are visible, so back faces are culled;
    this also avoids matplotlib's depth-sorting artefacts.
    """
    hull = P.hull
    tris = P.vertices[hull.simplices]
    n = hull.equations[:, :3]  # outward unit normals
    front = n @ _view_dir() > 1e-9
    light = _view_dir(ELEV + 40, AZIM - 50)
    intensity = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    rgb = np.array(mcolors.to_rgb(color))
    fc = np.clip(rgb * (0.3 + 0.95 * intensity[:, None]), 0, 1)[front]
    ax.add_collection3d(Poly3DCollection(tris[front], facecolors=fc, edgecolors=fc, linewidths=0.4))

    if edges:
        # an edge between two triangles is a facet edge if their normals differ
        segs = []
        for t in np.flatnonzero(front):
            for j, nb in enumerate(hull.neighbors[t]):
                if (nb > t or not front[nb]) and np.dot(n[t], n[nb]) < 1 - 1e-7:
                    a, b = np.delete(hull.simplices[t], j)
                    segs.append(P.vertices[[a, b]])
        ax.add_collection3d(Line3DCollection(segs, colors='#0f3d3a', linewidths=0.6, alpha=0.7))

    lim = 0.82
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, pad=-2, fontsize=11.5)
