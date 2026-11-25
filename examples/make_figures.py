"""Generate the figures in docs/img/ (run: ``uv run python examples/make_figures.py``)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

import convex_recon as cr  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"

INK = "#1e293b"
MUTED = "#94a3b8"
MEAS = "#f97316"
RECON = "#0d9488"
LAD = "#7c3aed"
BAD = "#dc2626"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)


# ---------------------------------------------------------------------------------------
# bodies
# ---------------------------------------------------------------------------------------


def body_2d():
    """An irregular heptagon with rounded corners: polygon + small disk."""
    P = cr.Polytope(
        np.array(
            [
                [-1.0, -0.3],
                [-0.5, -0.8],
                [0.6, -0.72],
                [1.02, -0.05],
                [0.7, 0.62],
                [-0.05, 0.86],
                [-0.78, 0.42],
            ]
        )
    )
    return P + cr.Ellipsoid(0.1 * np.eye(2), np.zeros(2))


def body_3d():
    """A rounded random polytope: polytope + small ball."""
    return cr.random_polytope(14, rng=5) + cr.Ellipsoid(0.12 * np.eye(3), np.zeros(3))


# ---------------------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------------------


def closed(P: cr.Polytope) -> np.ndarray:
    return P.boundary()


def setup_2d(ax, lim=1.3):
    ax.set_aspect("equal")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_truth(ax, K, label="true body"):
    b = closed(cr.to_polytope(K))
    ax.plot(b[:, 0], b[:, 1], color=INK, lw=2.0, label=label, zorder=4)


def draw_recon(ax, P, color=RECON, label="reconstruction"):
    b = closed(P)
    ax.fill(b[:, 0], b[:, 1], color=color, alpha=0.18, zorder=2, lw=0)
    ax.plot(b[:, 0], b[:, 1], color=color, lw=1.6, label=label, zorder=3)


def draw_measurements(ax, U, y, half_len=0.22, color=MEAS, alpha=None, label="measurement"):
    """Each measurement is a supporting line ``<x, u_i> = y_i``; draw a short piece of it."""
    k = len(y)
    alpha = alpha if alpha is not None else float(np.clip(12 / np.sqrt(k) * 0.25, 0.15, 0.9))
    p = U * y[:, None]
    t = np.column_stack([-U[:, 1], U[:, 0]])
    segs = np.stack([p - half_len * t, p + half_len * t], axis=1)
    for i, s in enumerate(segs):
        ax.plot(
            s[:, 0],
            s[:, 1],
            color=color,
            lw=1.0,
            alpha=alpha,
            zorder=1,
            label=label if i == 0 else None,
        )
    ax.scatter(p[:, 0], p[:, 1], s=6 if k > 100 else 12, color=color, zorder=5, lw=0)


ELEV, AZIM = 24, -58


def _view_dir(elev=ELEV, azim=AZIM) -> np.ndarray:
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    return np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def draw_3d(ax, P: cr.Polytope, color: str, title: str, edges: bool = True):
    """Render a convex polytope with flat shading and (optionally) its facet edges.

    For a convex body only front-facing triangles are visible, so back faces are culled;
    this also avoids matplotlib's depth-sorting artefacts.
    """
    hull = P.hull
    tris = P.vertices[hull.simplices]
    n = hull.equations[:, :3]  # outward unit normals
    front = n @ _view_dir() > 1e-9
    light = _view_dir(ELEV + 40, AZIM - 50)
    intensity = 0.25 + 0.75 * np.clip(n @ light, 0, 1)
    rgb = np.array(matplotlib.colors.to_rgb(color))
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
        from mpl_toolkits.mplot3d.art3d import Line3DCollection

        ax.add_collection3d(Line3DCollection(segs, colors="#0f3d3a", linewidths=0.6, alpha=0.7))
    lim = 0.82
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_axis_off()
    ax.set_title(title, pad=-2, fontsize=11.5)


# ---------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------


def fig_concept():
    """What a support-function measurement is."""
    K = body_2d()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    setup_2d(ax, 1.5)
    draw_truth(ax, K, "convex body K")
    b = closed(cr.to_polytope(K))
    ax.fill(b[:, 0], b[:, 1], color=INK, alpha=0.06)

    ang = np.deg2rad(38)
    u = np.array([np.cos(ang), np.sin(ang)])
    h = K.support(u[None])[0]
    x = K.support_points(u[None])[0]
    t = np.array([-u[1], u[0]])
    line = np.stack([h * u - 1.0 * t, h * u + 0.75 * t])
    ax.plot(
        line[:, 0],
        line[:, 1],
        color=MEAS,
        lw=2,
        label=r"supporting line $\langle x,u\rangle = h_K(u)$",
    )
    ax.annotate(
        "",
        xy=h * u,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=RECON, lw=2, mutation_scale=16),
    )
    ax.annotate(
        "",
        xy=0.45 * u,
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=2.4, mutation_scale=16),
    )
    ax.text(*(0.22 * u + 0.1 * t), r"$u$", fontsize=15, ha="center", va="center")
    ax.text(
        *(0.72 * h * u - 0.14 * t), r"$h_K(u)$", fontsize=15, color=RECON, ha="center", va="center"
    )
    ax.scatter(*x, color=MEAS, s=40, zorder=6)
    ax.scatter(0, 0, color=INK, s=18, zorder=6)
    ax.text(0.04, -0.12, "origin", fontsize=9, color="#475569")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_title(r"Support function: $h_K(u)=\max_{x\in K}\ \langle x,u\rangle$")
    fig.savefig(OUT / "concept.png")
    plt.close(fig)


def fig_hero():
    """Reconstructions from increasingly many noisy measurements."""
    K = body_2d()
    sigma = 0.05
    ks = [10, 40, 300]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, k in zip(axes, ks, strict=True):
        U = cr.circle_directions(k, offset=0.3)
        y = cr.measure(K, U, sigma, rng=k)
        rec = cr.reconstruct(U, y)
        setup_2d(ax)
        draw_measurements(ax, U, y)
        draw_recon(ax, rec.polytope)
        draw_truth(ax, K)
        err = cr.hausdorff_distance(K, rec.polytope)
        ax.set_title(f"k = {k} measurements")
        ax.text(
            0.5,
            0.02,
            f"Hausdorff error {err:.3f}",
            transform=ax.transAxes,
            ha="center",
            color="#475569",
            fontsize=10,
        )
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5, bbox_to_anchor=(-0.02, 1.0))
    fig.suptitle(
        rf"Least-squares reconstruction from noisy support values  ($\sigma$ = {sigma})",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.02, top=0.86)
    fig.savefig(OUT / "reconstruction_2d.png")
    plt.close(fig)


def fig_outliers():
    """Least squares vs least absolute deviations with gross outliers."""
    K = body_2d()
    k, sigma = 160, 0.02
    U = cr.circle_directions(k)
    rng = np.random.default_rng(3)
    y = cr.measure(K, U, sigma, rng=rng)
    bad = rng.choice(k, size=k // 10, replace=False)
    y[bad] += rng.uniform(0.4, 0.9, bad.size)
    good = np.setdiff1d(np.arange(k), bad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1))
    for ax, method, color, name in [
        (axes[0], "lsq", RECON, "Least squares (QP)"),
        (axes[1], "lad", LAD, "Least absolute deviations (LP)"),
    ]:
        rec = cr.reconstruct(U, y, method=method)
        setup_2d(ax, 1.9)
        draw_measurements(ax, U[good], y[good], label="measurement", alpha=0.35)
        draw_measurements(ax, U[bad], y[bad], color=BAD, alpha=0.9, label="outlier (10%)")
        draw_recon(ax, rec.polytope, color=color, label="reconstruction")
        draw_truth(ax, K)
        err = cr.hausdorff_distance(K, rec.polytope)
        ax.set_title(name)
        ax.text(
            0.5,
            0.02,
            f"Hausdorff error {err:.3f}",
            transform=ax.transAxes,
            ha="center",
            color="#475569",
            fontsize=10,
        )
    axes[0].legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.suptitle(f"Robustness: {k} measurements, 10% of them corrupted", fontsize=13)
    fig.savefig(OUT / "outliers.png")
    plt.close(fig)


def _fmt_time(dt: float) -> str:
    return f"{dt * 1000:.0f} ms" if dt < 1 else f"{dt:.1f} s"


def fig_3d():
    K = body_3d()
    sigma = 0.03
    fig = plt.figure(figsize=(13.5, 4.1))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    draw_3d(ax, cr.to_polytope(K), "#64748b", "true body\n(rounded polytope)", edges=False)
    for i, k in enumerate([60, 600]):
        U = cr.sphere_directions(k)
        y = cr.measure(K, U, sigma, rng=k)
        t = time.perf_counter()
        rec = cr.reconstruct(U, y)
        dt = time.perf_counter() - t
        err = cr.hausdorff_distance(K, rec.polytope)
        ax = fig.add_subplot(1, 3, i + 2, projection="3d")
        draw_3d(
            ax,
            rec.polytope,
            RECON,
            f"k = {k} directions\nHausdorff error {err:.3f}  ·  solved in {_fmt_time(dt)}",
        )
    fig.subplots_adjust(wspace=-0.05, left=0, right=1, top=1.0, bottom=-0.08)
    fig.savefig(OUT / "reconstruction_3d.png")
    plt.close(fig)


def _errors(K, dim, ks, trials, sigma):
    l2 = np.zeros((len(ks), trials))
    hd = np.zeros_like(l2)
    for a, k in enumerate(ks):
        for t in range(trials):
            rng = np.random.default_rng(1000 * k + t)
            # evenly spread directions, randomly rotated each trial
            U = cr.evenly_spread(k, dim)
            Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
            U = U @ Q
            rec = cr.reconstruct(U, cr.measure(K, U, sigma, rng=rng))
            l2[a, t] = cr.l2_distance(K, rec.polytope)
            hd[a, t] = cr.hausdorff_distance(K, rec.polytope)
        print(
            f"  {dim}D k={k}: L2 {np.median(l2[a]):.4f}  Hausdorff {np.median(hd[a]):.4f}",
            flush=True,
        )
    return l2, hd


def fig_convergence():
    sigma = 0.05
    setups = [
        (2, body_2d(), np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]), 20, 2 / 5),
        (3, body_3d(), np.array([16, 32, 64, 128, 256, 512]), 6, 1 / 3),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (dim, K, ks, trials, rate) in zip(axes, setups, strict=True):
        rate_txt = "2/5" if dim == 2 else "1/3"
        l2, hd = _errors(K, dim, ks, trials, sigma)
        for err, color, name in [(l2, RECON, "L2 error"), (hd, MEAS, "Hausdorff error")]:
            med = np.median(err, axis=1)
            lo, hi = np.percentile(err, [25, 75], axis=1)
            slope = np.polyfit(np.log(ks[2:]), np.log(med[2:]), 1)[0]
            ax.fill_between(ks, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(
                ks,
                med,
                "o-",
                color=color,
                lw=2,
                ms=4.5,
                label=f"{name}  (fitted slope {slope:.2f})",
            )
        ref = np.median(l2, axis=1)[1] * (ks / ks[1]) ** (-rate) * 1.6
        ax.plot(
            ks,
            ref,
            "--",
            color=INK,
            lw=1.2,
            label=rf"proven upper-bound rate $k^{{-{rate_txt}}}$ (L2)",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("number of measurements k")
        ax.set_ylabel("error (median, IQR band)")
        ax.set_title(f"{dim}D, $\\sigma$ = {sigma}, {trials} trials per k")
        ax.grid(True, which="major", color="#e2e8f0", lw=0.8)
        ax.legend(frameon=False, fontsize=9.5)
    fig.suptitle("Convergence as the number of measurements grows", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png")
    plt.close(fig)


def anim_2d():
    K = body_2d()
    sigma = 0.05
    ks = np.unique(np.round(np.geomspace(5, 400, 48)).astype(int))
    frames = []
    for k in ks:
        U = cr.circle_directions(int(k), offset=0.3)
        y = cr.measure(K, U, sigma, rng=int(k))
        rec = cr.reconstruct(U, y)
        frames.append((U, y, rec.polytope, cr.hausdorff_distance(K, rec.polytope)))
    frames += [frames[-1]] * 12  # pause on the last frame

    fig, ax = plt.subplots(figsize=(5, 5.2), dpi=90)

    def draw(i):
        ax.clear()
        U, y, P, err = frames[i]
        setup_2d(ax)
        draw_measurements(ax, U, y)
        draw_recon(ax, P)
        draw_truth(ax, K)
        ax.set_title(f"k = {len(y)}   Hausdorff error {err:.3f}")

    anim = FuncAnimation(fig, draw, frames=len(frames))
    anim.save(OUT / "reconstruction_2d.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


def main(which: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figs = {
        "concept": fig_concept,
        "hero": fig_hero,
        "outliers": fig_outliers,
        "3d": fig_3d,
        "convergence": fig_convergence,
        "gif": anim_2d,
    }
    for name in which or figs:
        t = time.perf_counter()
        figs[name]()
        print(f"{name}: {time.perf_counter() - t:.1f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
