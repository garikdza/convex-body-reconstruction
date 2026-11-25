"""Generate the README figures in docs/img/ (the benchmark plot is in convergence.py).

uv run python examples/figures.py                 # all figures (~20 s)
uv run python examples/figures.py hero outliers   # just some
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import src as cr  # noqa: E402
from shapes import rounded_heptagon, rounded_polytope  # noqa: E402
from src import plotting as pl  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / 'docs' / 'img'


def concept() -> None:
    """What a support-function measurement is."""
    K = rounded_heptagon()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    pl.setup_2d(ax, 1.5)
    pl.draw_body(ax, K, label='convex body K', fill=0.06)

    u = np.array([np.cos(np.deg2rad(38)), np.sin(np.deg2rad(38))])
    h = K.support(u[None])[0]
    x = K.support_points(u[None])[0]
    t = np.array([-u[1], u[0]])  # tangent direction
    line = np.stack([h * u - 1.0 * t, h * u + 0.75 * t])
    ax.plot(*line.T, color=pl.MEAS, lw=2, label=r'supporting line $\langle x,u\rangle = h_K(u)$')
    for tip, color, lw in [(h * u, pl.RECON, 2), (0.45 * u, pl.INK, 2.4)]:
        ax.annotate(
            '',
            xy=tip,
            xytext=(0, 0),
            arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=16),
        )
    ax.text(*(0.22 * u + 0.1 * t), r'$u$', fontsize=15, ha='center', va='center')
    ax.text(
        *(0.72 * h * u - 0.14 * t),
        r'$h_K(u)$',
        fontsize=15,
        color=pl.RECON,
        ha='center',
        va='center',
    )
    ax.scatter(*x, color=pl.MEAS, s=40, zorder=6)
    ax.scatter(0, 0, color=pl.INK, s=18, zorder=6)
    ax.text(0.04, -0.12, 'origin', fontsize=9, color=pl.SUBTLE)
    ax.legend(loc='lower left', frameon=False, fontsize=9.5)
    ax.set_title(r'Support function: $h_K(u)=\max_{x\in K}\ \langle x,u\rangle$')
    fig.savefig(OUT / 'concept.png')


def hero() -> None:
    """Reconstructions from increasingly many noisy measurements."""
    K, sigma = rounded_heptagon(), 0.05
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for ax, k in zip(axes, [10, 40, 300], strict=True):
        U = cr.circle_directions(k, offset=0.3)
        y = cr.measure(K, U, sigma, rng=k)
        rec = cr.reconstruct(U, y)
        pl.setup_2d(ax)
        pl.draw_measurements(ax, U, y)
        pl.draw_estimate(ax, rec.polytope)
        pl.draw_body(ax, K)
        ax.set_title(f'k = {k} measurements')
        pl.caption(ax, f'Hausdorff error {cr.hausdorff_distance(K, rec.polytope):.3f}')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9.5, bbox_to_anchor=(-0.02, 1.0))
    fig.suptitle(
        rf'Least-squares reconstruction from noisy support values  ($\sigma$ = {sigma})',
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(wspace=0.02, top=0.86)
    fig.savefig(OUT / 'reconstruction_2d.png')


def outliers() -> None:
    """Least squares vs least absolute deviations with gross outliers."""
    K, k, sigma = rounded_heptagon(), 160, 0.02
    U = cr.circle_directions(k)
    rng = np.random.default_rng(3)
    y = cr.measure(K, U, sigma, rng=rng)
    bad = rng.choice(k, size=k // 10, replace=False)
    y[bad] += rng.uniform(0.4, 0.9, bad.size)
    good = np.setdiff1d(np.arange(k), bad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.1))
    panels = [
        ('lsq', pl.RECON, 'Least squares (QP)'),
        ('lad', pl.LAD, 'Least absolute deviations (LP)'),
    ]
    for ax, (method, color, name) in zip(axes, panels, strict=True):
        rec = cr.reconstruct(U, y, method=method)
        pl.setup_2d(ax, 1.9)
        pl.draw_measurements(ax, U[good], y[good], alpha=0.35)
        pl.draw_measurements(ax, U[bad], y[bad], color=pl.BAD, alpha=0.9, label='outlier (10%)')
        pl.draw_estimate(ax, rec.polytope, color=color)
        pl.draw_body(ax, K)
        ax.set_title(name)
        pl.caption(ax, f'Hausdorff error {cr.hausdorff_distance(K, rec.polytope):.3f}')
    axes[0].legend(loc='upper left', frameon=False, fontsize=9.5)
    fig.suptitle(f'Robustness: {k} measurements, 10% of them corrupted', fontsize=13)
    fig.savefig(OUT / 'outliers.png')


def three_d() -> None:
    """True body next to reconstructions from few and many directions."""
    K, sigma = rounded_polytope(), 0.03
    fig = plt.figure(figsize=(13.5, 4.1))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    pl.draw_polytope_3d(ax, cr.to_polytope(K), pl.TRUTH_3D, 'true body\n(rounded polytope)', edges=False)
    for i, k in enumerate([60, 600]):
        U = cr.sphere_directions(k)
        y = cr.measure(K, U, sigma, rng=k)
        start = time.perf_counter()
        rec = cr.reconstruct(U, y)
        dt = time.perf_counter() - start
        took = f'{dt * 1000:.0f} ms' if dt < 1 else f'{dt:.1f} s'
        err = cr.hausdorff_distance(K, rec.polytope)
        ax = fig.add_subplot(1, 3, i + 2, projection='3d')
        title = f'k = {k} directions\nHausdorff error {err:.3f}  ·  solved in {took}'
        pl.draw_polytope_3d(ax, rec.polytope, pl.RECON, title)
    fig.subplots_adjust(wspace=-0.05, left=0, right=1, top=1.0, bottom=-0.08)
    fig.savefig(OUT / 'reconstruction_3d.png')


def animation() -> None:
    """GIF of the 2D reconstruction as k grows from 5 to 400."""
    K, sigma = rounded_heptagon(), 0.05
    frames = []
    for k in np.unique(np.round(np.geomspace(5, 400, 48)).astype(int)):
        U = cr.circle_directions(int(k), offset=0.3)
        y = cr.measure(K, U, sigma, rng=int(k))
        P = cr.reconstruct(U, y).polytope
        frames.append((U, y, P, cr.hausdorff_distance(K, P)))
    frames += [frames[-1]] * 12  # pause on the last frame

    fig, ax = plt.subplots(figsize=(5, 5.2), dpi=90)

    def draw(i: int) -> None:
        U, y, P, err = frames[i]
        ax.clear()
        pl.setup_2d(ax)
        pl.draw_measurements(ax, U, y)
        pl.draw_estimate(ax, P)
        pl.draw_body(ax, K)
        ax.set_title(f'k = {len(y)}   Hausdorff error {err:.3f}')

    FuncAnimation(fig, draw, frames=len(frames)).save(OUT / 'reconstruction_2d.gif', writer=PillowWriter(fps=8))


FIGURES = {
    'concept': concept,
    'hero': hero,
    'outliers': outliers,
    '3d': three_d,
    'gif': animation,
}


def main(names: list[str]) -> None:
    pl.use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names or FIGURES:
        start = time.perf_counter()
        FIGURES[name]()
        plt.close('all')
        print(f'{name}: {time.perf_counter() - start:.1f} s')


if __name__ == '__main__':
    main(sys.argv[1:])
