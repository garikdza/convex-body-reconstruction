"""Benchmark: reconstruction error as the number of measurements k grows.

Gardner, Kiderlen & Milanfar (2006) prove the L2 error is O(k^-2/5) in 2D and O(k^-1/3) in
3D. This script measures it and plots both against the proven rate.

    uv run python examples/convergence.py      # ~1.5 min, writes docs/img/convergence.png
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from shapes import rounded_heptagon, rounded_polytope  # noqa: E402

import src as cr  # noqa: E402
from src import plotting as pl  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "img"
SIGMA = 0.05

# (dimension, body, k values, trials per k, proven L2 rate, rate label)
SETUPS = [
    (2, rounded_heptagon(), [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096], 20, 2 / 5, "2/5"),
    (3, rounded_polytope(), [16, 32, 64, 128, 256, 512], 6, 1 / 3, "1/3"),
]


def errors(K: cr.ConvexBody, dim: int, ks, trials: int) -> tuple[np.ndarray, np.ndarray]:
    """L2 and Hausdorff errors, shape ``(len(ks), trials)`` each."""
    l2 = np.zeros((len(ks), trials))
    hd = np.zeros_like(l2)
    for a, k in enumerate(ks):
        for t in range(trials):
            rng = np.random.default_rng(1000 * k + t)
            # evenly spread directions, randomly rotated each trial
            Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
            U = cr.evenly_spread(k, dim) @ Q
            rec = cr.reconstruct(U, cr.measure(K, U, SIGMA, rng=rng))
            l2[a, t] = cr.l2_distance(K, rec.polytope)
            hd[a, t] = cr.hausdorff_distance(K, rec.polytope)
        print(f"{dim}D k={k}: L2 {np.median(l2[a]):.4f}  Hausdorff {np.median(hd[a]):.4f}")
    return l2, hd


def plot_panel(ax, ks, l2, hd, dim, trials, rate, rate_label) -> None:
    ks = np.asarray(ks)
    for err, color, name in [(l2, pl.RECON, "L2 error"), (hd, pl.MEAS, "Hausdorff error")]:
        med = np.median(err, axis=1)
        lo, hi = np.percentile(err, [25, 75], axis=1)
        slope = np.polyfit(np.log(ks[2:]), np.log(med[2:]), 1)[0]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(
            ks, med, "o-", color=color, lw=2, ms=4.5, label=f"{name}  (fitted slope {slope:.2f})"
        )
    # reference line with the proven slope, drawn above the data since it is an upper bound
    ref = np.median(l2, axis=1)[1] * (ks / ks[1]) ** (-rate) * 1.6
    ax.plot(
        ks,
        ref,
        "--",
        color=pl.INK,
        lw=1.2,
        label=rf"proven upper-bound rate $k^{{-{rate_label}}}$ (L2)",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of measurements k")
    ax.set_ylabel("error (median, IQR band)")
    ax.set_title(f"{dim}D, $\\sigma$ = {SIGMA}, {trials} trials per k")
    ax.grid(True, which="major", color="#e2e8f0", lw=0.8)
    ax.legend(frameon=False, fontsize=9.5)


def main() -> None:
    pl.use_style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (dim, K, ks, trials, rate, label) in zip(axes, SETUPS, strict=True):
        l2, hd = errors(K, dim, ks, trials)
        plot_panel(ax, ks, l2, hd, dim, trials, rate, label)
    fig.suptitle("Convergence as the number of measurements grows", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "convergence.png")


if __name__ == "__main__":
    main()
