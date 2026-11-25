"""
Reconstruction error against the number of measurements ``k``, in 2D and 3D, beside the rate Gardner, Kiderlen &
Milanfar (2006) prove for the L2 error: O(k^-2/5) in 2D and O(k^-1/3) in 3D (``figures/convergence.png``).

The errors are saved to ``figures/convergence.npz`` and later runs plot from them; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/convergence.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results import load_or_compute, parse_args
from shapes import rounded_heptagon, rounded_polytope
from src import evenly_spread, hausdorff_distance, l2_distance, measure, reconstruct
from style import HAUSDORFF_ERROR, INK, L2_ERROR, MINIMAL

SIGMA = 0.05
# per dimension: the k values, the draws per k, and the proven L2 rate as an exponent and as a label
SETUPS = {
    2: ((8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096), 20, 2 / 5, '2/5'),
    3: ((16, 32, 64, 128, 256, 512), 6, 1 / 3, '1/3'),
}
# the fitted slopes skip the smallest k values, before the error settles into its rate
FIT_FROM = 2
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'convergence.npz'


def compute() -> dict[str, np.ndarray]:
    """L2 and Hausdorff errors of every draw, shape ``(len(ks), draws)`` per dimension."""
    bodies = {2: rounded_heptagon(), 3: rounded_polytope()}
    results = {}
    for dim, (ks, draws, _, _) in SETUPS.items():
        K = bodies[dim]
        l2, hausdorff = np.zeros((len(ks), draws)), np.zeros((len(ks), draws))
        for a, k in enumerate(ks):
            for draw in range(draws):
                rng = np.random.default_rng(1000 * k + draw)
                # evenly spread directions, randomly rotated for each draw
                Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
                U = evenly_spread(k, dim) @ Q
                P = reconstruct(U, measure(K, U, SIGMA, rng=rng)).polytope
                l2[a, draw] = l2_distance(K, P)
                hausdorff[a, draw] = hausdorff_distance(K, P)
            print(f'{dim}D k={k:<5} L2 {np.median(l2[a]):.4f}  Hausdorff {np.median(hausdorff[a]):.4f}')
        results[f'ks_{dim}d'] = np.array(ks)
        results[f'l2_{dim}d'] = l2
        results[f'hausdorff_{dim}d'] = hausdorff
    return results


def plot(results: dict[str, np.ndarray]) -> None:
    """Draw the median errors with their interquartile bands on log axes, one panel per dimension."""
    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, (dim, (_, draws, rate, rate_label)) in zip(axes, SETUPS.items(), strict=True):
            ks = results[f'ks_{dim}d']
            for name, colour in [('l2', L2_ERROR), ('hausdorff', HAUSDORFF_ERROR)]:
                errors = results[f'{name}_{dim}d']
                median = np.median(errors, axis=1)
                q1, q3 = np.percentile(errors, [25, 75], axis=1)
                slope = np.polyfit(np.log(ks[FIT_FROM:]), np.log(median[FIT_FROM:]), 1)[0]
                print(f'{dim}D {name}: fitted slope {slope:.2f}')
                label = f'{"L2" if name == "l2" else "Hausdorff"} error (slope {slope:.2f})'
                ax.fill_between(ks, q1, q3, color=colour, alpha=0.12, linewidth=0)
                ax.plot(ks, median, color=colour, label=label)
                ax.scatter(ks, median, s=24, color=colour, zorder=3)
            # the proven rate is an upper bound, so its line is drawn above the L2 errors
            reference = 1.6 * np.median(results[f'l2_{dim}d'], axis=1)[1] * (ks / ks[1]) ** -rate
            ax.plot(ks, reference, '--', color=INK, linewidth=1.2, label=rf'proven L2 rate $k^{{-{rate_label}}}$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xticks(ks, [str(k) for k in ks])
            ax.minorticks_off()
            ax.set_xlabel('number of measurements k')
            ax.set_ylabel(f'error (median and IQR, {draws} draws)')
            ax.set_title(f'{dim}D, σ = {SIGMA}')
            ax.legend()
        fig.tight_layout()

        path = FIGURES / 'convergence.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
