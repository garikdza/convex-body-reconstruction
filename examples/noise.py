"""
Reconstruction error against the noise level σ with k = 400 fixed directions, for least squares and least absolute
deviations (``figures/noise.png``).

The errors are saved to ``figures/noise.npz`` and later runs plot from them; ``--recompute`` fits again.

Run with ``uv run --extra examples python examples/noise.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from results import load_or_compute, parse_args
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, l2_distance, noisy_support, reconstruct
from style import LAD, LSQ, MINIMAL

K_MEASUREMENTS = 400
SIGMAS = np.geomspace(0.005, 0.5, 11)
DRAWS = 10
# the seed of draw j at noise level a is SEED_STRIDE * a + j
SEED_STRIDE = 1000
FIGSIZE = (10, 4.2)
# opacity of the interquartile bands and size of the median dots
BAND_ALPHA = 0.15
DOT_SIZE = 20
# the dots go above the lines (zorder 2)
DOT_Z = 3
METHODS = (('lsq', LSQ, 'least squares'), ('lad', LAD, 'least absolute deviations'))
ERRORS = (('l2', 'L2 error'), ('hausdorff', 'Hausdorff error'))
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'noise.npz'


def compute() -> dict[str, np.ndarray]:
    """L2 and Hausdorff errors of every method and draw, shape ``(len(SIGMAS), DRAWS)`` per method."""
    body = rounded_heptagon()
    directions = circle_directions(K_MEASUREMENTS)
    results = {'sigmas': SIGMAS}
    for method, _, _ in METHODS:
        l2, hausdorff = np.zeros((len(SIGMAS), DRAWS)), np.zeros((len(SIGMAS), DRAWS))
        for a, sigma in enumerate(SIGMAS):
            for draw in range(DRAWS):
                # the same noise for both methods, so they are compared on identical measurements
                polytope = reconstruct(
                    directions, noisy_support(body, directions, sigma, rng=SEED_STRIDE * a + draw), method=method
                )
                l2[a, draw] = l2_distance(body, polytope)
                hausdorff[a, draw] = hausdorff_distance(body, polytope)
            print(f'{method} σ={sigma:<6.3f} L2 {np.median(l2[a]):.4f}  Hausdorff {np.median(hausdorff[a]):.4f}')
        results[f'l2_{method}'] = l2
        results[f'hausdorff_{method}'] = hausdorff
    return results


def plot(results: dict[str, np.ndarray]) -> None:
    """Draw the median errors with their interquartile bands on log axes, one panel per error measure."""
    sigmas = results['sigmas']
    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
        for ax, (error, title) in zip(axes, ERRORS, strict=True):
            for method, colour, name in METHODS:
                errors = results[f'{error}_{method}']
                q1, median, q3 = np.percentile(errors, [25, 50, 75], axis=1)
                ax.fill_between(sigmas, q1, q3, color=colour, alpha=BAND_ALPHA, linewidth=0)
                ax.plot(sigmas, median, color=colour, label=name)
                ax.scatter(sigmas, median, s=DOT_SIZE, color=colour, zorder=DOT_Z)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('noise σ')
            ax.set_ylabel(f'{title} (median and IQR, {errors.shape[1]} draws)')
            ax.set_title(f'{title}, k = {K_MEASUREMENTS}')
        axes[0].legend()
        fig.tight_layout()

        path = FIGURES / 'noise.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
