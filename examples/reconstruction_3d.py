"""
A rounded 3D polytope beside its reconstructions from 60 and 600 noisy measurements
(``figures/reconstruction_3d.png``).

The fitted support values are saved to ``figures/reconstruction_3d.npz`` and later runs plot from them;
``--recompute`` fits again.

Run with ``uv run --extra examples python examples/reconstruction_3d.py``.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from draw import draw_polytope_3d
from results import load_or_compute, parse_args
from shapes import rounded_polytope
from src import Polytope, hausdorff_distance, measure, reconstruct, sphere_directions, to_polytope
from style import LSQ, MINIMAL, TRUTH_3D

KS = (60, 600)
SIGMA = 0.03
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'reconstruction_3d.npz'


def compute() -> dict[str, np.ndarray]:
    """Fit each ``k`` in ``KS`` and keep the directions, the fitted support values and the solve time."""
    K = rounded_polytope()
    results = {'ks': np.array(KS)}
    for k in KS:
        U = sphere_directions(k)
        start = time.perf_counter()
        rec = reconstruct(U, measure(K, U, SIGMA, rng=k))
        results[f'seconds_{k}'] = np.array(time.perf_counter() - start)
        results[f'directions_{k}'] = U
        results[f'support_values_{k}'] = rec.support_values
    return results


def plot(results: dict[str, np.ndarray]) -> None:
    """Rebuild each estimate from its saved support values and draw it beside the true body."""
    K = rounded_polytope()
    with plt.rc_context(MINIMAL):
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_subplot(1, 3, 1, projection='3d')
        draw_polytope_3d(ax, to_polytope(K), TRUTH_3D)
        ax.set_title('true body\n')
        for i, k in enumerate(results['ks']):
            P = Polytope.from_halfspaces(results[f'directions_{k}'], results[f'support_values_{k}'])
            seconds = float(results[f'seconds_{k}'])
            error = hausdorff_distance(K, P)
            print(f'k={k:<4} Hausdorff error {error:.3f}, solved in {seconds:.2f} s')
            ax = fig.add_subplot(1, 3, i + 2, projection='3d')
            draw_polytope_3d(ax, P, LSQ, edge_color='#0d366b')
            took = f'{seconds * 1000:.0f} ms' if seconds < 1 else f'{seconds:.1f} s'
            ax.set_title(f'k = {k} measurements\nHausdorff error {error:.3f}, solved in {took}')
        fig.subplots_adjust(wspace=-0.05, left=0, right=1, top=1, bottom=-0.08)

        path = FIGURES / 'reconstruction_3d.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
