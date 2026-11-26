"""
A rounded 3D polytope beside its reconstructions from 60 and 600 noisy measurements
(``figures/reconstruction_3d.png``).

The vertices of the estimates are saved to ``figures/reconstruction_3d.npz`` and later runs plot from them;
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
from src import Polytope, hausdorff_distance, noisy_support, reconstruct, sphere_directions
from style import LSQ, LSQ_EDGE, MINIMAL, TRUTH_3D

KS = (60, 600)
SIGMA = 0.03
FIGSIZE = (12, 4)
# negative spacing and bottom margin pull the panels together and crop the empty space of the 3D axes
MARGINS = {'wspace': -0.05, 'left': 0, 'right': 1, 'top': 1, 'bottom': -0.08}
FIGURES = Path(__file__).parent / 'figures'
RESULTS = FIGURES / 'reconstruction_3d.npz'


def compute() -> dict[str, np.ndarray]:
    """Fit each ``k`` in ``KS`` and keep the vertices of each estimate and the solve time."""
    body = rounded_polytope()
    results = {'ks': np.array(KS)}
    for k in KS:
        directions = sphere_directions(k)
        start = time.perf_counter()
        polytope = reconstruct(directions, noisy_support(body, directions, SIGMA, rng=k))
        results[f'seconds_{k}'] = np.array(time.perf_counter() - start)
        results[f'vertices_{k}'] = polytope.vertices
    return results


def plot(results: dict[str, np.ndarray]) -> None:
    """Rebuild each estimate from its saved vertices and draw it beside the true body."""
    body = rounded_polytope()
    panels = len(results['ks']) + 1  # the true body, then one estimate per k
    with plt.rc_context(MINIMAL):
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot(1, panels, 1, projection='3d')
        draw_polytope_3d(ax, body.to_polytope(), TRUTH_3D)
        ax.set_title('true body\n')
        for i, k in enumerate(results['ks']):
            polytope = Polytope(results[f'vertices_{k}'])
            seconds = float(results[f'seconds_{k}'])
            error = hausdorff_distance(body, polytope)
            print(f'k={k:<4} Hausdorff error {error:.3f}, solved in {seconds:.2f} s')
            ax = fig.add_subplot(1, panels, i + 2, projection='3d')
            draw_polytope_3d(ax, polytope, LSQ, edge_color=LSQ_EDGE)
            took = f'{seconds * 1000:.0f} ms' if seconds < 1 else f'{seconds:.1f} s'
            ax.set_title(f'k = {k} measurements\nHausdorff error {error:.3f}, solved in {took}')
        fig.subplots_adjust(**MARGINS)

        path = FIGURES / 'reconstruction_3d.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


def main() -> None:
    args = parse_args(__doc__)
    plot(load_or_compute(RESULTS, compute, args.recompute))


if __name__ == '__main__':
    main()
