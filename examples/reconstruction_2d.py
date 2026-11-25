"""
Least-squares reconstructions of a rounded heptagon from 10, 40 and 300 noisy measurements
(``figures/reconstruction_2d.png``).

Run with ``uv run --extra examples python examples/reconstruction_2d.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from draw import caption, draw_body, draw_estimate, draw_measurements, setup_2d
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, measure, reconstruct
from style import LSQ, MINIMAL

KS = (10, 40, 300)
SIGMA = 0.05
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    K = rounded_heptagon()
    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, len(KS), figsize=(12, 4.3))
        for ax, k in zip(axes, KS, strict=True):
            U = circle_directions(k, offset=0.3)
            y = measure(K, U, SIGMA, rng=k)
            P = reconstruct(U, y).polytope
            error = hausdorff_distance(K, P)
            print(f'k={k:<4} Hausdorff error {error:.3f}')
            setup_2d(ax)
            draw_measurements(ax, U, y)
            draw_estimate(ax, P, LSQ)
            draw_body(ax, K)
            ax.set_title(f'k = {k} measurements')
            caption(ax, f'Hausdorff error {error:.3f}')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))
        fig.subplots_adjust(wspace=0.02)

        path = FIGURES / 'reconstruction_2d.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
