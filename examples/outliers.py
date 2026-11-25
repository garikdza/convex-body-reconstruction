"""
Least squares against least absolute deviations when 10% of 160 measurements are pushed far outward
(``figures/outliers.png``).

Run with ``uv run --extra examples python examples/outliers.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from draw import caption, draw_body, draw_estimate, draw_measurements, setup_2d
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, measure, reconstruct
from style import LAD, LSQ, MINIMAL, OUTLIER

K_MEASUREMENTS = 160
SIGMA = 0.02
OUTLIER_SHARE = 0.1
# outliers add a uniform draw from this range to the measurement
OUTLIER_SHIFT = (0.4, 0.9)
METHODS = (('lsq', LSQ, 'least squares (QP)'), ('lad', LAD, 'least absolute deviations (LP)'))
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    K = rounded_heptagon()
    U = circle_directions(K_MEASUREMENTS)
    rng = np.random.default_rng(3)
    y = measure(K, U, SIGMA, rng=rng)
    bad = rng.choice(K_MEASUREMENTS, size=int(OUTLIER_SHARE * K_MEASUREMENTS), replace=False)
    y[bad] += rng.uniform(*OUTLIER_SHIFT, bad.size)
    good = np.setdiff1d(np.arange(K_MEASUREMENTS), bad)

    with plt.rc_context(MINIMAL):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4.7))
        for ax, (method, colour, name) in zip(axes, METHODS, strict=True):
            P = reconstruct(U, y, method=method).polytope
            error = hausdorff_distance(K, P)
            print(f'{method}: Hausdorff error {error:.3f}')
            setup_2d(ax, 1.9)
            first = method == 'lsq'  # label the shared elements once, in the first panel
            draw_measurements(ax, U[good], y[good], alpha=0.35, label='measurement' if first else None)
            outlier_label = f'outlier ({OUTLIER_SHARE:.0%})' if first else None
            draw_measurements(ax, U[bad], y[bad], color=OUTLIER, alpha=0.9, label=outlier_label)
            draw_estimate(ax, P, colour, label=f'{method.upper()} estimate')
            draw_body(ax, K, label='true body' if first else None)
            ax.set_title(name)
            caption(ax, f'Hausdorff error {error:.3f}')
        fig.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))

        path = FIGURES / 'outliers.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
