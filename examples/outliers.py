"""
Least squares against least absolute deviations when 10% of 160 measurements are pushed far outward
(``figures/outliers.png``).

Run with ``uv run --extra examples python examples/outliers.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from draw import draw_body, draw_estimate, draw_measurements, setup_2d
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, noisy_support, reconstruct
from style import LAD, LSQ, MINIMAL, OUTLIER

K_MEASUREMENTS = 160
SIGMA = 0.02
OUTLIER_SHARE = 0.1
# each outlier adds a uniform draw from this range to its measurement
OUTLIER_SHIFT = (0.4, 0.9)
SEED = 3
FIGSIZE = (6, 6.4)
# half the side of the square shown, wide enough for the outliers
LIM = 1.9
# opacity of the measurement lines, of the outlier lines and of the true body's fill
MEASUREMENT_ALPHA = 0.3
OUTLIER_ALPHA = 0.9
BODY_FILL = 0.06
# the legend sits centred just below the axes
LEGEND_ANCHOR = (0.5, 0.04)
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    body = rounded_heptagon()
    directions = circle_directions(K_MEASUREMENTS)
    rng = np.random.default_rng(SEED)
    y = noisy_support(body, directions, SIGMA, rng=rng)
    bad = np.zeros(K_MEASUREMENTS, dtype=bool)
    bad[rng.choice(K_MEASUREMENTS, size=int(OUTLIER_SHARE * K_MEASUREMENTS), replace=False)] = True
    y[bad] += rng.uniform(*OUTLIER_SHIFT, bad.sum())

    with plt.rc_context(MINIMAL):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        setup_2d(ax, LIM)
        draw_measurements(ax, directions[~bad], y[~bad], alpha=MEASUREMENT_ALPHA)
        draw_measurements(
            ax, directions[bad], y[bad], color=OUTLIER, alpha=OUTLIER_ALPHA, label=f'outlier ({OUTLIER_SHARE:.0%})'
        )
        # the true body goes under the estimates, so the one that matches it stays visible
        draw_body(ax, body, fill=BODY_FILL, on_top=False)
        for method, colour, name in (('lsq', LSQ, 'least squares'), ('lad', LAD, 'least absolute deviations')):
            polytope = reconstruct(directions, y, method=method)
            error = hausdorff_distance(body, polytope)
            print(f'{method}: Hausdorff error {error:.3f}')
            draw_estimate(ax, polytope, colour, label=f'{name}, error {error:.2f}', fill=0)
        ax.set_title(f'{OUTLIER_SHARE:.0%} of the measurements are outliers')
        ax.legend(loc='upper center', bbox_to_anchor=LEGEND_ANCHOR, ncol=2)

        path = FIGURES / 'outliers.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
