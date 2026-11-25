"""
What one measurement is: a convex body, a direction ``u`` and the support function ``h_K(u)``, the signed distance from
the origin to the supporting line with outer normal ``u`` (``figures/support_function.png``).

Run with ``uv run --extra examples python examples/support_function.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from draw import draw_body, setup_2d
from shapes import rounded_heptagon
from style import INK, INK_SOFT, LSQ, MEASUREMENT, MINIMAL

ANGLE = np.deg2rad(38)
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    K = rounded_heptagon()
    u = np.array([np.cos(ANGLE), np.sin(ANGLE)])
    h = K.support(u[None])[0]
    contact = K.support_points(u[None])[0]
    t = np.array([-u[1], u[0]])  # along the supporting line

    with plt.rc_context(MINIMAL):
        fig, ax = plt.subplots(figsize=(5, 5))
        setup_2d(ax, 1.45)
        draw_body(ax, K, label='convex body K', fill=0.05)
        line = np.stack([h * u - 1.0 * t, h * u + 0.75 * t])
        ax.plot(*line.T, color=MEASUREMENT, label=r'supporting line $\langle x, u\rangle = h_K(u)$')
        for tip, colour in [(h * u, LSQ), (0.45 * u, INK)]:
            ax.annotate('', xy=tip, xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': colour, 'lw': 2})
        ax.text(*(0.22 * u + 0.1 * t), r'$u$', fontsize=14, ha='center', va='center')
        ax.text(*(0.72 * h * u - 0.15 * t), r'$h_K(u)$', fontsize=14, color=LSQ, ha='center', va='center')
        ax.scatter(*contact, color=MEASUREMENT, s=36, zorder=6)
        ax.scatter(0, 0, color=INK, s=16, zorder=6)
        ax.text(0.04, -0.12, 'origin', fontsize=9, color=INK_SOFT)
        ax.legend(loc='lower left')
        ax.set_title(r'support function $h_K(u) = \max_{x \in K} \langle x, u\rangle$')

        path = FIGURES / 'support_function.png'
        fig.savefig(path)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
