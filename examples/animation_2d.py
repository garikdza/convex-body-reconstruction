"""
The 2D reconstruction as the number of measurements grows from 5 to 400 (``figures/animation_2d.gif``).

Run with ``uv run --extra examples python examples/animation_2d.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from draw import draw_body, draw_estimate, draw_measurements, setup_2d
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, measure, reconstruct
from style import LSQ, MINIMAL

KS = np.unique(np.round(np.geomspace(5, 400, 48)).astype(int))
SIGMA = 0.05
# frames the last reconstruction is held for before the loop restarts
HOLD = 12
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    K = rounded_heptagon()
    frames = []
    for k in KS:
        U = circle_directions(int(k), offset=0.3)
        y = measure(K, U, SIGMA, rng=int(k))
        P = reconstruct(U, y).polytope
        frames.append((U, y, P, hausdorff_distance(K, P)))
    frames += [frames[-1]] * HOLD

    with plt.rc_context(MINIMAL):
        fig, ax = plt.subplots(figsize=(5, 5.2), dpi=90)

        def draw(i: int) -> None:
            U, y, P, error = frames[i]
            ax.clear()
            setup_2d(ax)
            draw_measurements(ax, U, y)
            draw_estimate(ax, P, LSQ)
            draw_body(ax, K)
            ax.set_title(f'k = {len(y)}    Hausdorff error {error:.3f}')

        path = FIGURES / 'animation_2d.gif'
        FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=PillowWriter(fps=8))
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
