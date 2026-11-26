"""
The 2D reconstruction as the number of measurements grows from 5 to 400, beside its error against k
(``figures/animation_2d.gif``).

The left panel shows the measurements, the estimate and the true body; the right panel traces the L2 and Hausdorff
errors of each estimate on log axes, beside the proven L2 rate k^-2/5. At the end the animation pauses, fades the
measurements out to leave the true body and the final estimate, and holds that view before it loops.

Run with ``uv run --extra examples python examples/animation_2d.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from draw import draw_body, draw_estimate, draw_measurements, setup_2d
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, l2_distance, noisy_support, reconstruct
from style import HAUSDORFF_ERROR, INK, L2_ERROR, LSQ, MEASUREMENT, MINIMAL, TRUTH

KS = np.unique(np.round(np.geomspace(5, 400, 48)).astype(int))
SIGMA = 0.05
FPS = 10
# at the end: seconds paused on the last frame, seconds to fade the measurements out, seconds holding the clean view
PAUSE, FADE, HOLD = 2.0, 1.5, 2.5
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    K = rounded_heptagon()
    fits = []
    for k in KS:
        U = circle_directions(int(k), offset=0.3)
        y = noisy_support(K, U, SIGMA, rng=int(k))
        P = reconstruct(U, y)
        fits.append((U, y, P))
    l2 = np.array([l2_distance(K, P) for _, _, P in fits])
    hausdorff = np.array([hausdorff_distance(K, P) for _, _, P in fits])

    # each frame is (index of the fit, opacity of the measurements)
    last = len(KS) - 1
    frames = [(i, 1.0) for i in range(len(KS))]
    frames += [(last, 1.0)] * round(PAUSE * FPS)
    frames += [(last, f) for f in np.linspace(1, 0, round(FADE * FPS) + 1)[1:]]
    frames += [(last, 0.0)] * round(HOLD * FPS)

    # the proven rate k^-2/5 is an upper bound, so its line is the lowest one above the L2 errors from k = 10 on,
    # once the error has settled into its rate
    reference = KS**-0.4 * np.max((l2 * KS**0.4)[KS >= 10])

    with plt.rc_context(MINIMAL):
        fig, (left, right) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 1.05]})
        # fixed margins: the axes are redrawn every frame, so a layout computed once would not fit later frames
        fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.12, wspace=0.18)

        # fixed legend handles: the drawn measurement lines get fainter as k grows and would barely show in a legend
        handles = [
            Line2D([], [], color=TRUTH, linewidth=1.8, label='true body'),
            Patch(facecolor=LSQ + '26', edgecolor=LSQ, linewidth=1.6, label='reconstruction'),
            Line2D([], [], color=MEASUREMENT, linewidth=1, marker='o', markersize=3, label='measurement'),
        ]

        def draw(frame: tuple[int, float]) -> None:
            i, fade = frame
            handles[2].set_alpha(fade)
            U, y, P = fits[i]
            left.clear()
            setup_2d(left)
            if fade > 0:
                draw_measurements(left, U, y, fade=fade)
            draw_estimate(left, P, LSQ)
            draw_body(left, K)
            left.set_title(f'k = {len(y)} measurements')
            # the measurement entry fades with the measurements and goes once they have
            shown = handles if fade > 0 else handles[:2]
            legend = left.legend(handles=shown, loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=3, handlelength=1.6)
            if fade > 0:
                legend.get_texts()[2].set_alpha(fade)

            right.clear()
            right.plot(KS, reference, '--', color=INK, linewidth=1.2, label=r'proven L2 rate $k^{-2/5}$')
            for errors, name, colour in [(l2, 'L2', L2_ERROR), (hausdorff, 'Hausdorff', HAUSDORFF_ERROR)]:
                right.plot(KS[: i + 1], errors[: i + 1], color=colour, label=f'{name} error {errors[i]:.3f}')
                right.scatter(KS[i], errors[i], s=30, color=colour, zorder=3)
            right.set_xscale('log')
            right.set_yscale('log')
            right.set_xlim(KS[0] * 0.9, KS[-1] * 1.1)
            right.set_ylim(min(l2.min(), hausdorff.min()) * 0.7, max(reference.max(), hausdorff.max()) * 1.3)
            ticks = [5, 10, 20, 50, 100, 200, 400]
            right.set_xticks(ticks, [str(t) for t in ticks])
            right.minorticks_off()
            right.set_xlabel('number of measurements k')
            right.set_ylabel(f'error, σ = {SIGMA}')
            right.set_title('error against k')
            right.legend(loc='lower left')

        path = FIGURES / 'animation_2d.gif'
        FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=FPS), dpi=90)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
