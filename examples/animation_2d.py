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
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from draw import (
    BODY_WIDTH,
    ESTIMATE_FILL,
    ESTIMATE_WIDTH,
    MEASUREMENT_WIDTH,
    draw_body,
    draw_estimate,
    draw_measurements,
    setup_2d,
)
from shapes import rounded_heptagon
from src import circle_directions, hausdorff_distance, l2_distance, noisy_support, reconstruct
from style import HAUSDORFF_ERROR, INK, L2_ERROR, LSQ, MEASUREMENT, MINIMAL, TRUTH

KS = np.unique(np.round(np.geomspace(5, 400, 48)).astype(int))
SIGMA = 0.05
# angle of the first direction, in radians
OFFSET = 0.3
FPS = 10
# at the end: seconds paused on the last frame, seconds to fade the measurements out, seconds holding the clean view
PAUSE, FADE, HOLD = 2.0, 1.5, 2.5
# the proven L2 rate k^-RATE, drawn through the error from k = RATE_FROM on
RATE = 0.4
RATE_FROM = 10
FIGSIZE = (10, 5)
WIDTH_RATIOS = (1, 1.05)
# fixed margins: the axes are redrawn every frame, so a layout computed once would not fit later frames
MARGINS = {'left': 0.02, 'right': 0.98, 'top': 0.9, 'bottom': 0.12, 'wspace': 0.18}
DPI = 90
# the left legend sits centred just below the axes
LEGEND_ANCHOR = (0.5, 0.0)
LEGEND_HANDLE_LENGTH = 1.6
LEGEND_MARKER_SIZE = 3
# the error panel: reference line width, size of the dot at the current k, ticks, and the factors that pad the
# axis limits beyond the data
REFERENCE_WIDTH = 1.2
DOT_SIZE = 30
# the dots go above the lines (zorder 2)
DOT_Z = 3
TICKS = (5, 10, 20, 50, 100, 200, 400)
X_PAD = (0.9, 1.1)
Y_PAD = (0.7, 1.3)
FIGURES = Path(__file__).parent / 'figures'


def main() -> None:
    body = rounded_heptagon()
    fits = []
    for k in KS:
        directions = circle_directions(int(k), offset=OFFSET)
        y = noisy_support(body, directions, SIGMA, rng=int(k))
        polytope = reconstruct(directions, y)
        fits.append((directions, y, polytope))
    l2 = np.array([l2_distance(body, polytope) for _, _, polytope in fits])
    hausdorff = np.array([hausdorff_distance(body, polytope) for _, _, polytope in fits])

    # each frame is (index of the fit, opacity of the measurements)
    last = len(KS) - 1
    frames = [(i, 1.0) for i in range(len(KS))]
    frames += [(last, 1.0)] * round(PAUSE * FPS)
    frames += [(last, f) for f in np.linspace(1, 0, round(FADE * FPS) + 1)[1:]]
    frames += [(last, 0.0)] * round(HOLD * FPS)

    # the proven rate is an upper bound, so its line is the lowest one above the L2 errors from k = RATE_FROM on,
    # once the error has settled into its rate
    reference = KS**-RATE * np.max((l2 * KS**RATE)[KS >= RATE_FROM])

    with plt.rc_context(MINIMAL):
        fig, (left, right) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={'width_ratios': WIDTH_RATIOS})
        fig.subplots_adjust(**MARGINS)

        # fixed legend handles: the drawn measurement lines get fainter as k grows and would barely show in a legend
        handles = [
            Line2D([], [], color=TRUTH, linewidth=BODY_WIDTH, label='true body'),
            Patch(
                facecolor=to_rgba(LSQ, ESTIMATE_FILL), edgecolor=LSQ, linewidth=ESTIMATE_WIDTH, label='reconstruction'
            ),
            Line2D(
                [],
                [],
                color=MEASUREMENT,
                linewidth=MEASUREMENT_WIDTH,
                marker='o',
                markersize=LEGEND_MARKER_SIZE,
                label='measurement',
            ),
        ]

        def draw(frame: tuple[int, float]) -> None:
            i, fade = frame
            handles[2].set_alpha(fade)
            directions, y, polytope = fits[i]
            left.clear()
            setup_2d(left)
            if fade > 0:
                draw_measurements(left, directions, y, fade=fade)
            draw_estimate(left, polytope, LSQ)
            draw_body(left, body)
            left.set_title(f'k = {len(y)} measurements')
            # the measurement entry fades with the measurements and goes once they have
            shown = handles if fade > 0 else handles[:2]
            legend = left.legend(
                handles=shown,
                loc='upper center',
                bbox_to_anchor=LEGEND_ANCHOR,
                ncol=len(shown),
                handlelength=LEGEND_HANDLE_LENGTH,
            )
            if fade > 0:
                legend.get_texts()[2].set_alpha(fade)

            right.clear()
            right.plot(KS, reference, '--', color=INK, linewidth=REFERENCE_WIDTH, label=r'proven L2 rate $k^{-2/5}$')
            for errors, name, colour in [(l2, 'L2', L2_ERROR), (hausdorff, 'Hausdorff', HAUSDORFF_ERROR)]:
                right.plot(KS[: i + 1], errors[: i + 1], color=colour, label=f'{name} error {errors[i]:.3f}')
                right.scatter(KS[i], errors[i], s=DOT_SIZE, color=colour, zorder=DOT_Z)
            right.set_xscale('log')
            right.set_yscale('log')
            right.set_xlim(KS[0] * X_PAD[0], KS[-1] * X_PAD[1])
            right.set_ylim(min(l2.min(), hausdorff.min()) * Y_PAD[0], max(reference.max(), hausdorff.max()) * Y_PAD[1])
            right.set_xticks(TICKS, [str(t) for t in TICKS])
            right.minorticks_off()
            right.set_xlabel('number of measurements k')
            right.set_ylabel(f'error, σ = {SIGMA}')
            right.set_title('error against k')
            right.legend(loc='lower left')

        path = FIGURES / 'animation_2d.gif'
        FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=FPS), dpi=DPI)
        plt.close(fig)
    print(f'wrote {path}')


if __name__ == '__main__':
    main()
