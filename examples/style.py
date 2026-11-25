"""
The minimal figure style of the example scripts: matplotlib settings and colours, with no matplotlib import.

A script draws inside ``plt.rc_context(MINIMAL)`` and takes its colours from here. The style uses a light surface,
no top or right spines, a solid hairline grid on y only, left-aligned bold titles and frameless legends.
"""

SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_SOFT = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'

# the roles every figure shares: the true body, the measurements and the estimates of the two methods
TRUTH = INK
TRUTH_3D = '#8c8b84'
MEASUREMENT = '#eb6834'
OUTLIER = '#c0392b'
LSQ = '#2a78d6'
LAD = '#1baf7a'
# the two error measures of the convergence figure
L2_ERROR = '#2a78d6'
HAUSDORFF_ERROR = '#eb6834'

MINIMAL: dict[str, object] = {
    'figure.facecolor': SURFACE,
    'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.edgecolor': AXIS,
    'axes.linewidth': 0.8,
    'axes.labelcolor': INK_SOFT,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.color': GRID,
    'grid.linewidth': 0.8,
    'grid.linestyle': '-',
    'axes.axisbelow': True,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'xtick.labelcolor': INK_SOFT,
    'ytick.labelcolor': INK_SOFT,
    'xtick.major.size': 0,
    'ytick.major.size': 0,
    'xtick.minor.size': 0,
    'ytick.minor.size': 0,
    'text.color': INK,
    'axes.titlecolor': INK,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.titlelocation': 'left',
    'axes.labelsize': 9.5,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.frameon': False,
    'legend.fontsize': 8.5,
    'legend.labelcolor': INK_SOFT,
    'legend.title_fontsize': 8.5,
    'lines.linewidth': 2,
    'lines.solid_capstyle': 'round',
    'font.family': 'DejaVu Sans',
}
