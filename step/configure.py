"""
configure

Date: 21-Feb-16
Authors: Y.M. Dijkstra
"""
import matplotlib as mpl
import step_config as cf
from cycler import cycler


def configure(axislimits=3):
    """Configures font and dpi for saving of all plots.
    Run before making plots using matplotlib

    Parameters:
        axislimits (int, optional) - maximum number of decimals on axis
    """
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['font.size'] = cf.fontsize
    mpl.rcParams['axes.titlesize'] = cf.fontsize
    mpl.rcParams['axes.labelsize'] = cf.fontsize
    mpl.rcParams['savefig.dpi'] = cf.savedpi

    mpl.rcParams['axes.formatter.limits'] = [-axislimits, axislimits]
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['font.family'] = cf.fontfamily

    # outer frame size and tickmarks
    mpl.rcParams['axes.linewidth'] = 0.5
    mpl.rcParams['xtick.direction'] = 'out'
    mpl.rcParams['ytick.direction'] = 'out'
    mpl.rcParams['xtick.major.size'] = 3.
    mpl.rcParams['ytick.major.size'] = 3.

    # tick label font size
    mpl.rcParams['legend.fontsize'] = cf.fontsize2
    mpl.rcParams['xtick.labelsize'] = cf.fontsize2
    mpl.rcParams['ytick.labelsize'] = cf.fontsize2

    # Revert to matplotlib 1.. style instead of 2.. style (see https://matplotlib.org/users/dflt_style_changes.html)
    if int(mpl.__version__.split('.')[0]) > 1:
        mpl.rcParams['axes.prop_cycle'] = cycler(color='bgrcmyk')
        mpl.rcParams['image.cmap'] = 'jet'
        mpl.rcParams['lines.linewidth'] = 1.0
        # mpl.rcParams['lines.dashed_pattern'] = [6, 6]
        # mpl.rcParams['lines.dashdot_pattern'] = [3, 5, 1, 5]
        # mpl.rcParams['lines.dotted_pattern'] = [1, 3]
        # mpl.rcParams['lines.scale_dashes'] = False
        mpl.rcParams['mathtext.fontset'] = 'cm'
        mpl.rcParams['mathtext.rm'] = 'serif'
        # mpl.rcParams['axes.autolimit_mode'] = 'round_numbers'
        mpl.rcParams['axes.xmargin'] = 0
        mpl.rcParams['axes.ymargin'] = 0
        # mpl.rcParams['legend.fontsize'] = 'large'
        mpl.rcParams['figure.titlesize'] = 'medium'

    return
