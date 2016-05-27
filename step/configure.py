"""
configure

Date: 21-Feb-16
Authors: Y.M. Dijkstra
"""
import matplotlib as mpl
import step_config as cf


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

    return
