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
    mpl.rcParams['legend.fontsize'] = cf.fontsize
    mpl.rcParams['savefig.dpi'] = cf.savedpi

    mpl.rcParams['axes.formatter.limits'] = [-axislimits, axislimits]
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    mpl.rcParams['font.family'] = cf.fontfamily
    return