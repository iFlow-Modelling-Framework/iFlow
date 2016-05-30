"""
show

Date: 21-Feb-16
Authors: Y.M. Dijkstra
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import step_config as cf

def show(block=True, hspace=0.6, wspace=0.4):
    """Replaces matplotlib.pyplot show().
    Corrects figure size, backgroundcolor and tightlayout

    Args:
        block:
    """
    figures=[manager.canvas.figure for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]

    for fig in figures:
        # axis ticks and legend frame line width
        for ax in fig.axes:
            ax.tick_params(axis='x', which='both', top='off')
            ax.tick_params(axis='y', which='both', right='off')
            try:
                ax.get_legend().get_frame().set_linewidth(0.5)
            except:
                pass

        # colors
        fig.set_facecolor('w')
        fig.set_edgecolor('k')

        # tight layout
        fig.set_tight_layout(True)
    plt.show(block=block)

    return