"""
show

Date: 21-Feb-16
Authors: Y.M. Dijkstra
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import step_config as cf


def show(block=True, fname=None, hspace=0.6, wspace=0.4):
    """Replaces matplotlib.pyplot show().
    Corrects figure size, backgroundcolor and tightlayout

    args:
        block: boolean

        fname: string
            filename of the figure. Must include file path and figure name. If provide the figure is saved.
    """
    nums = plt.get_fignums()
    for num in nums:
        fig = plt.figure(num=num)       # obtain figure reference from the fig number

        # reset the size and dpi (for display only)
        stdsize = plt.figure(num=num).get_size_inches()
        fig.set_dpi(cf.dpi)
        fig.set_size_inches(stdsize[1]*cf.wunit, stdsize[0]*cf.hunit, forward=True) # NB forward=True forwards the change in size and dpi to the plotting window
        plt.draw()
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
        if fig._suptitle is not None:
            fig.set_tight_layout({'rect':[0, 0, 1, 0.95]})
        else:
            fig.set_tight_layout(True)

    if fname != None:
        plt.savefig(fname, bbox_inches='tight', pad_inches=0.1, dpi=cf.savedpi)
    plt.show(block=block)

    return