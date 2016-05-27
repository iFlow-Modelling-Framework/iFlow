"""
show

Date: 21-Feb-16
Authors: Y.M. Dijkstra
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import step_config as cf

def show(block=True, hspace=0.6, wspace=0.4):
    figures=[manager.canvas.figure for manager in mpl._pylab_helpers.Gcf.get_all_fig_managers()]

    for fig in figures:
        # set size
        mng = fig.canvas.manager
        stdsize = fig.get_size_inches()
        fig.set_size_inches(stdsize[1]*cf.wunit, stdsize[0]*cf.hunit)#, forward=True)
        mng.resize(stdsize[1]*cf.wunit*cf.dpi, stdsize[0]*cf.hunit*cf.dpi)
        fig.set_dpi(cf.dpi)

        # colors
        fig.set_facecolor('w')
        fig.set_edgecolor('k')

        if fig._suptitle is not None:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
    plt.show(block=block)

    return