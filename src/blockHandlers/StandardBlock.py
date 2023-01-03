"""
Class StandardBlock
Implements the basic iFlowBlock (see iFlowBlock or the manual for documentation)


Date: 03-02-2022
Authors: Y.M. Dijkstra
"""
from src.blockHandlers.iFlowBlock import iFlowBlock


class StandardBlock(iFlowBlock):
    # Variables

    # Methods
    def __init__(self, callStack, memProfiler = None):
        iFlowBlock.__init__(self,callStack,memProfiler)
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def run(self, init=True, interrupt=False, startnumber=0):
        super().run(init=init, interrupt=interrupt, startnumber=startnumber)
        # Do a memory profile if profiler is set
        if self._memProfiler is not None:
            self._memProfiler.snapshot('Run %s'%self.getName(short=True))
        return


    # ####################################################################################################################
    # ## DEBUGGING METHODS
    # ####################################################################################################################
    # def plotMemory(self, colour, d_dc, d_data, d_vars):
    #     ## DEBUG MEMORY MANAGEMENT ##
    #     import matplotlib.pyplot as plt
    #
    #     plt.figure(1, figsize=(1,2))
    #     for module in self.callStack:
    #         if hasattr(module, 'getName'):
    #             print(module.getName())
    #             input =  module.getInput()
    #
    #             ## Register dc
    #             k1 = id(input)
    #             if k1 in d_dc.keys():
    #                 no1 = d_dc[k1]
    #             else:
    #                 no1 = len(d_dc.keys())
    #                 d_dc[k1] = no1
    #
    #             ## Register data
    #             k2 = id(input)
    #             if k2 in d_data.keys():
    #                 no2 = d_data[k2]
    #             else:
    #                 no2 = len(d_data.keys())
    #                 d_data[k2] = no2
    #
    #             # Plot link dc and data
    #             plt.plot([no1, no2], [0,1], 'o-', color = colour)
    #
    #             ## Register variables
    #             for var in input.data.keys():
    #                 if not isinstance(input.data[var], str) and not isinstance(input.data[var], int) and not isinstance(input.data[var], list) and not isinstance(input.data[var], float):
    #                     k3 = id(input.data[var])
    #                     if k3 in d_vars.keys():
    #                         no3 = d_vars[k3]
    #                     else:
    #                         no3 = len(d_vars.keys())
    #                         d_vars[k3] = no3
    #                         plt.text(no3, 2, var, rotation=90, fontsize=6)
    #
    #                     plt.plot([no2, no3], [1,2], 'o-', color = colour)
    #     return d_dc, d_data, d_vars




