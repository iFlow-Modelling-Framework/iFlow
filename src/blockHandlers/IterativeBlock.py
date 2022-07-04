"""
Class iterativeBlock
Subclass of iFlowBlock. Extends the iFlowBlock by creating a loop over the entire Block. The block contains one or more
IterativeModules. At the end of a run of the entire block, the stopping criteria of all IterativeModules is checked to
decide to do more iterations.

Date: 03-02-2022
Authors: Y.M. Dijkstra
"""
from .iFlowBlock import iFlowBlock
from src.config import MAXITERATIONS
from nifty.toList import toList

class IterativeBlock(iFlowBlock):
    # Variables

    # Methods
    def __init__(self, callStack, iterativemods, memProfiler = None):
        iFlowBlock.__init__(self,callStack, memProfiler)
        self.iterativemods = toList(iterativemods)
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def run(self, init=True):
        self.timer.tic()
        iteration = 0
        runloop = True
        while runloop and iteration < MAXITERATIONS:
            # Run the block
            super().run(init=(not bool(iteration)))

            # Update iteration and check stopping criterion
            iteration += 1
            keeprunning = [itMod.stopping_criterion(iteration) for itMod in self.iterativemods]
            if all(keeprunning):
                runloop = False

            # Do a memory profile if profiler is set
            if self._memProfiler is not None:
                self._memProfiler.snapshot('Run it. %s of %s'%(str(iteration), self.getName(short=True)))
        self.timer.toc()
        return

    def deepcopy(self, dataCopied=None):
        """ Return a deepcopy of this iFlowBlock with identical input and register information as on the moment of copying.
        All copies are deepcopies, so no object has the same reference as the original.

        All underlying modules are instantiated.

        Parameters:
            dataCopied (optional: tuple of list and DC): tracks the data from self._input of the modules that has already been copied
                firstly contains a list of original id() of copied elements, secondly contains the new DC.


        """
        callStackCopy = []
        for mod in self.callStack:
            modnew, dataCopied = mod.deepcopy(dataCopied=dataCopied)
            callStackCopy.append(modnew)

        newblock = self.__class__(callStackCopy, self.iterativemods, memProfiler=self._memProfiler)
        return newblock, dataCopied

    def getIterativeModules(self):
        """ Return list of the iterative module(s) in this block

        Returns: list of Modules.
        """
        return self.iterativemods




