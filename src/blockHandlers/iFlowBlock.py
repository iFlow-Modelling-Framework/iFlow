"""
Abstract class iFlowBlock
- implicitly implements the <<Interface>> Runable -
The iFlowBlock contains Runables (i.e. iFlowBlocks or Modules) in the correct for running. It implements several methods
to e.g. change input or deepcopy the block as well as a run method.

Date: 03-02-2022
Authors: Y.M. Dijkstra
"""
from src.DataContainer import DataContainer
from nifty.Timer import Timer
import logging


class iFlowBlock:
    # Variables

    # Methods
    def __init__(self, callStack, memProfiler = None):
        self.logger = logging.getLogger(__name__)
        self.callStack = callStack
        self._output = DataContainer()
        self._memProfiler = memProfiler
        self.timer = Timer()

        self.instantiateModule()
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def instantiateModule(self):
        for mod in self.callStack:
            mod.instantiateModule()
        return

    def run(self,init=True):
        self.timer.tic()
        for mod in self.callStack:
            mod.run(init=init)

            # handle result
            result = mod.getOutput()        # get result from the module

            self._output.merge(result)       # merge with block's result
            self.addInputData(result)       # distribute over elements of the block's moduleList
        self.timer.toc()
        return

    def reset(self):
        """ Remove instances of underlying modules but keep all data from input and potential previous runs. This allows
        for a total restart of the modules (without memory of class variables) but with a 'warm start'.
        """
        for mod in self.callStack:
            mod.reset()
        return

    def deepcopy(self, dataCopied=None):
        """ Return a deepcopy of this iFlowBlock with identical input and register information as on the moment of copying.
        All copies are deepcopies, so no object has the same reference as the original.

        All underlying modules are instantiated.

        Parameters:
            dataCopied (optional: tuple of list and DC): tracks the data from self._input of the modules that has already been copied
                firstly contains a list of original id() of copied elements, secondly contains the new DC. This is done to reduce
                memory use: instead of deepcopying self._input DCs in each module, entries that have already been copied in one module are used
                in the deepcopy of other modules. (NB. checks are done safely, only replacing objects that referred to the same memory ID in the
                original).
        """
        callStackCopy = []
        for mod in self.callStack:
            modnew, dataCopied = mod.deepcopy(dataCopied=dataCopied)
            callStackCopy.append(modnew)

        newblock = self.__class__(callStackCopy, memProfiler=self._memProfiler)
        return newblock, dataCopied

    def addInputData(self, input):
        """Adds 'input' to all elements in the moduleList
        Parameters:
            input (DataContainer):
        """
        for module in self.callStack:
            module.addInputData(input)
        return

    def updateSubmoduleListIteration(self, outputList):
        for mod in self.callStack:
            mod.updateSubmoduleListIteration(outputList)

    def printTimerResult(self, tabs=''):
        self.logger.info(self.timer.string(tabs+self.getName() + ' time elapsed: '))
        for mod in self.callStack:
            mod.printTimerResult(tabs+'\t')

    ####################################################################################################################
    ## Public queries
    ####################################################################################################################
    def getOutput(self):
        return self._output

    def getName(self, short=False):
        return 'Block(%s)'%self.callStack[0].getName(short=short)

    def getInput(self):
        DC = DataContainer()
        for mod in self.callStack:
            DC.merge(mod.getInput())
        return DC

    def getProfiler(self):
        return self._memProfiler








