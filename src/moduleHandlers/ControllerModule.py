"""
Class ControllerModule
Implementation of Module that receives the private classvariable __controlledBlock, which contains an iFlowBlock instance.
ControllerModule passes this block to the underlying module upon instantiation. The underlying module is responsible for
running this block. It has full access to this block
.

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import types
from nifty.dynamicImport import dynamicImport
from nifty import toList
from copy import deepcopy, copy
from src.util.diagnostics import KnownError
from src.moduleHandlers.Module import Module
from src.blockHandlers.StandardBlock import StandardBlock


class ControllerModule(Module):
    # Variables
    controlledModuleBlock = StandardBlock([])

    # Methods
    def __init__(self, input, register):
        """Load all variables needed for the module to (private)class variables.
        Parameters:
            input - (DataContainer) input variables and results of previous modules
            register - (DataContainer) register data of this module

        Exception:
            KnownError if module cannot be instantiated
            KnownError if no run() module was found in the module
            KnownError if an iterative method has no method stopping_criterion(iteration)
        """

        # Load the data from input and register to private class variables and initiate a timer for this module
        Module.__init__(self, input, register)
        self.__controlledBlock = None       # controlled block will be added later

        # self.runModule = True
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def instantiateModule(self):
        """Make and instantiate a module and check whether the input criteria are satisfied.
        """
        # find the module & run method
        moduleMain_ = dynamicImport(self._register.v('packagePath'), self._register.v('module'))
        try:
            self.module = moduleMain_(self._input, self.__controlledBlock)                      # Controlled block is passed to the module on instantiation
        except Exception as e:
            # Reraise a KnownError message received from the module
            if isinstance(e, KnownError):
                raise
            # Else raise a new KnownError
            else:
                raise KnownError('Could not instantiate module %s. Please check if the init method is correct and if all variables used in the init method are also available to the module.\nThis is a controller module. Please also check if the init allows for two arguments: input and the block of modules.'% self._register.v('module'), str(e))

        # Check if there is a run method
        if not (hasattr(self.module, 'run') and isinstance(self.module.run, types.MethodType)):
            raise KnownError('Module '+self._register.v('module')+' has no working run() method')

        return

    def addInputData(self, d):
        """Append the input data by d

        Parameters:
            d - (dict or DataContainer) data to append
        """
        self._input.merge(d)
        self.__controlledBlock.addInputData(d)

        return

    def printTimerResult(self, tabs=''):
        self.logger.info(self.timer.string(tabs+self.getName() + ' time elapsed: '))
        for mod in self.__controlledBlock.callStack:
            mod.printTimerResult(tabs+'\t')

    def isControllerModule(self):
        return True

    def getOutputVariables(self, *args, includePrepare=False, **kwargs):
        """List of output variable names (i.e. list of str) that this module will return
            For controller, this includes the variables under 'prepare'
                """
        if includePrepare:
            return list(set(self._returnmoduleRequirement('output')+self._returnmoduleRequirement('prepares')))
        else:
            return self._returnmoduleRequirement('output')

    def getInputRequirements(self, *args, includeControls=False, **kwargs):
        """List of output variable names (i.e. list of str) that this module will return
            For controller, this includes the variables under 'prepare'
                """
        if includeControls:
            return list(set(self._returnmoduleRequirement('input')+self._returnmoduleRequirement('controls')))
        else:
            return self._returnmoduleRequirement('input')

    def deepcopy(self, dataCopied=None):
        """Return a deepcopy of this module with input data at the moment of calling the deepcopy. The underlying module class still needs to be instantiated
        """
        moduleCopy, dataCopied = super().deepcopy(dataCopied=dataCopied)
        moduleCopy.addControlledBlock(self.__controlledBlock.deepcopy(dataCopied=dataCopied))

        return moduleCopy, dataCopied

    def addControlledBlock(self, block):
        self.__controlledBlock = block
        return

    def getControlledBlock(self):
        return self.__controlledBlock

