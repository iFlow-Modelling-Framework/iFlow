"""
Class IterativeModule
Implementation of Module that initiates an iteration loop in an IterativeBlock. Requires modules run_init(), that is run
instead of run() in the first iteration, and stopping_criterion(), that is called after the entire block is run.
IterativeModule checks for the existence of these methods in the underlying module.

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import types
from src.util.diagnostics import KnownError
from src import config as cf
from .Module import Module
from copy import deepcopy


class IterativeModule(Module):
    # Variables

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
        Module.__init__(self, input, register)
        self.iteratesWith = []

        return

    ####################################################################################################################
    #   Run and manipulate
    ####################################################################################################################
    def instantiateModule(self):
        """Make and instantiate a module and check whether the input criteria are satisfied.
        """
        super().instantiateModule()

        ## additional criteria
        if not (hasattr(self.module, 'stopping_criterion') and isinstance(self.module.stopping_criterion, types.MethodType)):
            raise KnownError('Module '+self._register.v('module')+' has no working stopping_criterion() method')
        if not (hasattr(self.module, 'run_init') and isinstance(self.module.run_init, types.MethodType)):
            raise KnownError('Module '+self._register.v('module')+' has no working run_init() method')

        return

    def run(self, init=False, **kwargs):
        """Invoke the module's run() method or run_init() method if available and init=True

        Parameters:
            init (bool, optional) - if True, invoke the run_init method

        Exception:
            KnownError exception if the output of the underlying module is not a dictionary of None-type

        Returns
            DataContainer with results of calculated module
        """
        if not hasattr(self, 'module'):
            raise KnownError('Module %s has not been instantiated. Use the method instantiateModule on the run block before running'%self.getName())

        # add submodules to run for this iteration
        self.addInputData({'submodules':self.submoduleList[(not init)]})

        self.timer.tic()
        try:
            if init:
                result = self.module.run_init()
            else:
                result = self.module.run()
        except Exception as e:
            if cf.IGNOREEXCEPTIONS:
                result = {'ERROR':True}
                self.logger.error('FATAL ERROR OCCURRED.\nSIMULATION CONTINUES BECAUSE IGNOREEXCEPTIONS IS SET TO TRUE IN THE SRC/CONFIG.PY.')
                pass
            else:
                raise

        if not isinstance(result, dict):
            raise KnownError('Output of module %s is invalid. Please make sure to return a dictionary.' % self.getName())
        self.timer.toc()

        self._output.merge(result)
        return

    def deepcopy(self, dataCopied=None):
        """Return a deepcopy of this module with input data at the moment of calling the deepcopy. The underlying module class still needs to be instantiated
        """
        moduleCopy, dataCopied = super().deepcopy(dataCopied=dataCopied)
        moduleCopy.iteratesWith = deepcopy(self.iteratesWith)

        return moduleCopy, dataCopied


    def stopping_criterion(self, iteration):
        """Invoke method 'stopping_criterion' of the underlying module. Pass iteration number down to the module.

        Parameters:
            iteration - (int) number of the current iteration.

        Exception:
            KnownError exception if underlying method does not return a boolean.

         Returns:
            bool whether to stop (=True) or continue (=False)
        """
        stop = self.module.stopping_criterion(iteration)
        if not isinstance(stop, bool):
            raise KnownError('Stopping critertion of module %s is invalid. Please make sure to return a boolean.'% self.getName())
        return stop

    def registerIterator(self, otherModules):
        """ Register this module as a module that iterates with other modules.

        Args:
            otherModules: (list of Module) other modules to iterate with

        """
        self.iteratesWith = otherModules
        return

    ####################################################################################################################
    #   Query methods
    ####################################################################################################################
    def getInputRequirements(self, *args, init=False, **kwargs):
        """List of input variables (i.e. list of str) that this module requires given the current list of submodules to run.

        Parameters:
            init - (bool, optional) Does not affect non-iterative modules.

        Returns:
            List of str with required input variable names.
        """
        if init:
            return self._returnmoduleRequirement('inputInit')
        else:
            return list(set(self._returnmoduleRequirement('inputInit')+self._returnmoduleRequirement('input')))

    def isIterative(self):
        return True

    def getIteratesWith(self):
        """ list of other modules that iterate together with this module

        Returns: List of Modules
        """
        return self.iteratesWith






