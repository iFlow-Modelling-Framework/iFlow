"""
Class UnregisteredModule
A module that does not require a registry entry. The module is always run at the end of a call stack
and is assumed to have no submodules. iFlow does not count on any output provided by this module.


Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import types
from nifty.dynamicImport import dynamicImport
from src.util.diagnostics import KnownError
from .Module import Module
from src.DataContainer import DataContainer


class UnregisteredModule(Module):
    # Variables

    # Methods
    def __init__(self, input):
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
        Module.__init__(self, input, DataContainer())
        self.runModule = True
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def instantiateModule(self):
        """For plot modules with registry entry: use code from Module.
           For plot modules without registry: load directly from the name
        """
        # find the module & run method
        name = self.getName().split('.')
        moduleMain_ = dynamicImport(name[0], name[1])
        try:
            self.module = moduleMain_(self._input)
        except Exception as e:
            # Reraise a KnownError message received from the module
            if isinstance(e, KnownError):
                raise
            # Else raise a new KnownError
            else:
                raise KnownError('Could not instantiate module name. Please check if the init method is correct and if all variables used in the init method are also available to the module.\nDo this by checking whether all variables are provided on input and by checking the call stack.', str(e))

        # Check if there is a run method
        if not (hasattr(self.module, 'run') and isinstance(self.module.run, types.MethodType)):
            raise KnownError('Module '+self.getName()+' has no working run() method')

        return

    #   Query methods ##################################################################################################
    def getName(self, short=False):
        """Returns the name (str) of the module"""
        modName = self._input.v('module')
        return modName

    def isUnregisteredModule(self):
        """Returns True is the underlying module is an unregistered module, else returns False"""
        return True

    def getIteratesWith(self):
        return self._input.v('iteratesWith')





