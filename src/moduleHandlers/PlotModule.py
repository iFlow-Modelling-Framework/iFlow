"""
Class PlotModule
Implementation of Module that is always run, regardless of output requirements.
Supports the 'iteratesWith' keyword in the input file to denote this module should be included in the iteration loop with
an iterativeModule if possible.

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import types
from nifty.dynamicImport import dynamicImport
from src.util.diagnostics import KnownError
from .Module import Module


class PlotModule(Module):
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
        # Load the data from input and register to private class variables and initiate a timer for this module
        Module.__init__(self, input, register)
        self.runModule = True
        self.iteratesWith = input.v('iteratesWith')

        return

    def getIteratesWith(self):
        return self._input.v('iteratesWith')









