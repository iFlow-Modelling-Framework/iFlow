"""
Class StandardModule
Implementation of Module without any additional features (see Module for documentation)

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
from .Module import Module


class StandardModule(Module):
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
        Module.__init__(self,input, register)
        return






