"""
Constant function
Implementation of FunctionBase. Accepts an arbitrary number of dimensions with arbitrary names.

Requires a parameter 'C0' on initialisation.

Date: 23-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from nifty.functionTemplates import FunctionBase


class Constant(FunctionBase):
    # Variables
        
    # Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.C0 = float(data.v('C0'))

        # call checkVariables method of FunctionBase to make sure that the input is correct
        FunctionBase.checkVariables(self, ('C0', self.C0))
        return

    def value(self, *args, **kwargs):
        return self.C0

    def derivative(self, *args, **kwargs):
        return 0.
