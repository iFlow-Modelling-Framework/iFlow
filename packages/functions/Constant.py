"""
Constant function
Implementation of FunctionBase. Accepts an arbitrary number of dimensions with arbitrary names.

Requires a parameter 'C0' on initialisation.

Original date: 23-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from .checkVariables import checkVariables


class Constant():
    # Variables
        
    # Methods
    def __init__(self, dimNames, data):
        self.C0 = float(data.v('C0'))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('C0', self.C0))    # check if input is complete
        return

    def value(self, **kwargs):
        return self.C0

    def derivative(self, **kwargs):
        return 0.

    def secondDerivative(self, **kwargs):
        return 0.
