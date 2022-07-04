"""
Exponential function in 1 variable between 0 and 1.
Implementation of FunctionBase.

Requires parameters 'C0': value for dep. variable value 0
                    'Lc': convergence length
                    'L': length of system

Original date: 23-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from numpy import exp
from .checkVariables import checkVariables


class Exponential():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C0 = float(data.v('C0'))
        self.Lc = float(data.v('Lc'))
        self.dimNames = dimNames

        # call checkVariables method of FunctionBase to make sure that the input is correct
        checkVariables(self.__class__.__name__, ('C0', self.C0), ('Lc', self.Lc), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return self.C0*exp(-x/self.Lc)

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return -self.C0/self.Lc*exp(-x/self.Lc)

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return self.C0/(self.Lc**2)*exp(-x/self.Lc)
