"""
Exponential function in 1 variable between 0 and 1.
Implementation of FunctionBase.

Requires parameters 'C0': value for dep. variable value 0
                    'Lc': convergence length
                    'L': length of system

Date: 23-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from numpy import exp
from nifty.functionTemplates import FunctionBase


class Exponential(FunctionBase):
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = data.v('L')
        self.C0 = data.v('C0')
        self.Lc = data.v('Lc')

        # call checkVariables method of FunctionBase to make sure that the input is correct
        FunctionBase.checkVariables(self, ('C0', self.C0), ('Lc', self.Lc), ('L', self.L))
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
        if kwargs['dim'] == 'x':
            x = x*self.L
            return -self.C0/self.Lc*exp(-x/self.Lc)
        elif kwargs['dim'] == 'xx':
            x = x*self.L
            return self.C0/(self.Lc**2)*exp(-x/self.Lc)
        else:
            FunctionBase.derivative(self)
            return 

