"""
Linear function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C0': value at variable value 0
                    'CL': value at variable value 1

Date: 23-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from nifty.functionTemplates import FunctionBase


class Linear(FunctionBase):
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = data.v('L')
        self.C0 = data.v('C0')
        self.CL = data.v('CL')
        FunctionBase.checkVariables(self, ('C0', self.C0), ('CL', self.CL), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        L = self.L
        x = x*L
        return self.C0*(L-x)/L+self.CL*x/L

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        if kwargs['dim'] == 'x':
            return -self.C0/self.L+self.CL/self.L
        else:
            return 0.