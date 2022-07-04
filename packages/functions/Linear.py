"""
Linear function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C0': value at variable value 0
                    'CL': value at variable value 1

Original date: 23-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from .checkVariables import checkVariables


class Linear():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C0 = float(data.v('C0'))
        self.CL = float(data.v('CL'))
        self.dimNames = dimNames


        checkVariables(self.__class__.__name__, ('C0', self.C0), ('CL', self.CL), ('L', self.L))
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
        return -self.C0/self.L+self.CL/self.L

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        return 0.