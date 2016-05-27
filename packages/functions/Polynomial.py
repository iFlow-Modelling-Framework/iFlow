"""
Polynomial function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C': list/array of coefficient of the polynomial ranging from the higher to the lowest order term
                         The length of the list determines the order of the polynomial
                    'L': length of system

Date: 23-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import numpy as np
from nifty.functionTemplates import FunctionBase


class Polynomial(FunctionBase):
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = data.v('L')
        self.C = np.array(data.v('C'))
        FunctionBase.checkVariables(self, ('C', self.C), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return np.polyval(self.C, x)

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = np.polyder(self.C)
        if kwargs['dim'] == 'x':
            return np.polyval(Cx, x)
        elif kwargs['dim'] == 'xx':
            Cxx = np.polyder(Cx)
            return np.polyval(Cxx, x)
        else:
            FunctionBase.derivative(self)
            return