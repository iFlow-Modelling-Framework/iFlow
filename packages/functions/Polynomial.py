"""
Polynomial function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C': list/array of coefficient of the polynomial ranging from the higher to the lowest order term
                         The length of the list determines the order of the polynomial
                    'L': length of system

Original date: 23-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import numpy as np
import nifty as ny
from .checkVariables import checkVariables


class Polynomial():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C = np.array(ny.toList(data.v('C')))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('C', self.C), ('L', self.L))
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
        return np.polyval(Cx, x)

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = np.polyder(self.C)
        Cxx = np.polyder(Cx)
        return np.polyval(Cxx, x)