"""
Linear function for 1 variable with slope slope1 between 0 and XL and slope2 between XL and L.
Implementation of FunctionBase.

Requires parameters 'C0': The depth at x=0
                    'slope1': The slope of the first part starting at x=0
                    'slope2': The slope of the second part starting at x=XL
                    'XL': end point of first linear part
                    'L': length of system

Original date: 13-12-18
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
from .checkVariables import checkVariables


class BiLinear():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C0 = data.v('C0')
        self.s1 = data.v('slope1')
        self.s2 = data.v('slope2')
        self.XL = float(np.array(data.v('XL')))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('C0', self.C0), ('slope1', self.s1), ('slope2', self.s2), ('XL', self.XL), ('L', self.L))    # check if input is complete
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        p = np.zeros(len(x))
        p[np.where(x<=self.XL)] = self.s1*x[np.where(x<=self.XL)] + self.C0
        p[np.where(x>self.XL)] = self.s2*(x[np.where(x>self.XL)]-self.XL) + self.s1*self.XL + self.C0
        return p

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        p = np.zeros(len(x))
        p[np.where(x<=self.XL)] = self.s1
        p[np.where(x>self.XL)] = self.s2
        return p

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        p = np.zeros(len(x))
        return p
