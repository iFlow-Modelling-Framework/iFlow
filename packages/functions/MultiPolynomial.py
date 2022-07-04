"""
combined polynomial functions for 1 variable between 0 and L. XL is a list of transition points.
Implementation of FunctionBase.

Requires parameters 'C0', 'C1', ...: list/array of coefficient of the polynomial ranging from the higher to the lowest order term
                         The length of the list determines the order of the polynomial
                    'XL': end points of each polynomial part
                    'L': length of system

Original date: 01-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from .checkVariables import checkVariables


class MultiPolynomial():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.XL = ny.toList(data.v('XL'))
        self.XL = [float(i) for i in self.XL]
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('XL', self.XL), ('L', self.L))

        self.C = []
        for i in range(0, len(self.XL)+1):
            self.C.append(np.array(data.v('C'+str(i))))
            if self.C[-1] is None:
                from src.util.diagnostics.KnownError import KnownError
                raise KnownError('Not enough elements for C given in depth profile.')
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        p = np.zeros(len(x))
        for i in range(0, len(self.XL)):
            p[np.where(x<=self.XL[-1-i])] = np.polyval(self.C[-2-i], x[np.where(x<=self.XL[-1-i])])
        p[np.where(x>self.XL[-1])] = np.polyval(self.C[-1], x[np.where(x>self.XL[-1])])
        return p

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = [np.polyder(i) for i in self.C]
        p = np.zeros(len(x))
        for i in range(0, len(self.XL)):
            p[np.where(x<=self.XL[-1-i])] = np.polyval(Cx[-2-i], x[np.where(x<=self.XL[-1-i])])
        p[np.where(x>self.XL[-1])] = np.polyval(Cx[-1], x[np.where(x>self.XL[-1])])
        return p

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = [np.polyder(i) for i in self.C]
        p = np.zeros(len(x))
        Cxx = [np.polyder(i) for i in Cx]
        for i in range(0, len(self.XL)):
            p[np.where(x<=self.XL[-1-i])] = np.polyval(Cxx[-2-i], x[np.where(x<=self.XL[-1-i])])
        p[np.where(x>self.XL[-1])] = np.polyval(Cxx[-1], x[np.where(x>self.XL[-1])])
        return p