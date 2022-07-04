"""
Polynomial function for 1 variable between 0 and XL/L. Between XL/L and 1 the function is linear, where the
linear function has the same level and slope as the polynomial at XL
Implementation of FunctionBase.

Requires parameters 'C': list/array of coefficient of the polynomial ranging from the higher to the lowest order term
                         The length of the list determines the order of the polynomial
                    'XL': end point of the polynomial part
                    'L': length of system

Original date: 01-07-15
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
from .checkVariables import checkVariables


class PolynomialLinear():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C = np.array(data.v('C'))
        self.XL = float(np.array(data.v('XL')))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('C', self.C), ('XL', self.XL), ('L', self.L))

        # coefficients for the linear function
        self.C_lin = np.asarray([np.polyval(np.polyder(self.C), self.XL), np.polyval(self.C, self.XL)])
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        p = np.zeros(len(x))
        p[np.where(x<=self.XL)] = np.polyval(self.C, x[np.where(x<=self.XL)])
        p[np.where(x>self.XL)] = np.polyval(self.C_lin, x[np.where(x>self.XL)]-self.XL)
        return p

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = np.polyder(self.C)
        Cx_lin = np.polyder(self.C_lin)
        p = np.zeros(len(x))
        p[np.where(x<=self.XL)] = np.polyval(Cx, x[np.where(x<=self.XL)])
        p[np.where(x>self.XL)] = np.polyval(Cx_lin, x[np.where(x>self.XL)]-self.XL)
        return p

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        Cx = np.polyder(self.C)
        Cx_lin = np.polyder(self.C_lin)
        p = np.zeros(len(x))
        Cxx = np.polyder(Cx)
        p[np.where(x<=self.XL)] = np.polyval(Cxx, x[np.where(x<=self.XL)])
        return p