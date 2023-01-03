"""
Polynomial function for 1 variable between 0 and XL/L. Between XL/L and 1 the function is linear, where the
linear function has the same level and slope as the polynomial at XL
Implementation of FunctionBase.

Requires parameters 'C': list/array of coefficient of the polynomial ranging from the higher to the lowest order term
                         The length of the list determines the order of the polynomial
                    'XL': end point of the polynomial part
                    'L': length of system

Date: 01-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from .checkVariables import checkVariables


class Multipoly():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C = np.array(data.v('C'))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('C', self.C), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        alpha = self.C[0]
        beta = self.C[1]
        gamma = self.C[2]
        xc = self.C[3]
        xl = self.C[4]
        xc2 = self.C[5]
        xl2 = self.C[6]
        return 0.5*alpha*(1+np.tanh((x-xc)/xl))+beta*(x-xc2)*0.5*(1+np.tanh((x-xc2)/xl2))+gamma
        

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        alpha = self.C[0]
        beta = self.C[1]
        gamma = self.C[2]
        xc = self.C[3]
        xl = self.C[4]
        xc2 = self.C[5]
        xl2 = self.C[6]
        p = (0.5*beta*(x-xc2)*self.sech((x-xc2)/xl2)**2)/xl2+0.5*beta*(np.tanh((x-xc2)/xl2)+1)+(0.5*alpha*self.sech((x-xc)/xl)**2)/xl
        return p

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        alpha = self.C[0]
        beta = self.C[1]
        gamma = self.C[2]
        xc = self.C[3]
        xl = self.C[4]
        xc2 = self.C[5]
        xl2 = self.C[6]

        p = (1.0*beta*self.sech((x-xc2)/xl2)**2)/xl2-(1.0*beta*(x-xc2)*self.sech((x-xc2)/xl2)**2*np.tanh((x-xc2)/xl2))/xl2**2-(1.0*alpha*self.sech((x-xc)/xl)**2*np.tanh((x-xc)/xl))/xl**2
        return p

    def sech(self, x):
        return 1/np.cosh(x)