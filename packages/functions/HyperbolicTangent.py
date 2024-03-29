"""
Hyperbolic tangent function in 1 variable between 0 and 1.
Implementation of FunctionBase.

Requires parameters 'C0': base level
                    'C1': step amplitude
                    'xc': position of step
                    'xl': length scale of step
                    'L': length

Original date: 07-07-16
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
from .checkVariables import checkVariables


class HyperbolicTangent():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.C0 = data.v('C0')
        self.C1 = data.v('C1')
        self.xc = float(data.v('xc'))
        self.xl = float(data.v('xl'))
        self.dimNames = dimNames

        # call checkVariables method of FunctionBase to make sure that the input is correct
        checkVariables(self.__class__.__name__, ('C0', self.C0), ('C1', self.C1),('xc', self.xc), ('xl', self.xl), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return self.C0 + self.C1*np.tanh((x-self.xc)/self.xl)

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return self.C1*(1-np.tanh((x-self.xc)/self.xl)**2)/self.xl

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return -self.C1*2.*np.tanh((x-self.xc)/self.xl)*(1-np.tanh((x-self.xc)/self.xl)**2.)/self.xl**2

