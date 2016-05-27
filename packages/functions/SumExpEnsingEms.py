"""
Sum of exponential functions for 1 variable between 0 and 1. This analytical function is used by E. Ensing for the
bottom profile of the Ems Estuary

Implementation of FunctionBase.

Requires parameters 'C0': Depth of the estuary at the mouth
                    'L': length of system

Date: 03-03-16
Authors: R.L. Brouwer
"""


import numpy as np
from nifty.functionTemplates import FunctionBase


class SumExpEnsingEms(FunctionBase):
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.__L = data.v('L')
        self.__C0 = data.v('C0')
        FunctionBase.checkVariables(self, ('L', self.__L), ('C0', self.__C0))
        self.__Lt = np.sqrt(9.81*self.__C0)/1.4e-4
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x * self.__L
        X1 = x / self.__Lt - 0.25
        X2 = x / self.__Lt - 0.5
        return self.__C0 * (5./6. - 0.45*x/self.__L + 0.2*np.exp(-10.*x/self.__Lt) - 0.12*np.exp(-80*X1**2) -
                            0.05*np.exp(-80*X2**2) + 0.5*((x**2-x*self.__L)/self.__Lt**2))

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x * self.__L
        X1 = x / self.__Lt - 0.25
        X2 = x / self.__Lt - 0.5
        return (self.__C0/self.__Lt) * (-0.45*self.__Lt/self.__L - 2*np.exp(-10.*x/self.__Lt) +
                                        19.2*X1*np.exp(-80*X1**2) + 8.*X2*np.exp(-80*X2**2) +
                                        (2*x-self.__L)/(2*self.__Lt))

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x * self.__L
        X1 = x / self.__Lt - 0.25
        X2 = x / self.__Lt - 0.5
        return (self.__C0/self.__Lt**2) * (1. + 20.*np.exp(-10.*x/self.__Lt) + (19.2 - 3072.*X1**2)*np.exp(-80*X1**2) +
                                           (8. - 1280.*X2**2)*np.exp(-80*X2**2))

