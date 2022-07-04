"""
Exponential of a rational function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C1':
                    'C2':
                    'L': length of system

Date: 29-06-15
Authors: I.Jalon-Rojas (modified from R.L. Brouwer)
"""
import numpy as np
from packages.functions.checkVariables import checkVariables


class ExpRationalFunc_Flat():
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        self.L = data.v('L')
        self.Le = data.v('Le')
        self.C1 = np.array(data.v('C1'))
        self.C2 = np.array(data.v('C2'))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('L', self.L), ('Le', self.Le), ('C1', self.C1), ('C2', self.C2))
        return

         
    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x * self.L
        Cte = np.exp(np.polyval(self.C1, self.Le)/np.polyval(self.C2, self.Le))
        return np.piecewise(x, [x <= self.Le, x > self.Le], [lambda x: 1000.*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 1000.*Cte])

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        C1x = np.polyder(self.C1)
        C2x = np.polyder(self.C2)

        return np.piecewise(x, [x <= self.Le, x > self.Le], [lambda x:   1000.*(np.polyval(C1x, x)*np.polyval(self.C2, x) -
              np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 0.])

    def secondDerivative(self, x, **kwargs):
        x = x*self.L
        #expC = np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x))
        C1x = np.polyder(self.C1)
        C2x = np.polyder(self.C2)

        C1xx = np.polyder(C1x)
        C2xx = np.polyder(C2x)
        return np.piecewise(x, [x <= self.Le, x > self.Le], [lambda x:  1000.*((np.polyval(C1x, x)*np.polyval(self.C2, x) -
              np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2**2 + (np.polyval(C1xx, x)*np.polyval(self.C2, x)**2 +
               2*np.polyval(self.C1, x)*np.polyval(C2x, x)**2 -
               2*np.polyval(C1x, x)*np.polyval(self.C2, x)*np.polyval(C2x, x) -
               np.polyval(self.C1, x)*np.polyval(self.C2, x)*np.polyval(C2xx, x)) / np.polyval(self.C2, x)**3)*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 0])



