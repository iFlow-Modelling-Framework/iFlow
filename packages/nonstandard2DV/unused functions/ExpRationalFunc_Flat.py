"""
Exponential of a rational function for x<XL and constant for x>XL for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C1':
                    'C2':
                    'L': length of system
                    'XL': x<XL exp. rat. function, x>XL constant

Date: June 2016 (Cleaned and added to iFlow at 26-07-2016 by Y.M. Dijkstra)
Authors: I. Jalon Rojas, R.L. Brouwer, Y.M. Dijkstra
"""
import numpy as np
from nifty.functionTemplates import FunctionBase


class ExpRationalFunc_Flat(FunctionBase):
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = data.v('L')
        self.XL = data.v('XL')
        self.C1 = np.array(data.v('C1'))
        self.C2 = np.array(data.v('C2'))
        FunctionBase.checkVariables(self, ('L', self.L), ('XL', self.XL), ('C1', self.C1), ('C2', self.C2))
        return

    def value(self, x, **kwargs):
        x = x * self.L
        Cte = np.exp(np.polyval(self.C1, self.XL)/np.polyval(self.C2, self.XL))
        return np.piecewise(x, [x <= self.XL, x > self.XL], [lambda x: 1000.*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 1000.*Cte])

    def derivative(self, x, **kwargs):
        x = x*self.L
        C1x = np.polyder(self.C1)
        C2x = np.polyder(self.C2)

        if kwargs['dim'] == 'x':
            return np.piecewise(x, [x <= self.XL, x > self.XL], [lambda x:   1000.*(np.polyval(C1x, x)*np.polyval(self.C2, x) -
                  np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 0.])

        elif kwargs['dim'] == 'xx':
            C1xx = np.polyder(C1x)
            C2xx = np.polyder(C2x)
            return np.piecewise(x, [x <= self.XL, x > self.XL], [lambda x:  1000.*((np.polyval(C1x, x)*np.polyval(self.C2, x) -
                  np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2**2 + (np.polyval(C1xx, x)*np.polyval(self.C2, x)**2 +
                   2*np.polyval(self.C1, x)*np.polyval(C2x, x)**2 -
                   2*np.polyval(C1x, x)*np.polyval(self.C2, x)*np.polyval(C2x, x) -
                   np.polyval(self.C1, x)*np.polyval(self.C2, x)*np.polyval(C2xx, x)) / np.polyval(self.C2, x)**3)*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x)), lambda x: 0])
        else:
            FunctionBase.derivative(self)
            return


