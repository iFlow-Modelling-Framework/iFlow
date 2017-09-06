"""
Exponential of a rational function for 1 variable between 0 and 1
Implementation of FunctionBase.

Requires parameters 'C1':
                    'C2':
                    'L': length of system

Date: 29-06-15
Authors: R.L. Brouwer
"""
import numpy as np
from nifty.functionTemplates import FunctionBase


class ExpRationalFunc(FunctionBase):
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = float(data.v('L'))
        self.C1 = np.array(data.v('C1'))
        self.C2 = np.array(data.v('C2'))
        FunctionBase.checkVariables(self, ('L', self.L), ('C1', self.C1), ('C2', self.C2))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return 1000*np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x))

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        expC = np.exp(np.polyval(self.C1, x)/np.polyval(self.C2, x))
        C1x = np.polyder(self.C1)
        C2x = np.polyder(self.C2)

        if kwargs['dim'] == 'x':
            Cx = (np.polyval(C1x, x)*np.polyval(self.C2, x) -
                  np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2
            return 1000*Cx*expC

        elif kwargs['dim'] == 'xx':
            C1xx = np.polyder(C1x)
            C2xx = np.polyder(C2x)
            Cx = (np.polyval(C1x, x)*np.polyval(self.C2, x) -
                  np.polyval(self.C1, x)*np.polyval(C2x, x))/np.polyval(self.C2, x)**2
            Cxx = (np.polyval(C1xx, x)*np.polyval(self.C2, x)**2 +
                   2*np.polyval(self.C1, x)*np.polyval(C2x, x)**2 -
                   2*np.polyval(C1x, x)*np.polyval(self.C2, x)*np.polyval(C2x, x) -
                   np.polyval(self.C1, x)*np.polyval(self.C2, x)*np.polyval(C2xx, x)) / np.polyval(self.C2, x)**3
            return 1000*(Cx**2 + Cxx)*expC
        else:
            FunctionBase.derivative(self)
            return


