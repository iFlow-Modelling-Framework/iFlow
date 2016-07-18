"""
Exponential function in 1 variable between 0 and 1.
Implementation of FunctionBase.

Requires parameters 'C0': value for dep. variable value 0
                    'Lc': convergence length
                    'L': length of system

Date: 23-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from numpy import exp, tanh
from nifty.functionTemplates import FunctionBase


class ExponentialTanhyp(FunctionBase):
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        FunctionBase.__init__(self, dimNames)
        self.L = data.v('L')
        self.C0 = data.v('C0')
        self.Lc = data.v('Lc')
        self.C1 = data.v('C1')
        self.xc = data.v('xc')
        self.xl = data.v('xl')

        # call checkVariables method of FunctionBase to make sure that the input is correct
        FunctionBase.checkVariables(self, ('C0', self.C0), ('Lc', self.Lc), ('C1', self.C1), ('xc', self.xc), ('xl', self.xl), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        x = x*self.L
        return self.C0*exp(-x/self.Lc)*(1-self.C1*tanh((x-self.xc)/self.xl))

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        if kwargs['dim'] == 'x':
            x = x*self.L
            return -self.C0*exp(-x/self.Lc)*(1-self.C1*tanh((x-self.xc)/self.xl))/self.Lc-self.C0*exp(-x/self.Lc)*self.C1*(1-tanh((x-self.xc)/self.xl)**2)/self.xl
        elif kwargs['dim'] == 'xx':
            x = x*self.L
            C0 = self.C0
            C1 = self.C1
            Lc = self.Lc
            xc = self.xc
            xl = self.xl
            return C0*exp(-x/Lc)*(1-C1*tanh((x-xc)/xl))/Lc**2+2*C0*exp(-x/Lc)*C1*(1-tanh((x-xc)/xl)**2)/(Lc*xl)+2*C0*exp(-x/Lc)*C1*tanh((x-xc)/xl)*(1-tanh((x-xc)/xl)**2)/xl**2
        else:
            FunctionBase.derivative(self)
            return 

