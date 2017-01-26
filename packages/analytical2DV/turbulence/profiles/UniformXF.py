"""
Uniform

Date: 10-Feb-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from nifty.functionTemplates import FunctionBase
import copy

class UniformXF(FunctionBase):
    # Variables

    # Methods
    def __init__(self, dimNames, data, m):
        """Initialise a uniform profile.

        Args:
            dimNames: see function
            data (DataContainer) - DataContainer with
                 - magnitude (Av0)
                 - grid
            m (real) - power for H/H0
        """
        FunctionBase.__init__(self, dimNames)
        self.m = m
        self.data = data
        self.H0 = self.data.v('grid', 'low', 'z', x=0) - self.data.v('grid', 'high', 'z', x=0)

        # change grid to remove z dimension
        self.data.data['grid'] = copy.copy(self.data.data['grid'])
        self.data.data['grid']['dimensions'] = dimNames
        if 'f' in dimNames:
            self.data.data['grid']['axis'] = copy.copy(self.data.data['grid']['axis'])

            lenf = self.data.data['grid']['axis']['f'].shape[-1]
            self.data.data['grid']['axis']['f'] = self.data.data['grid']['axis']['f'].reshape(1, lenf)
            self.data.data['grid']['contraction'] = np.asarray([[0, 0], [0, 0]])
        return

    def value(self, *args, **kwargs):
        x = args[0]
        if len(args) > 1:
            f = args[1]
        else:
            f = 0
        coef = self.data.v('coef', x=x, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)

        val = coef*((H/self.H0)**self.m)
        return val

    def derivative(self, *args, **kwargs):
        x = args[0]
        if len(args) > 1:
            f = args[1]
        else:
            f = 0

        if kwargs.get('dim') == 'x':
            coef = self.data.v('coef', x=x, f=f)
            H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)
            coefx = self.data.d('coef', x=x, f=f, dim='x')
            Hx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='x') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='x')

            val = coefx*((H/self.H0)**self.m) + coef *(self.m*Hx*(H/self.H0)**self.m / H)
        elif kwargs.get('dim') == 'xx':
            coef = self.data.v('coef', x=x, f=f)
            H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)
            coefx = self.data.d('coef', x=x, f=f, dim='x')
            Hx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='x') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='x')
            coefxx = self.data.d('coef', x=x, f=f, dim='xx')
            Hxx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='xx') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='xx')

            val = coefxx*(H/self.H0)**self.m + 2.*coefx*self.m*Hx*H**(self.m-1)/self.H0**self.m + coef*self.m*(self.m-1)*Hx**2*H**(self.m-2)/self.H0**self.m+coef*self.m*Hxx*H**(self.m-1)/self.H0**self.m

        elif 'z' in kwargs.get('dim'):
            val = 0.
        else:
            val = None
            FunctionBase.derivative(self)
        return val

    ### Depreciated v2.2 [dep01] ###
    def secondDerivative(self, x, f, **kwargs):
        """
        """
        if kwargs['dim'] == 'x':
            kwargs['dim']= 'xx'
        val = self.derivative(x, f, **kwargs)
        return val
    ### end ###
