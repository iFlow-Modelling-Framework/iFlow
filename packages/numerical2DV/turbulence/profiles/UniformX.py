"""
Uniform

Date: 10-Feb-16
Authors: Y.M. Dijkstra
"""
import nifty as ny
from nifty.functionTemplates import FunctionBase


class UniformX(FunctionBase):
    # Variables

    # Methods
    def __init__(self, dimNames, data, m):
        """Initialise a uniform profile with x dependency via H/H0

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
        return

    def value(self, x, **kwargs):
        coef = self.data.v('coef', x=x)
        if self.m != 0:
            H = self.data.v('grid', 'low', 'z', x=x) - self.data.v('grid', 'high', 'z', x=x)
            depthdep = (H/self.H0)**self.m
        else:
            depthdep = 1.
        val = coef * depthdep
        return val

    def derivative(self, x, **kwargs):
        if kwargs.get('dim') == 'x':
            coef = self.data.v('coef', x=x)
            H = self.data.v('grid', 'low', 'z', x=x) - self.data.v('grid', 'high', 'z', x=x)
            coefx = self.data.d('coef', x=x, dim='x')
            Hx = self.data.d('grid', 'low', 'z', x=x, dim='x') - self.data.d('grid', 'high', 'z', x=x, dim='x')

            val = coefx * ((H / self.H0) ** self.m) + coef * (self.m * Hx * (H / self.H0) ** self.m / H)
        elif kwargs.get('dim') == 'xx':
            coef = self.data.v('coef', x=x)
            H = self.data.v('grid', 'low', 'z', x=x) - self.data.v('grid', 'high', 'z', x=x)
            coefx = self.data.d('coef', x=x, dim='x')
            Hx = self.data.d('grid', 'low', 'z', x=x, dim='x') - self.data.d('grid', 'high', 'z', x=x, dim='x')
            coefxx = self.data.d('coef', x=x, dim='xx')
            Hxx = self.data.d('grid', 'low', 'z', x=x, dim='xx') - self.data.d('grid', 'high', 'z', x=x, dim='xx')

            val = coefxx * (H / self.H0) ** self.m + 2. * coefx * self.m * Hx * H ** (
            self.m - 1) / self.H0 ** self.m + coef * self.m * (self.m - 1) * Hx ** 2 * H ** (
            self.m - 2) / self.H0 ** self.m + coef * self.m * Hxx * H ** (self.m - 1) / self.H0 ** self.m

        elif 'z' in kwargs.get('dim'):
            val = 0.
        else:
            val = None
            FunctionBase.derivative(self)
        return val

    ### Depreciated v2.2 [dep01] ###
    def secondDerivative(self, x, **kwargs):
        """
        """
        if kwargs['dim'] == 'x':
            kwargs['dim']= 'xx'
        val = self.derivative(x, **kwargs)
        return val
    ### end ###
