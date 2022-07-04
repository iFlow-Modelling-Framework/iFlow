"""
Uniform

Date: 10-Feb-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import copy


class UniformXF():
    # Variables

    # Methods
    def __init__(self, data, m):
        """Initialise a uniform profile.

        Args:
            data (DataContainer) - DataContainer with
                 - magnitude (Av0)
                 - grid
            m (real) - power for H/H0
        """
        self.dimNames = ['x','f']
        self.m = m
        self.data = data
        self.H0 = self.data.v('grid', 'low', 'z', x=0) - self.data.v('grid', 'high', 'z', x=0)

        # change grid to remove z dimension
        self.data._data['grid'] = copy.copy(self.data._data['grid'])
        self.data._data['grid']['dimensions'] = self.dimNames
        if 'f' in self.dimNames:
            self.data._data['grid']['axis'] = copy.copy(self.data._data['grid']['axis'])

            lenf = self.data._data['grid']['axis']['f'].shape[-1]
            self.data._data['grid']['axis']['f'] = self.data._data['grid']['axis']['f'].reshape(1, lenf)
            self.data._data['grid']['contraction'] = np.asarray([[0, 0], [0, 0]])
        return

    def value(self, x, f=0, **kwargs):
        coef = self.data.v('coef', x=x, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)

        val = coef*((H/self.H0)**self.m)
        return val

    def derivative(self, x, f=0, **kwargs):
        coef = self.data.v('coef', x=x, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)
        coefx = self.data.d('coef', x=x, f=f, dim='x')
        Hx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='x') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='x')

        val = coefx*((H/self.H0)**self.m) + coef *(self.m*Hx*(H/self.H0)**self.m / H)
        return val

    def secondDerivative(self, x, f=0, **kwargs):
        coef = self.data.v('coef', x=x, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, f=f) - self.data.v('grid', 'high', 'z', x=x, f=f)
        coefx = self.data.d('coef', x=x, f=f, dim='x')
        Hx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='x') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='x')
        coefxx = self.data.d('coef', x=x, f=f, dim='xx')
        Hxx = self.data.d('grid', 'low', 'z', x=x, f=f, dim='xx') - self.data.d('grid', 'high', 'z', x=x, f=f, dim='xx')

        val = coefxx*(H/self.H0)**self.m + 2.*coefx*self.m*Hx*H**(self.m-1)/self.H0**self.m + coef*self.m*(self.m-1)*Hx**2*H**(self.m-2)/self.H0**self.m+coef*self.m*Hxx*H**(self.m-1)/self.H0**self.m

        return val


