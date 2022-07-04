"""
Parabolic

Original date: 10-Feb-16
Update: 04-02-22
Authors: Y.M. Dijkstra
"""


class ParabolicXZF():
    # Variables

    # Methods
    def __init__(self, data, m):
        """Initialise a parabolic profile.

        Args:
            coef (complex array) - needs to be a real/complex array in shape (x,1,f) (length of x may be 1)
            data (DataContainer) - DataContainer with
                 - roughness (z0*)
                 - roughness (zs*)
                 - magnitude (coef)
                 - grid
            m (real) - power for H/H0

        """
        self.dimNames = ['x', 'z', 'f']
        self.data = data
        self.m = m
        self.H0 = self.data.v('grid', 'low', 'z', x=0) - self.data.v('grid', 'high', 'z', x=0)

        return

    def value(self, x, z, f, **kwargs):
        z = z.reshape((1, len(z), 1))
        coef = self.data.v('coef', x=x, z=z, f=f)
        if self.m != 0:
            H = self.data.v('grid', 'low', 'z', x=x, z=z, f=f) - self.data.v('grid', 'high', 'z', x=x, z=z, f=f)
            depthdep = (H/self.H0)**self.m
        else:
            depthdep = 1.
        z0_dimless = self.data.v('z0*', x=x, z=z, f=f)
        zs_dimless = self.data.v('zs*', x=x, z=z, f=f)

        Av = coef*(zs_dimless-(-z))*(1+z0_dimless+(-z)) * depthdep
        return Av

    def derivative(self, x, z, f, **kwargs):
        """Derivative in z"""
        z = z.reshape((1, len(z), 1))
        coef = self.data.v('coef', x=x, z=z, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, z=z, f=f) - self.data.v('grid', 'high', 'z', x=x, z=z, f=f)
        z0_dimless = self.data.v('z0*', x=x, z=z, f=f)
        zs_dimless = self.data.v('zs*', x=x, z=z, f=f)

        Avz = coef/H*(-2.*(-z)-1.-z0_dimless+zs_dimless) * (H/self.H0)**self.m
        return Avz

    def secondDerivative(self, x, z, f, **kwargs):
        """ Second derivative in z"""
        coef = self.data.v('coef', x=x, z=z, f=f)
        H = self.data.v('grid', 'low', 'z', x=x, z=z, f=f) - self.data.v('grid', 'high', 'z', x=x, z=z, f=f)

        Avz = -2.*coef/H**2. * (H/self.H0)**self.m
        return Avz