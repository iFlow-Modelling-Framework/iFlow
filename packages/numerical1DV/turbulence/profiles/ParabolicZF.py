"""
Parabolic

Date: 10-Feb-16
Authors: Y.M. Dijkstra
"""
import nifty as ny
from nifty.functionTemplates import FunctionBase
import numpy as np


class ParabolicZF(FunctionBase):
    # Variables

    # Methods
    def __init__(self, dimNames, Av0, z0_dimless, H):
        """Initialise a parabolic profile.

        Args:
            dimNames: see function
            Av0 (complex array) - needs to be a real/complex array in the right shape
            z0_dimless (real) - dimensionless roughness height
            H (real) - Depth
        """
        FunctionBase.__init__(self, dimNames)
        self.Av0 = Av0
        self.z0_dimless = z0_dimless
        self.zs_dimless = self.zs_dimless(Av0[0], self.z0_dimless)
        self.H = H
        return

    def value(self, z, f, **kwargs):
        z = z.reshape((len(z), 1))
        Av = self.Av0[:, ny.toList(f)]*(self.zs_dimless-(-z))*(1+self.z0_dimless+(-z))
        return Av

    def derivative(self, z, f, **kwargs):
        if kwargs.get('dim') == 'z':
            z = z.reshape((len(z), 1))
            Avz = self.Av0[:, ny.toList(f)]/self.H*(-2.*(-z)-1.-self.z0_dimless+self.zs_dimless)

        elif kwargs.get('dim') == 'zz':
            Avz = -2.*self.Av0[:, ny.toList(f)]/self.H**2.
        else:
            Avz = None
            FunctionBase.derivative(self)       # derivatives in other directions not implemented
        return Avz

    def zs_dimless(self, Av00, z0_dimless):
        return np.minimum(10. ** -6,Av00) / (Av00 * (1 + z0_dimless)+10**-10)