"""

Date: 30-03-2020
Authors: Y.M. Dijkstra
"""
import numpy as np
import scipy.interpolate
import logging
import nifty as ny


class PresetLength:
    # Variables

    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        H = self.input.v('H0')
        B = self.input.v('B0')
        ssea = self.input.v('ssea')
        g = self.input.v('G')
        beta = self.input.v('BETA')
        c = np.sqrt(H*g*beta*ssea)
        Frnew = self.input.v('Q1')/(H*B*c)
        Kh = self.input.v('Kh')
        # Fr = np.array([0.000236403,0.000367479,0.00057123,0.000887953,0.001380285,0.002145595,0.003335237,0.005184486,0.008059065,0.012527477,0.019473434,0.030270633,0.047054423,0.073144116,0.113699444,0.176740993,0.274736422,0.427066185,0.663856377,1.05])
        # L = np.array([8.0E+07,6.4E+07,5.0E+07,4.0E+07,3.3E+07,2.4E+07,2.0E+07,1.6E+07,1.3E+07,8.7E+06,7.0E+06,5.7E+06,4.4E+06,3.5E+06,2.1E+06,1.5E+06,6.8E+05,2.0E+05,4.0E+04,8.0E+03])
        # Lnew = scipy.interpolate.interp1d(Fr, L, bounds_error=False, fill_value='extrapolate')
        # Lnew = float(Lnew(Frnew))
        Lnew = 6*Kh/(Frnew*c)

        print('-------------')
        print('Length is set to %s'%(np.round(Lnew)))
        print('-------------')

        ## To dict
        d = {}
        d['H'] = H
        d['B'] = B
        d['L'] = Lnew

        return d


