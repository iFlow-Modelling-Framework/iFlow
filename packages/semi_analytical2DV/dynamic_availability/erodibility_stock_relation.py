"""
function erodibility_stock_relation
compute <f> using average of instantaneous f. f is determined on the basis of the following rules:
1) if the bed is empty: all sediment in water column (erosion == deposition)
2) else: f=1

assumes c = chat * f, with int_{-H}^R chat = alpha1(1+alpha2*sin(2 omega t))
Input:
alpha2: M4/M0 ration in chat, see above
S: S/alpha1; scaled local stock

Date: 6-03-2017
Authors: R.L. Brouwer
"""
import numpy as np


def erodibility_stock_relation(alpha2, Shat):
    xi = np.arcsin((Shat - 1.) / alpha2)
    f = Shat * (0.5 - xi / np.pi) + 0.5 + xi / np.pi - alpha2 * np.cos(xi) / np.pi
    f[np.where(Shat < 1 - alpha2)[0]] = Shat[np.where(Shat < 1 - alpha2)[0]]
    f[np.where(Shat > 1 + alpha2)[0]] = 1.
    return np.real(f)

def erodibility_stock_relation_der(alpha2, Shat):
    xi = np.arcsin((Shat - 1.) / alpha2)
    dfdS = 0.5 - xi / np.pi
    dfdS[np.where(Shat <= 1 - alpha2)[0]] = 1.
    dfdS[np.where(Shat >= 1 + alpha2)[0]] = 0.
    return np.real(dfdS)