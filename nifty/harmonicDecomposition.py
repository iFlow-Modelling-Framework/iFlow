"""
harmonicDecomposition

Date: 16-10-15
Authors: R.L. Brouwer
"""

import numpy as np


def absoluteU(u, n):
    """Harmonic decomposition of |u(t)| = |U*cos(sigma*t-phi)| = |hat(u)*exp(sigma*t) + c.c.|, where
    hat(u) = U*exp(-i*phi) / 2. This function is used in iFlow amongst others to calculate the leading order erosion
    term for the concentration equation. In that case only values at the bed are relevant (z=-H).

    Parameters:
        u (ndarray) - complex amplitude of a variable that varies at M2 frequency only
        n (integer) - harmonic component (n = +/-k -> M_k).

    Returns:
        complex amplitude of the M_n tidal component of the absolute value of the M2 input variable.
        Note that there only exist components with frequencies that are an even multiple of the M2 tidal frequency, i.e.
        M0, M4, M8, etc. The real solution for the M_2n tidal signal then requires both +n and -n amplitude,
        alternatively one can take the +n-component and then add its complex conjugate.
    """
    if n % 2 == 0:
        k = n / 2
        absU = (2. / np.pi) * np.abs(u) * (u / np.conj(u))**k * (-1)**k / (1 - 4 * k**2)
    else:
        absU = 0
    return absU


def signU(u, n):
    """Harmonic decomposition of u / |u| = sg(u), where u is an M2 tidal signal. Here, u(t) = U*cos(sigma*t-phi) =
    hat(u)*exp(sigma*t) + c.c., where hat(u) = U*exp(-i*phi) / 2.

    Parameters:
        u (ndarray) - complex amplitude of a variable that varies at M2 frequency only
        n (integer) - harmonic component (n = +/-k -> M_k). All even tidal amplitudes M_2k are zero!

    Returns:
        complex amplitude of the M_n tidal component of the sign of the M2 input variable.
        Note that there only exist components with frequencies that are an odd multiple of the M2 tidal frequency, i.e.
        M2, M6, M10, etc. The real solution for the M_n tidal signal then requires both +n and -n amplitude,
        alternatively one can take the +n-component and then add its complex conjugate.
    """

    if n % 2 == 0:
        sgnU = 0
    else:
        # sgnU = -2 * (-1j)**(n + 1) * (u / abs(u))**n / (np.pi * n)
        k = (n-1)/2
        sgnU = (2 * (-1)**k / (np.pi * n)) * (u / abs(u))**n
        # sgnU[-1] = 0
    return sgnU
