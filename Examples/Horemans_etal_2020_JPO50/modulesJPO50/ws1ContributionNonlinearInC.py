def ws1ContributionNonlinearInC(A0, dA0, A2, dA2, beta, Kv, tau):
    r"""Computes the contribution in the first order settling velocity ws1 that does not scale linearly in c and
    is due to inertia effects, settling, and vertical diffusion.
    Parameters
    ----------
    A0 : complex float
        the subtidal amplitude corresponding to the zeroth order suspended sediment concentration.
    dA0 : complex float
        the derivative to z of the subtidal amplitude corresponding to the zeroth order suspended sediment
        concentration.
    A2 : complex float
        the M4 tidal amplitude corresponding to the zeroth order suspended sediment concentration.
    dA2 : complex float
        the derivative to z of the M4 tidal amplitude corresponding to the zeroth order suspended sediment
        concentration.
    Returns
    -------
    hatws10 : ndarray, shape (jmax+1, kmax+1)
        ws10 contribution which does not scale linearly in c and is due to inertia effects, settling, and vertical
        diffusion.
    hatws14 : ndarray, shape (jmax+1, kmax+1)
        ws14 contribution which does not scale linearly in c and is due to inertia effects, settling, and vertical
        diffusion.

    References
    ----------
    .. [1] D. M. L. Horemans, Y. M. Dijkstra, H. M. Schuttelaars, P. Meire and T. J. S. Cox, (2020): Unraveling the
    Essential Effects of Flocculation on Large-Scale Sediment Transport Patterns in a Tide-Dominated Estuary. Journal of
    Physical Oceanography, 50(7), 1957-1981. doi: 10.1175/jpo-d-19-0232.1
    """

    from numpy.lib import scimath
    import numpy as np
    from modulesJPO50.savitzky_golay import savitzky_golay

    # Apply smoothing of the suspended sediment concentration
    for zi in range(0, A2.shape[1]):
        A0[:,zi] = savitzky_golay(A0[:, zi], window_size=15, order=1)
        A2[:, zi] = savitzky_golay(A2[:, zi], window_size=15, order=1)
        dA0[:, zi] = savitzky_golay(dA0[:, zi], window_size=15, order=1)
        dA2[:, zi] = savitzky_golay(dA2[:, zi], window_size=15, order=1)

    # Compute the complex conjugate of the suspended sediment concentration
    A0C = np.conj(A0)
    A2C = np.conj(A2)
    dA0C = np.conj(dA0)
    dA2C = np.conj(dA2)

    # Define variables following Horemans et al. (2020)
    zdk1 = scimath.sqrt((-(dA0+dA0C))/dA2 + scimath.sqrt((dA0+dA0C)**2 - 4*dA2*dA2C)/dA2)/scimath.sqrt(2)

    zdk2 = scimath.sqrt((-(dA0+dA0C))/dA2 - scimath.sqrt((dA0+dA0C)**2 - 4*dA2*dA2C)/dA2)/scimath.sqrt(2)

    zk1 = scimath.sqrt((-(A0+A0C))/A2 + scimath.sqrt((A0+A0C)**2 - 4*A2*A2C)/A2)/scimath.sqrt(2)

    zk2 = scimath.sqrt((-(A0+A0C))/A2 - scimath.sqrt((A0+A0C)**2 - 4*A2*A2C)/A2)/scimath.sqrt(2)

    z = scimath.sqrt((-(A0+A0C))/A2 + scimath.sqrt((A0+A0C)**2 - 4*A2*A2C)/A2)/scimath.sqrt(2)

    Z1 = dA2*(z**2 - zdk1**2)*(z**2 - zdk2**2)

    Z2 = A2*(z + zk1)*(z**2 - zk2**2)

    dZ1 = 2*dA2*z*(z**2 - zdk1**2) + 2*dA2*z*(z**2 - zdk2**2)

    dZ2 = 2*A2*z*(z + zk1) + A2*(z**2 - zk2**2)

    ddZ1 = 8*dA2*z**2 + 2*dA2*(z**2 - zdk1**2) + 2*dA2*(z**2 - zdk2**2)

    ddZ2 = 4*A2*z + 2*A2*(z + zk1)

    F1 = Z1/Z2

    F2 = Z1**2/Z2**3

    dF2 = (-3*dZ2*Z1**2)/Z2**4 + (2*dZ1*Z1)/Z2**3

    ddF2 = -3*((-4*dZ2**2*Z1**2)/Z2**5 + (2*dZ1*dZ2*Z1)/Z2**4 + (ddZ2*Z1**2)/Z2**4) \
           - (6*dZ1*dZ2*Z1)/Z2**4 + (2*dZ1**2)/Z2**3 + (2*ddZ1*Z1)/Z2**3

    # Compute hatws10
    n=0
    A = F1 * (z ** (-1 - n) + z ** (-1 + n)) + (Kv * (
            -(F2 * (1 - n) * n * z ** (-1 - n)) + ddF2 * z ** (1 - n) + F2 * n * (1 + n) * z ** (-1 + n) + (
                2 * dF2 * (1 - n)) / z ** n + 2 * dF2 * (1 + n) * z ** n + ddF2 * z ** (1 + n))) / beta \
    +dA2C/A2C

    hatws10 = -2*tau*A

    # Compute hatws14
    n=2
    A = F1 * (z ** (-1 - n) + z ** (-1 + n)) + (Kv * (
            -(F2 * (1 - n) * n * z ** (-1 - n)) + ddF2 * z ** (1 - n) + F2 * n * (1 + n) * z ** (-1 + n) + (
                2 * dF2 * (1 - n)) / z ** n + 2 * dF2 * (1 + n) * z ** n + ddF2 * z ** (1 + n))) / beta \
    +(2*dA2C**2*Kv + A2C**2*dA0* beta + A2C**2*dA0C* beta - A0*A2C*dA2C* beta - A0C*A2C*dA2C* beta)/(2.*A2C**3* beta)
    B = 1j*(F1*(z**(-1 - n) - z**(-1 + n)) +
     (Kv*(-(F2*(1 - n)*n*z**(-1 - n)) + ddF2*z**(1 - n) + F2*(-1 - n)*n*z**(-1 + n) +
          (2*dF2*(1 - n))/z**n - 2*dF2*(1 + n)*z**n - ddF2*z**(1 + n)))/beta +
     (2*dA2C**2*Kv + A2C**2*dA0*beta + A2C**2*dA0C*beta - A0*A2C*dA2C*beta - A0C*A2C*dA2C*beta)/
      (2.*A2C**3*beta))

    hatws14 = -2*tau*(A-B*1j)

    return hatws10, hatws14