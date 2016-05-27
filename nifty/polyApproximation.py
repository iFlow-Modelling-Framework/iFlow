"""
polyApproximation
Approximate a function 'func' on the interval [-1, 1] using Chebyshev polynomials up to nth order
Returns the coefficients of a polynomial of the form c0 + c1*x + c2*x^2 + c3*x^3 + ...

Date: 21-Oct-15
Authors: Y.M. Dijkstra
"""
import math
import numpy as np
TOLERANCE = 1.e-4       # remove chebyshev coefficients smaller than this
ACCURACY = 100          # number of steps in numerical integration


def polyApproximation(func, n, TOLERANCE = TOLERANCE, ACCURACY = ACCURACY):
    fac = math.pi / ACCURACY
    c = 2./math.pi*np.asarray([fac * sum([func(math.cos(math.pi * (k + 0.5) / ACCURACY)) * math.cos(math.pi * j * (k + 0.5) / ACCURACY) for k in range(ACCURACY)]) for j in range(n+1)])
    c[0] = 0.5*c[0]

    # apply a correction to remove small coefficients
    for i in range(n+1):
        if abs(c[i]) < TOLERANCE:
            c[i]=0

    # convert chebyshev coefficients to polynomial basis coefficients
    c = np.polynomial.chebyshev.cheb2poly(c)
    c = np.append(c, [0]*(n+1-len(c)))
    return c