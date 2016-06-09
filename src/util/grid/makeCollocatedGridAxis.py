"""
makeCollocatedGridAxis

Date: 15-12-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import numpy.polynomial.legendre as lg


def makeCollocatedGridAxis(method, *args):
    """Make a single collocated grid axes. Several methods are implemented:
    1. equidistant: equidistant axis between 0 and 1. Requires nPoints for the number of grid points (including edge points 0 and 1).
    2. integer: integers from 0 to required argument integer
    3. logarithmic
    4. loglin
    5. LegendreGaussLobatto: spectral grid for Legendre polynomials. Corresponds to the zeros of P'_N + boundary points
    6. list: takes the list/array in args as the grid axis and makes sure to scale it from 0 to 1
    7. file: requires file path to an ascii file. The file format is:
                nPoints
                coordinate[0] >= 0
                coordinate[1]
                "       "
                coordinate[nPoints] <= 1


    Parameters:
        method (str)- names of the allowed axis types, see description above
        args - nPoints      if method='equidistant', 'integer', 'LegendreGaussLobatto'
              OR filePath     if method='file'

    Returns:
        grid axis as numpy array or None if method is not available

    """
    grid = None
    if method=='equidistant':
        nPoints = args[0]
        grid = np.linspace(0, 1, nPoints+1)

    if method=='logarithmic':
        nPoints = args[0]
        steepness = args[1]
        grid = (np.exp(steepness*np.linspace(0, 1, nPoints+1))-1.)/(np.exp(steepness)-1.)

    if method=='loglin':
        nPoints = args[0]           # args[0]: number of grid points
        dzlog = args[1][0]          # args[1]: dimensionless thickness of the logarithmic layer
        steepness = args[1][1]      # args[2]: steepness (negative results in refinement near upper part of axis)

        gamma = abs(steepness)
        dzlog_trans = np.log(dzlog*(np.exp(gamma)-1)+1)/gamma
        delta = dzlog_trans*gamma*(np.exp(gamma*dzlog_trans))/(np.exp(gamma)-1.)
        nLog = int(nPoints*delta/(1-dzlog+delta))
        dztoplog = delta/nLog
        grid = np.zeros(nPoints+1)
        grid[:nLog+1] = (np.exp(gamma*np.linspace(0, dzlog_trans, nLog+1))-1.)/(np.exp(gamma)-1.)
        grid[nLog+1:] = np.linspace(dzlog+dztoplog, 1, nPoints-nLog)
        if steepness < 0:
            grid = -grid[slice(None, None, -1)]+1

    elif method=='integer':
        nPoints = args[0]
        grid = np.arange(0, int(nPoints+1))

    elif method=='LegendreGaussLobatto':
        # coefficients of derivative of P_N
        N = int(args[0])
        coef = lg.legder([0]*N+[1])

        # grid is equal to roots of P'_N + boundary points
        grid = np.zeros(N+1)
        grid[1:-1] = np.polynomial.legendre.legroots(coef)
        grid[0] = -1.
        grid[-1] = 1.

        # transform grid from (-1,1) to (1,0) (i.e. -1 maps to 1 & 1 maps to 0)
        grid = np.asarray([i for i in reversed(grid)])
        grid = 0.5*(-grid+1)

    elif method=='list':
        grid = np.asarray(args[0])
        grid = (grid - grid[0])/(grid[-1] - grid[0])

    elif method=='file':
        filePath = args[0]
        with open(filePath,'r') as f:
            nPoints = int(f.readline())
            grid = np.zeros(nPoints)
            i = 0
            for line in f:
                grid[i] = line
                i += 1

    return grid




