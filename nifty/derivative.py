"""
Take the derivative from data in a numerical array

Date: 10-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from src.util.diagnostics import KnownError
import src.config as cf


def derivative(u, dimNo, grid, *args, **kwargs):
    """Compute the derivative of numerical array. The method of taking the derivative is specified in src.config (DERMETHOD)
    NB. when requesting a shape that has more dimensions that the data, this method fails. Needs fixing (TODO)

    Parameters:
        u (ndarray or scalar) - data to take derivative of.
                                length of the data on each axis should be either 1 or the same as the grid over that axis. Length 1 axes at the end may be omitted.
        dimNo (int or str) - number or name of dimension to take the derivative of
        grid (DataContainer) - DataContainer containing grid information
        args (ndarray/list, optional) - indices at which the derivative is requested.
                                      When omitted data is requested at the original grid.
                                      For omitting dimensions, the same rules hold as for the .v method of the DataContainer
        kwargs (optional): DERMETHOD: derivative method. If not set, this is taken from the config file

    Returns:
        ux (ndarray) - Numerical derivative of u in dimension dimNo on indices or original grid
    """

    # find dimension name corresponding to dimNo (or vv)
    if isinstance(dimNo, int):
        dim = grid.v('grid', 'dimensions')[dimNo]
    else: # else assume dimNo is a string with dimension name
        dim = dimNo
        dimNo = grid.v('grid', 'dimensions').index(dim)

    # take derivative along this axis, ignoring grid contraction
    der = axisDerivative(u, dim, dimNo, grid, *args, **kwargs)

    # add effects of grid contraction
    contr = grid.v('grid', 'contraction')[:,dimNo]
    for contrDimNo, i in enumerate(contr):
        if i == 1:
            # if u has less dimensions than the grid, ignore these dimensions in the axes
            axisrequest = {}
            for num, j in enumerate(grid.v('grid', 'dimensions')):
                if num >= len(u.shape):
                    axisrequest[j] = 0
                elif u.shape[num] == 1:
                    axisrequest[j] = [0]
            axisrequest['copy'] = 'all'             # dimensional axis copied over frequency domain (necessary from v 2.4)
            axis = ny.dimensionalAxis(grid, contrDimNo, **axisrequest)
            # take derivative of axis
            axis_der = axisDerivative(axis, dim, dimNo, grid, *args, **kwargs)
            # if u has more dimensions than the grid, append these
            axis_der = axis_der.reshape(axis_der.shape+(1,)*(len(u.shape) - len(axis_der.shape)))
            # grid contraction
            der = der - axis_der*axisDerivative(u, grid.v('grid', 'dimensions')[contrDimNo], contrDimNo, grid, *args, **kwargs)

    return der

def axisDerivative(u, dim, dimNo, grid, *args, **kwargs):
    DERMETHOD = kwargs.get('DERMETHOD') or cf.DERMETHOD
    # Preparation
    #   determine the size of u
    u = np.asarray(u)
    inds = [np.arange(0, n) for n in u.shape]

    #   replace indices by requested indices wherever available
    for n in range(0, min(len(args), len(inds))):
        inds[n] = np.asarray(args[n])

    #   determine the maximum index along the axis of derivation
    #   if this maximum index does not exist or is zero, then return a zero array
    try:
        maxIndex = u.shape[dimNo]-1
        if maxIndex == 0:
            raise Exception
    except:
        # if dimNo is out of range or corresponds to a length 1 dimension, return an array of zeros; data has a constant derivative
        ux = np.zeros([len(inds[n]) for n in range(0, len(inds))])
        return ux


    # central derivative
    if DERMETHOD == 'CENTRAL':  # central method with first order at the boundaries
        upInds = inds
        try:
            upInd = np.minimum(np.asarray(args[dimNo])+1, maxIndex)
        except:
            upInd = np.asarray(list(range(1, maxIndex+1))+[maxIndex])
        upInds[dimNo] = upInd

        downInds = inds[:]
        try:
            downInd = np.maximum(np.asarray(args[dimNo])-1, 0)
        except:
            downInd = np.asarray([0] + list(range(0, maxIndex)))
        downInds[dimNo] = downInd

        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy='all'), (grid.v('grid', 'high', dim, *upInds, copy='all')-grid.v('grid', 'low', dim, *upInds, copy='all')))+grid.v('grid', 'low', dim, *upInds, copy='all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy='all'), (grid.v('grid', 'high', dim, *downInds, copy='all')-grid.v('grid', 'low', dim, *downInds, copy='all')))+grid.v('grid', 'low', dim, *downInds, copy='all')

        ux = (u[np.ix_(*upInds)]-u[np.ix_(*downInds)])/(upaxis-downaxis)
    elif DERMETHOD == 'FORWARD':    # first order forward
        upInds = inds
        try:
            upInd = np.minimum(np.asarray(args[dimNo])+1, maxIndex)
        except:
            upInd = np.asarray(list(range(1, maxIndex+1))+[maxIndex])
        upInds[dimNo] = upInd

        downInds = inds[:]
        try:
            downInd = np.minimum(np.asarray(args[dimNo]), maxIndex-1)
        except:
            downInd = np.asarray(list(range(0, maxIndex)) + [maxIndex-1])
        downInds[dimNo] = downInd

        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy='all'), (grid.v('grid', 'high', dim, *upInds, copy='all')-grid.v('grid', 'low', dim, *upInds, copy='all')))+grid.v('grid', 'low', dim, *upInds, copy='all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy='all'), (grid.v('grid', 'high', dim, *downInds, copy='all')-grid.v('grid', 'low', dim, *downInds, copy='all')))+grid.v('grid', 'low', dim, *downInds, copy='all')

        ux = (u[np.ix_(*upInds)]-u[np.ix_(*downInds)])/(upaxis-downaxis)
    elif DERMETHOD == 'CENTRAL2':   # central method with second order at the boundaries

        midInds = inds[:]
        try:
            midInd = np.minimum(np.maximum(np.asarray(args[dimNo]), 1), maxIndex-1)
            beta = np.asarray([0]*len(midInd))
            beta[[i for i in args[dimNo] if i==0]] = 4.
            beta[[i for i in args[dimNo] if i==maxIndex]] = -4.
        except:
            midInd = np.asarray([1]+range(1, maxIndex)+[maxIndex-1])
            beta = np.asarray([0]*len(midInd))
            beta[0] = 4.
            beta[-1] = -4.
        midInds[dimNo] = midInd

        upInds = inds[:]
        try:
            upInd = np.maximum(np.minimum(np.asarray(args[dimNo])+1, maxIndex), 2)
            gamma = np.asarray([1.]*len(midInd))
            gamma[[i for i in args[dimNo] if i==0]] = -1.
            gamma[[i for i in args[dimNo] if i==maxIndex]] = 3.
        except:
            upInd = np.asarray([2]+range(2, maxIndex+1)+[maxIndex])
            gamma = np.asarray([1.]*len(midInd))
            gamma[0] = -1.
            gamma[-1] = 3.
        upInds[dimNo] = upInd

        downInds = inds[:]
        try:
            downInd = np.minimum(np.maximum(np.asarray(args[dimNo])-1, 0), maxIndex-2)
            alpha = np.asarray([-1.]*len(midInd))
            alpha[[i for i in args[dimNo] if i==0]] = -3.
            alpha[[i for i in args[dimNo] if i==maxIndex]] = 1.
        except:
            downInd = np.asarray([0] + range(0, maxIndex-1)+ [maxIndex-2])
            alpha = np.asarray([-1.]*len(midInd))
            alpha[0] = -3.
            alpha[-1] = 1.
        downInds[dimNo] = downInd

        alpha = alpha.reshape([1]*dimNo+[len(alpha)]+[1]*(len(u.shape)-dimNo-1))
        beta = beta.reshape([1]*dimNo+[len(beta)]+[1]*(len(u.shape)-dimNo-1))
        gamma = gamma.reshape([1]*dimNo+[len(gamma)]+[1]*(len(u.shape)-dimNo-1))

        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy = 'all'), (grid.v('grid', 'high', dim, *upInds, copy = 'all')-grid.v('grid', 'low', dim, *upInds, copy = 'all')))+grid.v('grid', 'low', dim, *upInds, copy = 'all')
        #midaxis = np.multiply(grid.v('grid', 'axis', dim, *midInds, copy = 'all'), (grid.v('grid', 'high', dim, *midInds, copy = 'all')-grid.v('grid', 'low', dim, *midInds, copy = 'all')))+grid.v('grid', 'low', dim, *midInds, copy = 'all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy = 'all'), (grid.v('grid', 'high', dim, *downInds, copy = 'all')-grid.v('grid', 'low', dim, *downInds, copy = 'all')))+grid.v('grid', 'low', dim, *downInds, copy = 'all')

        ux = (gamma*u[np.ix_(*upInds)]+beta*u[np.ix_(*midInds)]+alpha*u[np.ix_(*downInds)])/(upaxis-downaxis)
    else:
        raise KnownError("Numerical derivative scheme '%s' is not implemented" %(DERMETHOD))
    return ux