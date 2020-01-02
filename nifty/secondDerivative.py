"""
Take the derivative from data in a numerical array

Date: 10-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny
from src.config import SECONDDERMETHOD
from src.util.diagnostics import KnownError
from derivative import derivative
from derivative import axisDerivative


def secondDerivative(u, dimNo, grid, *args):
    """Compute the 2nd derivative of numerical array. This implements now only the 2nd derivative over the same dimenion
    The method of taking the derivative is specified in src.config (SECONDDERMETHOD)
    NB. when requesting a shape that has more dimensions that the data, this method fails. Needs fixing (TODO)
    NB. allows for grid contractions, but only in a single direction

    Parameters:
        u (ndarray or scalar) - data to take derivative of.
                                length of the data on each axis should be either 1 or the same as the grid over that axis. Length 1 axes at the end may be omitted.
        dimNo (int or str) - number or name of dimension to take the derivative of
        grid (DataContainer) - DataContainer containing grid information
        args (ndarray/list, optional) - indices at which the derivative is requested.
                                      When omitted data is requested at the original grid.
                                      For omitting dimensions, the same rules hold as for the .v method of the DataContainer

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
    der = axisSecondDerivative(u, dim, dimNo, grid, *args)

    # add effects of grid contraction
    contr = grid.v('grid', 'contraction')[:,dimNo]
    for contrDimNo, i in enumerate(contr):
        if i == 1:
            # if u has less dimensions than the grid, ignore these dimensions in the axes
            axisrequest = {}
            for j in grid.v('grid', 'dimensions')[len(u.shape):]:
                axisrequest[j] = 0
            axis = ny.dimensionalAxis(grid, contrDimNo, **axisrequest)
            # take derivative of axis
            u_i = derivative(u, contrDimNo, grid, *args)    # first derivative wrt i
            u_di = derivative(derivative(u, contrDimNo, grid), dimNo, grid, *args)                                           # mixed derivative
            u_ii = axisSecondDerivative(u, grid.v('grid', 'dimensions')[contrDimNo], contrDimNo, grid, *args)                                     # 2nd derivative wrt i
            axis_d = axisDerivative(axis, dim, dimNo, grid, *args)
            axis_dd = axisSecondDerivative(axis, dim, dimNo, grid, *args)

            # grid contraction
            der = der - 2.*u_di*axis_d - u_ii*axis_d**2-u_i*axis_dd   # TODO does not work yet for arrays with more dimensions than the grid

    return der

def axisSecondDerivative(u, dim, dimNo, grid, *args):
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
    #  NB SHOULD BE IMPROVED ON THE EDGES OF THE DOMAIN
    if SECONDDERMETHOD == 'CENTRAL':
        upInds = inds
        try:
            upInd = np.maximum(np.minimum(np.asarray(args[dimNo])+1, maxIndex), 2)
        except:
            upInd = np.asarray([2] + range(2, maxIndex+1)+[maxIndex])
        upInds[dimNo] = upInd

        midInds = inds[:]
        try:
            midInd = np.maximum(np.minimum(np.asarray(args[dimNo]), maxIndex-1), 1)
        except:
            midInd = np.asarray([1] + range(1, maxIndex)+[maxIndex-1])
        midInds[dimNo] = midInd

        downInds = inds[:]
        try:
            downInd = np.maximum(np.minimum(np.asarray(args[dimNo])-1, maxIndex-2), 0)
        except:
            downInd = np.asarray([0] + range(0, maxIndex-1)+[maxIndex-2])
        downInds[dimNo] = downInd

        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy='all'), (grid.v('grid', 'high', dim, *upInds, copy='all')-grid.v('grid', 'low', dim, *upInds, copy='all')))+grid.v('grid', 'low', dim, *upInds, copy='all')
        midaxis = np.multiply(grid.v('grid', 'axis', dim, *midInds, copy='all'), (grid.v('grid', 'high', dim, *midInds, copy='all')-grid.v('grid', 'low', dim, *midInds, copy='all')))+grid.v('grid', 'low', dim, *midInds, copy='all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy='all'), (grid.v('grid', 'high', dim, *downInds, copy='all')-grid.v('grid', 'low', dim, *downInds, copy='all')))+grid.v('grid', 'low', dim, *downInds, copy='all')
        dxup = upaxis-midaxis
        dxdown = midaxis-downaxis
        dxav = .5*(dxup+dxdown)

        umid = u[np.ix_(*midInds)]
        ux = (u[np.ix_(*upInds)]-umid)/(dxup*dxav) - (umid - u[np.ix_(*downInds)])/(dxdown*dxav)
    else:
        raise KnownError("Numerical derivative scheme '%s' is not implemented" %(SECONDDERMETHOD))
    return ux