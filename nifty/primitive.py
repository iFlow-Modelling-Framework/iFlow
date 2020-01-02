"""


Date: 10-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
import src.config as cf
from src.util.diagnostics import KnownError
import scipy.integrate
import scipy.interpolate


def primitive(u, dimNo, low, high, grid, *args, **kwargs):
    """Compute the integral of numerical array between low and high. The method is specified in src.config (INTMETHOD)
    NB. 'indices' now only as indices (not as coordinates)
    NB. when requesting a shape that has more dimensions that the data, this method fails. Needs fixing (TODO)
    NB. low/high only as indices. Needs coordinates (incl interpolation) later

    Parameters:
        u (ndarray) - data to take integral of
        dimNo (int or str) - number or name of dimension to take the integral of
        low (int) - index of lower integration boundary grid point
        high (int) - index of upper integration boundary grid point
        data (DataContainer) - DataContainer containing grid information
        args (ndarray/list, optional) - indices at which the integral is requested.
                                      Dimension of integration should be included to keep the order of dimensions correct, but this information is not used
                                      When omitted data is requested at the original grid.
                                      For omitting dimensions, the same rules hold as for the .v method of the DataContainer
        kwargs (optional): INTMETHOD: interpolation method. If not set, this is taken from the config file

    Returns:
        Ju (ndarray) - Numerical integral of u in dimension dimNo evaluated between grid indices 'low' and 'high'
    """
    INTMETHOD = kwargs.get('INTMETHOD') or cf.INTMETHOD
    # find dimension name corresponding to dimNo (or vv)
    if isinstance(dimNo, int):
        dim = grid.v('grid', 'dimensions')[dimNo]
    else: # else assume dimNo is a string with dimension name
        dim = dimNo
        dimNo = grid.v('grid', 'dimensions').index(dim)

    # preparation: determine the size of u and the maximum index along the axis of derivation
    # if this maximum index does not exist or is zero, then return a zero array
    u = np.asarray(u)
    inds = [np.arange(0, n) for n in u.shape]

    # replace indices by requested indices wherever available
    for n in range(0, min(len(args), len(inds))):
        inds[n] = np.asarray(args[n])

    # trapezoidal integration
    if INTMETHOD == 'TRAPEZOIDAL':
        # indices of upper and lower grid points
        upInds = inds
        downInds = inds[:]
        if high > low:
            upInds[dimNo] = [low] + list(range(low+1, high+1))
            downInds[dimNo] = [low] + list(range(low, high))
        else:
            upInds[dimNo] = list(range(high, low)) + [low]
            downInds[dimNo] = list(range(high+1, low+1)) + [low]

        # take grid axis at the grid points required
        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy = 'all'), (grid.v('grid', 'high', dim, *upInds, copy = 'all')-grid.v('grid', 'low', dim, *upInds, copy = 'all')))+grid.v('grid', 'low', dim, *upInds, copy = 'all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy = 'all'), (grid.v('grid', 'high', dim, *downInds, copy = 'all')-grid.v('grid', 'low', dim, *downInds, copy = 'all')))+grid.v('grid', 'low', dim, *downInds, copy = 'all')

        Ju = 0.5*(upaxis-downaxis)*(u[np.ix_(*upInds)]+u[np.ix_(*downInds)])

    elif INTMETHOD == 'INTERPOLSIMPSON':
        # indices of upper and lower grid points
        upInds = inds
        downInds = inds[:]

        axis = grid.v('grid', 'axis', dim)
        axis = axis.reshape(np.product(axis.shape))
        axis_mid = np.zeros(axis.shape)
        if high > low:
            upInds[dimNo] = [low] + list(range(low+1, high+1))
            downInds[dimNo] = [low] + list(range(low, high))
            axis_mid = 0.5*(axis[downInds[dimNo]] - axis[upInds[dimNo]])+axis[upInds[dimNo]]
        else:
            upInds[dimNo] = list(range(high, low)) + [low]
            downInds[dimNo] = list(range(high+1, low+1)) + [low]
            axis_mid = 0.5*(axis[downInds[dimNo]] - axis[upInds[dimNo]])+axis[upInds[dimNo]]


        # take grid axis at the grid points required
        upaxis = np.multiply(grid.v('grid', 'axis', dim, *upInds, copy = 'all'), (grid.v('grid', 'high', dim, *upInds, copy = 'all')-grid.v('grid', 'low', dim, *upInds, copy = 'all')))+grid.v('grid', 'low', dim, *upInds, copy = 'all')
        downaxis = np.multiply(grid.v('grid', 'axis', dim, *downInds, copy = 'all'), (grid.v('grid', 'high', dim, *downInds, copy = 'all')-grid.v('grid', 'low', dim, *downInds, copy = 'all')))+grid.v('grid', 'low', dim, *downInds, copy = 'all')

        uquad = scipy.interpolate.interp1d(axis, u, 'quadratic', axis=dimNo)
        Ju = 1./6*(upaxis-downaxis)*(u[np.ix_(*upInds)]+4.*uquad(axis_mid)+u[np.ix_(*downInds)])
    else:
        raise KnownError("Numerical integration scheme '%s' is not implemented" %(INTMETHOD))
    return Ju

