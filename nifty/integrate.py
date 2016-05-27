"""


Date: 10-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from toList import toList
from primitive import primitive
from src.util.diagnostics import KnownError
import src.config as cf
import scipy.integrate
import nifty as ny

def integrate(u, dimNo, low, high, grid, *args, **kwargs):
    """Compute the integral of numerical array between low and high. The method is specified in src.config (INTMETHOD)
    NB. 'indices' now only as indices (not as coordinates)
    NB. when requesting a shape that has more dimensions that the data, this method fails. Needs fixing (TODO)
    NB. low/high only as indices. Needs coordinates (incl interpolation) later

    Parameters:
        u (ndarray) - data to take integral of
        dimNo (int or str) - number or name of dimension to take the integral of
        low (int) - index of lower integration boundary grid point
        high (1d-array or int) - (list of) index/indices of upper integration boundary grid point
        data (DataContainer) - DataContainer containing grid information
        args (ndarray/list, optional) - indices at which the integral is requested.
                                      Dimension of integration should be included to keep the order of dimensions correct, but this information is not used
                                      When omitted data is requested at the original grid.
                                      For omitting dimensions, the same rules hold as for the .v method of the DataContainer
        kwargs (optional): INTMETHOD: integration method. If not set, this is taken from the config file

    Returns:
        Ju (ndarray) - Numerical integral of u in dimension dimNo evaluated between grid indices 'low' and 'high'.
            Shape of Ju is equal to u, with the exception that axis 'dimNo' has the same length as 'high'
    """
    INTMETHOD = kwargs.get('INTMETHOD') or cf.INTMETHOD
    # if string of dimension is provided, convert to number of axis
    if isinstance(dimNo, basestring):
        dimNo = grid.v('grid', 'dimensions').index(dimNo)

    if INTMETHOD == 'TRAPEZOIDAL' or INTMETHOD == 'INTERPOLSIMPSON':
        # determine 'maximum' of high, i.e. that value furthest from low
        high = np.asarray(toList(high))
        if max(high) > low:
            maxHigh = max(high)
            incr = 1
        else:
            maxHigh = min(high)
            incr = -1

        # Determine primitive function over each cell
        Ju = primitive(u, dimNo, low, maxHigh, grid, INTMETHOD=INTMETHOD, *args)

        # sum the primitive over each cell with respect to low
        size = [j for j in Ju.shape]
        size[dimNo] = len(high)

        if incr == -1:
            uInt = np.zeros(Ju.shape, dtype=u.dtype)
            uInt[[slice(None)]*dimNo+[slice(1, None)]+[Ellipsis]] = np.cumsum(Ju, axis=dimNo)[[slice(None)]*dimNo+[slice(None, -1)]+[Ellipsis]]
            high = high - min(high)
            uInt = (-uInt + uInt[[slice(None)]*dimNo+[[-1]]+[Ellipsis]])[[slice(None)]*dimNo+[high]+[Ellipsis]]
        else:
            uInt = np.cumsum(Ju, axis=dimNo)
            high = high - low
            uInt = uInt[[slice(None)]*dimNo+[high]+[Ellipsis]]

    elif INTMETHOD == 'SIMPSON':
        # make z the same shape as u
        #   if u has less dimensions or length-1 dimensions, remove them from z
        axisrequest = {}
        for num, j in enumerate(grid.v('grid', 'dimensions')):
            if num >= len(u.shape):
                axisrequest[j] = 0
            elif u.shape[num] == 1:
                axisrequest[j] = [0]
        axis = ny.dimensionalAxis(grid, 'z', **axisrequest)
        #   if u has more dimensions, append them
        axis = axis.reshape(axis.shape+(1,)*(len(u.shape) - len(axis.shape)))*np.ones(u.shape)

        # integrate
        uInt = np.zeros(u.shape[:dimNo]+(len(ny.toList(high)),)+u.shape[dimNo+1:], dtype=u.dtype)
        high = np.asarray(toList(high))
        if max(high) > low:
            incr = 1
        else:
            incr = -1
        for i, hi in enumerate(high+incr):
            if hi < 0:
                hi = None
            slices = [slice(None)]*dimNo+[slice(low, hi, incr)]+[Ellipsis]
            uInt[[slice(None)]*dimNo+[i]+[Ellipsis]] = scipy.integrate.simps(u[slices], axis[slices], axis=dimNo)
    else:
        raise KnownError("Numerical integration scheme '%s' is not implemented" %(INTMETHOD))
    return uInt
