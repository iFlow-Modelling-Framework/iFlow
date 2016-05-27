"""
Date: 21-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from src.util.grid import makeCollocatedGridAxis
from toList import toList
import numpy as np


def makeRegularGrid(dimensions, axisType, axisSize, axisOther, enclosures, contraction = None):
    """Make separate regular grid axes and enclosures for each of the dimensions in the method registry.
    Now only supports equidistant collocated axes using 'axisType' equal to equidistant

    Parameters:
        dimensions - (str or list of str) dimension names
        axisType - (str or list of str) type of axis. Now supports 'equidistant'
        axisSize - (int or list of int) length of axis
        axisOther -
        enclosures - (tuple or list of tuples) One tuple for each axis. A tuple consists of two elements:
                    lower and upper boundary (scalar, array or function), where lower corresponds to
                    the dimensionless 0 point and upper to the dimensionless 1 point
        contraction - (list of lists, optional) Requires one sublist per dimension. Sublist i contains the dimension names
                    on which the enclosure of dimension i depends. Assumes empty lists if not provided

    Returns:
        Dictionary with grid data. This includes:
        - gridtype (str) - name of grid type
        - dimensions (list of str) - list of dimensions in the correct order
        - axis: x in dimensions (ndarray) - grid axes between 0 and 1
        - high: x in dimensions (any) - dimension-full limit belonging to point 1 on the dimensionless axis
        - low: x in dimensions (any) - dimension-full limit belonging to point 0 on the dimensionless axis
        - maxIndex: x in dimensions (int) - maximum index number
        - contraction: (len(dimensions) x len(dimensions) ndarray) - matrix containing the grid contraction information;
                        has a 1 if dimension in rows depend on dimension in column (diagonal is 0 by definition)
    """
    grid = {}

    # set type and variables order
    grid['gridtype'] = 'Regular'
    grid['dimensions'] = toList(dimensions)
    grid['axis'] = {}
    grid['high'] = {}
    grid['low'] = {}
    grid['maxIndex'] = {}

    # loop over dimensions to fill the rest of the dictionary
    for i, dim in enumerate(grid['dimensions']):
        type = toList(axisType)[i]  # type, e.g. equidistant
        size = toList(axisSize)[i]  # number of grid cells
        other = toList(axisOther)[i]  # other grid arguments
        axis = makeCollocatedGridAxis(type, size, other)     # make an axis between 0 and 1

        grid['maxIndex'][dim] = reduce(lambda x, y: x*y, axis.shape)-1
        grid['axis'][dim] = axis.reshape((1,)*i+(len(axis),))
        if toList(enclosures)[i]:
            grid['low'][dim] = toList(enclosures)[i][0]
            grid['high'][dim] = toList(enclosures)[i][1]

        # grid contraction matrix
        grid['contraction'] = np.zeros((len(grid['dimensions']), len(grid['dimensions'])))
        if contraction:
            for i, dim in enumerate(grid['dimensions']):
                for j in contraction[i]:
                    grid['contraction'][i, grid['dimensions'].index(j)] = 1.

    return grid