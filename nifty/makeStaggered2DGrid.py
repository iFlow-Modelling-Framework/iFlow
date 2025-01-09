"""
Date: 21-07-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from src.util.grid import makeCollocatedGridAxis
from .toList import toList
import numpy as np
from functools import reduce
from copy import deepcopy

def makeStaggered2DGrid(dimensions, axisType, axisSize, axisOther, enclosures, contraction = None, copy = None):
    """Make separate staggered grid axes and enclosures for each dimension.

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
        copy - (list, optional) indicated whether lower-dimensional arrays should be copied over these axes.
                1: yes, copy. 0: no, only keep in zero-index. In not filled, a default 1 is inserted for every axis

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
        - copy: list of len(dimensions) - see input
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

        if i<2:
            axis_mid = np.zeros(len(axis)+1)
            axis_mid[0] = axis[0]
            axis_mid[-1] = axis[-1]
            axis_mid[1:-1] = 0.5*(axis[1:]+axis[:-1])
        else:
            axis_mid = axis

        grid['maxIndex'][dim] = len(axis_mid)-1
        grid['axis'][dim] = axis_mid.reshape((1,)*i+(len(axis_mid),))
        if toList(enclosures)[i]:
            grid['low'][dim] = toList(enclosures)[i][0]
            grid['high'][dim] = toList(enclosures)[i][1]

        # grid contraction matrix
        grid['contraction'] = np.zeros((len(grid['dimensions']), len(grid['dimensions'])))
        if contraction:
            for i, dim in enumerate(grid['dimensions']):
                for j in contraction[i]:
                    grid['contraction'][i, grid['dimensions'].index(j)] = 1.

        # grid copy vector
        if copy == None:
            copy = [1]*len(toList(dimensions))
        grid['copy'] = toList(copy)

    # u and v grids
    grid_u = deepcopy(grid)
    grid_v = deepcopy(grid)
    for i, dim in enumerate(grid['dimensions']):
        type = toList(axisType)[i]  # type, e.g. equidistant
        size = toList(axisSize)[i]  # number of grid cells
        other = toList(axisOther)[i]  # other grid arguments
        axis = makeCollocatedGridAxis(type, size, other)     # make an axis between 0 and 1
        if i <2:
            axis_mid = np.zeros(len(axis)+1)
            axis_mid[0] = axis[0]
            axis_mid[-1] = axis[-1]
            axis_mid[1:-1] = 0.5*(axis[1:]+axis[:-1])
            if i==0:
                axis_v = axis_mid
            else:
                axis_v = axis
            if i==1:
                axis_u = axis_mid
            else:
                axis_u = axis
        else:
            axis_u = axis
            axis_v = axis

        grid_u['maxIndex'][dim] =  len(axis_u)-1
        grid_u['axis'][dim] = axis_u.reshape((1,)*i+(len(axis_u),))
        grid_v['maxIndex'][dim] =  len(axis_v)-1
        grid_v['axis'][dim] = axis_v.reshape((1,)*i+(len(axis_v),))
    return grid, grid_u, grid_v