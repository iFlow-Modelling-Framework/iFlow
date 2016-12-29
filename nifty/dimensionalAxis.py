"""
dimensionalAxis

Date: 08-Nov-15
Authors: Y.M. Dijkstra
"""
import numpy as np


def dimensionalAxis(grid, dim, *args, **kwargs):
    """

    :param grid:
    :param dim:
    :return:
    """
    # convert integer dimension to string if necessary
    if isinstance(dim, int):
        dim = grid.v('grid','dimensions')[dim]

    # optional argument copy in kwargs. value 'all' is allowed.
    copy = kwargs.get('copy')

    # if indices
    if args:
        indices = []
        for num, i in enumerate(grid.v('grid','dimensions')):
            if kwargs.get(i) is not None:
                indices.append(args[num])
            else:
                indices.append(range(0, grid.v('grid', 'maxIndex', i)+1))
        return np.multiply(grid.v('grid', 'axis', dim ,*indices, copy=copy), (grid.v('grid', 'high', dim, *indices, copy=copy)-grid.v('grid', 'low', dim, *indices, copy=copy)))+grid.v('grid', 'low', dim, *indices, copy=copy)

    # if coordinates
    else:
        dimlessaxis = {}
        for i in grid.v('grid','dimensions'):
            if kwargs.get(i) is not None:
                dimlessaxis[i] = kwargs.get(i)
            else:
                dimlessaxis[i] = grid.v('grid', 'axis', i).reshape(grid.v('grid', 'maxIndex', i)+1)
        dimlessaxis['copy'] = copy
        #dimlessaxis[dim] = grid.v('grid', 'axis', dim).reshape(grid.v('grid', 'maxIndex', dim)+1)
        return np.multiply(grid.v('grid', 'axis', dim ,**dimlessaxis), (grid.v('grid', 'high', dim, **dimlessaxis)-grid.v('grid', 'low', dim, **dimlessaxis)))+grid.v('grid', 'low', dim, **dimlessaxis)