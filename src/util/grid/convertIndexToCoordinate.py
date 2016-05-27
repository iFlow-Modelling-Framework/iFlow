"""
convertIndexToCoordinate

Date: 15-12-15
Authors: Y.M. Dijkstra
"""
from nifty.toList import toList


def convertIndexToCoordinate(grid, indices):
    """Convert matrix indices to coordinates on the grid provided in data. Presently only works for regular grids.

    Parameters:
        grid (dictionary): contains the grid data
        indices (tuple): set of coordinates in order of dimensions. May contain less or equal number of elements as dimensions

    Returns:
        coordinates (dict): dict of form (dim1:..., dim2=... etc)
    """

    if grid['gridtype'] == 'Regular':
        coordinates = {}
        for n, dim in enumerate(grid['dimensions']):
            try:
                # set entry in indices to coordinates if possible
                ind = indices[n]
            except IndexError:
                pass
            else:
                coordinates[dim] = grid['axis'][dim][(slice(None),)*n+(ind,)].reshape(len(toList(ind)))

    return coordinates


