"""
callDataOnGrid

Date: 15-12-15
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
from functools import reduce


def callDataOnGrid(dataContainer, keys, grid, gridname, reshape):
    """Call data corresponding to keys from dataContainer on a certain grid

    Parameters:
        dataContainer (DataContainer) - contains all data
        keys (tuple of str or str) - single (!) key plus optional subkeys
        grid (DataContainer) - a data container containing a grid satisfying grid conventions.
                               This function currently only supports 'Regular' grid types
        gridname (str) - key under which the grid can be found in the grid DataContainer
        reshape (bool) - reshape the requested data to grid size or keep in original format?

    Returns:
        Data requested on grid in varying format (str, ndarray, list, etxc).
        flag:   -1: error: no suitable grid
                0: warning: could not convert, returned original data
                1: conversion succesful
    """
    axes = {}
    if grid.v(gridname, 'gridtype') == 'Regular':
        axes = {}
        for dim in ny.toList(grid.v(gridname, 'dimensions')):
            axes[dim] = grid.v(gridname, 'axis', dim)
            axes[dim] = axes[dim].reshape(reduce(lambda x, y: x*y, axes[dim].shape))
    try:
        data = dataContainer.v(*keys, reshape=reshape, **axes)
        flag = 1
    except:
        data = dataContainer.v(*keys)
        flag = 0

    return data, flag

