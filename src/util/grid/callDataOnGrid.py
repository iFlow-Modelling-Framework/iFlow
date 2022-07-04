"""
callDataOnGrid

Original date: 15-12-15
Updated: 04-02-22
Authors: Y.M. Dijkstra
"""
import logging
import nifty as ny
from functools import reduce


def callDataOnGrid(dataContainer, keys, gridname, reshape):
    """Call data corresponding to keys from dataContainer on a certain grid

    Parameters:
        dataContainer (DataContainer) - contains all data (incl grid)
        keys (tuple of str or str) - single (!) key plus optional subkeys
        gridname (str) - key under which the grid can be found in the grid DataContainer
        reshape (bool) - reshape the requested data to grid size or keep in original format?

    Returns:
        Data requested on grid in varying format (str, ndarray, list, etxc).
        flag:   -1: error: no suitable grid
                0: warning: could not convert, returned original data
                1: conversion succesful
    """
    coordinates = {}
    if dataContainer.v(gridname, 'gridtype') == 'Regular':
        for dim in ny.toList(dataContainer.v(gridname, 'dimensions')):
            coordinates[dim] = dataContainer.v(gridname, 'axis', dim)
            coordinates[dim] = coordinates[dim].reshape(reduce(lambda x, y: x*y, coordinates[dim].shape))
    try:
        data = dataContainer.v(*keys, reshape=reshape, **coordinates)
        flag = 1
    except:
        data = dataContainer.v(*keys)
        flag = 0

    return data, flag

