"""
Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import numpy as np
from src.util.diagnostics import KnownError
from copy import copy


def derivativeOfFunction(u, dimNo, grid, gridname='grid', epsilon=1e-4, **kwargs):
    """Compute the derivative of a function. Uses a central method with equidistant distance set by epsilon along axis
    given by dimNo. At the edges of the grid, the method moves the centre points of the derivative to the interior
    domain so it can still take a central derivative.

    Parameters:
        u (function) - data to take derivative of.
        dimNo (int or str) - number or name of dimension to take the derivative of
        grid (DataContainer) - DataContainer containing grid information
        gridname (str) - name of the grid
        epsilon (float) - grid spacing for numerical approximation.
        kwargs - contains named coordinates where derivative is requested.

    Returns:
        ux (ndarray) - Numerical derivative of u in dimension dimNo on coordinates in kwargs
    """
    # find dimension name corresponding to dimNo (or vv)
    if isinstance(dimNo, int):
        dim = grid.v(gridname, 'dimensions')[dimNo]
    else: # else assume dimNo is a string with dimension name
        dim = dimNo
    coordinates = {}
    for i in grid.v(gridname, 'dimensions'):
        if i in kwargs.keys():
            coordinates[i] = kwargs[i]

    # take derivative along this axis, ignoring grid contraction
    axis_dimless = grid.v(gridname, 'axis', dim).flatten()
    ax_length = abs(axis_dimless[-1]-axis_dimless[0])

    # check if axis of differentiation is not outside of bounds of the grid: can prevent a common mistake of defining a
    # function on a wrong grid.
    if any(np.asarray(kwargs[dim])>np.max(axis_dimless)) or any(np.asarray(kwargs[dim])<np.min(axis_dimless)):
        KnownError('Derivative of function called with arguments outside the grid domain.')

    # set axis; in such a way that distance is always 2*epsilon for simplicity
    dx = 2*abs(epsilon)*ax_length
    axis_up = np.minimum(kwargs[dim] + 0.5*dx, np.max(axis_dimless))
    axis_down = np.maximum(axis_up - dx, np.min(axis_dimless))
    axis_up = axis_down + dx
    kwargs_up = copy(kwargs)
    kwargs_down = copy(kwargs)
    kwargs_up[dim] = axis_up
    kwargs_down[dim] = axis_down

    axis_scale = grid.v(gridname, 'high', dim, **coordinates) -grid.v(gridname, 'low', dim, **coordinates)

    grid.addData('u', u)    # add to DC to make sure reshaping rules are satisfied
    der = (grid.v('u',**kwargs_up)-grid.v('u',**kwargs_down))/(axis_scale*dx)

    return der

def secondDerivativeOfFunction(u, dimNo, grid, gridname='grid', epsilon = 1e-4, **kwargs):
    """Computes second derivative of a function using a central method (also at edges). See documentation of
    derivativeOfFunction for further information.
    """

    # find dimension name corresponding to dimNo (or vv)
    if isinstance(dimNo, int):
        dim = grid.v(gridname, 'dimensions')[dimNo]
    else: # else assume dimNo is a string with dimension name
        dim = dimNo
    coordinates = {}
    for i in grid.v(gridname, 'dimensions'):
        if i in kwargs.keys():
            coordinates[i] = kwargs[i]

    # take derivative along this axis, ignoring grid contraction
    axis_dimless = grid.v(gridname, 'axis', dim).flatten()
    ax_length = abs(axis_dimless[-1]-axis_dimless[0])

    # check if axis of differentiation is not outside of bounds of the grid: can prevent a common mistake of defining a
    # function on a wrong grid.
    if any(np.asarray(kwargs[dim])>np.max(axis_dimless)) or any(np.asarray(kwargs[dim])<np.min(axis_dimless)):
        KnownError('Derivative of function called with arguments outside the grid domain.')

    # set axis; in such a way that distance is always 2*epsilon for simplicity
    dx = abs(epsilon)*ax_length
    axis_up = np.minimum(kwargs[dim] + dx, np.max(axis_dimless))
    axis_down = np.maximum(axis_up - 2*dx, np.min(axis_dimless))
    axis_up = axis_down + 2*dx
    axis_mid = 0.5*(axis_up+axis_down)
    kwargs_up = copy(kwargs)
    kwargs_down = copy(kwargs)
    kwargs_mid = copy(kwargs)
    kwargs_up[dim] = axis_up
    kwargs_down[dim] = axis_down
    kwargs_mid[dim] = axis_mid

    axis_scale = grid.v(gridname, 'high', dim, **coordinates) -grid.v(gridname, 'low', dim, **coordinates)

    grid.addData('u', u)    # add to DC to make sure reshaping rules are satisfied
    der = (grid.v('u',**kwargs_up)-2*grid.v('u',**kwargs_mid)+grid.v('u',**kwargs_down))/(axis_scale*dx)**2

    return der