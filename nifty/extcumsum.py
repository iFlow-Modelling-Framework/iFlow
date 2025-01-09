import numpy as np


def extcumsum(a, dx, axis=0):
    """
    Simple integration a*dx, cumulative, with size one bigger than 'a' along axis
    'dx' can be an array a.shape or a.shape[axis]
    """
    if axis<0:
        raise Exception('nifty extcumsum does not accept negative values for keyword axis')

    if len(dx.shape)==1:
        dx = dx.reshape([1]*axis+[len(dx)]+[1]*(len(a.shape)-axis-1))

    shape = list(a.shape)
    shape[axis] += 1
    i = np.zeros(shape, dtype=a.dtype)
    i[(slice(None),)*axis + (slice(1,None), )] = np.cumsum(a*dx, axis=axis)
    return i