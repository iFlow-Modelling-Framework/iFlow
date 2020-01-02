"""
complexAmplitudeProduct

Date: 12-01-2016
Authors: Y.M. Dijkstra
"""
import numpy as np


def complexAmplitudeProduct(u, v, dim, includeNegative=False):
    """Multiplication two signals using their complex Fourier coefficients.
    This method requires u and v to contain positive coefficients only (incl 0).
    The resulting signal is also represented by a (truncated) set of Fourier coefficients.
    The signals can depend on multiple variables with the Fourier coefficients given in axis 'dim.

    Parameters:
        u (ndarray) - complex positive Fourier coefficients of the first signal
        v (ndarray) - complex positive Fourier coefficients of the second signal.
            Needs to be of almost the same shape as u.
            Exceptions: 1) v can have possible 'length 1' axes where u has a 'length n' axis, or vice versa.
                        2) 'length 1' dimensions at the end of v may be omitted, even if u has 'length n' dimensions there, or vice versa.
        dim (int) - axis number along which the Fourier coefficients are found
        includeNegative (bool, optional) - u and v include negative Fourier components

    Returns:
        complex ndarray with the same shape as u and v (or the maximum length of each of the axes of u and v, if these are different).
    """
    # if any of the arrays has a 'length 0' dimension, immediately return an empty array of the right size
    if (0 in u.shape) or (0 in v.shape):
        uv = np.zeros([x.shape for x in [(u, v)[np.argmax([len(u.shape), len(v.shape)])]]][0])
        return uv

    # if negative components are included, call a different function
    if includeNegative:
        uv = complexAmplitudeProductNegative(u, v, dim)
        return uv

    # set shape of return array and of u and v in case of omitted dimensions
    dif = len(u.shape)-len(v.shape)
    u = u.reshape(u.shape+(1,)*(-dif))
    v = v.reshape(v.shape+(1,)*(dif))
    size = [max(u.shape[i], v.shape[i]) for i in range(0, len(u.shape))]
    uv = np.zeros(size, dtype=complex)
    vcon = np.conj(v)
    ucon = np.conj(u)

    for i in range(0, u.shape[dim]):
        # a-part
        sliceUV = (slice(None),)*dim + (i,) + (Ellipsis,)

        slice1 = slice(None, i+1)
        slice2 = slice(i, None, -1)
        uv[sliceUV] = 0.5*np.sum(u[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*v[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)

        # b-part
        end = -i or None
        multiplier = 0.25+0.25*bool(i)
        slice1 = slice(i, None)
        slice2 = slice(None, end)
        uv[sliceUV] += multiplier*np.sum(u[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*vcon[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)
        uv[sliceUV] += multiplier*np.sum(v[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*ucon[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)
    return uv

def complexAmplitudeProductNegative(u, v, dim):
    """Same as above, now with u and v containing both negative and positive Fourier components:


    Parameters:
        u (ndarray) - complex positive and negative Fourier coefficients of the first signal
        v (ndarray) - complex positive and negative Fourier coefficients of the second signal.
            Needs to be of the same shape as u with possible 'length 1' axes where u has a 'length n' axis, or vice versa.
        dim (int) - axis number along which the Fourier coefficients are found
        includeNegative (bool, optional) - u and v include negative Fourier components

    Returns:
        complex ndarray with the same shape as u and v (or the maximum length of each of the axes of u and v, if these are different).
    """

    dif = len(u.shape)-len(v.shape)
    u = u.reshape(u.shape+(1,)*(-dif))
    v = v.reshape(v.shape+(1,)*(dif))
    size = [max(u.shape[i], v.shape[i]) for i in range(0, len(u.shape))]
    uv = np.empty(size, dtype=complex)
    fmax = int((uv.shape[dim]-1)/2)
    vcon = np.conjugate(v)
    ucon = np.conjugate(u)

    for i in range(0, u.shape[dim]):
        # a-part
        sliceUV = (slice(None),)*dim + (i,) + (Ellipsis,)

        beg = max(0, i-fmax) or None
        end = min(u.shape[dim]-1, i+fmax)
        slice2 = slice(beg, end+1)
        try:
            slice1 = slice(end, beg-1, -1)
        except:
            slice1 = slice(end, beg, -1)
        uv[sliceUV] = 0.5*np.sum(u[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*v[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)

        # b-part
        beg = max(0, i-fmax) or None
        end = min(u.shape[dim]-1, i+fmax)
        slice1 = slice(beg, end+1)
        try:
            slice2 = slice(-end-1, -beg)
        except:
            slice2 = slice(-end-1, beg)

        uv[sliceUV] += 0.25*np.sum(u[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*vcon[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)
        uv[sliceUV] += 0.25*np.sum(v[(slice(None),)*dim + (slice1,)+(Ellipsis,)]*ucon[(slice(None),)*dim + (slice2,)+(Ellipsis,)], axis=dim)

    return uv

