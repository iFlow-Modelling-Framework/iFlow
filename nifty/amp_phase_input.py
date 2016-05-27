"""
amp_phase_input

Date: 03-Mar-16
Authors: Y.M. Dijkstra
"""
import numpy as np
import nifty as ny

def amp_phase_input(amp, phase, shape):
    """ Compile complex quantity u from the amplitude and phase.
    Reshape this to 'shape', where the information is put in the last dimension of 'shape'

    Args:
        amp (scalar, list, 1D array): amplitude of first components of u. number of elements may be smaller or the same as the size of the last dimension of u
        phase (scalar, list, 1D array): phase (deg) of first components of u. number of elements may be smaller or the same as the size of the last dimension of u
        shape (tuple): shape of u.
    Returns:
        u - complex array of shape 'shape
    """
    nodim = len(shape)
    amp = ny.toList(amp)
    phase = ny.toList(phase)
    amp = np.asarray(ny.toList(amp[:min(len(amp), shape[-1])]))
    amp = amp.reshape((1,)*(nodim-1)+(len(amp),))
    phase = np.asarray(ny.toList(phase[:min(len(amp), shape[-1])]))
    phase = phase.reshape((1,)*(nodim-1)+(len(phase),))
    u = np.zeros(shape, dtype=complex)
    u[[slice(None)]*(nodim-1)+[slice(None, amp.shape[-1])]] = amp
    u[[slice(None)]*(nodim-1)+[slice(None, phase.shape[-1])]] = u[[slice(None)]*(nodim-1)+[slice(None, phase.shape[-1])]]*np.exp(-1j*phase/180.*np.pi)
    return u