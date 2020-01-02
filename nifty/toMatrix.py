import numpy as np
import nifty as ny
def toMatrix(a, includeNegative=False):
    if includeNegative:
        # DOES NOT WORK
        jmax = a.shape[0]-1
        ftot = a.shape[2]
        fmax = int((ftot-1)/2)

        Z = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
        for j in range(0, jmax+1):
            Z[j, 0, range(0, ftot), range(0, ftot)] = a[j, 0, fmax]
            for n in range(1, fmax+1):
                Z[j, 0, range(n, ftot), range(0, ftot-n)] = 0.5*a[j, 0, fmax+n] + 0.5*np.conj(a[j, 0, fmax-n])
                Z[j, 0, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(a[j, 0, fmax+n]) + 0.5*a[j, 0, fmax-n]
        # a = ny.eliminateNegativeFourier(a, 2)
        # jmax = a.shape[0]-1
        # ftot = (a.shape[2]-1)*2+1
        # fmax = a.shape[2]-1
        # Z = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
        # for j in range(0, jmax+1):
        #     Z[j, 0, range(0, ftot), range(0, ftot)] = a[j, 0, 0]
        #     for n in range(1, fmax+1):
        #         Z[j, 0, range(n, ftot), range(0, ftot-n)] = 0.5*a[j, 0, n]
        #         Z[j, 0, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(a[j, 0, n])

    else:
        jmax = a.shape[0]-1
        ftot = (a.shape[2]-1)*2+1
        fmax = a.shape[2]-1
        Z = np.zeros((jmax+1, 1, ftot, ftot), dtype=complex)
        for j in range(0, jmax+1):
            Z[j, 0, range(0, ftot), range(0, ftot)] = a[j, 0, 0]
            for n in range(1, fmax+1):
                Z[j, 0, range(n, ftot), range(0, ftot-n)] = 0.5*a[j, 0, n]
                Z[j, 0, range(0, ftot-n), range(n, ftot)] = 0.5*np.conj(a[j, 0, n])
    return Z