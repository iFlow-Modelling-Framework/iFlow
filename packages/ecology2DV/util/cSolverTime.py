import numpy as np
import scipy.linalg


def cSolverTime(T, F, G1, G2, H, HCprev, X0, bctype, QL, dt, C0, data):
    """
    Make one time step in solving:
    HX_t + (TX + FX_x)_x = G1X + G2
    BC: X(0) = X0
        TX(L)+FX_x(L) = QL

    with initial condition C0 and time step dt

    Args:
        T: (jmax+1) advective transport subtidal
        F: (jmax+1) diffusive transport subtidal
        G: (jmax+1) net growth subtidal (1: implicit, 2: explicit)
        H: (jmax+1) coefficient in front of inertial term
        HCprev: (jmax+1) inertial term at previous time
        bctype: (string) type of upstream boundary: flux or concentration
        P0: (1) depth-averaged P at seaward boundary
        QL: (1) P flux at landward boundary
        dt:  time step
        C0: (jmax+1) initial condition for C
        data: (dc) grid data

    Returns:

    """
    jmax = data.v('grid', 'maxIndex', 'x')

    x = data.v('grid', 'axis', 'x')
    dx = (x[1:]-x[:-1])*data.v('L')

    ##### LEFT HAND SIDE #####
    # init
    A = np.zeros([3, jmax+1])

    # dx vectors
    dx_down = (dx[:jmax-1])
    dx_up = (dx[1:jmax])
    dx_av = 0.5*(dx[1:jmax]+dx[:jmax-1])

    ## components of equation
    # adv
    a = - np.maximum(T, 0)[:-2]/dx_down
    b =   np.maximum(T, 0)[1:-1]/dx_down - np.minimum(T, 0)[1:-1]/dx_up
    c =                                    np.minimum(T, 0)[2:]/dx_up

    # dif
    a +=   0.5*(F[1:-1]+F[:-2])/(dx_down*dx_av)
    b += - 0.5*(F[1:-1]+F[:-2])/(dx_down*dx_av) - 0.5*(F[1:-1]+F[2:])/(dx_up*dx_av)
    c +=                                          0.5*(F[1:-1]+F[2:])/(dx_up*dx_av)

    # source
    b += -np.minimum(G1[1:-1], 0)
    b += -np.minimum(G2[1:-1], 0)/(C0[1:-1]+1.e-15)

    # inertia
    b += H[1:-1]/dt

    # Build matrix
    A[0, 2:] = c
    A[1, 1:-1] = b
    A[2, :-2] = a

    ## BC
    # sea
    A[1, 0] = 1.

    # land
    if bctype == 'flux':
        A[1, -1] = T[-1] + F[-1]/dx[-1]
        A[2, -2] =       - F[-1]/dx[-1]
    else:
        A[1, -1] = 1.

    ## RHS
    rhs = np.zeros((jmax+1))
    rhs[1:-1] = HCprev[1:-1]/dt
    rhs[1:-1] += np.maximum(G2[1:-1], 0)
    rhs[1:-1] += np.maximum(G1[1:-1], 0)*C0[1:-1]
    rhs[0] = X0
    rhs[-1] = QL
    C = scipy.linalg.solve_banded((1, 1), A, rhs, overwrite_ab=False, overwrite_b=False)

    return C