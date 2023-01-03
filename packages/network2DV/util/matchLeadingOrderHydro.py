import numpy as np
from copy import deepcopy
import nifty as ny


def matchLeadingOrderHydro(input, channel_list):
    Vertex = input.v('network_settings', 'label', 'Vertex')
    Sea = input.v('network_settings', 'label', 'Sea')   # labels of sea channels
    River = input.v('network_settings', 'label', 'River')    # labels of river channels
    tide = input.v('network_settings', 'hydrodynamics', 'M2')
    nch = input.v('network_settings', 'numberofchannels')  # number of channels

    # gather data from all channels
    zeta0_0 = []
    zeta0_L = []
    Q0_0 = []
    Q0_L = []
    for channel in channel_list:
        dc = channel.getOutput()
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        zeta0_0.append([dc.v('zeta0', 'tide', 0, 0, 1), dc.v('zeta0_reverse', 0, 0, 1)])
        zeta0_L.append([dc.v('zeta0', 'tide', jmax, 0, 1), dc.v('zeta0_reverse', jmax, 0, 1)])

        # depth-integrated horizontal velocity of mode
        u0 = dc.v('u0', 'tide', range(0, jmax+1), range(0, kmax+1), 1)
        u0_2 = dc.v('u0_reverse', range(0, jmax+1), range(0, kmax+1), 1)
        u0_DA = ny.integrate(u0, 'z', kmax, 0, dc)[0,0]
        uL_DA = 0

        # depth-integrated horizontal velocity of mode 2
        uL_2_DA = ny.integrate(u0_2, 'z', kmax, 0, dc)[-1,0]
        u0_2_DA = 0

        B_0 = dc.v('B', 0)
        B_L = dc.v('B',jmax)

        # transport
        Q0_0.append([u0_DA*B_0, u0_2_DA*B_0])
        Q0_L.append([uL_DA*B_L, uL_2_DA*B_L])

    # Match
    vertex_nosign = deepcopy(Vertex)
    sign = deepcopy(Vertex)
    for i in range(len(Vertex)):
        for x in range(len(Vertex[i])):
            vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
            sign[i][x] = np.sign(Vertex[i][x])

    M = np.zeros((2*nch, 2*nch), dtype=complex)
    v = np.zeros(2*nch, dtype=complex)
    v[0:len(Sea)] = tide
    row = 0

    # zeta0_L[# of Channel][# of Mode]
    # prescribe tidal surface elevation at the sea
    for i in Sea:
        M[row, [2*i, 2*i+1]] = np.array([zeta0_0[i][0], zeta0_0[i][1]])
        row += 1

    # u=0 at tidal limit/weir <=> Q0 = 0
    for i in River:
        M[row, [2*i, 2*i+1]] = np.array([Q0_L[i][0], Q0_L[i][1]])
        row += 1

    # continuous SSH
    for j in range(len(Vertex)):
        loc = vertex_nosign[j]  # indices of channels that are connected at branching point
        # loc[i]: index for channel
        for i in range(len(loc)-1):
            if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                M[row, [2*loc[i], 2*loc[i]+1]] = [zeta0_L[loc[i]][0], zeta0_L[loc[i]][1]]
            elif sign[j][i] == -1:
                M[row, [2*loc[i], 2*loc[i]+1]] = [zeta0_0[loc[i]][0], zeta0_0[loc[i]][1]]
            if sign[j][i-1] == 1:
                M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta0_L[loc[i-1]][0], -zeta0_L[loc[i-1]][1]]
            elif sign[j][i-1] == -1:
                M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta0_0[loc[i-1]][0], -zeta0_0[loc[i-1]][1]]
            row += 1

    # mass conservation
    for j in range(len(Vertex)):
        for i in range(len(vertex_nosign[j])):
            k = vertex_nosign[j][i] # channel index
            if sign[j][i] == 1:
                M[row, [2*k, 2*k+1]] = [Q0_L[k][0], Q0_L[k][1]]
            elif sign[j][i] == -1:
                M[row, [2*k, 2*k+1]] = [-Q0_0[k][0], -Q0_0[k][1]]
        row += 1
    C = np.reshape(np.linalg.solve(M, v), (nch, 2))

    # Compute new water levels and velocities
    for i, channel in enumerate(channel_list):
        d = dict()
        d['zeta0'] = {}
        d['u0'] = {}
        d['w0'] = {}
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['xx'] = {}
        d['__derivative']['z'] = {}
        d['__derivative']['zz'] = {}
        d['__derivative']['zzx'] = {}
        d['__derivative']['x']['zeta0'] = {}
        d['__derivative']['xx']['zeta0'] = {}
        d['__derivative']['x']['u0'] = {}
        d['__derivative']['z']['u0'] = {}
        d['__derivative']['zz']['u0'] = {}
        d['__derivative']['zzx']['u0'] = {}
        d['__derivative']['z']['w0'] = {}

        dc = channel.getOutput()
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        zeta00 = dc.v('zeta0', 'tide', range(jmax+1), [0], range(fmax+1))
        zeta01 = dc.v('zeta0_reverse', range(jmax+1), [0], range(fmax+1))
        zeta00x = dc.d('zeta0', 'tide', range(jmax+1), [0], range(fmax+1), dim='x')
        zeta01x = dc.d('zeta0_reverse', range(jmax+1), [0], range(fmax+1), dim='x')
        zeta00xx = dc.d('zeta0', 'tide', range(jmax+1), [0], range(fmax+1), dim='xx')
        zeta01xx = dc.d('zeta0_reverse', range(jmax+1), [0], range(fmax+1), dim='xx')
        d['zeta0']['tide']   = zeta00 * C[i,0] + zeta01 * C[i,1]
        d['__derivative']['x']['zeta0']['tide'] = zeta00x * C[i,0] + zeta01x * C[i,1]
        d['__derivative']['xx']['zeta0']['tide'] = zeta00xx * C[i,0] + zeta01xx * C[i,1]

        u00 = dc.v('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
        u01 = dc.v('u0_reverse', range(jmax+1), range(kmax+1), range(fmax+1))
        u00x = dc.d('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1), dim='x')
        u01x = dc.d('u0_reverse', range(jmax+1), range(kmax+1), range(fmax+1), dim='x')
        u00z = dc.d('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1), dim='z')
        u01z = dc.d('u0_reverse', range(jmax+1), range(kmax+1), range(fmax+1), dim='z')
        u00zz = dc.d('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1), dim='zz')
        u01zz = dc.d('u0_reverse', range(jmax+1), range(kmax+1), range(fmax+1), dim='zz')
        u00zzx = dc.d('u0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1), dim='zzx')
        u01zzx = dc.d('u0_reverse', range(jmax+1), range(kmax+1), range(fmax+1), dim='zzx')
        d['u0']['tide'] = u00 * C[i,0] + u01 * C[i,1]
        d['__derivative']['x']['u0']['tide'] = u00x * C[i,0] + u01x * C[i,1]
        d['__derivative']['z']['u0']['tide'] = u00z * C[i,0] + u01z * C[i,1]
        d['__derivative']['zz']['u0']['tide'] = u00zz * C[i,0] + u01zz * C[i,1]
        d['__derivative']['zzx']['u0']['tide'] = u00zzx * C[i,0] + u01zzx * C[i,1]

        w00 = dc.v('w0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
        w01 = dc.v('w0_reverse', range(jmax+1), range(kmax+1), range(fmax+1))
        w00z = dc.d('w0', 'tide', range(jmax+1), range(kmax+1), range(fmax+1), dim='z')
        w01z = dc.d('w0_reverse', range(jmax+1), range(kmax+1), range(fmax+1), dim='z')
        d['w0']['tide']   = w00 * C[i,0] + w01 * C[i,1]
        d['__derivative']['z']['w0']['tide']  = w00z * C[i,0] + w01z * C[i,1]

        # Add data to the channel
        channel.addInputData(d)
        channel._output.merge(d)

    return