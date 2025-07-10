"""
Network module for matching the first order water motion.

Date: 13-Feb-2020
Authors: J. Wang
"""
import copy
import logging
import numpy as np
from src.DataContainer import DataContainer
from src.util.diagnostics import KnownError
from nifty import toList
import numbers
import os
from itertools import product
from copy import deepcopy
import matplotlib.pyplot as plt
import nifty as ny
import scipy.linalg


def matchFirstOrderHydro(input, channel_list):
    """
    Return the channel matching coefficients.
    For M0 components, the final solution:
        zeta = zeta1 + C[0] * zeta_river + C[1]
        u = u1 + * C[0] * u_river
    where
    zeta1, u1: the solution for single channel (is 0 for river and dynamic pressure)
    C[0]: subtidal transport
    C[1]: a constant s.t. water level is continuous

    If the component contributes to M4, then,
        zeta = C[2] * zeta1[:,:,2] + C[3] * zeta2 + zetas[:,:,2]
        u = C[2] * u1[:,:,2] + C[3] * u2 + us[:,:,2]
    where
    zetas, us: the solution for single channel (is 0 for river and dynamic pressure)
    zeta1: first order tide for single channel
    zeta2: as zeta1, but solved with reversed boundary conditions
    """

    # Load settings
    C = {}
    G = input.v('G')
    Vertex = input.v('network_settings', 'label', 'Vertex')
    Sea = input.v('network_settings', 'label', 'Sea')   # labels of sea channels
    River = input.v('network_settings', 'label', 'River')    # labels of river channels
    discharge = input.v('network_settings', 'hydrodynamics', 'Q1')
    nch = input.v('network_settings', 'numberofchannels')  # number of channels
    tide = input.v('network_settings', 'hydrodynamics', 'M4')

    # gather data from all channels
    coefQ_0 = []
    coefQ_L = []
    riverZeta = {}
    riverU = {}
    u0da_0 = []
    u0da_L = []
    coef0 = {}
    Q1_0 = {}
    zeta1_0 = {}
    coefL = {}
    Q1_L = {}
    zeta1_L = {}
    for i, channel in enumerate(channel_list):
        d = {}
        dc = channel.getOutput()
        jmax = dc.v('grid', 'maxIndex', 'x')
        kmax = dc.v('grid', 'maxIndex', 'z')
        fmax = dc.v('grid', 'maxIndex', 'f')

        riverZeta[str(i)] = deepcopy(dc.v('zeta1', 'river', range(jmax+1), [0], range(fmax+1)))
        riverU[str(i)] = deepcopy(dc.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1)))

        coefQ_0.append(dc.v('zeta1', 'river', 0, 0, 0))
        coefQ_L.append(dc.v('zeta1', 'river', jmax, 0, 0))

        u1 =   dc.v('u1', 'tide',   range(jmax+1), range(kmax+1), 2)
        u1_2 = dc.v('u1_reverse', range(jmax+1), range(kmax+1), 2)

        zeta1 =   dc.v('zeta1', 'tide',   range(jmax+1), 0, 2)
        zeta1_2 = dc.v('zeta1_reverse', range(jmax+1), 0, 2)

        # Leading order M2 cuurent will be only used by
        u0 = dc.v('u0', 'tide',   range(jmax+1), range(kmax+1), 1)
        u0_DA = ny.integrate(u0, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]
        uL_DA = ny.integrate(u0, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]
        u0da_0.append(u0_DA / dc.v('H', 0))
        u0da_L.append(uL_DA / dc.v('H', jmax))

        # store zeta1 at 0 and L without transport
        # loop over every contirbution, excluding dynamic pressure
        for mod in dc.getKeysOf('zeta1'):
            if mod not in coef0:
                coef0[mod] = []
                Q1_0[mod] = []
                zeta1_0[mod] = []
            # if mod not in coefL:
                coefL[mod] = []
                Q1_L[mod] = []
                zeta1_L[mod] = []

            if mod == 'tide':
                zeta1_0[mod].append([zeta1[0], zeta1_2[0]])
                zeta1_L[mod].append([zeta1[-1], zeta1_2[-1]])

                ### depth-integrated horizontal velocity of mode 2
                u0_DA = ny.integrate(u1, 'z', kmax, 0, dc, [0], range(kmax+1), [2])[0][0]
                uL_DA = 0

                ### depth-integrated horizontal velocity of mode 2
                uL_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [jmax], range(kmax+1), [2])[0][0]
                u0_2_DA = 0

                B_0 = dc.v('B', 0)
                B_L = dc.v('B',jmax)
                ### transport
                Q1_0[mod].append([u0_DA*B_0, u0_2_DA*B_0])
                Q1_L[mod].append([uL_DA*B_L, uL_2_DA*B_L])

            else: # adv, nostress, Stokes, baroc
                coef0[mod].append((dc.v('zeta1', mod, 0, 0, 0)))
                coefL[mod].append((dc.v('zeta1', mod, -1, 0, 0)))

                if mod in ['adv', 'stokes', 'nostress']: # add M4
                    u1_s =   dc.v('u1', mod,  range(jmax+1), range(kmax+1), 2)
                    zeta1_s = dc.v('zeta1', mod, range(jmax+1), 0, 2)

                    zeta1_0[mod].append([zeta1[0], zeta1_2[0], zeta1_s[0]])
                    zeta1_L[mod].append([zeta1[-1], zeta1_2[-1], zeta1_s[-1]])

                    ### depth-integrated horizontal velocity of mode
                    u0_DA = ny.integrate(u1, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]
                    uL_DA = ny.integrate(u1, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]

                    ### depth-integrated horizontal velocity of mode 2
                    uL_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]
                    u0_2_DA = ny.integrate(u1_2, 'z', kmax, 0, dc, [0], range(kmax+1), 2)[0][0]
                    u0sda = ny.integrate(u1_s, 'z', kmax, 0, dc, [0],    range(kmax+1), 2)[0][0]
                    uLsda = ny.integrate(u1_s, 'z', kmax, 0, dc, [jmax], range(kmax+1), 2)[0][0]
                    B_0 = dc.v('B', 0)
                    B_L = dc.v('B',jmax)

                    ### transport
                    Q1_0[mod].append([u0_DA*B_0, u0_2_DA*B_0, u0sda*B_0])
                    Q1_L[mod].append([uL_DA*B_L, uL_2_DA*B_L, uLsda*B_L])

            # Add a dummy subtidal signal for the dynamic pressure
            # d_dp = {}
            # d_dp['zeta1'] = {}
            # d_dp['u1'] = {}
            # d_dp['zeta1']['dp'] = deepcopy(dc.v('zeta1', 'river', range(jmax+1), [0], range(fmax+1)))
            # d_dp['u1']['dp'] = deepcopy(dc.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1)))
            # channel.addInputData(d_dp)
            # channel._output.merge(d_dp)

    # if the external M4 forcing is not given:
    if tide is None:
        tide = 0

    vertex_nosign = deepcopy(Vertex)
    sign = deepcopy(Vertex)
    for i in range(len(Vertex)):
        for x in range(len(Vertex[i])):
            vertex_nosign[i][x] = abs(Vertex[i][x]) - 1
            sign[i][x] = np.sign(Vertex[i][x])

    for mod in coef0.keys():
        if mod == 'river':
            Mr = np.zeros((2 * nch, 2 * nch))
            vr = np.zeros(2 * nch)
            vr[0:len(River)] = discharge    # prescribed discharge
            row = 0

            for i in River:
                Mr[row,[2*i]] = 1
                row += 1

            for i in Sea:
                Mr[row,[[2*i, 2*i+1]]] = [np.real(coefQ_0[i]), 1]
                row += 1

            # continuous SSH
            for j in range(len(Vertex)):
                loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                # loc[i]: index for channel
                for i in range(len(loc)-1):
                    if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                        Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(coefQ_L[loc[i]]), 1]
                    elif sign[j][i] == -1:
                        Mr[row, [2*loc[i], 2*loc[i]+1]] = [np.real(coefQ_0[loc[i]]), 1]
                    if sign[j][i-1] == 1:
                        Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(coefQ_L[loc[i-1]]), -1]
                    elif sign[j][i-1] == -1:
                        Mr[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(coefQ_0[loc[i-1]]), -1]
                    row += 1
            # mass cons.
            for j in range(len(Vertex)):
                for i in range(len(vertex_nosign[j])):
                    k = vertex_nosign[j][i]
                    if sign[j][i] == 1:
                        Mr[row, 2*k] = 1
                    elif sign[j][i] == -1:
                        Mr[row, 2*k] = -1
                row += 1
            C['river'] = np.reshape(np.linalg.solve(Mr, vr), (nch, 2))

        elif mod == 'tide':
            M = np.zeros((2*nch, 2*nch), dtype=complex)
            v = np.zeros(2*nch, dtype=complex)
            v[0:len(Sea)] = tide
            row = 0

            # zeta0_L[# of Channel][# of Mode]
            # prescribe tidal surface elevation at the sea
            for i in Sea:
                M[row, [2*i, 2*i+1]] = np.array([zeta1_0[mod][i][0], zeta1_0[mod][i][1]])
                row += 1

            # u=0 at tidal limit/weir <=> Q0 = 0
            for i in River:
                M[row, [2*i, 2*i+1]] = np.array([Q1_L[mod][i][0], Q1_L[mod][i][1]])
                row += 1

            # continuous SSH
            for j in range(len(Vertex)):
                loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                # loc[i]: index for channel
                for i in range(len(loc)-1):
                    if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                        M[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_L[mod][loc[i]][0], zeta1_L[mod][loc[i]][1]]
                    elif sign[j][i] == -1:
                        M[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_0[mod][loc[i]][0], zeta1_0[mod][loc[i]][1]]
                    if sign[j][i-1] == 1:
                        M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_L[mod][loc[i-1]][0], -zeta1_L[mod][loc[i-1]][1]]
                    elif sign[j][i-1] == -1:
                        M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_0[mod][loc[i-1]][0], -zeta1_0[mod][loc[i-1]][1]]
                    row += 1

            # mass conservation
            for j in range(len(Vertex)):
                for i in range(len(vertex_nosign[j])):
                    k = vertex_nosign[j][i] # channel index
                    if sign[j][i] == 1:
                        M[row, [2*k, 2*k+1]] = [Q1_L[mod][k][0], Q1_L[mod][k][1]]
                    elif sign[j][i] == -1:
                        M[row, [2*k, 2*k+1]] = [-Q1_0[mod][k][0], -Q1_0[mod][k][1]]
                row += 1
            C[mod] = np.reshape(np.linalg.solve(M, v), (nch, 2))

        elif mod != 'dp':
            M = np.zeros((2 * nch, 2 * nch), dtype=complex)
            v = np.zeros(2 * nch, dtype=complex)
            row = 0

            # transport = 0 in river channels
            for i in River:
                M[row,[2*i]] = 1
                row += 1
            # 0 SSH at sea
            for i in Sea:
                # M[row,[[2*i, 2*i+1]]] = [coefQ_L[i], 1]
                M[row,[[2*i, 2*i+1]]] = [coefQ_0[i], 1]
                # v[row] = -coefL[mod][row]
                v[row] = -coef0[mod][row]
                row += 1
            # continuous SSH
            for j in range(len(Vertex)):
                loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                # loc[i]: index for channel
                for i in range(len(loc)-1):
                    if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                        M[row, [2*loc[i], 2*loc[i]+1]] = [coefQ_L[loc[i]], 1]
                        v[row] -= coefL[mod][loc[i]]
                    elif sign[j][i] == -1:
                        M[row, [2*loc[i], 2*loc[i]+1]] = [coefQ_0[loc[i]], 1]
                        v[row] -= coef0[mod][loc[i]]
                    if sign[j][i-1] == 1:
                        M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-coefQ_L[loc[i-1]], -1]
                        v[row] += coefL[mod][loc[i-1]]
                    elif sign[j][i-1] == -1:
                        M[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-coefQ_0[loc[i-1]], -1]
                        v[row] += coef0[mod][loc[i-1]]
                    row += 1
            # mass cons.
            for j in range(len(Vertex)):
                for i in range(len(vertex_nosign[j])):
                    k = vertex_nosign[j][i]
                    if sign[j][i] == 1:
                        M[row, 2*k] = 1
                    elif sign[j][i] == -1:
                        M[row, 2*k] = -1
                row += 1

            C[mod] = np.reshape(np.linalg.solve(M, v), (nch, 2))

            if mod in ['adv', 'stokes', 'nostress']:
                M4 = np.zeros((2*nch, 2*nch), dtype=complex)
                v4 = np.zeros(2*nch, dtype=complex)
                row = 0

                # zeta0_L[mod][# of Channel][# of Mode]
                # prescribe tidal surface elevation at the sea
                for i in Sea:
                    M4[row, [2*i, 2*i+1]] = np.array([zeta1_0[mod][i][0], zeta1_0[mod][i][1]])
                    v4[row] -= zeta1_0[mod][i][2]
                    row += 1

                # u=0 at tidal limit/weir
                for i in River:
                    M4[row, [2*i, 2*i+1]] = np.array([Q1_L[mod][i][0], Q1_L[mod][i][1]])
                    v4[row] -= Q1_L[mod][i][2]
                    row += 1

                # continuous SSH
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            M4[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_L[mod][loc[i]][0], zeta1_L[mod][loc[i]][1]]
                            v4[row] -= zeta1_L[mod][loc[i]][2]
                        elif sign[j][i] == -1:
                            M4[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_0[mod][loc[i]][0], zeta1_0[mod][loc[i]][1]]
                            v4[row] -= zeta1_0[mod][loc[i]][2]
                        if sign[j][i-1] == 1:
                            M4[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_L[mod][loc[i-1]][0], -zeta1_L[mod][loc[i-1]][1]]
                            v4[row] -= -zeta1_L[mod][loc[i-1]][2]
                        elif sign[j][i-1] == -1:
                            M4[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_0[mod][loc[i-1]][0], -zeta1_0[mod][loc[i-1]][1]]
                            v4[row] -= -zeta1_0[mod][loc[i-1]][2]
                        row += 1

                # mass conservation
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i] # channel index
                        if sign[j][i] == 1:
                            M4[row, [2*k, 2*k+1]] = [Q1_L[mod][k][0], Q1_L[mod][k][1]]
                            v4[row] -= Q1_L[mod][k][2]
                        elif sign[j][i] == -1:
                            M4[row, [2*k, 2*k+1]] = [-Q1_0[mod][k][0], -Q1_0[mod][k][1]]
                            v4[row] -= -Q1_0[mod][k][2]
                    row += 1
                C[mod] = np.append(C[mod], np.reshape(np.linalg.solve(M4, v4), (nch, 2)), axis=1)

            # Solve for the contribution by cont. dp only if adv is solved
            if mod == 'adv':
                u0 = np.real(0.25 * np.abs(u0da_0)**2)
                uL = np.real(0.25 * np.abs(u0da_L)**2)
                u0M4 = 0.25 * np.complex64(u0da_0)**2
                uLM4 = 0.25 * np.complex64(u0da_L)**2

                # first match M0
                M0dp = np.zeros((2 * nch, 2 * nch))
                v0dp = np.zeros(2 * nch)
                row = 0
                for i in River:
                    M0dp[row,[2*i]] = 1
                    row += 1

                for i in Sea:
                    M0dp[row,[[2*i, 2*i+1]]] = [np.real(coefQ_0[i]), 1]
                    row += 1

                # continuous dynamic pressure
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            M0dp[row, [2*loc[i], 2*loc[i]+1]] = [np.real(coefQ_L[loc[i]]), 1]
                            v0dp[row] -= uL[loc[i]] / G
                        elif sign[j][i] == -1:
                            M0dp[row, [2*loc[i], 2*loc[i]+1]] = [np.real(coefQ_0[loc[i]]), 1]
                            v0dp[row] -= u0[loc[i]] / G
                        if sign[j][i-1] == 1:
                            M0dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(coefQ_L[loc[i-1]]), -1]
                            v0dp[row] += uL[loc[i-1]] / G
                        elif sign[j][i-1] == -1:
                            M0dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-np.real(coefQ_0[loc[i-1]]), -1]
                            v0dp[row] += u0[loc[i-1]] / G
                        row += 1
                # mass cons.
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i]
                        if sign[j][i] == 1:
                            M0dp[row, 2*k] = 1
                        elif sign[j][i] == -1:
                            M0dp[row, 2*k] = -1
                    row += 1
                C['dp'] = np.reshape(np.linalg.solve(M0dp, v0dp), (nch, 2))


                # match M4 by dynamic pressure
                M4dp = np.zeros((2*nch, 2*nch), dtype=complex)
                v4dp = np.zeros(2*nch, dtype=complex)
                row = 0

                # zeta0_L[# of Channel][# of Mode]
                # 0 elevation at the sea
                for i in Sea:
                    M4dp[row, [2*i, 2*i+1]] = np.array([zeta1_0['tide'][i][0], zeta1_0['tide'][i][1]])
                    v4dp[row] = 0
                    row += 1

                # u=0 at tidal limit/weir <=> Q0 = 0
                for i in River:
                    M4dp[row, [2*i, 2*i+1]] = np.array([Q1_L['tide'][i][0], Q1_L['tide'][i][1]])
                    row += 1

                # continuous dynamic pressure
                for j in range(len(Vertex)):
                    loc = vertex_nosign[j]  # indices of channels that are connected at branching point
                    # loc[i]: index for channel
                    for i in range(len(loc)-1):
                        if sign[j][i] == 1: # if sign = 1, use _0, if sign = -1, use _L
                            M4dp[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_L['tide'][loc[i]][0], zeta1_L['tide'][loc[i]][1]]
                            v4dp[row] -= uLM4[loc[i]] / G
                        elif sign[j][i] == -1:
                            M4dp[row, [2*loc[i], 2*loc[i]+1]] = [zeta1_0['tide'][loc[i]][0], zeta1_0['tide'][loc[i]][1]]
                            v4dp[row] -= u0M4[loc[i]] / G
                        if sign[j][i-1] == 1:
                            M4dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_L['tide'][loc[i-1]][0], -zeta1_L['tide'][loc[i-1]][1]]
                            v4dp[row] += uLM4[loc[i-1]] / G
                        elif sign[j][i-1] == -1:
                            M4dp[row, [2*loc[i-1], 2*loc[i-1]+1]] = [-zeta1_0['tide'][loc[i-1]][0], -zeta1_0['tide'][loc[i-1]][1]]
                            v4dp[row] += u0M4[loc[i-1]] / G
                        row += 1

                # mass conservation
                for j in range(len(Vertex)):
                    for i in range(len(vertex_nosign[j])):
                        k = vertex_nosign[j][i] # channel index
                        if sign[j][i] == 1:
                            M4dp[row, [2*k, 2*k+1]] = [Q1_L['tide'][k][0], Q1_L['tide'][k][1]]
                        elif sign[j][i] == -1:
                            M4dp[row, [2*k, 2*k+1]] = [-Q1_0['tide'][k][0], -Q1_0['tide'][k][1]]
                    row += 1
                CM4dp = np.reshape(np.linalg.solve(M4dp, v4dp), (nch, 2))
                C['dp'] = np.append(C['dp'], CM4dp , axis=1)

    # Compute final u1 and zeta1
    for i, channel in enumerate(channel_list):
        d = dict()
        d['zeta1'] = {}
        d['u1'] = {}

        for mod in C.keys():
            data = scaleChannels(mod, i, channel, C[mod][i], riverZeta, riverU)

            d['zeta1'][mod]   = data['zeta1']
            d['u1'][mod]   = data['u1']

        # Add data to the channel
        channel.addInputData(d)
        channel._output.merge(d)
    return

def scaleChannels(mod, i, channel, C, riverZeta, riverU):
    # self.logger.info('Scaling ' + mod + ' channel ' + str(i+1))
    dc = channel.getOutput()
    jmax = dc.v('grid', 'maxIndex', 'x')
    kmax = dc.v('grid', 'maxIndex', 'z')
    fmax = dc.v('grid', 'maxIndex', 'f')

    d={}
    dictU = {}

    if mod == 'river':
        # uriver = dc.v('u1', 'river', range(jmax+1), range(kmax+1), range(fmax+1))
        temp = np.zeros((1,1,fmax+1))
        temp[0,0,0] = C[1]
        dictU = {
            'zeta1': {mod: deepcopy(riverZeta[str(i)] * C[0] + temp)},
            'u1':    {mod: deepcopy(riverU[str(i)] * C[0] )},
            'Q': {mod: np.real(C[0])}
        }
        d['network'] = {}
        d['network'][str(i)] = DataContainer()
        d['network'][str(i)].merge(dictU)

    elif mod == 'tide':
        zeta1 = dc.v('zeta1', 'tide', range(jmax+1), [0], range(fmax+1))
        zeta2 = dc.v('zeta1_reverse', range(jmax+1), [0], range(fmax+1))
        u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
        u2 = dc.v('u1_reverse', range(jmax+1), range(kmax+1), range(fmax+1))

        dictU = {
            'zeta1': {
                mod: deepcopy(C[0]*zeta1 + C[1]*zeta2)
            },
            'u1': {
                mod: deepcopy(C[0]*u1 + C[1]*u2)
            }
        }
        d['network'] = {}
        d['network'][str(i)] = DataContainer()
        d['network'][str(i)].merge(dictU)

    else: # adv, stokes, nostress, baroc
        if mod in ['adv', 'stokes', 'nostress']:
            zeta1 = dc.v('zeta1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
            zeta2 = dc.v('zeta1_reverse', range(jmax+1), range(kmax+1), 2)
            zetas = dc.v('zeta1', mod, range(jmax+1), [0], range(fmax+1))

            u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
            u2 = dc.v('u1_reverse', range(jmax+1), range(kmax+1), 2)
            us = dc.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1))

            # M0 component
            zeta1[:,:,0] = (zetas[:,:,0]  + riverZeta[str(i)][:,:,0] * C[0] + C[1])
            u1[:,:,0] = us[:,:,0]  + riverU[str(i)][:,:,0] * C[0]

            # M4 component
            zeta1[:,:,2] = C[2] * zeta1[:,:,2] + C[3] * zeta2 + zetas[:,:,2]
            u1[:,:,2]    = C[2] * u1[:,:,2]    + C[3] * u2    + us[:,:,2]

            dictU = {
                'zeta1': {mod: deepcopy(zeta1)},
                'u1': {mod: deepcopy(u1)},
                'Q': {mod: np.real(C[0])}
            }

        elif mod == 'dp':
            zeta1 = dc.v('zeta1', 'tide', range(jmax+1), [0], range(fmax+1))
            zeta2 = dc.v('zeta1_reverse', range(jmax+1), [0], 2)

            u1 = dc.v('u1', 'tide', range(jmax+1), range(kmax+1), range(fmax+1))
            u2 = dc.v('u1_reverse', range(jmax+1), range(kmax+1), 2)

            zeta1[:,:,0] = riverZeta[str(i)][:,:,0] * C[0] + C[1]
            zeta1[:,:,2] = C[2] * zeta1[:,:,2] + C[3] * zeta2
            u1[:,:,0] = riverU[str(i)][:,:,0] * C[0]
            u1[:,:,2] = C[2] * u1[:,:,2] + C[3] * u2

            dictU = {
                'zeta1': {mod: deepcopy(zeta1)},
                'u1': {mod: deepcopy(u1)},
                'Q': {mod: np.real(C[0])}
            }

        elif mod == 'baroc': # baroc
            zeta1 = dc.v('zeta1', mod, range(jmax+1), range(kmax+1), range(fmax+1))
            u1 = dc.v('u1', mod, range(jmax+1), range(kmax+1), range(fmax+1))
            temp = np.zeros((1,1,fmax+1), dtype=complex)
            temp[0,0,0] = C[1]
            dictU = {
                'zeta1': {mod: deepcopy(zeta1  + riverZeta[str(i)] * C[0] + temp)},
                'u1':    {mod: deepcopy(u1 + riverU[str(i)] * C[0])},
                'Q': {mod: np.real(C[0])}
            }

    return dictU