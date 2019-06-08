"""
SalinityCoefficients
xc and xl for the Ems according to Talke 2009 (oxygen paper)

Date: 06-04-2018
Authors: Y.M. Dijkstra
"""
import numpy as np
import scipy.interpolate


class SalinityCoefficients:
    # Variables

    # Methods
    def __init__(self, input):
        """
        """
        self.input = input
        return

    def run(self):
        Q = self.input.v('Q1')
        Qtalke = np.array([ 10,   20,   40,   60,   80,   100, 160,  240, 320,  600])
        xltalke = np.array([15.7, 14.2, 12.7, 12,   11.5, 11,  10.3, 9.7, 9.3,  8.4])*1000
        xctalke = np.array([47.9, 43.1, 38.7, 36.4, 34.9, 33.7,31.4, 29.5,28.2, 25.6])*1000-36000.
        xl = scipy.interpolate.interp1d(Qtalke, xltalke)
        xc = scipy.interpolate.interp1d(Qtalke, xctalke)

        Q = np.maximum(np.minimum(Q, np.max(Qtalke)), np.min(Qtalke))
        xlval = xl(abs(Q))
        xcval = xc(abs(Q))

        # save data
        d = {}
        d['xl'] = xlval
        d['xc'] = xcval

        return d

# def fun(Q, *args):
#     return args[0]*Q**args[1]
#
# def tanhyp(x, slim, ssea, xc, xl):
#     return ssea/2.*(1-np.tanh((x-xc)/xl)) - slim
#
#
# import scipy.optimize
# import step as st
# import matplotlib.pyplot as plt
# Qtalke = np.array([ 10,   20,   40,   60,   80, 100, 160, 240, 320, 600])
# xltalke = np.array([15.7, 14.2, 12.7, 12,   11.5, 11, 10.3, 9.7, 9.3, 8.4])*1000
# xctalke = np.array([47.9, 43.1, 38.7, 36.4, 34.9, 33.7, 31.4, 29.5, 28.2, 25.6])*1000-36000.
# slim = np.linspace(0, 15, 50)
#
# xintru = np.zeros(len(Qtalke))
# coef = np.zeros(len(slim))
# x = np.linspace(0, 640000, 200)
# for j, s in enumerate(slim):
#     for i, Q in enumerate(Qtalke):
#         xintru[i] = scipy.optimize.fsolve(tanhyp, 40000., (s, 30., xctalke[i], xltalke[i]))
#
#     cof,_ =  scipy.optimize.curve_fit(fun, Qtalke, xintru, [0.001, -.3])
#     coef[j] = cof[1]
#     print s
#     print cof
#
# st.configure()
# plt.figure(1, figsize=(1,2))
# plt.plot(slim, coef, 'o')
# st.show()
    # plt.figure(1, figsize=(1,2))
    # plt.plot(Ql, ss)
    # xl = scipy.interpolate.interp1d(Qtalke, xltalke)
    # xc = scipy.interpolate.interp1d(Qtalke, xctalke)
    # ssea = 30
    # x = np.linspace(0, 64000., 1000)
    # Ql = Qtalke#np.linspace(30, 150, 25)
    # ss = np.zeros(len(Ql))
    # import step as st
    # import matplotlib.pyplot as plt
    # for i,Q in enumerate(Ql):
    #     s = ssea/2.*(1-np.tanh((x-xc(Q))/xl(Q)))
    #     ss[i] = s[0]
    #
    # st.configure()
    # plt.figure(1, figsize=(1,2))
    # plt.plot(Ql, ss)
    # print Ql
    # print ss
    #
    # st.show()




