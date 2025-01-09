"""
Combination of a hyperbolic tangent funtion and linear function
u(x) = 0.5*alpha*(1+tanh((x-xc)/xl) + 0.5*beta*x*(1+tanh((x-xc)/xl) + gamma

Requires parameters: alpha, beta, gamma, xc and xl

Date: September 2018
Authors: Y.M. Dijkstra
"""
import numpy as np
from packages.functions.checkVariables import checkVariables


class TangentLinear():
    #Variables
        
    #Methods
    def __init__(self, dimNames, data):
        self.L = float(data.v('L'))
        self.alpha = float(data.v('alpha'))
        self.beta = float(data.v('beta'))
        self.gamma = float(data.v('gamma'))
        self.xc = float(data.v('xc'))
        self.xl = float(data.v('xl'))
        self.dimNames = dimNames

        checkVariables(self.__class__.__name__, ('alpha', self.alpha), ('beta', self.beta), ('xc', self.xc), ('xl', self.xl), ('L', self.L))
        return

    def value(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        L = self.L
        x = x*L
        return 0.5*self.alpha*(1+np.tanh((x-self.xc)/self.xl))+0.5*self.beta*x*(1+np.tanh((x-self.xc)/self.xl))+self.gamma

    def derivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        L = self.L
        x = x * L
        return (0.5*self.beta*x*self.sech((x-self.xc)/self.xl)**2)/self.xl+(0.5*self.alpha*self.sech((x-self.xc)/self.xl)**2)/self.xl+0.5*self.beta*(np.tanh((x-self.xc)/self.xl)+1)

    def secondDerivative(self, x, **kwargs):
        """
        Parameters:
            x - value between 0 and 1
        """
        L = self.L
        x = x * L
        return (1.0*self.beta*self.sech((x-self.xc)/self.xl)**2)/self.xl-(1.0*self.beta*x*self.sech((x-self.xc)/self.xl)**2*np.tanh((x-self.xc)/self.xl))/self.xl**2-(1.0*self.alpha*self.sech((x-self.xc)/self.xl)**2*np.tanh((x-self.xc)/self.xl))/self.xl**2

    def sech(self, x):
        return 1./np.cosh(x)

# x = np.linspace(0, 64000, 100)
# from src.DataContainer import DataContainer
# import matplotlib.pyplot as plt
# import step as st
# st.configure()
# d = {}
# d['alpha'] = -2.78
# d['beta'] = -7.13e-5
# d['gamma'] = 10.
# d['xl'] = 5000.
# d['xc'] = 13000.
# d['L'] = 64000.
#
# dc = DataContainer(d)
# c = TangentLinear(['x'], dc)
# v = c.value(x/d['L'])
# vy = np.gradient(v, x[1]-x[0])
# vyy = np.gradient(vy, x[1]-x[0])
#
# vx= c.derivative(x/d['L'], dim='x')
# vxx= c.derivative(x/d['L'], dim='xx')
#
# plt.figure(1, figsize=(1,2))
# plt.plot(x, vy, label= 'manual')
# plt.plot(x, vx, label='code')
# plt.legend()
# st.show()