"""
CriticalErosion

Date: 07-03-2017
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
import nifty as ny
import step as st
import matplotlib.pyplot as plt


class CombinedDepth:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):

        ################################################################################################################
        ## Width
        ################################################################################################################
        # load data from the DataContainer with input
        #    name and package of functions for depth and width
        widthPackage, widthName = ny.splitModuleName(self.input.v('B0', 'type'))

        widthData = self.input.slice('B0', excludeKey=True)
        widthData.addData('L', self.input.v('L'))

        widthMain_ = ny.dynamicImport(widthPackage, widthName)
        width = widthMain_('x', widthData)

        ################################################################################################################
        ## Depth
        ################################################################################################################
        ## Make a grid for the depth
        dimensions = ['x']
        enclosures = [(0, self.input.v('L'))]
        contraction = [[]]  # Enclosures of each dimension depend on these parameters
        copy = [1]  # Copy lower-dimensional arrays over these dimensions. 1: yes, copy. 0: no, only keep in the zero-index

        # grid
        axisTypes = ['equidistant']
        axisSize = [self.input.v('resolution')]
        axisOther = [None]

        grid = ny.makeRegularGrid(dimensions, axisTypes, axisSize, axisOther, enclosures, contraction, copy)

        ## Depth
        alpha = self.input.v('alpha')

        # determine the number of profiles to combine
        profile_no = 1
        readProfiles = True
        while readProfiles:
            if self.input.v('H'+str(profile_no)) is None:
                readProfiles = False
            else:
                profile_no+=1

        # generate each profile as object
        depthObject = []
        for i in range(1, profile_no):
            depthData = self.input.slice('H'+str(i), excludeKey=True)
            depthData.addData('L', self.input.v('L'))

            depthPackage, depthName = ny.splitModuleName(self.input.v('H'+str(i), 'type'))
            depthMain_ = ny.dynamicImport(depthPackage, depthName)
            depthObject.append(depthMain_('x', depthData))

        # load actual depths
        x = grid['axis']['x']
        H = []
        Hx = []
        Hxx = []
        for i in range(0, len(depthObject)):
            H.append(depthObject[i].value(x))
            Hx.append(depthObject[i].derivative(x, dim='x'))
            Hxx.append(depthObject[i].derivative(x, dim='xx'))

        # load alpha limit per profile
        alpha_lim = []
        for i in range(1, profile_no):
            alpha_lim.append(float(self.input.v('alpha'+str(i))))

        # determine range that alpha is in
        if alpha>max(alpha_lim) or alpha<=min(alpha_lim):        # if alpha is outside the interpolation range, take the highest and lowest alpha to extrapolate
            indlow = alpha_lim.index(min(alpha_lim))
            indhigh = alpha_lim.index(max(alpha_lim))
        else:
            for i in range(0, len(alpha_lim)):
                if alpha<=alpha_lim[i+1] and alpha>alpha_lim[i]:
                    indlow = i
                    indhigh = i+1
                    break
        alpha_rel = (float(alpha) - alpha_lim[indlow]) / (alpha_lim[indhigh] - alpha_lim[indlow])

        Ha = (1-alpha_rel)*H[indlow] + alpha_rel*H[indhigh]
        Hax = (1 - alpha_rel) * Hx[indlow] + alpha_rel * Hx[indhigh]
        Haxx = (1 - alpha_rel) * Hxx[indlow] + alpha_rel * Hxx[indhigh]

        # print alpha_rel, alpha_lim[indlow], alpha_lim[indhigh], Ha[-1]

        nf = ny.functionTemplates.NumericalFunctionWrapper(Ha, grid)
        nf.addDerivative(Hax, 'x')
        nf.addDerivative(Haxx, 'xx')

        ################################################################################################################
        ## Dictionary
        ################################################################################################################
        d = {}
        d['H'] = nf.function
        d['B'] = width.function
        d['L'] = self.input.v('L')

        # st.configure()
        # plt.figure(1, figsize=(1,2))
        # plt.plot(x, -Ha)

        # st.show()

        return d