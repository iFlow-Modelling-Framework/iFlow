"""
Geometry for Flow 2DV including a flat fluvial region for a distance long enough to dissipate tides completely
Load width, depth and length data in a dictionary

Date: 23-07-15
Update: 04-02-22
Authors: I. Jalon-Rojas (modified from Geometry2DV, Y.M. Dijkstra)
"""
from nifty import dynamicImport
from nifty import splitModuleName


class Geometry2DV_Flat:
    # Variables

    # Methods
    def __init__(self, input):
        """
        """
        self.input = input
        return

    def run(self):
        """
        """
        # load data from the DataContainer with input
        #    name and package of functions for depth and width
        depthPackage, depthName = splitModuleName(self.input.v('H0', 'type'))
        widthPackage, widthName = splitModuleName(self.input.v('B0', 'type'))

        #    input variables
        depthData = self.input.slice('H0', excludeKey=True)
        depthData.addData('L', self.input.v('L'))
        depthData.addData('Le', self.input.v('Le'))

        widthData = self.input.slice('B0', excludeKey=True)
        widthData.addData('L', self.input.v('L'))
        widthData.addData('Le', self.input.v('Le'))

        # instantiate width/depth objects
        depthMain_ = dynamicImport(depthPackage, depthName)
        widthMain_ = dynamicImport(widthPackage, widthName)
        depth = depthMain_('x', depthData)
        width = widthMain_('x', widthData)

        # save data
        d = {}
        d['B'] = width.value
        d['H'] = depth.value
        d['__derivative'] = {}
        d['__derivative']['x'] = {}
        d['__derivative']['x']['B'] = width.derivative
        d['__derivative']['x']['H'] = depth.derivative
        d['__derivative']['xx'] = {}
        d['__derivative']['xx']['B'] = width.secondDerivative
        d['__derivative']['xx']['H'] = depth.secondDerivative

        d['L'] = self.input.v('L')
        d['Le'] = self.input.v('Le')
        d['Lf'] = self.input.v('L')-self.input.v('Le')

        return d



