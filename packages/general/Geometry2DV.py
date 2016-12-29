"""
Geometry for Flow 2DV
Load width, depth and length data in a dictionary

Date: 23-07-15
Authors: Y.M. Dijkstra
"""
from nifty import dynamicImport
from nifty import splitModuleName


class Geometry2DV:
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

        widthData = self.input.slice('B0', excludeKey=True)
        widthData.addData('L', self.input.v('L'))

        # instantiate width/depth objects
        depthMain_ = dynamicImport(depthPackage, depthName)
        widthMain_ = dynamicImport(widthPackage, widthName)
        depth = depthMain_('x', depthData)
        width = widthMain_('x', widthData)

        # save data
        d = {}
        d['B'] = width.function
        d['H'] = depth.function
        d['-H'] = depth.negfunction
        d['L'] = self.input.v('L')

        return d



