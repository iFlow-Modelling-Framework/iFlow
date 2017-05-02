"""
Geometry for Flow 2DV
Load crosssec, depth and length data in a dictionary

Date: 23-07-15
Authors: Y.M. Dijkstra
"""
from nifty import dynamicImport
from nifty import splitModuleName
from nifty.functionTemplates import FunctionBase


class Geometry2DVCS:
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
        #    name and package of functions for depth and crosssec
        depthPackage, depthName = splitModuleName(self.input.v('H0', 'type'))
        crosssecPackage, crosssecName = splitModuleName(self.input.v('A0', 'type'))

        #    input variables
        depthData = self.input.slice('H0', excludeKey=True)
        depthData.addData('L', self.input.v('L'))

        crosssecData = self.input.slice('A0', excludeKey=True)
        crosssecData.addData('L', self.input.v('L'))

        # instantiate crosssec/depth objects
        depthMain_ = dynamicImport(depthPackage, depthName)
        crosssecMain_ = dynamicImport(crosssecPackage, crosssecName)
        depth = depthMain_('x', depthData)
        crosssec = crosssecMain_('x', crosssecData)
        width = WidthFunction('x', depth.function, crosssec.function)

        # save data
        d = {}
        d['B'] = width.function
        d['H'] = depth.function
        d['L'] = self.input.v('L')

        return d

class WidthFunction(FunctionBase):

    def __init__(self, dimNames, depth, crosssec):
        FunctionBase.__init__(self, dimNames)
        from src.DataContainer import DataContainer
        grid ={}
        grid['dimensions'] = ['x']

        self.dc = DataContainer({'depth':depth, 'crosssec':crosssec, 'grid':grid})
        return

    def value(self, x, **kwargs):
        return self.dc.v('crosssec', x=x)/self.dc.v('depth', x=x)

    def derivative(self, x, **kwargs):
        if kwargs['dim'] == 'x':
            ax = self.dc.d('crosssec', x=x, dim='x')
            a =  self.dc.v('crosssec', x=x)
            hx = self.dc.d('depth', x=x, dim='x')
            h = self.dc.v('depth', x=x)
            return (ax*h-hx*a)/h**2.
        if kwargs['dim'] == 'xx':
            axx = self.dc.d('crosssec', x=x, dim='xx')
            ax = self.dc.d('crosssec', x=x, dim='x')
            a =  self.dc.v('crosssec', x=x)
            hxx = self.dc.d('depth', x=x, dim='xx')
            hx = self.dc.d('depth', x=x, dim='x')
            h = self.dc.v('depth', x=x)
            return axx/h-2*ax*hx/h**2+2*a*hx**2/h**3-a*hxx/h**2
        else:
            FunctionBase.derivative(self)
        return






