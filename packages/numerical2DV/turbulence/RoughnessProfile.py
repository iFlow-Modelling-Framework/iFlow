"""
Set a profile in x-direction for the roughness parameter.
- Name of the roughness parameter should be provided on input
- Profile should be described by a function

Date: 26-07-16
Authors: Y.M. Dijkstra
"""
from nifty import dynamicImport
from nifty import splitModuleName


class RoughnessProfile:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        roughnesspar = self.input.v('roughnessParameter')

        # load data from the DataContainer with input
        #    name and package of functions for roughness
        roughPackage, roughName = splitModuleName(self.input.v(roughnesspar, 'type'))

        #    input variables
        roughData = self.input.slice(roughnesspar, excludeKey=True)
        roughData.addData('L', self.input.v('L'))

        # instantiate objects
        roughMain_ = dynamicImport(roughPackage, roughName)
        rough = roughMain_('x', roughData)

        # save data
        d = {}
        d[roughnesspar] = rough.function

        return d



