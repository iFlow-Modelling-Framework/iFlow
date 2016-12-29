"""
Output module for transfering data to external applications
Data conversion of standard output module is  overridden. This module saves everything to numerical data

Date: 18-06-15
Authors: Y.M. Dijkstra
"""
from src.util.grid import callDataOnGrid
from Output import Output


class OutputExternal(Output):
    # Variables

    # Methods
    def __init__(self, input):
        Output.__init__(self, input)
        return

    def _convertData(self, saveData, grid, outputgrid, _, dontConvert, convertGrid = False):
        """Override convertData. Only allow for conversion to purely numerical data
        """
        # merge grids into one DC
        grid.merge(outputgrid)

        # take all subkeys
        subkeys = saveData.getAllKeys()

        for keys in subkeys:
            # Convert data to grid if key is in 'dontConvert', else convert to outputgrid
            if keys[0] in dontConvert or keys[0] in ['grid']:
                gridname = 'grid'
            else:
                gridname = self.outputgridName

            # call on grid 'gridname'
            data, _ = callDataOnGrid(saveData, keys, grid, gridname, False)

            # merge into saveData
            saveData.merge(self._buildDicts(keys, data))

        return