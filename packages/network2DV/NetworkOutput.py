"""
Class NetworkOutput

Original date: 10-07-25
Original authors: Y.M. Dijkstra
"""
import logging
from packages.general.Output import Output


class NetworkOutput(Output):
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        super().__init__(input)
        return

    def run(self):
        """invoke the saveData() method to save by using Pickle.

        Returns:
            Empty dictionary.
        """
        self.logger.info('Saving output')

        network_output = {}
        for key in self.input.getKeysOf('network_output'):
            data = self.input.v('network_output', key)
            network_output[key] = self.prepData(data)
        self.input._data.pop('network_output')
        saveData = self.prepData(self.input)
        saveData.addData('network_output',network_output)
        d = self.saveData(saveData)
        return d



