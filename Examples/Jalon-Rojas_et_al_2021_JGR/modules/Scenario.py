"""
Scenario
Load year, river flow and tidal range in a dictionary

Date: 23-05-16
Authors: I.Jalon-Rojas
"""


class Scenario:
    # Variables

    # Methods
    def __init__(self, input):
        """
        """
        self.input = input
        return

    def run(self):
        d = {}

        d['year'] = self.input.v('year')
        d['TRm'] = self.input.v('TRm')
        d['Q'] = self.input.v('Q')

        return d



