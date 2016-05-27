"""

Date: 25-05-16
Authors: R.L. Brouwer, Y.M. Dijkstra
"""


class BypassDatacontainer:
    # Variables

    # Methods
    def __init__(self, input, name):
        self.input = input
        self.name = name

    def callDataContainer(self, **kwargs):
        if kwargs.get('operation'):
            if kwargs.get('operation') == 'd':
                v = self.input.d(self.name, **kwargs)
            elif kwargs.get('operation') == 'n':
                v = self.input.n(self.name, **kwargs)
        else:
            v = self.input.v(self.name, **kwargs)
        return v
