"""
Class DFunction

provides a callable function that references to taking -1*  a function. It stores the underlying function.

Date: 03-02-22
Authors: Y.M. Dijkstra
"""


class NFunction:
    def __init__(self, function):
        self.dimNames = function.__self__.dimNames
        self.function = function
        return

    def nfunction(self, **kwargs):
        return - self.function(**kwargs)


