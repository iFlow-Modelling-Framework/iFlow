"""
Timer class
provides methods tic, tac, toc for timing

Date: 09-07-15
Authors: Y.M. Dijkstra
"""
from time import time


class Timer:
    # Variables

    # Methods
    def __init__(self):
        self.reset()

    def reset(self):
        self.timespent = 0.

    def tic(self):
        '''Start timer
        '''
        self.t0 = time()

    def toc(self):
        '''Add the elapsed time to the timer
        '''
        self.timespent += time() - self.t0

    def disp(self, message):
        '''Print elapsed time
        '''
        if self.timespent < 1:
            print(message+": %.1f msec" % (self.timespent*1000))
        else:
            print(message+": %.1f sec" % (self.timespent))

    def string(self, message):
        '''Print elapsed time
        '''
        if self.timespent < 1:
            return message+": %.1f msec" % (self.timespent*1000)
        else:
            return message+": %.1f sec" % (self.timespent)

