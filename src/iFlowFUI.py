"""
iFlow Functional User Interface
Handles iFlow called as a function

Date: 22-04-20
Authors: Y.M. Dijkstra
"""
import logging
from .Program import Program
from src.util.diagnostics.NoInputFileException import NoInputFileException
import time
from nifty.Timer import Timer
import os
import importlib
from src.util.grid.convertData import convertData


class iFlowFUI:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self):
        return

    def start(self, version, d, filePath=None, cwdpath=''):
        """Display the menu, wait for user input and then run the program

        Parameters:
            version - version number
        """
        # merge path
        totalPath = os.path.join(cwdpath, filePath)

        # call program selector
        program = Program(cwdpath, totalPath, d)
        timer = Timer()
        timer.tic()
        try:
            program.run()
        except NoInputFileException as e:
            print(e.message)
        timer.toc()
        self.logger.info(timer.string('\n'
                                      'iFlow run time'))
        dc = program.getResults()
        convertData(dc, dc.slice('grid'), dc.slice('outputgrid'), {}, {}, convertGrid = False, keepNF=False, keepSubkeys=False, outputgridName='outputgrid')

        return dc.data






