"""
iFlow main file
Starts user interface and programme. Also contains version information.
Catches KnownError and displays exceptions, which are iFlow specific error messages.

Original date: 02-11-15
Updated: 04-01-22
Original authors: Y.M. Dijkstra, R.L. Brouwer
Update authors: Y.M. Dijkstra
"""
import os
import logging
import time

from .iFlowTUI import iFlowTUI
from .iFlowBuilder import iFlowBuilder
from src.util.diagnostics import LogConfigurator
from nifty.Timer import Timer
from src.config import Profiler

from src.util.diagnostics.NoInputFileException import NoInputFileException
from src.util.diagnostics import MemoryProfiler


class iFlow:
    # Variables      
    __version = '3.0'
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, *args):
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def version(self):
        """ Print current iFlow version
        """
        print('iFlow version '+str(self.__version))

    def startTUI(self):
        """
        Start a Textual User Interface (TUI) on the console to enter working directory and input file.
        Directly continues to start the programme.
        """

        # start user interface
        userInterface = iFlowTUI()

        cwdpath, inputfile = userInterface.start(self.__version)
        iFlowBlock = self.initialise(inputfile, cwdpath)

        iFlowBlock.instantiateModule()
        iFlowBlock.run()

        time.sleep(0.05)    # short pause to make sure that log is indeed flushed before final statement
        input("Done. Press enter to close all windows and end the program.")
        logging.shutdown()
        return

    def initialise(self, inputfile=None, cwdpath='', d=dict()):
        """ """
        # merge path
        timer_prep = Timer()
        timer_prep.tic()
        totalPath = os.path.join(cwdpath, inputfile)

        # prepare logging system
        self.makeLogger(totalPath)       # Make logger

        # prepare memmory profiler if switched on in config file
        if Profiler:
            memProfiler = MemoryProfiler.MemoryProfiler()
            memProfiler.startProfiling()
        else:
            memProfiler = None

        # call program selector
        builder = iFlowBuilder(cwdpath, totalPath, d, memProfiler)


        # Read input and make call stack
        iFlowRunBlock = builder.makeCallStack()
        del builder

        timer_prep.toc()
        self.logger.debug(timer_prep.string('Time preparing call stack'))

        return iFlowRunBlock

    ####################################################################################################################
    ## other methods
    ####################################################################################################################
    def makeLogger(self, inputpath):
        # make console logger
        logConf = LogConfigurator(__package__)
        logConf.makeConsoleLog()

        # Make the path to the diag file
        path = inputpath.split('\\')
        path[-1] = '/'.join(path[-1].split('/')[:-1])
        path = '\\'.join(path)

        # check if the path exists
        if not os.path.exists(path):
            raise NoInputFileException('ERROR: Could not find path to input file %s.\nPlease check if you have selected the correct working directory and path to the input file.' % (path))

        # make file logger using the input filepath
        logConf.makeDiagFile(inputpath)










