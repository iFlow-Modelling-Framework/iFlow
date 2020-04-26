"""
iFlow main file
Starts user interface and contains version information
Also catches KnownError and displays exceptions, which are iFlow specific error messages.

Date: 02-11-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from .iFlowTUI import iFlowTUI
from .iFlowFUI import iFlowFUI
from src.util.diagnostics import ExceptionHandler
from src.util.diagnostics.KnownError import KnownError


class iFlow: 
    # Variables      
    __version = '3.0'

    # Methods
    def __init__(self, *args):
        return

    def StartTUI(self):
        # start user interface
        userInterface = iFlowTUI()
        # userInterface.start(self.__version)
        try:
            userInterface.start(self.__version)
        except KnownError as e:  # TODO extend to all exceptions
            # print(str(e))
            exceptionHandler = ExceptionHandler()
            exceptionHandler.handle(e)
        return

    def StartFUI(self, inputfile=None, cwdpath='', d=dict()):
        userInterface = iFlowFUI()
        d = userInterface.start(self.__version, d, inputfile, cwdpath)
        return d






