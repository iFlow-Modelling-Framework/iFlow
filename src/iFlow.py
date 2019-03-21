"""
iFlow main file
Starts user interface and contains version information
Also catches KnownError and displays exceptions, which are iFlow specific error messages.

Date: 02-11-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from iFlowTUI import iFlowTUI
from src.util.diagnostics import ExceptionHandler
from src.util.diagnostics.KnownError import KnownError


class iFlow: 
    # Variables      
    __version = '2.6'

    # Methods
    def __init__(self):
        """Start iFlow user interface and add src to python path"""
        # start user interface
        userInterface = iFlowTUI()
        try:
            userInterface.start(self.__version)
        except KnownError as e:  # TODO extend to all exceptions
            exceptionHandler = ExceptionHandler()
            exceptionHandler.handle(e)
