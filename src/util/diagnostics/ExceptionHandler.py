"""
Exception Handler
writes error messages and further information to the diagnostics channels

Date: 28-04-15
Authors: Y.M. Dijkstra
"""
import traceback
import logging
from src.util.diagnostics.KnownError import KnownError


class ExceptionHandler:
    # Variables      
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self):
        return

    def handle(self, e):
        """Write a neat error message based on the exception 'e'.
         Makes a distinction between KnownError instances, for which a minimal message is shown, and other exceptions,
         which are fully displayed except for their call list.
        """
        if isinstance(e, KnownError):
            # d = {'type': e.getOriginalErrorType(),
            #      'trace': traceback.format_exc(e)}
            self.logger.error('ERROR\n'+e.message+'\n\nIs this information not sufficient or inaccurate, see the debug file for traceback.')
        else:
            self.logger.exception('An unknown error occurred. Please see the traceback below.')
        return