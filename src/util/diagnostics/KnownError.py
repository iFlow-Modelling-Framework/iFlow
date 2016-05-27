"""
Known Error
Contains information on know exceptions


Date: 28-04-15
Authors: Y.M. Dijkstra
"""
import traceback

class KnownError(Exception):
    
    # Variables
        

    # Methods
    def __init__(self, message, *args):
        """sava data of the known error

        Parameters:
            message - custom error message
            e - (optional) original error message
        """
        self.message = message

        self.e = None
        for i in args:
            self.e = i
            
        return

    def getOriginalErrorType(self):
        return type(self.e).__name__

