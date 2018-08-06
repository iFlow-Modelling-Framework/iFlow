"""
Log Configurator
Creates a top-level logger and contains methods for configuring the logger

Date: 28-04-15
Authors: Y.M. Dijkstra
"""
import logging
import os
from src.util.diagnostics import DiagFormatter


class LogConfigurator:   
    # Variables
    

    # Methods
    def __init__(self,name):
        """create root level logger and set level

        Parameters:
            name - name of root level logger. Equals the name of the top level package (e.g. src)
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        return

    def getCount(self):
        return self.logger.handlers[0].level2count

    def makeConsoleLog(self):
        """Configure the format for the console logger. Makes a handler for messages on the console
        """
        console = logging.StreamHandler() #MsgCounterHandler()
        console.setLevel(logging.INFO)
        consoleFormat   = DiagFormatter('console')
        console.setFormatter(consoleFormat)
        if not any([isinstance(i, logging.StreamHandler) for i in self.logger.handlers]):       # add a streamHandler logger if one does not exist
            self.logger.addHandler(console)
        return

    def makeDiagFile(self, filepath):
        """Create a new diag file in same directory as input file and add handlers for INFO, WARNING and ERROR

        Parameters:
            filepath - path to the input file
        """
        # extract path and extension of input file 
        filepath.replace('\\','/')
        length = filepath.rfind('/')            # finds index of last occurrence of '/' or returns -1
        logFileLocation = filepath[:length+1]
        
        logFileExtension = filepath[-3:]
        logFilePath = (logFileLocation+'diag.'+logFileExtension)

        # clear file if needed
        try:
            os.remove(logFilePath)
        except:
            pass

        # format and add handlers
        fileHandler = logging.FileHandler(logFilePath)
        fileHandler.setLevel(logging.INFO)
        fileHandlerFormat = DiagFormatter('file')
        fileHandler.setFormatter(fileHandlerFormat)
        self.logger.addHandler(fileHandler)
        
        return

# class MsgCounterHandler(logging.StreamHandler):
#     level2count = None
#
#     def __init__(self, *args, **kwargs):
#         super(MsgCounterHandler, self).__init__(*args, **kwargs)
#         self.level2count = {}
#
#     def emit(self, record):
#         super(MsgCounterHandler, self).emit(record)
#         l = record.levelname
#         if (l not in self.level2count):
#             self.level2count[l] = 0
#         self.level2count[l] += 1
