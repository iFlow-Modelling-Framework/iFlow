"""
Log Configurator
Creates a top-level logger and contains methods for configuring the logger


Original date: 28-04-15
Updated: 03-02-22
Authors: Y.M. Dijkstra
"""
import logging
import os
from src.config import ConsoleLoggingLevel, FileLoggingLevel
from src.util.diagnostics import DiagFormatter


class LogConfigurator:   
    # Variables
    

    # Methods
    def __init__(self,name):
        """create root level logger and set level

        Parameters:
            name - name of root level logger. Equals the name of the top level package (e.g. src)
        """
        self.logger = logging.root   # handle to root logger

        self.ConsoleLoggingLevelUp = ConsoleLoggingLevel.upper()
        self.FileLoggingLevelUp = FileLoggingLevel.upper()

        if not self.ConsoleLoggingLevelUp in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'OFF']:
            logging.info('Incorrect value for variable LoggingLevel in the iFlow config file. Please set to DEBUG, INFO, WARNING, ERROR, CRITICAL (see python logging system) or OFF. Logging level now set to INFO.')
            self.ConsoleLoggingLevelUp = 'INFO'
        if not self.FileLoggingLevelUp in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'OFF']:
            logging.info('Incorrect value for variable LoggingLevel in the iFlow config file. Please set to DEBUG, INFO, WARNING, ERROR, CRITICAL (see python logging system) or OFF. Logging level now set to INFO.')
            self.FileLoggingLevelUp = 'INFO'

        self.__setRootLevel(self.ConsoleLoggingLevelUp, self.FileLoggingLevelUp)
        for i in self.logger.handlers:
            self.logger.removeHandler(i)
        return

    def makeConsoleLog(self):
        """Configure the format for the console logger. Makes a handler for messages on the console
        """
        if not self.ConsoleLoggingLevelUp == 'OFF':
            console = logging.StreamHandler()
            self.__setLogLevel(self.ConsoleLoggingLevelUp, console)

            consoleFormat   = DiagFormatter('console')
            console.setFormatter(consoleFormat)         # assign formatter to handler


            # if not any([isinstance(i, logging.StreamHandler) for i in self.logger.handlers]):       # add a streamHandler logger if one does not exist
            self.logger.addHandler(console)
        return

    def makeDiagFile(self, filepath):
        """Create a new diag file in same directory as input file and add handlers for INFO, WARNING and ERROR

        Parameters:
            filepath - path to the input file
        """
        # extract path and extension of input file
        if not self.FileLoggingLevelUp == 'OFF':
            filepath.replace('\\','/')
            length = filepath.rfind('/')            # finds index of last occurrence of '/' or returns -1
            logFileLocation = filepath[:length+1]
            logFileName = filepath[length+1:-4]
            if logFileLocation[-1] != '/':
                logFileLocation = logFileLocation+'/'

            logFileExtension = filepath[-3:]
            logFilePath = (logFileLocation+'diag_'+logFileName + '.'+logFileExtension)

            # clear file if needed
            try:
                os.remove(logFilePath)
            except:
                pass

            # format and add handlers
            fileHandler = logging.FileHandler(logFilePath)
            self.__setLogLevel(self.FileLoggingLevelUp, fileHandler)

            fileHandlerFormat = DiagFormatter('file')
            fileHandler.setFormatter(fileHandlerFormat)

            self.logger.addHandler(fileHandler)
        return

    def __setLogLevel(self, level, handler):
        if level == 'DEBUG':
            handler.setLevel(logging.DEBUG)
        elif level == 'INFO':
            handler.setLevel(logging.INFO)
        elif level == 'WARNING':
            handler.setLevel(logging.WARNING)
        elif level == 'ERROR':
            handler.setLevel(logging.ERROR)
        elif level == 'CRITICAL':
            handler.setLevel(logging.CRITICAL)
        return

    def __setRootLevel(self, level1, level2):
        if level1 == 'DEBUG' or level2 == 'DEBUG':
            logging.basicConfig(level=logging.DEBUG)
        elif level1 == 'INFO' or level2 == 'INFO':
            logging.basicConfig(level=logging.INFO)
        elif level1 == 'WARNING' or level2 == 'WARNING':
            logging.basicConfig(level=logging.WARNING)
        elif level1 == 'ERROR' or level2 == 'ERROR':
            logging.basicConfig(level=logging.ERROR)
        elif level1 == 'CRITICAL' or level2 == 'CRITICAL':
            logging.basicConfig(level=logging.CRITICAL)
        return

