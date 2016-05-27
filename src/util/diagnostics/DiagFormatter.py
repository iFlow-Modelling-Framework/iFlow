"""
Class DiagFormatter
Formatter for the diagnostic messages on console or file

Date: 28-04-15
Authors: Y.M. Dijkstra
"""
import logging

class DiagFormatter(logging.Formatter):

    infoFile = '%(message)s (in %(name)s)'
    warningFile = 'WARNING: %(message)s (in %(name)s)'
    errorFile = ('\nERROR: %(message)s'+
                '\n---------------------------' +
                '\nError type: %(type)s' +
                '\nError traceback: %(trace)s')
    errorConsole  = '\nERROR: %(message)s'
    warningConsole = 'WARNING: %(message)s'
    infoConsole = '%(message)s'

    def __init__(self,medium):
        self.medium = medium
        return

    def format(self, record):
        #test if additional values are present
        try:
            record.trace
        except:
            record.trace = ""
        try:
            record.type
        except:
            record.type = ""

        #implement formatting rules
        if record.levelno == logging.INFO:
            if self.medium == 'console':
                self._fmt = self.infoConsole
            elif self.medium == 'file':
                self._fmt = self.infoFile

        elif record.levelno == logging.WARNING:
            if self.medium == 'console':
                self._fmt = self.warningConsole
            elif self.medium == 'file':
                self._fmt = self.warningFile

        elif record.levelno == logging.ERROR:
            if self.medium == 'console':
                self._fmt = self.errorConsole
            elif self.medium == 'file':
                
                self._fmt = self.errorFile

        # the default formatting class implements the format
        format = logging.Formatter.format(self, record)
        return format