"""
ReadSingle

Date: 11-Nov-15
Authors: Y.M. Dijkstra
"""
import nifty as ny
import os
from src.util.diagnostics import KnownError
import src.config_menu as cfm


class ReadSingle:
    # Variables

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        cwdpath = cfm.CWD
        folder = os.path.join(cwdpath, self.input.v('folder'))

        file = self.input.v('file')
        # If files is 'all' then get all files that have a .p extension
        if file == 'all':
            file = [ f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f)) and f.endswith('.p') ]

        # check if only one file was requested
        if len(ny.toList(file)) > 1:
            raise KnownError('Found multiple files to load in module %s. This module can only load one file' % (self.__module__))

        # take the single file (in case files has a list type)
        file = ny.toList(file)[0]

        if self.input.v('variables') == 'all':
            varlist = None
        else:
            varlist = ny.toList(self.input.v('variables'))+['grid']+['__variableOnGrid']+['__derivative']
        d = ny.pickleload(os.path.join(folder, file), varlist)

        return d