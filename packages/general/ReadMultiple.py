"""
ReadMultiple

Date: 11-Nov-15
Authors: Y.M. Dijkstra
"""
import nifty as ny
import os
from src.DataContainer import DataContainer
import logging


class ReadMultiple:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input, submodulesToRun):
        self.input = input
        return

    def run(self):
        self.logger.info('Reading files')
        folder = self.input.v('folder')
        files = ny.toList(self.input.v('files'))
        # If files is 'all' then get all files that have a .p extension
        if files[0] == 'all':
            files = [ f[:-2] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f)) and f.endswith('.p') ]

        # load data from the files
        experimentdata = {}
        for file in files:
            if self.input.v('variables') == 'all':
                varlist = None
            else:
                varlist = ny.toList(self.input.v('variables'))+['grid']
            d = ny.pickleload(os.path.join(folder, file), varlist)
            experimentdata[file] = DataContainer(d)

        d = {}
        d['experimentdata'] = experimentdata
        return d