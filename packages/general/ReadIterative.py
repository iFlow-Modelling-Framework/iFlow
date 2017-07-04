"""
ReadIterative

Date: 29-6-2017
Authors: Y.M. Dijkstra
"""
import nifty as ny
import os
import logging


class ReadIterative:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def stopping_criterion(self, iteration):
        self.iteration = iteration
        if iteration >= len(self.files):
             return True
        else:
            return False

    def run_init(self):
        self.logger.info('Reading files')
        cwdpath = self.input.v('CWD') or ''  # path to working directory
        self.folder = os.path.join(cwdpath, self.input.v('folder'))

        self.files = ny.toList(self.input.v('files'))
        # If files is 'all' then get all files that have a .p extension
        if self.files[0] == 'all':
            self.files = [ f[:-2] for f in os.listdir(self.folder) if os.path.isfile(os.path.join(self.folder,f)) and f.endswith('.p') ]

        d = self.read(0)
        d['experimentlength'] = len(self.files)

        return d

    def run(self):
        return self.read(self.iteration)

    def read(self, iteration):
        file = self.files[iteration]
        if self.input.v('variables') == 'all':
            varlist = None
        else:
            varlist = ny.toList(self.input.v('variables'))+['grid']
        d = ny.pickleload(os.path.join(self.folder, file), varlist)
        d['experimentnumber'] = iteration
        return d