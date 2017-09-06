"""
ReadMultiple

Date: 11-Nov-15
Update: 13-07-17 - allow for reading files from multiple folders
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
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        self.logger.info('Reading files')
        ## set folder(s)
        cwdpath = self.input.v('CWD') or ''  # path to working directory
        folders = ny.toList(self.input.v('folder'))
        folders = [os.path.join(cwdpath, i) for i in folders]

        ## determine unique part of folder names
        folder_name = []
        for folder in folders:
            tempname = folder.replace('\\', '/')
            folder_name.append(tempname.split('/'))
        nameparts = list(set([item for sublist in folder_name for item in sublist]))
        nameinall = [all([i in j for j in folder_name]) for i in nameparts]
        nameparts = [prt for i, prt in enumerate(nameparts) if nameinall[i]]
        for i, folder in enumerate(folder_name):
            for name in nameparts:
                try:
                    folder_name[i].remove(name)
                except:
                    pass
            if len(folder_name[i])>0:
                folder_name[i].append('')

        ## Loop over folders
        experimentdata = {}
        for i, folder in enumerate(folders):
            ## scan files
            files = ny.toList(self.input.v('files'))
            # If files is 'all' then get all files that have a .p extension, else take the listing files
            if files[0] == 'all':
                files = [f[:-2] for f in os.listdir(folder) if os.path.isfile(os.path.join(folder,f)) and f.endswith('.p') ]
            else:
                files = [f for f in ny.toList(self.input.v('files')) if os.path.isfile(os.path.join(folder,f)) and f.endswith('.p')]

            ## load data from the files
            for file in files:
                if self.input.v('variables') == 'all':
                    varlist = None
                else:
                    varlist = ny.toList(self.input.v('variables'))+['grid']
                d = ny.pickleload(os.path.join(folder, file), varlist)
                experimentdata['/'.join(folder_name[i]) + file] = DataContainer(d)

        d = {}
        d['experimentdata'] = experimentdata
        return d