"""
Class OutputMat
Saves output to a .mat file for Matlab.

Date: 04-02-22
Authors: Y.M. Dijkstra
"""
import os
import src.config_menu as cfm
import logging
import numbers
import numpy as np
import scipy.io as sio
from .Output import Output


class OutputMat:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        self.ext = '.mat'      # file extension
        if self.input.v('outputgrid') is None:
            self.outputgridName = self.input.v('outputgrid')
        else:
            self.outputgridName = 'outputgrid'
        return

    def run(self):
        """invoke the saveData() method to save by using Pickle.

        Returns:
            Empty dictionary.
        """
        self.logger.info('Saving output to .mat')
        self.input.addData('saveAnalytical', None)
        OutputClass = Output(self.input)
        saveData = OutputClass.prepData()

        newSaveData = {}
        for keys in saveData.getAllKeys():
            subkeys_merge = ['_'.join(i) for i in keys]
            data = saveData.v(*keys)
            if data is not None:
                newSaveData[subkeys_merge] = data
            else:
                newSaveData[subkeys_merge] = np.nan

        self.saveData(newSaveData)
        return

    def saveData(self, saveData):
        ################################################################################################################
        # Make the output directory if it doesnt exist
        ################################################################################################################
        cwdpath = cfm.CWD       # path to working directory
        self.path = os.path.join(cwdpath, self.input.v('path'))
        if not self.path[-1]=='/':
            self.path += '/'
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        ################################################################################################################
        # set file name and save data
        ################################################################################################################
        filename = self.__makeFileName()
        # write
        filepath = (self.path + filename)
        try:
            sio.savemat(filepath, saveData, appendmat=True, do_compression=True, oned_as='row', long_field_names=True)
        except:
            raise

        ################################################################################################################
        # return
        ################################################################################################################
        d = {}
        d['outputDirectory'] = self.path
        return d

    def __makeFileName(self):
        # Prepare output file name format
        outputformat = ''.join(self.input.v('filename'))
        outputnames = []
        while outputformat.find('@{')>0:
            start = outputformat.find('@{')
            end = outputformat.find('}')
            outputnames.append(outputformat[start+2:end])
            outputformat = outputformat.replace(outputformat[start:end+1], '%s')

        counter = [0]*len(outputnames)    # set a counter for the case where the outputnames cannot be evaluated to a number; the counter then replaces this.

        # set output file name
        outnames = []
        for i, key in enumerate(outputnames):
            key = key.strip('"')
            key = key.strip("'")
            key = key.split(',')
            key = [self.__tryint(qq) for qq in key]
            outnames.append(self.input.v(*key))
            if not isinstance(outnames[-1], numbers.Number):
                try:
                    outnames[-1] = float(outnames[-1])
                except:
                    outnames[-1] = counter[i]
                    counter[i]+=1
        filename = outputformat % tuple(outnames)

        # check if file name already exists. If it does, append the filename by '_' + the first number >1 for which the path does not exist
        if os.path.exists(self.path+filename+self.ext):
            i=2
            while os.path.exists(self.path+filename+'_'+str(i)+self.ext):
                i += 1
            filename = filename+'_'+str(i)

        return filename

    def __tryint(self, i):
        try:
            i = int(i)
        except:
            pass
        return i

