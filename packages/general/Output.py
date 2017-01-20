"""
Output Writer
Saves output as a numpy structure to the output location on the output grid

Date: 12-11-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import os
import cPickle as pickle
import logging
import numbers
from src.util.grid import callDataOnGrid
from src.DataContainer import DataContainer
from nifty import toList
import nifty as ny
from copy import deepcopy
import types
from src.Reader import Reader
import numpy as np


class Output:
    # Variables
    logger = logging.getLogger(__name__)
    ext = '.p'      # file extension
    outputgridName = 'outputgrid'

    # Methods
    def __init__(self, input):
        self.input = input
        return

    def run(self):
        """invoke the saveData() method to save by using Pickle.

        Returns:
            Empty dictionary.
        """
        self.logger.info('Saving output')

        ################################################################################################################
        # Make all variables from config, input and modules available (note, config not available in output module, see Program.py)
        ################################################################################################################
        # read input file
        reader = Reader()
        reader.open(self.input.v('inputFile'))
        data = reader.read('module')
        reader.close()

        # merge the datacontainers of all modules & make the module tags into a list of modules
        inputvars = DataContainer()
        module = []
        for d in data:
            module.append(d.v('module'))
            inputvars.merge(d)
        inputvars.addData('module', module)

        # merge input vars with self.input in hierarchy config, input, input for this module, vars calculated in other modules (low - high)
        # + make a list of all keys of input and config vars; these are saved always and later appended by selected module calc. vars.
        data = self.__loadConfig()
        data.merge(inputvars)
        outputKeys = self.__checkInputOverrides(data) # checks if input is overwritten and provides the keys of not or correctly overwritten input vars
        data.merge(self.input)
        del inputvars, reader
        # now all variables from config, input and modules are in 'data'

        ################################################################################################################
        # Isolate part of DC to write; put this in saveData
        ################################################################################################################
        saveData = DataContainer()

        # vars to save
        outputVariables = toList(self.input.v('requirements'))
        outputKeys = list(set(outputKeys + self.__getSubmoduleRequirements(outputVariables)))        # convert the requested output variables to key tuples including submodule requirements
        for key in outputKeys:
            if len(key)>1:
                saveData.merge({key[0]: data.slice(*key).data})
            else:
                saveData.merge(data.slice(*key))

        # add grid and outputgrid if available; needed for interpolating data to outputgrid later
        saveData.merge(self.input.slice('grid'))
        saveData.merge(self.input.slice(self.outputgridName))

        # make a deepcopy of the data to be saved
        # NB. very memory inefficient, but needed not to overwrite existing data
        saveData = deepcopy(saveData)

        ################################################################################################################
        # Convert data using output grid (if this is provided)
        ################################################################################################################
        grid = saveData.slice('grid')
        outputgrid = saveData.slice(self.outputgridName)
        saveAnalytical = toList(self.input.v('saveAnalytical')) or []
        dontConvert = toList(self.input.v('dontConvert')) or []
        if 'all' in saveAnalytical:
            saveAnalytical = outputVariables
        if 'all' in dontConvert:
            dontConvert = outputVariables
        self._convertData(saveData, grid, outputgrid, saveAnalytical, dontConvert)

        # rename the outputgrid to grid and replace the original grid in saveData
        saveData.addData('grid', saveData.data[self.outputgridName])
        saveData.data.pop(self.outputgridName)

        ################################################################################################################
        # Make the output directory if it doesnt exist
        ################################################################################################################
        cwdpath = data.v('CWD') or ''       # path to working directory
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
        filepath = (self.path + filename + self.ext)
        try:
            with open(filepath, 'wb') as fp:
                pickle.dump(saveData.data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            raise

        ################################################################################################################
        # return
        ################################################################################################################
        d = {}
        d['outputDirectory'] = self.path
        return d

    def __getSubmoduleRequirements(self, outputVariables):
        # refactor output requirements to key tuples to check against dataContainer keys. Load this into reqKeys
        reqKeys = []
        for var in toList(outputVariables):
            submoduleRequirements = self.input.v('submodules', var)
            if submoduleRequirements == None or submoduleRequirements == 'all':
                reqKeys.append((var,))
            else:
                for submod in toList(submoduleRequirements):
                    reqKeys.append((var,submod))
        return reqKeys

    def _convertData(self, saveData, grid, outputgrid, saveAnalytical, dontConvert, convertGrid = False):
        """
        """
        # merge grids into one DC
        grid.merge(outputgrid)

        # take all subkeys
        subkeys = saveData.getAllKeys()

        for keys in subkeys:
           # check saving method for different instance types
            value = saveData.v(*keys)

            #   a. Numerical function
            if isinstance(value, types.MethodType) and isinstance(value.__self__, ny.functionTemplates.NumericalFunctionBase):
                data = value.__self__       # save reference to instance
                nfgrid = data.dataContainer.slice('grid')
                if keys[0] in dontConvert:      # if in dontconvert, propagate that all underlying elements are not converted
                    dontConvert_nf = list(set([i[0] for i in data.dataContainer.getAllKeys()]))
                else:                           # else, add the output grid to the nf
                    dontConvert_nf = []
                    data.dataContainer.merge(outputgrid.data)
                self._convertData(data.dataContainer, nfgrid, outputgrid, [], dontConvert_nf, convertGrid = True) # recursively convert the data inside the numerical function

            #   b. Other function + saveAnalytical
            elif isinstance(value, types.MethodType) and isinstance(value.__self__, ny.functionTemplates.FunctionBase) and keys[0] in saveAnalytical:
                data = value.__self__       # save reference to instance

                # convert all public data stored inside DCs inside the function
                classvars = [var for var in vars(data) if not var[0]=='_'] # public class vars
                for var in classvars:
                    if isinstance(data.__dict__[var], DataContainer):
                        # check for possible endless recursion, where the DC inside the function contains the function itself
                        if data.__dict__[var].v(*keys) is None:
                            self._convertData(data.__dict__[var], grid, outputgrid, saveAnalytical, dontConvert)
                        else:
                            self.logger.error('Could not save variable %s as analytical function; variable is not saved.\n'
                                                'Reason: the function stores itself as class variable, leading to an endless recursion.\n'
                                                'Please check the module that defines this function and make sure it saves only the minimum required amount of data in class variables.\n'
                                                'Alternatively, save this variable as numerical data' % keys[0])
                            data = None
                            break
                classvars = [var for var in vars(data) if var[0]=='_'] # private class vars
                for var in classvars:
                    data.__dict__.pop(var)                              # remove private class vars

            #   c. Arrays, functions + not saveAnalytical and other
            else:
                # Convert data to grid if key is in 'dontConvert', else convert to outputgrid
                if keys[0] in dontConvert or keys[0] in ['grid']:
                    gridname = 'grid'
                else:
                    gridname = self.outputgridName

                # call on grid 'gridname'
                data, _ = callDataOnGrid(saveData, keys, grid, gridname, False)

            # merge into saveData
            saveData.merge(self._buildDicts(keys, data))

        if convertGrid and saveData.v('grid') and 'grid' not in dontConvert:                      # if we are in a numerical function, the grid should be converted to the output grid
            saveData.addData('grid', saveData.data[self.outputgridName])  # rename the outputgrid to grid and replace the original grid in saveData
            saveData.data.pop(self.outputgridName)

        return

    def _buildDicts(self, keys, data):
        """ """
        d1 = {}
        d2 = {}
        d1[keys[-1]] = data
        for i in range(len(keys)-2,-1,-1):
            d2[keys[i]] = d1
            d1 = d2
            d2 = {}

        return d1

    def __loadConfig(self):
        import src.config as cf
        configvars = [var for var in dir(cf) if not var.startswith('__')]
        d = {}
        for var in configvars:
            exec('d[var] = cf.'+var)
        return DataContainer(d)

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
            exec('outnames.append(self.input.v('+key+'))')
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

    def __checkInputOverrides(self, inputData):
        # check that sub-vars of input vars are not removed by module calculated variables.
        # This is often a sign of overwriting a variable with a different data type
        modKeys = self.input.getAllKeys()
        inputKeys = inputData.getAllKeys()
        inputKeysCopy = inputData.getAllKeys()
        warnings = [] # list of variables that already have warnings, prevents multiple warnings per variable

        for inkey in inputKeysCopy:
            check = True
            # if the same key+subkeys exists in self.input, check that they are the same datatype
            if inkey in modKeys:
                if isinstance(self.input.v(inkey), types.MethodType) or isinstance(inputData.v(inkey), types.MethodType):
                    check = isinstance(self.input.v(inkey), types.MethodType) and isinstance(inputData.v(inkey), types.MethodType)  # True if both are functions
                else:
                    check = np.asarray(self.input.v(inkey)).shape == np.asarray(inputData.v(inkey)).shape
            # elseif only the first element of the key corresponds, the input must have been overwritten by a different datatype
            elif inkey[0] in [key[0] for key in modKeys]:
                check = False

            if not check:
                inputKeys.remove(inkey)
                if not inkey[0] in warnings:
                    warnings.append(inkey[0])
                    self.logger.warning('Input/config variable %s has been overwritten by a different data type or differently shaped data. '
                                    'This variable is now only written to output if explicitly requested in input file. '
                                    'It is advised to change the name of the input/config variable.' % inkey[0])
        return inputKeys

