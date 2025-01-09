"""
Class Output
Saves output as a numpy structure to the output location on the output grid

Original date: 12-11-15
Updated: 04-02-22
Original authors: Y.M. Dijkstra, R.L. Brouwer
Update authors: Y.M. Dijkstra
"""
import os
import src.config_menu as cfm
import pickle as pickle
import logging
import numbers
from src.util.grid import callDataOnGrid
from src.DataContainer import DataContainer
from nifty import toList
import types
from src.Reader import Reader
import numpy as np
from copy import deepcopy


class Output:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self, input):
        self.input = input
        self.ext = '.p'      # file extension
        return

    def run(self):
        """invoke the saveData() method to save by using Pickle.

        Returns:
            Empty dictionary.
        """
        self.logger.info('Saving output')

        saveData = self.prepData()
        d = self.saveData(saveData)
        return d

    def prepData(self):
        ################################################################################################################
        # Get all variables from input, config and modules
        #   config and input variables are saved always (since iFlow3 also when overwritten by data of different type in modules)
        #   data from modules are first all merged and then filtered for 'requirements'
        ################################################################################################################
        # read input file and load to inputVars
        reader = Reader()
        reader.open(self.input.v('inputFile'))
        data = reader.read('module')
        reader.close()
        inputvars = DataContainer()

        # make the module tags into a list of modules
        module = []
        for d in data:
            module.append(d.v('module'))
            inputvars.merge(d)
        inputvars.addData('module', module)

        # read config vars and merge with input (config vars are overwritten if needed)
        allVars = self.__loadConfig()
        allVars.merge(inputvars)
        configInputKeys = allVars.getAllKeys()

        # merge data from the modules (overwrite input/config data)
        self.__checkInputOverrides(allVars, self.input)  # check if there is overwritten data and write warning if overwritten with different data type. May signal unexpected behaviour.
        allVars.merge(self.input)
        del inputvars, reader, data

        ################################################################################################################
        # Isolate part of DC to write; put this in saveData
        ################################################################################################################
        saveData = DataContainer()

        # make list of vars to save
        outputVariables = [i for i in allVars.getAllKeys() if i[0] in toList(self.input.v('requirements'))]
        outputVariables = outputVariables + configInputKeys
        derKeys = [i for i in allVars.getAllKeys() if i[0]=='__derivative']
        derKeysUpd = [i for i in derKeys if any([j for j in outputVariables if i[2:min(len(i)-2,len(j))+2]==j[:min(len(i)-2,len(j))]])]
        outputVariables = outputVariables + derKeysUpd

        # add data from the list outputVariables to saveData
        for key in outputVariables:
            saveData.merge(self._buildDicts(key, allVars.v(*key)))

        # add __variableOnGrid
        saveData.merge(self.input.slice('__variableOnGrid'))

        # add __outputGrid
        saveData.merge(self.input.slice('__outputGrid'))

        # add all grids (incl grid and outputgrid) if available to saveData and to a dedicated DC
        grids = DataContainer()
        gridslist = list(set(['grid'] + list(self.input._data['__variableOnGrid'].values()) + list(self.input._data['__outputGrid'].values())))
        for i in gridslist:
            grids.merge(self.input.slice(i))
            saveData.merge(self.input.slice(i))
            grid_addition = {}
            grid_addition['__variableOnGrid'] = {}
            grid_addition['__variableOnGrid'][i] = i
            saveData.merge(grid_addition)

        for i in gridslist:
            # add reference level to all grids
            if allVars.v('R') is not None:
                allVars.merge({i:{'low':{'z':saveData.v('R', x=saveData.v(i, 'axis', 'x'))}}})     # add reference level to outputgrid
                saveData.merge({i:{'low':{'z':saveData.v('R', x=saveData.v(i, 'axis', 'x'))}}})     # add reference level to outputgrid
                grids.merge({i:{'low':{'z':saveData.v('R', x=saveData.v(i, 'axis', 'x'))}}})     # add reference level to outputgrid

        ################################################################################################################
        # Convert data using output grid (if this is provided)
        ################################################################################################################
        convertedSaveData = self._convertData(saveData, grids)

        return convertedSaveData

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
        filepath = (self.path + filename + self.ext)
        try:
            with open(filepath, 'wb') as fp:
                pickle.dump(saveData._data, fp, protocol=pickle.HIGHEST_PROTOCOL)
        except:
            raise

        ################################################################################################################
        # return
        ################################################################################################################
        d = {}
        d['outputDirectory'] = self.path
        return d

    def _convertData(self, saveData, grids):
        """
        """
        convertedData = DataContainer()

        # take all subkeys except for those included in grids and the key '__variableOnGrid'
        subkeys = [i for i in saveData.getAllKeys() if i[0][:2]!='__'] + [i for i in saveData.getAllKeys() if i[0][:2]=='__'] # make sure underscored variables are at the end
        gridkeys = [i for i in subkeys if i[0] in grids._data.keys() and i[0][:2]!='__']
        variableOnGrid = DataContainer()
        for key in [i for i in variableOnGrid._data.keys() if i[:2]=='__']:
            variableOnGrid._data.pop(key)

        # check keywords saveAnalytical and dontConvert
        saveAnalytical = toList(self.input.v('saveAnalytical')) or []
        dontConvert = toList(self.input.v('dontConvert')) or []
        if 'all' in saveAnalytical:
            saveAnalytical = subkeys
        elif len(saveAnalytical)>0:
            saveAnalytical = [i for i in saveData.getAllKeys() if i[0] in saveAnalytical] + [i for i in saveData.getAllKeys() if i[0]=='__derivative' and i[2] in saveAnalytical]
        if 'all' in dontConvert:
            dontConvert = subkeys
        elif len(dontConvert)>0:
            dontConvert = [i for i in saveData.getAllKeys() if i[0] in saveAnalytical] + [i for i in saveData.getAllKeys() if i[0]=='__derivative' and i[2] in saveAnalytical]

        for keys in subkeys:     # reverse sort to make sure underscored keys are at the end and hence not overwritten.
            # check saving method for different instance types
            value = saveData.v(*keys)

            #   a. Function + saveAnalytical
            #       From iFlow3: store function just as it is and don't convert underlying data.
            #       One should not save too much data in a function or otherwise have it converted to numerical data
            if isinstance(value, types.MethodType) and keys in saveAnalytical:
                valuecopy = deepcopy(value)

                # remove private class variables from the copied data
                classvars = [var for var in vars(valuecopy.__self__) if var[0]=='_'] # get private class vars
                for var in classvars:
                    valuecopy.__self__.__dict__.pop(var)
                data = (valuecopy.__self__, valuecopy.__name__)

                for i in range(1,len(keys)+1)[::-1]:
                    v = saveData.v('__variableOnGrid', *keys[:i])
                    if v is not None:
                        variableOnGrid.merge(self._buildDicts(keys[:i], v))

            #   b. values not to convert
            elif keys in dontConvert:
                data = value
                for i in range(1, len(keys)+1)[::-1]:
                    v = saveData.v('__variableOnGrid', *keys[:i])
                    if v is not None:
                        variableOnGrid.merge(self._buildDicts(keys[:i], v))

            #   c. Arrays, functions + not saveAnalytical and other
            else:
                if keys in gridkeys:
                    gridname = keys[0]  # grids should not be evaluated
                else:
                    inputgrid = 'grid' # for all other variables, determine the outputgrid. First set default and check whether this is correct below
                    for i in range(1,len(keys)+1)[::-1]:
                        v = saveData.v('__variableOnGrid', *keys[:i])
                        if v is not None:
                            inputgrid = v
                            break
                    gridname = saveData.v('__outputGrid', inputgrid)

                data, _ = callDataOnGrid(saveData, keys, gridname, False)

            # merge into saveData
            convertedData.merge(self._buildDicts(keys, data))
            convertedData.addData('__variableOnGrid', variableOnGrid._data)

        # replace grids by their outputgrids and merge into the dataset
        d = {}
        for g in saveData.getKeysOf('__outputGrid'):
            outgridname = saveData.v('__outputGrid', g)
            d.update({g:convertedData._data[outgridname]})
            convertedData._data.pop(outgridname)
        convertedData.merge(d)

        return convertedData

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
        from src import config as cf
        configvars = [var for var in dir(cf) if not var.startswith('__')]
        d = {}
        for var in configvars:
            d[var] = eval('cf.'+var)
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

    def __checkInputOverrides(self, inputData, modInput):
        # check that sub-vars of input vars are not removed by module calculated variables.
        # This is often a sign of overwriting a variable with a different data type
        modKeys = modInput.getAllKeys()
        inputKeys = inputData.getAllKeys()
        warnings = [] # list of variables that already have warnings, prevents multiple warnings per variable

        for inkey in inputKeys:
            check = True
            # if the same key+subkeys exists in modInput, check that they are the same datatype
            if inkey in modKeys:
                if isinstance(modInput.v(inkey), types.MethodType) or isinstance(inputData.v(inkey), types.MethodType):
                    check = isinstance(modInput.v(inkey), types.MethodType) and isinstance(inputData.v(inkey), types.MethodType)  # True if both are functions
                else:
                    check = np.asarray(modInput.v(inkey)).shape == np.asarray(inputData.v(inkey)).shape
            # elseif only the first element of the key corresponds, the input must have been overwritten by a different datatype
            elif inkey[0] in [key[0] for key in modKeys]:
                check = False

            if not check:
                if not inkey[0] in warnings:
                    warnings.append(inkey[0])
                    self.logger.warning('Input/config variable %s has been overwritten by a different data type or differently shaped data in the modules. '
                                    'This variable is written to output but may take more disk space than anticipated.'
                                    'It is advised to change the name of the input/config variable to prevent this.' % inkey[0])
        return

    def __tryint(self, i):
        try:
            i = int(i)
        except:
            pass
        return i

