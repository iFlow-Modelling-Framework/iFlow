"""
Numerical data read from file
Now only supports 1D data import from txt files.

Date: 17-07-15
Authors: Y.M. Dijkstra
"""
import numpy as np
from nifty.functionTemplates import NumericalFunctionBase
from src.Reader import Reader
from src.util.diagnostics import KnownError
from src.DataContainer import DataContainer


class Gridded1D(NumericalFunctionBase):
    #Variables

    #Methods
    def __init__(self, dimNames, data):
        NumericalFunctionBase.__init__(self, dimNames)
        self.__file = data.v('file')
        self.__varname = data.v('varname')
        NumericalFunctionBase.checkVariables(self, ('file', self.__file), ('varname', self.__varname))

        if self.__file[-4:] == '.txt':
            self.__readtxt()
        else:
            raise KnownError('Unknown file extension of file '+str(self.__file))
        return

    def __readtxt(self):
        # read preamble
        reader = Reader()
        reader.open(self.__file)
        loadedData = reader.read('--')[0]   # DataContainer instance with read preamble data

        # determine the columns to read from the input file
        usecols=()
        columns = loadedData.v('columns')
        dimensions = loadedData.v('grid', 'dimensions')

        try:
            for i, var in enumerate(dimensions):
                usecols += (columns.index(var), )
        except ValueError as e:
            raise KnownError('Error in reading "'+self.__file+'". Not all dimensions are provided as data.', e)
        try:
            usecols += (columns.index(self.__varname),)
        except ValueError as e:
            raise KnownError('Error in reading "'+self.__file+'". Variable "'+self.__varname+'" not provided as data.', e)

        # read data
        startrow = self.__findDataStart(reader, 'data')
        value = np.loadtxt(self.__file, skiprows=startrow+1, usecols=usecols)

        # add data to data container. Only add grid data and requested data
        for i, var in enumerate(columns):
            if var in dimensions:
                # build grid from the read data
                dc = DataContainer({'grid': {'axis': {var: value[:, i]}}})
                loadedData.merge(dc)
            elif var == self.__varname:
                # add read values for the variable '__varname' to the numerical function
                self.addValue(value[:, i])
        # add grid to the numerical function
        self.addGrid(loadedData)

        reader.close()
        return

    def __findDataStart(self, reader, chapter):
        reader.filePointer.seek(0)    # restarts reading the file
        startChapter = ['--', chapter]
        for i, line in enumerate(reader.filePointer):
            if startChapter[0] in line[:len(startChapter[0])] and startChapter[1] in line:
                return i