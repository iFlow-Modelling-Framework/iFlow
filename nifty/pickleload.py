"""
pickleload

Original date: 29-Feb-16
Update: 04-02-22
Authors: Y.M. Dijkstra
"""
import pickle as pickle
from src.util.diagnostics.KnownError import KnownError
import src.DataContainer


def pickleload(filepath, variables):
    d = {}
    if filepath[-2:] != '.p':
        filepath = filepath+'.p'

    try:
        try:
            with open(filepath,'rb') as fp:
                alldata = pickle.load(fp)
        except UnicodeDecodeError:
            with open(filepath, 'rb') as fp:
                alldata = pickle.load(fp, encoding="latin1")

    except IOError as e:
        raise KnownError('Could not find file %s' % (filepath), e)
    except pickle.UnpicklingError as e:
        raise KnownError('File %s is not a valid Pickle file and cannot be loaded.' % (filepath), e)

    # Check if requested variables are available and load them to dict d
    if variables is None:
        variables = alldata.keys()
        d = alldata
    else:
        for key in variables:
            # verify that requested key exists, else raise an exception
            if key not in alldata:
                raise KnownError('Could not load variable %s from file %s' % (key, filepath))
            # load data
            d[key] = alldata[key]

    # convert instances to functions
    __convertfunction(d, variables)

    return d

def __convertfunction(data, variables):
    for key in variables:
        # if dict, go a level deeper
        if isinstance(data[key], dict):
            __convertfunction(data[key], data[key].keys())

        # if tuple with the first argument having 'dimNames', assume this is a class with an analytical function
        elif isinstance(data[key], tuple) and hasattr(data[key][0], 'dimNames'):
            # a. also change instances in datacontainers within the instance
            classinst = data[key][0]
            classvars = vars(classinst)
            for var in classvars:
                if isinstance(classinst.__dict__[var], src.DataContainer.DataContainer):
                    __convertfunction(classinst.__dict__[var]._data, classinst.__dict__[var]._data.keys())
            # b. convert to function
            data[key] = eval('classinst.'+data[key][1])
    return