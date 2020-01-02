"""
pickleload

Date: 29-Feb-16
Authors: Y.M. Dijkstra
"""
import pickle as pickle
import types
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
        # if instance, make it a function
        elif hasattr(data[key], 'function'):
            # a. also change instances in datacontainers within the instance
            classvars = vars(data[key])
            for var in classvars:
                if isinstance(data[key].__dict__[var], src.DataContainer.DataContainer):
                    __convertfunction(data[key].__dict__[var].data, data[key].__dict__[var].data.keys())
            # b. convert to function
            data[key] = data[key].function
    return