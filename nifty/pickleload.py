"""
pickleload

Date: 29-Feb-16
Authors: Y.M. Dijkstra
"""
import cPickle as pickle
import types
from src.util.diagnostics.KnownError import KnownError
import src.DataContainer


def pickleload(filepath, variables):
    d = {}
    try:
        with open(filepath+'.p','rb') as fp:
            alldata = pickle.load(fp)
    except IOError as e:
        raise KnownError('Could not find file %s.p' % (filepath), e)
    except pickle.UnpicklingError as e:
        raise KnownError('File %s.p is not a valid Pickle file and cannot be loaded.' % (filepath), e)

    # Check if requested variables are available and load them to dict d
    if variables is None:
        variables = alldata.keys()
        d = alldata
    else:
        for key in variables:
            # verify that requested key exists, else raise an exception
            if key not in alldata:
                raise KnownError('Could not load variable %s from file %s' % (key, file))
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
        elif isinstance(data[key], types.InstanceType):
            # a. also change instances in datacontainers within the instance
            classvars = vars(data[key])
            for var in classvars:
                if isinstance(data[key].__dict__[var], src.DataContainer.DataContainer):
                    __convertfunction(data[key].__dict__[var].data, data[key].__dict__[var].data.keys())
            # b. convert to function
            data[key] = data[key].function
    return