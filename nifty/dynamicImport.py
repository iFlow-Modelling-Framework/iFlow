"""
28-04-15 (update 18-01-2017)
@author: Y.M. Dijkstra
"""
# import importlib as imp
from src.util.diagnostics import KnownError
from src.util.localpath import localpath
import imp


def dynamicImport(path, name):
    """Dynamically load a class from its path.

    Parameters:
        path - path to the class to be loaded. This includes the class name without .py
               Accepts folder description separated by ., \ or /.
               Also accepts None; an empty path
        name - name of the class file (without .py)

    Returns:
        pointer to the class' __init__ method

    Raises:
        Exception is the class could not be found
    """
    loader_details = (imp.machinery.ExtensionFileLoader, imp.machinery.EXTENSION_SUFFIXES)

    # set . as the path separator
    if path is None or path == '':
        packagePath = ''
    else:
        packagePath = path.replace('\\', '.')
        packagePath = packagePath.replace('/', '.')
        if packagePath[-1:] != '.':                 # add . at the end if not already
            packagePath += '.'

    # import and retrieve constructor method
    try:
        workpath = (packagePath+name).split('.')            # new in v2.4 to make sure modules are only searched for in a controlled set of packages
        # find and load package and subpackages
        i = 0
        while i < len(workpath)-1:
            try:
                if i == 0:
                    modpack = imp.find_module(workpath[i], localpath)
                else:
                    modpack = imp.find_module(workpath[i], [modpack[1]])
            except ImportError as e:
                raise KnownError('Error while loading module ' + '.'.join(workpath)+'.\nPlease check if '+str(workpath[-1]+' exists in package '+'.'.join(workpath[:-1]))+'.\nError given is: '+str(e))
            imp.load_module('.'.join(workpath[:i+1]), *modpack)
            i += 1
        # find and load the module
        modlocation = imp.find_module(name, [modpack[1]])
        programPointer = imp.load_module(packagePath+name, *modlocation)
        programMain_ = getattr(programPointer, name)

    except ImportError as e:
        errorlocation = str(e).split(' ')[-1]
        errorlocation = errorlocation.split('.')[-1]

        if errorlocation==name:
            raise KnownError("Could not find the class file with name '%s.py' in package '%s'." % (name, path), e)
        else:
            raise           # re-raises the original exception
    except AttributeError as e:
        errorlocation = str(e).split(' ')[-1]
        errorlocation = errorlocation.split('.')[-1]
        errorlocation = errorlocation.replace("'", "")
        errorlocation = errorlocation.replace('"', '')
        if errorlocation == name:
            raise KnownError("Could not find the class name '%s' in package '%s'" % (name, path), e)
        else:
            raise


    return programMain_