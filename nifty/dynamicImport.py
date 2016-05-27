"""
28-04-15
@author: Y.M. Dijkstra
"""
from src.util.diagnostics import KnownError


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
        Exception is the class could not be flound (TODO)
    """
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
        programPointer = __import__((packagePath+'%s') % name, fromlist=[packagePath])
        programMain_ = getattr(programPointer, name)
    except ImportError as e:
        errorlocation = e.message.split(' ')[-1]
        errorlocation = errorlocation.split('.')[-1]

        if errorlocation==name:
            raise KnownError("Could not find the class file with name '%s.py' in package '%s'." % (name, path), e)
        else:
            raise           # re-raises the original exception
    except AttributeError as e:
        errorlocation = e.message.split(' ')[-1]
        errorlocation = errorlocation.split('.')[-1]
        errorlocation = errorlocation.replace("'", "")
        errorlocation = errorlocation.replace('"', '')
        if errorlocation == name:
            raise KnownError("Could not find the class name '%s' in package '%s'" % (name, path), e)
        else:
            raise


    return programMain_