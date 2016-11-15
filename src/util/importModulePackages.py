"""
importModulePackages

Date: 15-12-15
Authors: Y.M. Dijkstra
"""
import os
import pkgutil
import sys
from nifty import toList
from src.util.diagnostics import LogConfigurator
import packages          # DEFAULT PACKAGE LOCATION


def importModulePackages(cwd, imports):
    """import module packages from:
    (1) the standard folder 'packages',
    (2) the working directory 'cwd' and
    (3) the paths given in 'imports'
    by adding them to sys.path (= temporary path variable for run-time of the session)

    Parameters:
        imports (list of DataContainers) - DataContainer list containing lines of import statements in the input file
    """
    # add default location
    #  NB.  need to add package itself and its parent package.
    #       The parent is needed because dynamic import uses the package name to identify the module
    #       The parent package is loaded before the child packages to make sure that the program can find packages with any name
    if sys.platform[:3] == 'win':
        slash1 = '/'
        slash2 = '\\'
    else:
        slash1 = '\\'
        slash2 = '/'

    # 1. import packages in 'packages' folder
    #   list packages
    modlist = []
    for importer, modname, ispkg in pkgutil.walk_packages(path=packages.__path__, onerror=lambda x: None):
        if ispkg:
            modlist.append(modname)

    #  add to path and set logging system
    sys.path.append(packages.__path__[0])
    for modname in modlist:
        sys.path.append(packages.__path__[0] + slash2 + modname)            # add to path
        logConf = LogConfigurator(modname)                                  # logger
        logConf.makeConsoleLog()

    # 2. add working directory
    modlist = []
    cwd = cwd.replace(slash1, slash2)
    for importer, modname, ispkg in pkgutil.walk_packages(path=[cwd], onerror=lambda x: None):
        if ispkg:
            modlist.append(modname)

    #  add to path and set logging system
    sys.path.append(cwd)
    for modname in modlist:
        sys.path.append(cwd + slash2 + modname)            # add to path
        logConf = LogConfigurator(modname)                 # logger
        logConf.makeConsoleLog()

    # 3. add data from 'import' in input file
    for DC in imports:
        path = DC.v('import')
        path = ' '.join(toList(path))

        # replace separators / or . by \
        path = path.replace(slash1, slash2)
        # path = path.replace('.', slash2)      # YMD 15-11-2016. I would like to allow for . in file names
        path = path.split(slash2)

        # add parent and child paths
        sys.path.append(os.path.abspath(slash2.join(path[:-1])))
        sys.path.append(os.path.abspath(slash2.join(path)))

        # make logger
        # logConf = LogConfigurator(path[-1])                                 # logger
        # logConf.makeConsoleLog()
    return
