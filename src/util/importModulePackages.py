"""
importModulePackages

Date: 15-12-15
Authors: Y.M. Dijkstra
"""
import os
import pkgutil
import sys
import packages          # DEFAULT PACKAGE LOCATION
from nifty import toList
from src.util.diagnostics import LogConfigurator
from src.util.localpath import localpath


def importModulePackages(cwd, imports):
    """import module packages from:
    (1) the standard folder 'packages',
    (2) the working directory 'cwd' and
    (3) the paths given in 'imports'
    by adding them to localpath (= temporary path variable for run-time of the session (since v2.4, before this was added to sys.path))

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

    loggingpath = []

    # 1. add working directory
    modlist = []
    cwd = cwd.replace(slash1, slash2)
    for importer, modname, ispkg in pkgutil.walk_packages(path=[cwd], onerror=lambda x: None):
        if ispkg:
            modlist.append(modname)

    #  add to path
    localpath.append(cwd)
    for modname in modlist:
        localpath.append(cwd + slash2 + modname)
        loggingpath.append(cwd + slash2 + modname)

    # 2. add data from 'import' in input file
    for DC in imports:
        path = DC.v('import')
        path = ' '.join(toList(path))

        # replace separators / or . by \
        path = path.replace(slash1, slash2)
        # path = path.replace('.', slash2)      # YMD 15-11-2016. I would like to allow for . in file names
        path = path.split(slash2)

        # add parent and child paths
        localpath.append(os.path.abspath(slash2.join(path[:-1])))
        localpath.append(os.path.abspath(slash2.join(path)))
        loggingpath.append(os.path.abspath(slash2.join(path)))

    # 3. import packages in 'packages' folder
    #   list packages
    modlist = []
    for importer, modname, ispkg in pkgutil.walk_packages(path=packages.__path__, onerror=lambda x: None):
        if ispkg:
            modlist.append(modname)

    #  add to path and set logging system
    localpath.append(packages.__path__[0])
    for modname in modlist:
        localpath.append(packages.__path__[0] + slash2 + modname)
        loggingpath.append(packages.__path__[0] + slash2 + modname)

    # add handlers to loggers in all paths
    for pth in loggingpath:
        p = pth.split('/')
        p = p[-1].split('\\')
        p = p[-1]
        # make logger
        logConf = LogConfigurator(p)                                 # logger
        logConf.makeConsoleLog()
    return
