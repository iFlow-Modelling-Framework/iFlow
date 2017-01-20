"""
Class RegistryChecker
Read entries in the module registry and refactor placeholders (@) in the registry.
This class also makes the first check on whether a module package exists.

Checks whether:
- the module package can be found
- the registry file exists
- the registry entry of the module can be found and is unique
Throws a KnownError exception with intructions whenever these conditions are not met

Date: 03-11-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import sys
import Reader
from src.util.diagnostics import KnownError
from nifty import toList
from src.util.localpath import localpath


class RegistryChecker:
    # Variables      

    # Methods
    def __init__(self):
        """
        """
        self.__reader = Reader.Reader()
        return

    def readRegistryEntry(self, moduleName):
        """Read the registry entry corresponding to the moduleName
        Check if this registry entry is valid for using dynamic import
        Set variable packagePath using the modulePackage extended by a packagePath

        Parameters:
            modulePackage (str) - absolute path to the module package
            moduleName (str) - name of the module without any package information

        Returns:
            DataContainer containing the register data appended by a packagePath key with the absolute path.
        """
        # split module name into package and registry entry name
        if sys.platform[:3] == 'win':
            slash1 = '/'
            slash2 = '\\'
        else:
            slash1 = '\\'
            slash2 = '/'
        if len(moduleName.split('.')) < 2:
            raise KnownError('"'+moduleName + '" is not a valid module entry. Please specify a module as: module [package].[module name]')
        modulePackage = moduleName.split('.')[:-1]
        moduleName = moduleName.split('.')[-1]

        # make registry path and open registry
        registryPath = None
        for path in localpath:     # scan the localpath (since v2.4, before this contained the reverse system path)
            path.replace(slash1, slash2)
            pathList = path.split(slash2)
            if all([pathList[-n] == modulePackage[-n] for n in range(1, len(modulePackage)+1)]):
                registryPath = path + slash2 +'registry.reg'
                break
        if registryPath is None:
            raise KnownError("Module package %s could not be found."
                             "\nCheck if the package is imported correctly using 'import [path]/%s' in the input file." % ('.'.join(modulePackage), '/'.join(modulePackage)) )

        self.__reader.open(registryPath)

        # read registry data
        registerData = self.__reader.read('module', moduleName)
        if len(registerData) > 1:
            raise KnownError("Found multiple registry entries for module '"+moduleName+"'.")
        elif len(registerData) < 1:
            raise KnownError("Could not find registry entry of module '"+moduleName+"'.")
        registerData = registerData[0]

        # take the package path from the registry
        packagePath = registerData.v('packagePath')
        if packagePath is not None:
            packagePath = packagePath.replace('/', '.')
            packagePath = packagePath.replace('\\', '.')
            if packagePath[-1] == '.':                     # remove possible trailing .
                packagePath = packagePath[:-1]
            if packagePath[0] != '.':                      # add starting .
                packagePath = '.'+packagePath
        else:
            packagePath=''

        # update packagePath
        packagePath = '\\'.join(modulePackage)+packagePath
        registerData.addData('packagePath', packagePath)

        # Transfer general data on input/output to submodules if there are any
        d={}
        if registerData.v('submodules'):
            for submod in registerData.v('submodules'):
                d[submod]={}
                d[submod]['input']  = list(set(toList(registerData.v('input')  or []) + toList(registerData.v(submod, 'input')  or [])))
                d[submod]['output'] = list(set(toList(registerData.v('output') or []) + toList(registerData.v(submod, 'output') or [])))
        registerData.merge(d)

        self.__reader.close()
        return registerData

    def refactorRegistry(self, data, reg, **kwargs):
        """Check for any @ symbols in registry after keys 'input', 'output', denoting that input data should be filled in here
        The placeholder @ can be used in two ways:
        1. @varname. @varname will be replaced by the value of variable 'varname'. This variable should be given in argument 'data'.
        2. @name1.varname. Idem, but with varname in kwargs[name1].

        Parameters:
            data (DataContainer) - Input data from input file
            reg (DataContainer) - entry from registry
            kwargs (dict of DataContainers, optional) - additional source to search for replacing @ symbols using method 2. above
        """
        for tag in ['input', 'output', 'inputInit']:
            # tags for whole module
            inputList = toList(reg.v(tag))
            inputList = self.__refactorUtil(inputList, data, **kwargs)
            reg.addData(tag, inputList)

            # tags for submodule
            submodules = reg.v('submodules')
            if submodules:
                for mod in submodules:
                    inputList = toList(reg.v(mod, tag))
                    inputList = self.__refactorUtil(inputList, data, **kwargs)
                    reg.merge({mod:{tag: inputList}})
        return

    def __refactorUtil(self, inputList, data, **kwargs):
        """Replace tags @{}, +{} and if{}. Do this recursively by removing an item from the inputList, replacing one occurrence
         of @{} and +{} and putting the result at the back of inputList.
        """
        i = 0
        while i<len(inputList):
            item = inputList[i]
            if item.find('@')>=0 or item.find('+{')>=0 or item.find('if{')>=0:
                inputList.pop(i)
                if item.find('@')>=0 and item.find('@{')<0:
                    if '.' in item:
                        dict = item[1:].split('.')[0]
                        key =  item[1:].split('.')[-1]
                        item = toList(kwargs[dict].v(key))
                    else:
                        item = toList(data.v(item[1:]))
                    inputList = inputList + item

                elif item.find('@{')>=0:
                    start = item.find('@{')
                    end = item.find('}')
                    if '.' in item[start+2:end]:
                        dict = item[start+2:end].split('.')[0]
                        key =  item[start+2:end].split('.')[-1]
                        item = [item[:start]+j+item[end+1:] for j in toList(kwargs[dict].v(key))]
                    else:
                        item = [item[:start]+str(j)+item[end+1:] for j in toList(data.v(item[start+2:end]), forceNone=True)]    #20-01-2017 added forceNone=True
                    inputList = inputList + item

                elif item.find('+{')>=0:
                    start = item.find('+{')
                    end = item.find('}')
                    setrange = range(*eval(item[start+2:end]))
                    item = item[:start]+'%s'+item[end+1:]
                    for k in setrange:
                        inputList = inputList + toList(item % k)

                elif item.find('if{')>=0:                   # added YMD 02-11-2016
                    start = item.find('if{')
                    end = item.find('}')
                    item = item[start+3:end]

                    item = item.split(',')
                    if len(item)!=2:
                        raise KnownError('error in registry in if{}. Does not have two comma-separated arguments. Make sure there are no spaces in the if-statement')
                    if eval(item[1]):
                        inputList = inputList + toList(item[0])

            else:
                i += 1
        return list(set(inputList))