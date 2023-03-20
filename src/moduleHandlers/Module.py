"""
Abstract class Module
- implicitly implements the <<Interface>> Runable -

Connection between the iFlow core and the actual classes implementing the content;
Module stores the input and register data of a module and initiates it. It also determines wheter a module should run and which
submodules should run. The class provides a range of query methods to request input/register data (which is stored in private
variables, to prevent changes from outside). It also provides methods for determining if the module should run, based on the
requested output variables.

Module checks whether the underlying module has a run() method. It throws a KnownError if this is not the case.
Module also throws an error if the underlying module does not return a dictionary or None-type.

Original date: 26-02-16
Updated: 03-02-22
Original authors: Y.M. Dijkstra, R.L. Brouwer
Update authors: Y.M. Dijkstra
"""
import types
from nifty.dynamicImport import dynamicImport
from nifty import toList, Timer
from src.util.diagnostics import KnownError
from copy import deepcopy
from src.DataContainer import DataContainer
import logging
from src import config as cf


class Module:
    # Variables


    # Methods
    def __init__(self, input, register):
        """Load all variables needed for the module to (private)class variables.
        Parameters:
            input - (DataContainer) input variables and results of previous modules
            register - (DataContainer) register data of this module

        Exception:
            KnownError if module cannot be instantiated
            KnownError if no run() module was found in the module
            KnownError if an iterative method has no method stopping_criterion(iteration)
        """
        self.logger = logging.getLogger(__name__)

        # Load the data from input and register to private class variables and initiate a timer for this module
        self._input = input
        self._register = register
        submodList = self._makeSubmoduleListInit()
        self.submoduleList = [submodList, submodList]   # list with two elements: first for first time in an iteration, second for subsequent times.
        self._output = DataContainer()
        self.timer = Timer()
        self.runModule = False

        return

    ####################################################################################################################
    #   Run and manipulate
    ####################################################################################################################
    def run(self, init=False, **kwargs):
        """Invoke the module's run() method or run_init() method if available and init=True

        Exception:
            KnownError exception if the output of the underlying module is not a dictionary or None-type

        Returns
            DataContainer with results of calculated module
        """
        if not hasattr(self, 'module'):
            raise KnownError('Module %s has not been instantiated. Use the method instantiateModule on the run block before running'%self.getName())

        # add submodules to run for this iteration
        self.addInputData({'submodules':self.submoduleList[(not init)]})

        # run
        self.timer.tic()
        try:
            self._output = self.module.run()
        except Exception as e:
            if cf.IGNOREEXCEPTIONS:
                self._output = {'ERROR':True}
                self.logger.error('FATAL ERROR OCCURRED.\nSIMULATION CONTINUES BECAUSE IGNOREEXCEPTIONS IS SET TO TRUE IN THE SRC/CONFIG.PY.')
                pass
            else:
                raise
        self.timer.toc()

        # make a dataContainer for the result
        if self._output is None:
            self._output = {}
        elif not isinstance(self._output, dict):
            raise KnownError('Output of module %s is invalid. Please make sure to return a dictionary.' % self.getName())

        return

    def instantiateModule(self):
        """Make and instantiate a module and check whether the input criteria are satisfied.
        """

        # find the module & run method
        moduleMain_ = dynamicImport(self._register.v('packagePath'), self._register.v('module'))
        try:
            self.module = moduleMain_(self._input)
        except Exception as e:
            # Reraise a KnownError message received from the module
            if isinstance(e, KnownError):
                raise
            # Else raise a new KnownError
            else:
                raise KnownError('Could not instantiate module %s. Please check if the init method is correct and if all variables used in the init method are also available to the module.\nDo this by checking whether all variables are provided on input and by checking the call stack.'% self._register.v('module'), str(e))

        # check if module satisfies criteria
        if not (hasattr(self.module, 'run') and isinstance(self.module.run, types.MethodType)):
            raise KnownError('Module '+self._register.v('module')+' has no working run() method')
        
        return

    def reset(self):
        try:
            del self.module
        except:
            pass
        self.instantiateModule()
        return

    def deepcopy(self, dataCopied=None):
        """Return a deepcopy of this module with input data at the moment of calling the deepcopy. The underlying module
        class still needs to be instantiated.

        See iFlowBlock.deepcopy for more details.
        """
        # deepcopy the DC self._input. Then immediately overwrite the entries that were copied before
        input = deepcopy(self._input)
        if dataCopied is not None:
            for key in self._input.getAllKeys():
                if id(self._input.v(*key)) in dataCopied[0]:
                    d = self.__buildDict(key, dataCopied[1].v(*key))
                    input.merge( d )
                else:
                    d = self.__buildDict(key, input.v(*key))
                    dataCopied[0].append(id(self._input.v(*key)))
                    dataCopied[1].merge( d )
        else:
            dataCopied = ([],input)
            for key in self._input.getAllKeys():
                dataCopied[0].append(id(self._input.v(*key)))

        moduleCopy = self.__class__(input, deepcopy(self._register))
        moduleCopy.runModule = deepcopy(self.runModule)

        return moduleCopy, dataCopied

    def __buildDict(self, key, data):
        d = {}
        if len(key)>1:
            d[key[0]] = self.__buildDict(key[1:], data)
        else:
            d[key[0]] = data
        return d

    def addInputData(self, d):
        """Append the input data by d

        Parameters:
            d - (dict or DataContainer) data to append
        """
        self._input.merge(d)
        return

    def updateSubmoduleListIteration(self, outputList):
        submodList = []
        for mod in self.submoduleList[0]:
            if self._register.v(mod,'inputInit') is not None:
                inputReq = list(set(toList(self._register.v(mod,'input')) + toList(self._register.v(mod,'inputInit'))))
                inputReq = list(set(inputReq + self._register.v('input')+ toList(self._register.v('inputInit'))))
            else:
                inputReq = toList(self._register.v(mod,'input'))
                inputReq = list(set(inputReq + toList(self._register.v('input'))))

            overlap = [i for i in outputList if i in inputReq]
            if len(overlap)>0:
                submodList.append(mod)
        self.submoduleList[1] = submodList
        return

    def printTimerResult(self, tabs=''):
        self.logger.info(self.timer.string(tabs+self.getName() + ' time elapsed: '))

    ####################################################################################################################
    #   Query methods
    ####################################################################################################################
    def getName(self, short=False):
        """Returns the name (str) of the module given by the register. Only the main package and module name are used;
        sub-package names are left-out.

        @args: short (bool, optional): if True, the name is shortened and the package name is removed.

        """
        if not short:
            path = self._register.v('packagePath').split('.')
            modName = path[0]+'.'+self._register.v('module')
            return modName
        else:
            return self._register.v('module')


    def getOutput(self):
        """Returns the module output. NB. as this variable is passed without (deep)copying, this module output may be
        overwritten in other modules. Hence, requesting the module output does not guarantee that the actual module
        output is returned. (this is done to save data use) """
        return self._output

    def getOutputVariables(self, *args, **kwargs):
        """List of output variable names (i.e. list of str) that this module will return
        given the list of submodules to run
        """
        return self._returnmoduleRequirement('output')

    def getInputRequirements(self, *args, **kwargs):
        """List of input variables (i.e. list of str) that this module requires given the current list of submodules to run.
        If init is set to true this will try to obtain the requirements for input initially (iterative modules)
        or will return the normal input requirement (non-iterative modules)

        Returns:
            List of str with required input variable names.
        """
        return self._returnmoduleRequirement('input')

    def getAvailableVariableNames(self):
        """Returns all names of input variables available to this module. This is given as a list of tuples with the
        tuple containing a hierarchy of variable names. Example: [('grid', 'axis', 'x'), (myvar, )]"""
        return self._input.getAllKeys()

    def getIteratesWith(self):
        """Returns the module that this module may iterate with (only for output module and visualisation module) or None if it does not iterate with anything (also None for other modules)."""
        return None

    def getInput(self):
        return self._input

    def getRunModule(self):
        return self.runModule

    def isIterative(self):
        """Returns boolean saying whether the module is iterative."""
        return False

    def isOutputModule(self):
        """Returns True is the underlying module is an output module, else returns False"""
        return False

    def isControllerModule(self):
        """Returns True is the underlying module is an output module, else returns False"""
        return False

    def isUnregisteredModule(self):
        """Returns True is the underlying module is an unregistered module, else returns False"""
        return False

    def getSubmodulesToRun(self, iter = 0):
        """List of submodules that will be run or empty list if no submodules exist

        Parameters:
            iter (int, optional) -  iteration number, default 0. If >0 return self.__submodulesToRunIter
        """
        if iter == 0:
            return self.submoduleList[0]
        else:
            return self.submoduleList[1]

    ####################################################################################################################
    #   Private methods
    ####################################################################################################################
    def _makeSubmoduleListInit(self):
        ''' Make a list of the submodules based on the optional keywords 'submodules' and 'excludeSubmodules'
        Use all submodules, unless the input tells something else
        '''
        if self._register.v('submodules'):
            submodReqList = toList(self._input.v('submodules'))
            submodExcludeList = toList(self._input.v('excludeSubmodules'))
            if submodReqList != [] and submodReqList != ['all']:
                sublist = [i for i in submodReqList if i in self._register.v('submodules')]
            elif submodExcludeList != [] and submodExcludeList != ['none'] and submodExcludeList != ['None']:
                sublist = [i for i in self._register.v('submodules')if i not in submodExcludeList]
            else:
                sublist = toList(self._register.v('submodules'))
        else:
            sublist = []

        return sublist

    def _returnmoduleRequirement(self, property):
        """List property 'property' of module and submodules as given in the registry.
        Searches in (1) the general module properties
         or (2) the submodule properties of the submodules in self.submoduleList.

        Parameters:
            property - (string) property to search for

        Returns:
            list of properties
        """
        reqList = []

        # 1. search in module registry data
        addData = self._register.v(property)
        if addData is not None:
            reqList = list(set(reqList + toList(addData)))

        # 2. search for the property per submodule in the submodule list
        for mod in self.submoduleList[0]:
            addData = self._register.v(mod, property)
            if addData is not None:
                reqList = list(set(reqList + toList(addData)))      # add but prevent double entries


        return reqList






