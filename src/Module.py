"""
Class Module
Interface for modules; interfaces the ModuleList and actual modules.
Module stores the input and register data of a module, initiates. It also determines wheter a module should run and which
submodules should run. The class provides a range of query methods to request input/register data (which is stored in private
variables, to prevent changes from outside). It also provides methods for determining if the module should run, based on the
requested output variables.

Module checks whether the underlying module has a run() method and, in case of an iterative module, run_init()
and stopping_criterion() methods. It throws a KnownError if this is not the case. Module also throws an error if the
underlying module does not return a dictionary or None-type. Id when the stopping criterion of iterative modules do not
return a boolean.

Date: 26-02-16
Authors: Y.M. Dijkstra, R.L. Brouwer
Update  28-12-2016: submodules to run per module, not per variable

"""
import types
from nifty.dynamicImport import dynamicImport
from nifty import toList, Timer
from src.util.diagnostics import KnownError
from . import DataContainer
import inspect
import logging
import src.config as cf


class Module:
    # Variables

    # Methods
    def __init__(self, input, register, outputReq):
        """Load all variables needed for the module to (private)class variables.
        Parameters:
            input - (DataContainer) input variables and results of previous modules
            register - (DataContainer) register data of this module
            outputReq - (DataContainer) output requirements (from input file)

        Exception:
            KnownError if module cannot be instantiated
            KnownError if no run() module was found in the module
            KnownError if an iterative method has no method stopping_criterion(iteration)
        """
        # check if module is an output module
        if register.v('outputModule') == 'True':
            self.__isOutputModule = True
            self.iteratesWith = input.v('iteratesWith')
            alwaysRun = True
        else:
            self.__isOutputModule = False
            self.iteratesWith = None
            alwaysRun = False

        # check if the module is a visualisation module. If so, set alwaysRun to True
        if register.v('visualisationModule') == 'True':
            self.iteratesWith = input.v('iteratesWith')
            alwaysRun = True

        # Load the data from input and register to private class variables and initiate a timer for this module
        self.__input = input
        self.__register = register
        self.__outputReq = outputReq
        self.__toRun(alwaysRun)
        self.timer = Timer()

        # set if this module is iterative; it is when itself or any of its submodules is iterative
        self.iterative = False
        if self.__register.v('iterative') == 'True':
            self.iterative = True
        for submod in self.__submodulesToRunInit:
            if self.__register.v(submod, 'iterative') == 'True':
                self.iterative = True
        return

    def instantiateModule(self):
        """Make and instantiate a module and check whether the input criteria are satisfied.
        """
        # find the module & run method
        moduleMain_ = dynamicImport(self.__register.v('packagePath'), self.__register.v('module'))
        try:
            if len(inspect.getargspec(moduleMain_.__init__)[0]) == 2:
                self.module = moduleMain_(self.__input)                             # VERSION 2.4. ONLY ONE ARGUMENT, SUBMODULES NOW IN DATACONTAINER 'INPUT' (SUPPORTS DYNAMIC SUBMODULE LIST IN ITERATIONS)
            else:
                self.module = moduleMain_(self.__input, self.submodulesToRun)       # VERSION 2.3. SUBMODULES AS EXTRA ARGUMENT. WILL BECOME OBSOLETE [dep02]
                self.logger = logging.getLogger(__name__)
                self.logger.warning('Module ' + self.getName() + ' still takes 3 arguments upon initialisation.\nAs of v2.4 it should take 2 arguments. SubmodulesToRun is now found in the DataContainer keyword "submodules".')
        # except Exception as e:
        #      if isinstance(e, )
        except Exception as e:
            # Reraise a KnownError message received from the module
            if isinstance(e, KnownError):
                raise
            # Else raise a new KnownError
            else:
                raise KnownError('Could not instantiate module %s. Please check if the init method is correct and if all variables used in the init method are also available to the module.\nDo this by checking whether all variables are provided on input and by checking the call stack.'% self.__register.v('module'), str(e))

        if not (hasattr(self.module, 'run') and isinstance(self.module.run, types.MethodType)):
            raise KnownError('Module '+self.__register.v('module')+' has no working run() method')
        if self.isIterative():
            if not (hasattr(self.module, 'stopping_criterion') and isinstance(self.module.run, types.MethodType)):
                raise KnownError('Module '+self.__register.v('module')+' has no working stopping_criterion() method')

        return

    #   Query methods
    def getName(self):
        """Returns the name (str) of the module given by the register, where only the main package and module
        name are used; sub-package names are left-out."""
        modName = self.__register.v('packagePath')+'.'+self.__register.v('module')
        return modName

    def getSubmodulesToRun(self, iter = 0):
        """List of submodules that will be run or empty list if no submodules exist

        Parameters:
            iter (int, optional) -  iteration number, default 0. If >0 return self.__submodulesToRunIter
        """
        if iter == 0:
            return self.__submodulesToRunInit
        else:
            return self.__submodulesToRunIter

    def getOutputVariables(self, iter = 0):
        """List of output variable names (i.e. list of str) that this module will return
        given the current list of submodules to run

        Parameters:
            iter (int, optional) -  iteration number, default 0. If >0 return self.__submodulesToRunIter
        """
        return self.__returnSubmoduleRequirement('output', self.getSubmodulesToRun(iter))

    def getOutputRequirements(self):
        """Returns a list of output variables (i.e. list of str) required on output according to the input file under tag
        'requirements' in module output"""
        return toList(self.__outputReq.v('requirements'))

    def getInputRequirements(self, init=False):
        """List of input variables (i.e. list of str) that this module requires given the current list of submodules to run.
        If init is set to true this will try to obtain the requirements for input initially (iterative modules)
        or will return the normal input requirement (non-iterative modules)

        Parameters:
            init - (bool, optional) In case of iterative module, take initial input requirements. Does not affect
                    non-iterative modules. Default = False.

        Returns:
            List of str with required input variable names.
        """
        if init and self.isIterative():
            return self.__returnSubmoduleRequirement('inputInit', self.submodulesToRun)
        elif self.isIterative():
            return list(set(self.__returnSubmoduleRequirement('inputInit', self.submodulesToRun)+self.__returnSubmoduleRequirement('input', self.submodulesToRun)))
        else:
            return self.__returnSubmoduleRequirement('input', self.submodulesToRun)

    def getAvailableVariableNames(self):
        """Returns all names of input variables available to this module. This is given as a list of tuples with the
        tuple containing a hierarchy of variable names. Example: [('grid', 'axis', 'x'), (myvar, )]"""
        return self.__input.getAllKeys()

    def getIteratesWith(self):
        """Returns the module that this module may iterate with (only for output module and visualisation module) or None if it does not iterate with anything (also None for other modules)."""
        return self.iteratesWith

    def isIterative(self):
        """Returns boolean saying whether the module is iterative.
        A module is iterative when any of its submodules to run is iterative."""
        return self.iterative

    def isOutputModule(self):
        """Returns True is the underlying module is an output module, else returns False"""
        return self.__isOutputModule

    def isIteratorModule(self):
        """Returns if the module is an iterator module."""
        if self.__register.v('iteratorModule') == 'True':
            return True
        else:
            return False

    #   Other public methods
    def setSubmoduleRunList(self, iter):
        self.submodulesToRun = self.getSubmodulesToRun(iter)

    def getInput(self):
        return self.__input

    def addInputData(self, d):
        """Append the input data by d

        Parameters:
            d - (dict or DataContainer) data to append
        """
        self.__input.merge(d)
        return

    def run(self, init=False):
        """Invoke the module's run() method or run_init() method if available and init=True

        Parameters:
            init (bool, optional) - if True, invoke the run_init method

        Exception:
            KnownError exception if the output of the underlying module is not a dictionary of None-type

        Returns
            DataContainer with results of calculated module
        """
        self.timer.tic()
        try:
            if init and self.isIterative():         #21-07-2017 YMD correction: check if iterative, not if run_init exists
                result = self.module.run_init()
            else:
                result = self.module.run()
        except Exception as e:
            if cf.IGNOREEXCEPTIONS:
                result = {'ERROR':True}
                self.logger.error('FATAL ERROR OCCURRED.\nSIMULATION CONTINUES BECAUSE IGNOREEXCEPTIONS IS SET TO TRUE IN THE SRC/CONFIG.PY.')
                pass
            else:
                raise

        # make a dataContainer for the result
        # try:                                          ## YMD 10-12-2019 not necessary to convert to DC. Keep as dict.
        #     self.result = DataContainer.DataContainer(result)
        # except:
        #     raise KnownError('Output of module %s is invalid. Please make sure to return a dictionary.' % self.getName())
        if not isinstance(result, dict):
            raise KnownError('Output of module %s is invalid. Please make sure to return a dictionary.' % self.getName())
        self.timer.toc()
        return result

    def stopping_criterion(self, iteration):
        """Invoke method 'stopping_criterion' of the underlying module. Pass iteration number down to the module.

        Parameters:
            iteration - (int) number of the current iteration.

        Exception:
            KnownError exception if underlying method does not return a boolean.

         Returns:
            bool whether to stop (=True) or continue (=False)
        """
        stop = self.module.stopping_criterion(iteration)
        if not isinstance(stop, bool):
            raise KnownError('Stopping critertion of module %s is invalid. Please make sure to return a boolean.'% self.getName())
        return stop

    def updateToRun(self, inputList):
        """Update if this module should be run and which submodules should be run.
        The update is based on whether this module provides output variables that are present in argument inputList
        Note:
             All submodules will be run if the the general module output (i.e. not for submodules) is required.
             However, if the submodules to run are specified on input, this will not be overruled.

        Parameters:
            inputList - (list) list of variables
        """
        # check if general output by module is required for other modules
        varlist = [i for i in toList(self.__register.v('output')) if i in inputList] # if there is any variable in the output that is also given in inputList, run the module
        if len(varlist)>0:
            self.runModule = True
            self.__submodulesToRunInit = self.__submoduleList()

        # additionally, check if individual submodules contribute to required output
        sublist = self.__submoduleList()
        for i in sublist:
            varlist = [j for j in toList(self.__register.v(i, 'output')) if j in inputList]
            if len(varlist)>0:
                self.runModule = True
                self.__submodulesToRunInit = list(set(self.__submodulesToRunInit+toList(i)))
        self.submodulesToRun = self.__submodulesToRunInit
        return

    def updateToRunIter(self, inputList):
        """Update submodules to run in iterations after the initialisation.
        The update is based on the input variables updated in the iteration and the dependencies of the submodules

        Parameters:
            inputList - (list) list of variables
        """
        self.__submodulesToRunIter = []

        # check if general input by module is changed in the loop
        varlist = [i for i in toList(self.__register.v('input')) if i in inputList]
        if len(varlist) > 0:
            self.__submodulesToRunIter = self.__submoduleList()

        # additionally, check if individual submodules contribute to required output
        sublist = self.__submoduleList()
        for i in sublist:
            varlist = [j for j in toList(self.__register.v(i, 'input')) if j in inputList]
            if len(varlist)>0:
                self.__submodulesToRunIter = list(set(self.__submodulesToRunIter+toList(i)))
        return

    #   Private methods
    def __toRun(self, alwaysRun=False):
        """Check if this module should be run based on the output requirements.
        If so, also set the submodules that should be run (if there are any submodules)

        Notes:
        This is not the final verdict on whether this module will run and what list of submodules will be run.
        An update will be done from the module list

       Sets the class variables
            runModule - (bool) should this module run
            _submodulesToRun - (list) list of submodules to run or an empty list
        """
        self.__submodulesToRunInit = []
        self.runModule = alwaysRun  # default setting: False, except when it is forced to be run from outside

        # check if the module has submodules
        if self.__register.v('submodules'):
            sublist = self.__submoduleList()

            # loop over variables in the output requirements
            for var in toList(self.__outputReq.v('requirements')):
                submoduleList = []

                # gather the submodules that have 'var' as output
                for submod in sublist:
                    outputList = toList(self.__register.v(submod, 'output'))
                    if var in outputList:
                        submoduleList.append(submod)

                self.__submodulesToRunInit = list(set(self.__submodulesToRunInit + submoduleList))
            if self.__submodulesToRunInit:
                self.runModule = True        # if any submodules should be run, then so does the module

        # check general module output
        #   loop over variables in the output requirements
        for var in toList(self.__outputReq.v('requirements')):
            outputList = toList(self.__register.v('output'))
            if var in outputList:
                self.runModule = True

        # set submodules to run
        self.submodulesToRun = self.__submodulesToRunInit
        return

    def __returnSubmoduleRequirement(self, property, submoduleList, toRun=True):
        """List property 'property' of submodules as given in the registry.
        Searches in (1) the general module properties
         or (2) the submodule properties of the submodules in submodulelist.

        Parameters:
            property - (string) property to search for
            submoduleList - (list) list of submodules
            toRun - (bool, optional) only take properties of submodules that are on the run list
                    (or of the whole module if it is on the run list). Default: True

        Returns:
            list of properties
        """
        reqList = []

        # 1. search in module registry data
        addData = self.__register.v(property)
        if addData is not None:
            reqList = list(set(reqList + toList(addData)))

        # 2. search for the property per submodule in the submodule list
        for mod in submoduleList:
            if not toRun or mod in self.__submodulesToRunInit:
                addData = self.__register.v(mod, property)
                if addData is not None:
                    reqList = list(set(reqList + toList(addData)))      # add but prevent double entries


        return reqList

    def __submoduleList(self):
        ''' Make a list of the submodules based on the optional keywords 'submodules' and 'excludeSubmodules'
        Use all submodules, unless the input tells something else
        '''
        if self.__register.v('submodules'):
            submodReqList = toList(self.__input.v('submodules'))
            submodExcludeList = toList(self.__input.v('excludeSubmodules'))
            if submodReqList != [] and submodReqList != ['all']:
                sublist = [i for i in submodReqList if i in self.__register.v('submodules')]
            elif submodExcludeList != [] and submodExcludeList != ['none'] and submodExcludeList != ['None']:
                sublist = [i for i in self.__register.v('submodules')if i not in submodExcludeList]
            else:
                sublist = toList(self.__register.v('submodules'))
        else:
            sublist = []

        return sublist

    def updateIteratorRegistry(self, moduleList):
        """Update the register of the special iterator module (IteratorModule in register).
        This has an input variable modules that it controls and it takes its registry data"""
        if self.isIteratorModule():
            modules = toList(self.__input.v('modules'))
            modules = [i for j in modules for i in moduleList if (i.getName().split('.')[0]+'.'+i.getName().split('.')[-1] == j)]
            inputInit = toList(self.__register.v('inputInit'))
            input = toList(self.__register.v('input'))
            output = toList(self.__register.v('output'))
            run = []
            for module in modules:
                # get registry data of the dependent modules
                inputInit += toList(module.getInputRequirements(True))
                input += toList(module.getInputRequirements())
                output += toList(module.getOutputVariables())

                # remove the dependent module from the module list
                moduleList.remove(module)

            # compile final list of register data
            self.__register.addData('inputInit', list(set(inputInit)))
            self.__register.addData('input', list(set(input)))
            self.__register.addData('output', list(set(output)))
            self.__toRun()

            # append input data with the module object references, instead of the module names
            self.addInputData({'modules': modules})

        return moduleList




