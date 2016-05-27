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
"""
import types
from nifty.dynamicImport import dynamicImport
from nifty import toList
from src.util.diagnostics import KnownError
import DataContainer


class Module:
    # Variables      
    
    # Methods
    def __init__(self, input, register, outputReq, alwaysRun = False, outputModule = False):
        """Load all variables needed for the module to (private)class variables.
        Parameters:
            input - (DataContainer) input variables and results of previous modules
            register - (DataContainer) register data of this module
            outputReq - (DataContainer) output requirements (from input file)
            alwaysRun - (bool, optional) Set to True to force the module to run. Default: False.
            outputModule - (bool, optional) Set to True to indicate that the underlying module is an output module. Default: False.

        Exception:
            KnownError if module cannot be instantiated
            KnownError if no run() module was found in the module
            KnownError if an iterative method has no method stopping_criterion(iteration)
        """
        # check if module is an output module
        if outputModule:
            self.__isOutputModule = True
            alwaysRun = True
            self.outputIterationModule = input.v('iteratesWith')
        else:
            self.__isOutputModule = False

        self.__input = input
        self.__register = register
        self.__outputReq = outputReq
        self.__toRun(alwaysRun)
        return

    def instantiateModule(self):
        """Make and instantiate a module and check whether the input criteria are satisfied.
        """
        # find the module & run method
        moduleMain_ = dynamicImport(self.__register.v('packagePath'), self.__register.v('module'))
        try:
            self.module = moduleMain_(self.__input, self.__submodulesToRun)
        except Exception as e:
            # Reraise a KnownError message received from the module
            if isinstance(e, KnownError):
                raise
            # Else raise a new KnownError
            else:
                raise KnownError('Could not instantiate module %s. Please check if the init method is correct and if all variables used in the init method are also available to the module.\nDo this by checking whether all variables are provided on input and by checking the call stack.'% self.__register.v('module'), e)

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

    def getSubmodulesToRun(self):
        """List of submodules that will be run or empty list if no submodules exist"""
        return self.__submodulesToRun

    def getOutputVariables(self):
        """List of output variable names (i.e. list of str) that this module will return
        given the current list of submodules to run"""
        return self.__returnSubmoduleRequirement('output', self.__submodulesToRun)

    def getOutputRequirements(self):
        """Returns a list of output variables (i.e. list of str) required on output according to the input file under tag
        'variables' in module output"""
        return toList(self.__outputReq.v('variables'))

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
            return self.__returnSubmoduleRequirement('inputInit', self.__submodulesToRun)
        else:
            return self.__returnSubmoduleRequirement('input', self.__submodulesToRun)

    def getAvailableVariableNames(self):
        """Returns all names of input variables available to this module. This is given as a list of tuples with the
        tuple containing a hierarchy of variable names. Example: [('grid', 'axis', 'x'), (myvar, )]"""
        return self.__input.getAllKeys()

    def isIterative(self):
        """Returns boolean saying whether the module is iterative.
        A module is iterative when any of its submodules to run is iterative."""
        iterative = False
        if self.__register.v('iterative') == 'True':
            iterative = True
        for submod in self.__submodulesToRun:
            if self.__register.v(submod, 'iterative') == 'True':
                iterative = True
        return iterative

    def isOutputModule(self):
        """Returns True is the underlying module is an output module, else returns False"""
        return self.__isOutputModule

    #   Other public methods
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
        if init and hasattr(self.module, 'run_init') and callable(self.module.run_init):
            result = self.module.run_init()
        else:
            result = self.module.run()

        # make a dataContainer for the result
        try:
            self.result = DataContainer.DataContainer(result)
        except:
            raise KnownError('Output of module %s is invalid. Please make sure to return a dictionary.' % self.getName())

        return self.result

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
        for var in varlist:
            self.runModule = True
            if 'all' in toList(self.__outputReq.v('submodules', var)) or not toList(self.__outputReq.v('submodules', var)):
                self.__submodulesToRun = self.__register.v('submodules') or []      # add all submodules to runlist (or set [] if there are no submodules)
            else:
                self.__submodulesToRun = list(set(self.__submodulesToRun+toList(self.__outputReq.v('submodules', var))))

        # additionally, check if individual submodules contribute to required output
        if self.__register.v('submodules'):
            for i in self.__register.v('submodules'):
                varlist = [j for j in toList(self.__register.v(i, 'output')) if j in inputList]
                for var in varlist:
                    self.runModule = True
                    subreq = toList(self.__outputReq.v('submodules', var))
                    if 'all' in subreq or not subreq or i in  subreq:
                        self.__submodulesToRun = list(set(self.__submodulesToRun+toList(i)))

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
        self.__submodulesToRun = []
        self.runModule = alwaysRun  # default setting: False, except when it is forced to be run from outside

        # check if the module has submodules
        if self.__register.v('submodules'):
            # loop over variables in the output requirements
            for var in toList(self.__outputReq.v('variables')):
                submoduleList = []

                # gather the submodules that have 'var' as output
                for submod in self.__register.v('submodules'):
                    outputList = toList(self.__register.v(submod, 'output'))
                    if var in outputList:
                        submoduleList.append(submod)

                # check if output has a filter on submodules
                submodReqList = toList(self.__outputReq.v('submodules', var))

                if submodReqList:
                    if submodReqList[0] != 'all':
                        # if a filter is provided and does not say 'all'
                        submoduleList = [j for j in submoduleList if j in submodReqList]

                self.__submodulesToRun = list(set(self.__submodulesToRun + submoduleList))
            if self.__submodulesToRun:
                self.runModule = True        # if any submodules should be run, then so does the module

        else:
            # loop over variables in the output requirements
            for var in toList(self.__outputReq.v('variables')):
                outputList = toList(self.__register.v('output'))
                if var in outputList:
                    self.runModule = True
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
            if not toRun or mod in self.__submodulesToRun:
                addData = self.__register.v(mod, property)
                if addData is not None:
                    reqList = list(set(reqList + toList(addData)))      # add but prevent double entries


        return reqList
