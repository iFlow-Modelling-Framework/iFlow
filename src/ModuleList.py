"""
Class Module List
Contains list of modules and provides methods for adding modules to the list, building the call stack and running the call stack
The validity of the input will be checked while building the call stack. This will check whether all variables to a
module are provided in either other modules, the input file or the config file. It will also check whether a call stack
can be built from the required methods. Known errors with intructions will be generated if the input is invalid.

Date: 26-02-16
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
import logging
from Module import Module
from src.util.diagnostics import KnownError
from nifty import toList
from config import MAXITERATIONS


class ModuleList:
    # Variables      
    moduleList = []
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self):
        return

    def addModule(self, inputData, register, output, alwaysRun = False, outputModule = False):
        """Add a module to the module list

        Parameters:
            input - (DataContainer)  input variables and results from preparatory modules
            register - (DataContainer) registry data of this module
            output - (DataContainer) with output requirements for this module
            alwaysRun - (bool, optional) Set to True to force the module to run. Default: False.
            outputModule - (bool, optional) Set to True to indicate that the underlying module is an output module. Default: False.
        """
        module = Module(inputData, register, output, alwaysRun, outputModule)
        self.moduleList.append(module, )
        return

    def buildCallStack(self):
        """Build the call stack of calculation modules. The method follows these steps
        1. update the submodules that each module should run using the method '__updateSubmodulesToRun'.
            This is done by starting from the modules necessary for providing the requested output. These modules already
            have the property 'runModule' set to True. Using a while loop, modules required for these modules also get
            the property 'runModule'=True. This also determines which submodules are required. All submodules that might
            be used are called.
        2. Check input variables. The register of all modules that are enlisted for running (i.e. runModule=True) are
           checked to verify the input. Input variables can be provided either by other modules (i.e. in 'output' tag
           in module registry), on input or in the config.py file. A KnownError Exception is raised when the input is
           invalid.
        3. Initialise. First determine the the required module for each module (both for initial run (iterative mods) and
            consecutive runs. Then prepare the lists required in step 3.
        4. place modules in the call stack. This works with a two list system: a list of unplaced modules and
            a call stack (=list of placed modules).
            Iteratively do the following check:
            a. Sort the modules (see below at *)
            b. Place the first module in the list that does not require other unplaced modules.
            c. If the placed module is iterative, determine the requirements for closing the iteration loop
        5. check if the any change occurred in the last iteration or if the unplaced list is depleted.
            If no change or depleted list, stop the iteration.
            Provide a warning (but continue the program) if not all output requirements have been met or if not
            all specified methods are used.

        Notes: the modules themselves are responsible for knowing whether to run or not and which submodules to run.
            In building the call stack, ask module for information on whether it runs and what input/output needs/provides
            based its submodules to run. i.e. ModuleList has no knowledge of the submodules.

        Exceptions:
            KnownError expception if the input is invalid. The exception contains intructions on the missing variables
            Note that no exception is thrown when the call stack is incomplete, the program will then only provide a
            warning, but continues running.
        """
        # 1. update the submodules to run based on the input requirements of all modules
        self.__updateSubmodulesToRun()

        # 2. check input requirements; have all variables been provided
        self.__checkInputRequirements()

        # 3. Init
        #   Determine which modules are required for the modules to run
        inputInitMods = self.__loadInputRequirements(init=True)
        inputMods = self.__loadInputRequirements()

        #   Make lists
        unplacedList = [i for i in self.moduleList if i.runModule]  # list with modules not placed in the call stack
        self.callStack = ([], [], [])    # sublists with 0: modules in call stack, 1: iteration number, 2: 'start' if this is the place to start an iteration (i.e. run stopping criterion)
        outputList = []  # output of all modules in the call stack
        outputReqList = []  # list of required output variables
        for module in self.moduleList:
            outputReqList = list(set(outputReqList + toList(module.getOutputRequirements())))

        #   iterative module inits
        iterativeList = [i.isIterative() for i in unplacedList]  # is the module in the unplacedList iterative or not (bool)
        iterationReqList = [[]]
        iterationNo = 0

        # 4. place modules in call stack
        while unplacedList:
            listSizeBefore = len(unplacedList)

            # a. sort modules depending
            if iterationNo == 0:
                # no iteration
                # sort the iterativeList and unplacedList so that non-iterative modules are always placed first
                iterativeList, unplacedList = (list(x) for x in zip(*sorted(zip(iterativeList, unplacedList))))
            else:
                # in an iteration
                # sort so that modules that can contribute to the iteration requirements come first.
                # if more modules can contribute, take non-iteratives first
                contributionList = [(i in iterationReqList[-1]) for i in unplacedList]
                iterativeList = [not i for i in iterativeList]      # sorting is reversed for contribution list, therefore also reverse iterativelist. Reverse back later
                contributionList, iterativeList, unplacedList = (list(x) for x in zip(*sorted(zip(contributionList, iterativeList, unplacedList), reverse=True)))
                iterativeList = [not i for i in iterativeList]

            # b. place modules that can be placed according to their input requirements
            for i, mod in enumerate(unplacedList):
                if not [j for j in inputInitMods[mod.getName()] if j not in self.callStack[0]]:       # if a module does not require initial input that is not already in outList
                    if iterativeList[i]:
                        # start a new iteration loop
                        iterationNo += 1
                        iterationReqList.append([])

                    self.callStack[0].append(mod)                                           # place in call stack
                    self.callStack[1].append(iterationNo)                                   # iteration loop this module is in; 0 means no iteration
                    self.callStack[2].append(iterativeList[i]*'start')                      # 'start' if this is the starting point of a loop
                    unplacedList.pop(i)                                                     # remove from list of unplaced modules
                    iterativeList.pop(i)                                                    # " "

                    outputList = list(set(outputList + mod.getOutputVariables()))             # add its output to the list of calculated output variables
                    iterationReqList[iterationNo] = list(set(iterationReqList[iterationNo]+ [j for j in inputMods[mod.getName()] if j not in self.callStack[0]]))  # update and append list of modules required for this iteration loop
                    for k in range(1, iterationNo+1):
                        iterationReqList[k] = [j for j in iterationReqList[k] if j not in self.callStack[0]]  # update list of variables required for all iteration loops
                    break   # break loop if a module is placed

            # c. close iteration loop if in iteration but no requirements anymore
            while iterationNo > 0 and not iterationReqList[iterationNo]:
                # add output module if iterative output is requested and it belongs to this iteration loop
                outputModule = [mod for mod in self.moduleList if mod.isOutputModule()]              # find the output module
                if outputModule:
                    outputModule = outputModule[0]
                    outputIterationModule = outputModule.outputIterationModule                              # find with which module the output iterates or None if non-iterative
                    iterStartModuleName_list = [mod.getName() for i, mod in enumerate(self.callStack[0]) if (self.callStack[2][i]=='start' and self.callStack[1][i]==iterationNo)]    # find the module(s) that started the current iteration loop
                    for iterStartModuleName in iterStartModuleName_list:
                        if outputIterationModule and outputIterationModule == iterStartModuleName:              # if output iterates with current iterative module, add output module and remove from unplaced list
                            self.callStack[0].append(outputModule)
                            self.callStack[1].append(iterationNo)
                            self.callStack[2].append('')
                            try:
                                unplacedList.remove(outputModule)
                            except Exception as e:
                                raise KnownError('Output module is supposed to iterate with module %s, which is not used.' % outputIterationModule, e)

                # set iteration level back by 1
                iterationReqList.pop(iterationNo)
                iterationNo -= 1

            listSizeAfter = len(unplacedList)

            # 5. check the progress made in the last iteration
            if listSizeBefore == listSizeAfter or not unplacedList:
                # a. there are vars required on output, but not calculated
                if any([j for j in outputReqList if j not in outputList]):
                    missingVars = [j for j in outputReqList if j not in outputList]

                    # Mention that not all variables can be calculated, but carry on anyway
                    message = ('Not all variables that are required on output can be calculated using the current input settings. '
                                '\nCould not calculate %s.'
                                '\nComputation will continue.\n' % (', '.join(missingVars)))
                    self.logger.warning(message)
                    unplacedList = []

                # b. output requirements (more than) fulfilled. Stop the building of the call stack.
                unusedlist = [i for i in self.moduleList if not i.runModule]
                if any(unusedlist):
                    # Nb. not necessarily all modules will be run, provide a warning if modules have not been placed in the call stack
                    self.logger.warning('Not all modules will be used.'
                                        '\nModule(s) ' + (', '.join([mod.getName() for mod in unusedlist])) + ' will not be run'
                                        '\nComputation will continue.\n')
                # c. output requirements (more than) fulfilled. Stop the building of the call stack.
                if any(unplacedList):
                   raise KnownError('Modules %s could not be placed in the call stack, but seem to contribute to the requested output. '
                                    'Please check the input and output requirements of your modules.' % ', '.join([mod.getName() for mod in unplacedList]))

        self.logger.info('Call stack was built successfully')

        # Print call stack to log. Includes information on loops
        self.logger.info('** Call stack **')
        for i, mod in enumerate(self.callStack[0]):
            self.logger.info(str(i+1)+'\t'+mod.getName()+ '\t' + ', '.join(toList(mod.getSubmodulesToRun())) + '\t' + 'Loop '.join(['']+[(str)(j) for j in [self.callStack[1][i]] if j > 0])+ '\t' +self.callStack[2][i])
        self.logger.info('')
        return

    def runCallStack(self):
        """First instantiate all modules in the call stack

        Then run the call stack using the recursive method '__runLoop'.
        Take two steps:
        1. Rebuild the call stack in reverse to determine blocks of modules that belong to a single loop or
        that are indidivual non-iterative modules
        2. Run the identified blocks.

        The output of each module is appended to the input of all modules.
        NB. now only supports only non-iterative call stacks
        """
        # instantiate
        for mod in self.callStack[0]:
            mod.instantiateModule()

        # run
        self.logger.info('** Run **')
        self.__runLoop(0, len(self.callStack[0])-1, 0)
        return

    def __runLoop(self, imin, imax, loopNo, **kwargs):
        """Rebuild the call stack by using recursion and then run
        All modules in a single iteration loop are grouped and such a group is then called by this method

        Parameters:
            imin (int) - index of first element in group wrt to call stack
            imax (int) - index of last element in group wrt to call stack
            loopNo (int) - level of recursion
        """
        # 1. Init
        #   build the call list
        callList = []
        end = -1
        for i in range(imax, imin-1, -1):
            if self.callStack[1][i] == loopNo:
                module = self.callStack[0][i]
                callList.append((module.run, (), module.getName()))
            else:
                end = max(end, i)
                if self.callStack[1][i]==loopNo+1 and self.callStack[2][i] == 'start':
                    callList.append((self.__runLoop, (i, end, loopNo+1)))
                    end = -1
        callList.reverse()

        # 2. run the callList as long as iteration is positive. Set to negative if stopping criterion has been met
        for iteration in range(0, MAXITERATIONS*bool(loopNo)+1):
            # check stopping criterion
            if iteration > 0:
                if self.callStack[0][imin].stopping_criterion(iteration):
                    return

            for tup in callList:
                method = tup[0]
                arguments = tup[1]
                result = method(*arguments, init=(not bool(iteration)))  # only add init=True for iteration==0

                if result is not None:
                    for module in self.callStack[0]:
                        module.addInputData(result)
        return

    def __checkInputRequirements(self):
        """Private methods checkInputRequirements.
        See buildCallStack step 2 for information.
        """
        for mod in [i for i in self.moduleList if i.runModule]:
            #  find input variables for module 'mod'
            inputList = list(set(mod.getInputRequirements()+mod.getInputRequirements(init=True)))

            # check against variables calculated by other modules
            outputList = []
            for j in [k for k in self.moduleList if k.runModule]:
                outputList+=j.getOutputVariables()
            inputList = [i for i in inputList if i not in outputList]

            # check against input variables and config file
            inputvars = [l[0] for l in mod.getAvailableVariableNames()]
            inputList = [i for i in inputList if i not in inputvars]

            # for output module, check whether the missing variables are output variables. If so, then continue without error; a warning is provided later when building the call stack
            if mod.isOutputModule():
                inputList = [i for i in inputList if i not in mod.getOutputRequirements()]

            # if there
            if inputList:
                message = ("Not all required variables are given. Missing '%s' in module '%s'.\nVariables can be calculated in other modules, given on input or specified in the iFlow config file" % (', '.join(inputList), str(mod.getName())))
                raise KnownError(message)
        return

    def __loadInputRequirements(self, init=False):
        """Make a dictionary containing 'module name': 'list of required modules' pairs.
        The required variables are determined on the basis of the modules information on whether to run and which submodules to run.

        Parameters:
            init - (bool, optional) take modules required for initial run in case of iterative methods.

        Return:
            dictionary with 'module name': 'list of required module(s)' pairs.
        """
        d = {}
        for mod in [i for i in self.moduleList if i.runModule]:
            inputReq = mod.getInputRequirements(init=init)
            # Add module to list of input requirements if it provides data for 'mod'
            d[mod.getName()] = [inmod for inmod in self.moduleList if ([i for i in inputReq if i in inmod.getOutputVariables()] and inmod!=mod)]
        return d

    def __updateSubmodulesToRun(self):
        """Update the submodules to run in each module. Add all submodules that provide output that is used as input for other submodules.
        This method takes an iterative three step process. Iteratively:
            a. make the list of input required by the current modules/submodules
            b. for each module, add submodules that provide output variables that are already in the list of input variables
            c. Stop iterating if the input list has not changed
        """
        inputList_before = []
        inputList_after = []
        changed = True

        while changed:
            # a.
            for mod in self.moduleList:
                if mod.runModule:
                    inputList_after = list(set(inputList_after + mod.getInputRequirements()))

            # b.
            for mod in self.moduleList:
                mod.updateToRun(inputList_after)

            # c.
            if not any([j for j in inputList_after if j not in inputList_before]):
                # i.e. if no change occurred
                changed = False
            else:
                inputList_before = inputList_after

        return
