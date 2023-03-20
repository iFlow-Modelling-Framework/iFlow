"""
Class iFlowBuilder
Directs the set-up of an iFlow program given an input file.
The input file path should be provided when instantiating. It will then direct reading the input and collecting
the requested modules.

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
# load matplotlib backend suited to the STeP plotting routine. Check if backend can be loaded to prevent matplotlib's lengthy warning message.
import sys
import matplotlib as mpl
if 'matplotlib.pyplot' not in sys.modules and 'matplotlib.pylab' not in sys.modules:
    from matplotlib import use
    use('TkAgg')
elif mpl.get_backend != 'TkAgg':
    print('Warning: iFlow could not load its preferred backend. iFlow\'s own plotting routines may not behave as expected. Probably your IDE (e.g. Spyder) loads its own backend. Please consider disabling the backend of your IDE.')

import logging
from src import config
from copy import copy,deepcopy
from nifty import toList

from src.util.diagnostics import KnownError
from src.util import importModulePackages

from .DataContainer import DataContainer
from .RegistryChecker import RegistryChecker
from .Reader import Reader
from .moduleHandlers.Module import Module

from src.moduleHandlers.StandardModule import StandardModule
from src.moduleHandlers.IterativeModule import IterativeModule
from src.moduleHandlers.OutputModule import OutputModule
from src.moduleHandlers.ControllerModule import ControllerModule
from src.moduleHandlers.PlotModule import PlotModule
from src.blockHandlers.StandardBlock import StandardBlock
from src.blockHandlers.IterativeBlock import IterativeBlock
from src.moduleHandlers.UnregisteredModule import UnregisteredModule


class iFlowBuilder:
    # Variables
    logger = logging.getLogger(__name__)


    # Methods
    def __init__(self, cwd, inputFilePath, additionalInputData=None, memProfiler=None):
        """create utilities required by the program,
        i.e. a registry checker, reader and module list
        """
        self.moduleList = []
        self.cwd = cwd
        self.inputFilePath = inputFilePath
        self.additionalInputData = additionalInputData

        self._registryChecker = RegistryChecker()
        self._inputReader = Reader()

        self._memProfiler = memProfiler

        # import step as st
        # st.configure()
        return

    ####################################################################################################################
    ## Public methods
    ####################################################################################################################
    def makeCallStack(self):
        """
        """
        self.__loadModules()                        # Read input and collect modules in moduleList

        self.__setModulesToRun()                    # Determine which modules are needed to satisfy the output requirements and requirements of other modules
        self.__checkInputRequirements()             # Check input requirements; have all variables been provided

        iFlowBlock = self.__buildCallStack(self.moduleList)                     # Build the call stack
        if self._memProfiler is not None:
            self._memProfiler.snapshot('Initialise iFlow')
        return iFlowBlock


    ####################################################################################################################
    ## Private methods 1: read data and collect and check modules
    ####################################################################################################################
    def __loadModules(self):
        """
        Read input file and initiate the requested modules.
        """
        # 1 # open input file
        self._inputReader.open(self.inputFilePath)

        # 2 # load imports
        imports = self._inputReader.readline('import')
        importModulePackages(self.cwd, imports)

        # 3 # load output requirements
        outputDataBlocks = self._inputReader.read('requirements') # returns a list of DataContainers for each 'requirements' statement
        outputReq = DataContainer()    # empty if no 'requirements' were found
        for dc in outputDataBlocks:
            outputReq.merge(dc)        # merge multiple 'requirements'
        self.outputReqList = toList(outputReq.v('requirements'))

        # 4 # read data from config and input file (NB. input file data may overwrite config data)
        configvars = [var for var in dir(config) if not var.startswith('__')]
        d = {}
        for var in configvars:
            d[var] = eval('config.'+var)
        d['CWD'] = self.cwd
        configData = DataContainer(d)

        # 5 # Read registered modules
        keyword = ['module', 'plot']
        for key in keyword:
            inputData = self._inputReader.read(key, stop=keyword)
            for dataContainer in inputData:
                # for each tag 'module' in input:
                #   iterate over all module types specified
                moduleList = toList(dataContainer.v(key))

                for moduleName in moduleList:
                    # make a new copy of the data container so that each module has a unique data container
                    data = DataContainer()
                    data.merge(configData)
                    data.merge(dataContainer)       # input data may overwrite config data
                    data.addData('module', moduleName)

                    # load their registry
                    if key == 'module':
                        registerData = self._registryChecker.readRegistryEntry(moduleName)
                        self._registryChecker.refactorRegistry(data, registerData, output=outputReq)    # Registry records containing placeholder '@' will be refactored here before instantiating a module.

                        # Add to data for output modules
                        if registerData.v('outputModule') == 'True':
                            data.addData('inputFile', self.inputFilePath)       # output needs the input file, add this to its data
                            data.merge(outputReq)                               # add output requirements to the input for the ouput module

                    else:
                        registerData = None

                    # add additional input data (if available) given on iFlow start-up
                    if self.additionalInputData is not None:
                        data.merge(self.additionalInputData)

                    # Determine the type of module based on the registerData
                    if registerData is None:
                        module = UnregisteredModule(data)
                    elif registerData.v('iterative') == 'True':
                        module = IterativeModule(data, registerData)
                    elif registerData.v('controller') == 'True':
                        module = ControllerModule(data, registerData)
                    elif registerData.v('visualisationModule') == 'True':
                        module = PlotModule(data, registerData)
                    elif registerData.v('outputModule') == 'True':
                        module = OutputModule(data, registerData)
                    else:
                        module = StandardModule(data, registerData)
            
                    # add the module
                    self.moduleList.append(module)

        # 6 # load modules iterating together
        iteratorBlocks = self._inputReader.read('iterator')
        for iter in iteratorBlocks:
            iterator_mods = [i for i in self.moduleList if (i.getName() in iter.v('iterator') and i.isIterative())]
            for i in iterator_mods:
                i.registerIterator([j for j in iterator_mods if j!=i])

        # 7 # close file
        self._inputReader.close()

        return

    def __setModulesToRun(self):
        """Update the modules to run. Add all modules that provide output that is used as input for other modules
        recursively to satisfy the output criteria and criteria to run essential modules (output/plotting/unregistered).

        This method takes an iterative three step process. Iteratively:
            a. make the list of input required by the current modules
            b. for each module, check if it contributes to the required input
            c. Stop iterating if the input list has not changed
        """
        inputList_before = []
        inputList_after = self.outputReqList        # create list of required variables for output

        changed = True
        while changed:
            # a.
            for mod in [i for i in self.moduleList if i.getRunModule()]:
                inputList_after = list(set(inputList_after + mod.getInputRequirements(includeControls=True, init=False)))

            # b.
            for mod in self.moduleList:
                if len(list(set(mod.getOutputVariables(includePrepare=True)) & set(inputList_after)))>0:
                    mod.runModule = True    # turn the module on

            # c.
            if not any([j for j in inputList_after if j not in inputList_before]):
                # i.e. if no change occurred
                changed = False
            else:
                inputList_before = inputList_after

        return

    def __checkInputRequirements(self):
        """Check input variables. The register of all modules that are enlisted for running (i.e. runModule==True) are
           checked to verify the input. Input variables can be provided either by other modules (i.e. in 'output' tag
           in module registry), on input or in the config.py file.

           A KnownError Exception is raised when the input is invalid.
        """
        moduleRunList = [i for i in self.moduleList if i.getRunModule()]
        for mod in moduleRunList:
            #  find input variables for module 'mod'
            inputList = list(set(mod.getInputRequirements()))

            # check against variables calculated by other modules
            outputList = []
            for j in moduleRunList:
                outputList+=j.getOutputVariables(includePrepare=True)           # This check includes the 'prepare' of a controller. These prepared variables are only available in the controlled block and not globally, so this is not a water tight check
            inputList = [i for i in inputList if i not in outputList]

            # check against input variables and config file
            inputvars = [l[0] for l in mod.getAvailableVariableNames()]
            inputList = [i for i in inputList if i not in inputvars]

            # for output module, check whether the missing variables are output variables. If so, then continue without error; a warning is provided later when building the call stack
            if mod.isOutputModule():
                inputList = [i for i in inputList if i not in self.outputReqList]

            # if there are variables in the inputlist that cannot be provided by any source, show an error and abort.
            if inputList:
                message = ("Not all required variables are given. Missing '%s' in module '%s'.\nVariables can be calculated in other modules, given on input or specified in the iFlow config file" % (', '.join(inputList), str(mod.getName())))
                raise KnownError(message)
        return

    ####################################################################################################################
    ## Private methods 2: Call Stack
    ####################################################################################################################
    def __buildCallStack(self, moduleList):
        """Build the call stack of calculation modules. The method follows these steps
        1. Initialise. First determine the the required module for each module (both for initial run (iterative mods) and
            consecutive runs. Then prepare the lists required in step 3.
        2. place modules in the call stack. This works with a two list system: a list of unplaced modules and
            a call stack (=list of placed modules).
            Iteratively do the following check:
            a. Sort the modules (see below at *)
            b. Place the first module in the list that does not require other unplaced modules.
            c. If the placed module is iterative, determine the requirements for closing the iteration loop
        3. check if the any change occurred in the last iteration or if the unplaced list is depleted.
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
        ## 1. Init
        #   Determine which modules are required for the modules to run

        inputInitMods = self.__loadInputRequirements(init=True, prepare=True, iterator=True)
        unplacedList = [i for i in moduleList if i.getRunModule()]     # list with modules not placed in the call stack

        ## 2. Build
        block, unplacedList, outputList = self.__buildBlock(unplacedList, inputInitMods)

        ## 3. Check for errors or nonconformities
        # a. there are vars required on output, but not calculated
        if any([j for j in self.outputReqList if j not in outputList]):
            missingVars = [j for j in self.outputReqList if j not in outputList]

            # Mention that not all variables can be calculated, but carry on anyway
            message = ('Not all variables that are required on output can be calculated using the current input settings. '
                        '\nCould not calculate %s.'
                        '\nComputation will continue.\n' % (', '.join(missingVars)))
            self.logger.warning(message)
            unplacedList = []

        # b. output requirements (more than) fulfilled. Stop the building of the call stack.
        unusedlist = [i for i in moduleList if not i.getRunModule()]
        if any(unusedlist):
            # Nb. not necessarily all modules will be run, provide a warning if modules have not been placed in the call stack
            self.logger.warning('Not all modules will be used.'
                                '\nModule(s) ' + (', '.join([mod.getName() for mod in unusedlist])) + ' will not be run'
                                '\nComputation will continue.\n')

        # c. output requirements (more than) fulfilled. Stop the building of the call stack.
        if any(unplacedList):
           raise KnownError('Modules %s could not be placed in the call stack, but seem to contribute to the requested output. '
                            'Please check the input and output requirements of your modules.' % ', '.join([mod.getName() for mod in unplacedList]))

        ## 4. Print result to log
        # Console information
        self.logger.info('Call stack was built successfully')

        # Print call stack to log. Includes information on loops
        self.logger.info('** Call stack **')
        self.callStackCounter = 1
        self.__printBlockCallStackToConsole(block, self.callStackCounter, '\t', '\t', '\t')
        self.logger.info('')

        return block

    def __printBlockCallStackToConsole(self, block, counter, depth, depth_first, depth_last):
            for i, mod in enumerate(block.callStack):
                if isinstance(mod, Module):

                    submods = mod.getSubmodulesToRun()
                    submods_iter = mod.getSubmodulesToRun(iter=1)
                    submod_str = ', '.join(submods_iter+['*'+i for i in submods if i not in submods_iter])

                    if (isinstance(block, StandardBlock) and i==0) or (isinstance(block, IterativeBlock) and i<len(toList(block.getIterativeModules()))):
                        self.logger.info(str(self.callStackCounter)+depth_first+mod.getName()+ '\t' + '('*bool(len(toList(mod.getSubmodulesToRun()))) + submod_str+ ')'*bool(len(toList(mod.getSubmodulesToRun()))))
                    elif i==len(block.callStack)-1:
                        self.logger.info(str(self.callStackCounter)+depth_last+mod.getName()+ '\t' + '('*bool(len(toList(mod.getSubmodulesToRun()))) + submod_str+ ')'*bool(len(toList(mod.getSubmodulesToRun()))))
                    else:
                        self.logger.info(str(self.callStackCounter)+depth+mod.getName()+ '\t' + '('*bool(len(toList(mod.getSubmodulesToRun()))) + submod_str+ ')'*bool(len(toList(mod.getSubmodulesToRun()))))
                    self.callStackCounter += 1

                    if mod.isControllerModule():
                        controlledBlock = mod.getControlledBlock()
                        self.__printBlockCallStackToConsole(controlledBlock, self.callStackCounter, depth+'\t'+' # ', depth+'\t'+' # ', depth+'\t'+' # ')

                if isinstance(mod, IterativeBlock):
                    if i==0:
                        self.__printBlockCallStackToConsole(mod, self.callStackCounter, depth+'|'+'\t', depth_first+'|->'+'\t', depth+'|<-'+'\t')
                    elif i==len(block.callStack)-1:
                        self.__printBlockCallStackToConsole(mod, self.callStackCounter, depth+'|'+'\t', depth+'|->'+'\t', depth_last+'|<-'+'\t')
                    else:
                        self.__printBlockCallStackToConsole(mod, self.callStackCounter, depth+'|'+'\t', depth+'|->'+'\t', depth+'|<-'+'\t')


    def __buildBlock(self, unplacedList, inputMods):
        """

        Args:
            unplacedList:
            inputList:

        Returns:

        """
        callStack = []
        outputList = []     # list of available output variables
        while unplacedList:
            listSizeBefore = len(unplacedList)
            _, sortingList = self.__iterativeDependence(unplacedList)  # set importance per iterative loop # 06-08-2018: in separate module to evaluate everytime the callstack is updated; helps dealing with situation of two independent loops inside another loop

            # 1. sort modules
            sortingList, unplacedList = (list(x) for x in zip(*sorted(zip(sortingList, unplacedList), key=lambda pair: pair[0])))

            # 2. place modules that can be placed according to their input requirements
            for i, mod in enumerate(unplacedList):
                if not [j for j in inputMods[mod]]:       # if a module does not require initial input that is not already in outList

                    # Check if the module is iterative or a controller: make a new block or put mod in call stack
                    if mod.isIterative():               # for iterative module: create a new block that includes the iterative module and put the block in the call stack
                        # check if this module iterates together with others
                        co_iterators = [i for i in unplacedList if i in toList(mod.getIteratesWith())]
                        newblock, unplacedList, outputListAdd = self.__buildIterativeBlock(unplacedList, inputMods, mods=co_iterators+[mod])
                        callStack.append(newblock)

                    elif mod.isControllerModule():      # for controller: put the controller in the call stack but not the controlled block; this will be passed to the controller to instantiate and run manually.
                        callStack.append(mod)
                        unplacedList.pop(i)
                        self.updateInputMods(inputMods, mod)
                        newblock, unplacedList, outputListControlledModules = self.__buildControlledBlock(unplacedList, inputMods, mod)
                        outputListAdd = mod.getOutputVariables() # Do not include the output of the controlled modules in the output of this block.
                        mod.addControlledBlock(newblock)

                    else:
                        callStack.append(mod)                                           # place in call stack
                        unplacedList.pop(i)                                             # remove from list of unplaced modules
                        self.updateInputMods(inputMods, mod)
                        outputListAdd = mod.getOutputVariables()
                    outputList = list(set(outputList + outputListAdd))                  # add output of the added module/block to the list of calculated output variables

                    # break loop if a module is placed
                    break

            listSizeAfter = len(unplacedList)

            ## 3. check the progress made in the last iteration
            if listSizeBefore == listSizeAfter or not unplacedList:
                block = StandardBlock(callStack, self._memProfiler)
                return block, unplacedList, outputList


    def __buildIterativeBlock(self, unplacedList, inputMods, mods):
        """

        Args:
            unplacedList:
            inputList:
            mods:

        Returns:

        """
        callStack = []
        outputList = []

        ## Place iterative module(s) and check their requirements
        dependence, sortingList = self.__iterativeDependence(unplacedList)
        requiredModules = []
        for mod in mods:
            callStack.append(mod)
            unplacedList.remove(mod)                                             # remove from list of unplaced modules
            self.updateInputMods(inputMods, mod)
            outputList = list(set(outputList + mod.getOutputVariables()))
            requiredModules = list(set(requiredModules + dependence[mod]))

        while unplacedList and requiredModules:
            listSizeBefore = len(unplacedList)

            # 1. sort modules
            # in an iteration or controlled block
            # sort so that modules that can contribute to the iteration requirements come first.
            # if more modules can contribute, take non-iteratives first
            contributionList = [(i in requiredModules) for i in unplacedList]
            sortingList = [-i for i in sortingList]      # sorting is reversed for contribution list, therefore also reverse sortingList. Reverse back later        # TODO CHECK BY DEBUGGING
            contributionList, sortingList, unplacedList = (list(x) for x in zip(*sorted(zip(contributionList, sortingList, unplacedList), reverse=True, key=lambda pair: pair[0])))
            sortingList = [-i for i in sortingList]

            # 2. place modules that can be placed according to their input requirements
            for i, mod in enumerate(unplacedList):
                if not [j for j in inputMods[mod]]:       # if a module does not require initial input that is not already in outList

                    # Check if the module is iterative or a controller: make a new block or put mod in call stack
                    if mod.isIterative():         # for iterative module: create a new block that includes the iterative module and put the block in the call stack
                        # check if this module iterates together with others
                        co_iterators = [i for i in unplacedList if i in toList(mod.getIteratesWith())]
                        newblock, unplacedList, outputListAdd = self.__buildIterativeBlock(unplacedList, inputMods, mods=co_iterators+[mod])
                        callStack.append(newblock)

                    elif mod.isControllerModule():      # for controller: put the controller in the call stack but not the controlled block; this will be passed to the controller to instantiate and run manually.
                        callStack.append(mod)
                        unplacedList.pop(i)
                        self.updateInputMods(inputMods, mod)
                        newblock, unplacedList, outputListControlledModules = self.__buildControlledBlock(unplacedList, inputMods, mod)
                        outputListAdd = mod.getOutputVariables() # Do not include the output of the controlled modules in the output of this block.
                        mod.addControlledBlock(newblock)

                    else:
                        callStack.append(mod)                                           # place in call stack
                        unplacedList.pop(i)                                             # remove from list of unplaced modules
                        self.updateInputMods(inputMods, mod)
                        outputListAdd = mod.getOutputVariables()

                    outputList = list(set(outputList + outputListAdd))                  # add output of the added module/block to the list of calculated output variables
                    requiredModules = [i for i in requiredModules if i in unplacedList]

                    # break loop if a module is placed
                    break
            listSizeAfter = len(unplacedList)

            ## 3. check the progress made in the last iteration
            if listSizeBefore == listSizeAfter or (not unplacedList and requiredModules):
                raise KnownError('The loop for iterative module(s) %s could not be closed. Please check the input and output requirements of this module and check if all the necessary modules are specified in the input file.' % ', '.join([mod.getName() for mod in mods]))

        # 4. close iteration there are no requirements anymore
        # handle 'iteratesWith' statement
        for mod in copy(unplacedList):
            if mod.getIteratesWith() in [i.getName() for i in mods]:
                callStack.append(mod)
                unplacedList.remove(mod)
                self.updateInputMods(inputMods, mod)
                outputListAdd = mod.getOutputVariables()
                outputList = list(set(outputList + outputListAdd))

        # 5. Check if all submodules need to be run after the initial iteration and change if necessary
        #   use that outputList contains all output variables in this block and child-blocks
        for mod in callStack:
            mod.updateSubmoduleListIteration(outputList)

        block = IterativeBlock(callStack, mods, self._memProfiler)
        return block, unplacedList, outputList


    def __buildControlledBlock(self, unplacedList, inputMods, controller):
        """ Build block that is under control of a ControllerModule. The Controller module itself is not part of this block

        Args:
            unplacedList:
            inputList:

        Returns:

        """
        callStack = []
        inputMods = {key: copy(inputMods[key]) for key in inputMods.keys()} # inputMods list should only be updated inside the controlled block, not propagated to parent blocks.

        ## Place iterative module(s) and check their requirements
        dependence, sortingList = self.__iterativeDependence(unplacedList+[controller])
        outputList = controller.getOutputVariables(includePrepare=False)  # Nb the prepared variables are not necessarily output for parent blocks
        requiredModules = dependence[controller]

        while unplacedList and requiredModules:
            listSizeBefore = len(unplacedList)

            # 1. sort modules
            # in an iteration or controlled block
            # sort so that modules that can contribute to the iteration requirements come first.
            # if more modules can contribute, take non-iteratives first
            contributionList = [(i in requiredModules) for i in unplacedList]
            sortingList = [-i for i in sortingList]      # sorting is reversed for contribution list, therefore also reverse sortingList. Reverse back later        # TODO CHECK BY DEBUGGING
            contributionList, sortingList, unplacedList = (list(x) for x in zip(*sorted(zip(contributionList, sortingList, unplacedList), reverse=True, key=lambda pair: pair[0])))
            sortingList = [-i for i in sortingList]

            # 2. place modules that can be placed according to their input requirements
            for i, mod in enumerate(unplacedList):
                if not [j for j in inputMods[mod]]:       # if a module does not require initial input that is not already in outList

                    # Check if the module is iterative or a controller: make a new block or put mod in call stack
                    if mod.isIterative():         # for iterative module: create a new block that includes the iterative module and put the block in the call stack
                        # check if this module iterates together with others
                        co_iterators = [i for i in unplacedList if i in toList(mod.getIteratesWith())]
                        newblock, unplacedList, outputListAdd = self.__buildIterativeBlock(unplacedList, inputMods, mods=co_iterators+[mod])
                        callStack.append(newblock)

                    elif mod.isControllerModule():      # for controller: put the controller in the call stack but not the controlled block; this will be passed to the controller to instantiate and run manually.
                        callStack.append(mod)
                        unplacedList.pop(i)
                        self.updateInputMods(inputMods, mod)
                        newblock, unplacedList, outputListControlledModules = self.__buildControlledBlock(unplacedList, inputMods, mod)
                        outputListAdd = mod.getOutputVariables() # Do not include the output of the controlled modules in the output of this block.
                        mod.addControlledBlock(newblock)

                    else:
                        callStack.append(mod)                                           # place in call stack
                        unplacedList.pop(i)                                             # remove from list of unplaced modules
                        self.updateInputMods(inputMods, mod)
                        outputListAdd = mod.getOutputVariables()

                    outputList = list(set(outputList + outputListAdd))                  # add output of the added module/block to the list of calculated output variables
                    requiredModules = [i for i in requiredModules if i in unplacedList]

                    # break loop if a module is placed
                    break
            listSizeAfter = len(unplacedList)

            ## 3. check the progress made in the last iteration
            if listSizeBefore == listSizeAfter or (not unplacedList and requiredModules):
                raise KnownError('The loop for iterative module(s) %s could not be closed. Please check the input and output requirements of this module and check if all the necessary modules are specified in the input file.' % ', '.join([mod.getName() for mod in mods]))

        # 4. close iteration there are no requirements anymore
        # handle 'iteratesWith' statement
        for mod in unplacedList:
            if mod.getIteratesWith() == controller.getName():
                callStack.append(mod)
                unplacedList.remove(mod)
                self.updateInputMods(inputMods, mod)
                outputListAdd = mod.getOutputVariables()
                outputList = list(set(outputList + outputListAdd))

        # 5. Controllers may induce iterations, therefore check if all submodules need to be run after the initial
        #   iteration and change if necessaryuse that outputList contains all output variables in this block
        #   and child-blocks
        for mod in callStack:
            output_prep_list = list(set(outputList + controller.getOutputVariables(includePrepare=True)))
            mod.updateSubmoduleListIteration(output_prep_list)

        block = StandardBlock(callStack, self._memProfiler)
        return block, unplacedList, outputList


    def __loadInputRequirements(self, init=False, control=False, prepare=False, iterator=False):
        """Make a dictionary containing 'module name': 'list of required modules' pairs.
        The required variables are determined on the basis of the module's information on whether to run and which submodules to run.

        Parameters:
            init - (bool, optional) take modules required for initial run in case of iterative methods.
            control - (bool ,optional)
            iterator - (bool, optional) if True, add the input requirements of co-iterating modules to the requirements of this module as well

        Return:
            dictionary with 'module name': 'list of required module(s)' pairs.
        """
        d = {}
        runmods = [i for i in self.moduleList if i.getRunModule()]
        for mod in runmods:
            inputReq = mod.getInputRequirements(init=init, includeControls=control)

            if iterator and mod.isIterative():      # if 'iterator' is True, add the input requirements of the co-iterating modules as well
                itermods = mod.getIteratesWith()
                for i in itermods:
                    inputReq = inputReq +  i.getInputRequirements(init=init, includeControls=control)
                inputReq = list(set(inputReq))

            # Add module to list of input requirements if it provides data for 'mod'
            d[mod] = [inmod for inmod in runmods if ([i for i in inputReq if i in inmod.getOutputVariables(includePrepare=prepare)] and inmod!=mod)]
        return d

    def __iterativeDependence(self, unplacedList):
        ## rate iterative modules on interdependency using level. This replaces True/False in iterativelist -> not only denote if a module is iterative, but also how much interdependency it has
        #     0 (or False): not iterative
        #     >0, <1: controller (higher numbers are more interdependent)
        #     1 iterative not dependent on any other iterative module
        #     2, 3, .. dependent on one or more iterative modules - try to place interdependent modules as late as possible for optimal runtimes

        ################################################################################################################
        ## Iterative mod
        ################################################################################################################
        iterativeList = [i.isIterative() for i in unplacedList]  # is the module in the unplacedList iterative or not (bool)

        unratedIterative = list(set([i for i in unplacedList if i.isIterative()]))  # list of iterative modules
        levelIter = len(unratedIterative)  # determine maximum level

        dependenceIter = {}  # initialise list of interdependencies
        sortingList = copy(iterativeList)

        # determine all modules that are required for a closing loop of this iterative module, i.e. all modules required for input and inputInit and their inputs
        for mod in unratedIterative:
            inputInitMods  = self.__loadInputRequirements(init=True)
            inputMods  = self.__loadInputRequirements()

            inp = [i for i in list(set(inputInitMods[mod] + inputMods[mod])) if i in unplacedList]
            dif = len(inp)
            while dif:
                lenold = len(inp)
                inp = [i for i in (list(set(inp + [qq for q in inp for qq in inputInitMods[q]] + [qq  for q in inp for qq in inputMods[q]]))) if i in unplacedList]
                dif = len(inp)-lenold

            try:
                inp.remove(mod)  # remove self if in list
            except:
                pass
            dependenceIter[mod] = inp

        # check for each iterative module if it is dependent on others and assign levels
        dependence_tmp = copy(dependenceIter)  # make a temporary list, because we will remove elements
        while levelIter > 0:
            tmp = []  # temporary list of modules assigned this loop
            for j, mod in enumerate(unplacedList):
                if iterativeList[j] == True:
                    if not any([mod in inp for inp in dependence_tmp.values()]):
                        sortingList[j] = levelIter
                        tmp.append(mod)
            try:
                [dependence_tmp.pop(i) for i in tmp]
            except:
                pass
            levelIter -= 1

        ################################################################################################################
        ## Controller mod
        ################################################################################################################
        controllerList = [i.isControllerModule() for i in unplacedList]  # is the module in the unplacedList a controller or not (bool)

        unratedController = [i for i in unplacedList if i.isControllerModule()]  # list of controller modules
        levelControl = len(unratedController)  # determine maximum level
        dependenceController = {}  # initialise list of interdependencies

        # determine all modules that are required for a closing the controlled block
        for mod in list(set(unratedController)):
            inputMods = list(set(self.__loadInputRequirements()[mod] + self.__loadInputRequirements(control=True)[mod]))
            inp = [i for i in inputMods if i in unplacedList]
            try:
                inp.remove(mod)  # remove self if in list
            except:
                pass
            dependenceController[mod] = inp

        # check for each controller module if it is dependent on others and assign levels
        dependence_tmp = copy(dependenceController)  # make a temporary list, because we will remove elements
        while levelControl > 0:
            tmp = []  # temporary list of modules assigned this loop
            for j, mod in enumerate(unplacedList):
                if controllerList[j] == True:
                    # if not any([mod in inp for inp in iterativeDependence_tmp.values()]):
                    if not any([mod in inp for inp in dependence_tmp.values()]):
                        sortingList[j] = (1.-1./levelControl)
                        tmp.append(mod)
            [dependence_tmp.pop(i) for i in tmp]
            levelControl -= 1

        # merge to dependency dicts
        dependenceIter.update(dependenceController)
        return dependenceIter, sortingList

    def updateInputMods(self, inputMods, mod):
        for q in inputMods.keys():
            try:
                inputMods[q].remove(mod)
            except:
                pass
        return