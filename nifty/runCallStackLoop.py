"""
runCallStackLoop

Date: 07-Mar-16
Authors: Y.M. Dijkstra
"""
import inspect
import src.config as cf


def runCallStackLoop(startUpdate=None):
    """Run the modules in the loop of the calling module. The calling module itself is not run.
    NB. the calling module has to be iterative

    Parameters:
        startUpdate (dict, optional) - start by updating the datacontainers of all modules in the loop of this module
          using a dictionary
    """
    stack = inspect.stack()
    # get moduleList
    for i in stack:
        if 'ModuleList' in i[1]:
            callStack = i[0].f_locals['self'].callStack
            break
    # get calling module
    callingMod = stack[1][0].f_locals['self'].__module__

    # set imin as the next module in the call stack after this module
    imin = [i for i, mod in enumerate(callStack[0]) if mod.getName() == callingMod][0]+1
    loopno = callStack[1][imin]

    # determine the end of the iteration loop of this module
    imax = imin
    for i in range(imin, len(callStack[0])):
        if callStack[1][i] == 1 and callStack[2][i] != 'start':
            imax = i
        else:
            break

    # make an update to the datacontainers of the modules imin:imax+1
    if startUpdate is not None:
        for module in callStack[0][imin:imax+1]:
            module.addInputData(startUpdate)

    __runLoop(callStack, imin, imax, loopno, noStartModule=True)
    return

def __runLoop(callStack, imin, imax, loopNo, noStartModule=False, **kwargs):
    """Rebuild the call stack by using recursion and then run
    All modules in a single iteration loop are grouped and such a group is then called by this method

    Parameters:
        imin (int) - index of first element in group wrt to call stack
        imax (int) - index of last element in group wrt to call stack
        loopNo (int) - level of recursion
    """
    MAXITERATIONS = cf.MAXITERATIONS

    # 1. Init
    #   build the call list
    callList = []
    end = -1
    for i in range(imax, imin-1, -1):
        if callStack[1][i] == loopNo:
            module = callStack[0][i]
            callList.append((module.run, (), module.getName()))
        else:
            end = max(end, i)
            if callStack[1][i]==loopNo+1 and callStack[2][i] == 'start':
                callList.append((__runLoop, (callStack, i, end, loopNo+1)))
                end = -1
    callList.reverse()

    # 2. run the callList as long as iteration is positive. Set to negative if stopping criterion has been met
    for iteration in range(0, MAXITERATIONS*bool(loopNo)+1):
        # if start module of iteration is not included (noStartModule=True), return to this module
        if iteration > 0 and noStartModule:
            return
        # else, check stopping criterion of iterative module
        elif iteration > 0:
            if callStack[0][imin].stopping_criterion(iteration):
                return


        for tup in callList:
            method = tup[0]
            arguments = tup[1]
            result = method(*arguments, init=(not bool(iteration)))  # only add init=True for iteration==0

            if result is not None:
                for module in callStack[0]:
                    module.addInputData(result)
    return