"""
FunctionBase
Parent class for analytical functions of an arbitrary number of variables with arbitrary names
Implementation:
    In child classes, implement an __init__(dimnames, data) method.
    In this, call the parents init method with argument dimnames (i.e. Functionbase.__init__(self, dimnames)
    It is also advised to copy parameters in data (DataContainer) to class variables in the child's init

    Further implement the value, derivative and secondDerivative methods (all optional).
    These can be functions of any number of dimensions passed as unnamed arguments, see *).
    They can further be depending on additional parameters passed as named arguments in a dict (i.e. kwargs)
    Please make sure that the function is initialised (see above) with the same number of dimensions as used by value/derivative/secondDerivative
    *)  The function should be called by named dimensions, but the arguments are then passed as unnamed arguments.
        This means that the name of the dimensions used in the implementation is irrelevant.

Use:
    In a calling script, make a function variable, say 'fun', of an implementation of FunctionBase, say 'LinearFunction'.
    The variable can be loaded with a reference to the function by typing:
        >> tempvar = LinearFunction(['x'], data)
        >> fun = tempvar.function

    Transfer fun to a DataContainer instance to access the function values by using .v(), .d() or .dd().
    Alternatively, access the function directly by using fun as a function call using named(!) arguments, e.g.
        >> value = fun(x=x)

Example:
    Let Linear be a function of one variable that implements FunctionBase. Here we want to use 'z' as the variable
    Then initialise as
        fun = Linear('z', [any parameters])

    reference to function
        fun = fun.function

    calling function
        fun(z=0.1)

    implementation of value (NB. name of dimension used here is arbitrary)
        def value(self, x):
            return x

Date: 16-07-15
Authors: Y.M. Dijkstra
"""
from src.util.diagnostics import KnownError
from ..toList import toList


class FunctionBase:
    #Variables

    #Methods
    def __init__(self, dimNames):
        """Initialise number and names of dimensions

        Parameters:
            dimNames (list of strings or single string) - names of variables used by the function in the correct order
        """
        self.dimNames = toList(dimNames)
        return

    ####################################################################################################################
    # Function references
    ####################################################################################################################
    def function(self, **kwargs):
        # evaluate function
        try:
            returnval = self.__evaluateFunction(**kwargs)
        except FunctionEvaluationError:
            returnval = self.__setReturnReference(kwargs.get('operation'))
        return returnval

    def negfunction(self, **kwargs):
        """Same as function, but then provides a reference to -1* the value of the function
        """
        # reset operations
        if kwargs.get('operation') == 'n':      # if the negative of a negfunction is called, return to function.
            kwargs.pop('operation')
        elif kwargs.get('operation') == 'd':
            kwargs['operation'] = 'dn'
        else:
            kwargs['operation'] = 'n'

        # evaluate
        try:
            returnval = self.__evaluateFunction(**kwargs)
        except FunctionEvaluationError:
            returnval = self.__setReturnReference(kwargs.get('operation'))
        return returnval

    def derfunction(self, **kwargs):
        """Same as function, but then provides a reference its derivative
        Still requires the axis of the derivative in a named argument 'dim'
        """
        # reset operations
        if kwargs.get('operation') == 'n':      # if the negative of a negfunction is called, return to function.
            kwargs['operation'] = 'dn'
        else:
            kwargs['operation'] = 'd'           # i.e. .d() and .v() of a derfunction returns the same

        # check derivative axis
        if kwargs.get('dim') is None:
            raise KnownError('Called a derivative function without dim argument')
        elif not all(i in self.dimNames for i in kwargs.get('dim')):
            raise KnownError('Called a derivative function with an incorrect dim argument')

        # evaluate
        try:
            returnval = self.__evaluateFunction(**kwargs)
        except FunctionEvaluationError:
            returnval = self.__setReturnReference(kwargs.get('operation'))
        return returnval


    def dnfunction(self, **kwargs):
        """Same as function, but then provides a reference the negative and derivative
        Still requires the axis of the derivative in a named argument 'dim'
        """
        # reset operations
        if kwargs.get('operation') == 'n':      # if the negative of a negfunction is called, return to function.
            kwargs['operation'] = 'd'
        else:
            kwargs['operation'] = 'dn'           # i.e. .d() and .v() of a dnfunction returns the same

        # evaluate
        try:
            returnval = self.__evaluateFunction(**kwargs)
        except FunctionEvaluationError:
            returnval = self.__setReturnReference(kwargs.get('operation'))
        return returnval

    ####################################################################################################################
    # Other public methods
    ####################################################################################################################
    def addNumericalDerivative(self, grid, *args, **kwargs):
        # evaluate function on grid and put in datacontainer 'grid'
        v = self.value(*args,**kwargs)
        grid.addData('value', v)

        # convert unnamed argument to named argument
        coord={}
        dimensions = grid.v('grid', 'dimensions')
        for i, axis in enumerate(args):
            coord[dimensions[i]]=axis
        kwargs.update(coord)

        # take derivative
        der = grid.d('value', **kwargs)

        return der

    def checkVariables(self, *args):
        """Check if variables are set.
        For each variable this will check if the variable is different from None.
        An error is raised if the variable is None

        Parameters:
            args (tuples of two elements: (str, any)) - set of tuples of two elements.
                        The first element describes the variable name in a string, the second is the value of any type
        """
        for pair in args:
            if pair[1] is None:
                message = ("Not all required variables are given. Missing '%s' in module '%s'" % (pair[0], self.__class__.__name__))
                raise KnownError(message)
        return

    ####################################################################################################################
    # General private methods
    ####################################################################################################################
    def __evaluateFunction(self, **kwargs):
        """Directs the function to the value or any operation on the function (derivative etc.).
        However it returns a reference to the function is the number of dimensions in kwargs is smaller than the
        number of dimensions of the function

        Parameters:
            named arguments with coordinates corresponding to the function argument names
            kwargs['operation'] - (string, optional) operation d or n

        Returns:
            scalar or array or function reference - value of function or reference if function cannot be evaluated
        """
        # check size of request in kwargs (coordinates) and compare this to the number dimensions of the function
        #   Count the number of dimensions in kwargs (this makes sure that other parameters or
        #   irrelevant dimensions are ignored)
        requestSize = sum([dim in kwargs for dim in self.dimNames])
        if requestSize < len(self.dimNames):
            raise FunctionEvaluationError

        # convert coordinates to function arguments
        indices, kwargs = self.__convertIndices(**kwargs)

        # direct to actual function
        if kwargs.get('operation') is None:
            returnval = self.value(*indices, **kwargs)
        elif kwargs.get('operation') == 'd':
            returnval = self.derivative(*indices, **kwargs)
        elif kwargs.get('operation') == 'n':
            returnval = -self.value(*indices, **kwargs)
        elif kwargs.get('operation') == 'dn':
            returnval = -self.derivative(*indices, **kwargs)
        else:
            raise FunctionEvaluationError
        return returnval

    def __convertIndices(self, **kwargs):
        '''Convert named arguments (kwargs) to unnamed arguments in the correct order of the function.
        This allows e.g. that, for a function fun(x, z), called as fun(y=0, z=1, x=0.5), the result is f(0.5, 1)
        '''
        indices = ()
        for dim in self.dimNames:
            indices += (kwargs.pop(dim),)
        return indices, kwargs

    def __setReturnReference(self, operation=None):
        '''Determine what function reference to return if no value can be returned.
        '''
        if not operation:
            returnval = self.function
        elif operation == 'n':
            returnval = self.negfunction
        elif operation == 'd':
            returnval = self.derfunction
        elif operation == 'dn':
            returnval = self.dnfunction
        else:
            raise KnownError('Function called with unknown operation. (This error indicates an incorrectly defined function)')
        return returnval

    ####################################################################################################################
    # Empty methods to be implemented
    ####################################################################################################################
    def value(self, *args, **kwargs):
        raise KnownError('value of class %s not implemented' % (self.__class__.__name__))

    def derivative(self, *args, **kwargs):
        raise KnownError('derivative %s of class %s not implemented' % (kwargs['dim'], self.__class__.__name__))

    ### depreciated v2.2 [dep01] ###
    def secondDerivative(self, *args, **kwargs):
        raise KnownError('Method depreciated since v2.2. Use derivative or .d() instead.\n'
                         'Second derivative of class %s not implemented' % (self.__class__.__name__))
    ### end ###

class FunctionEvaluationError(Exception):
    def __init__(self, *args):
        return





