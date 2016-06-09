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


    def function(self, **kwargs):
        """Acts as reference to the function.
        Directs the function to the value or any operation on the function (derivative etc.).
        However it returns a reference to the function is the number of dimensions in kwargs is smaller than the
        number of dimensions of the function

        Parameters:
            x - x-coordinate or coordinates
            kwargs['operation'] - (string, optional) operation d or n

        Returns:
            (scalar or list, depending on x) value of function
        """
        returnval = self.function

        # check size of request in kwargs and compare this to the number dimensions of the function
        requestSize = sum([dim in kwargs for dim in self.dimNames])  # count the number of dimensions in kwargs (this makes sure that other parameters or irrelevant dimensions are ignored)
        if requestSize >= len(self.dimNames):
            # convert named arguments to unnamed arguments in the correct order
            indices = ()
            for dim in self.dimNames:
                indices += (kwargs.pop(dim),)

            # direct to actual function
            if kwargs.get('operation'):
                if kwargs.get('operation') == 'd':
                    returnval = self.derivative(*indices, **kwargs)
                ### dd depreciated since v2.2 [dep01] ###
                elif kwargs.get('operation') == 'dd':
                    returnval = self.secondDerivative(*indices, **kwargs)
                ### end ###
                elif kwargs.get('operation') == 'n':
                    returnval = self.returnNegative(*indices, **kwargs)
            else:
                returnval = self.value(*indices, **kwargs)
        return returnval

    def negfunction(self, **kwargs):
        """Same as function, but then provides a reference to -1* the value of the function
        """
        try:
            returnval = -self.function(**kwargs)
        except:
            returnval = self.negfunction
        return returnval

    def returnNegative(self, *args, **kwargs):
        return -self.value(*args, **kwargs)

    def value(self, *args, **kwargs):
        raise KnownError('value of class %s not implemented' % (self.__class__.__name__))
        return

    def derivative(self, *args, **kwargs):
        raise KnownError('derivative %s of class %s not implemented' % (kwargs['dim'], self.__class__.__name__))
        return

    ### depreciated v2.2 [dep01] ###
    def secondDerivative(self, *args, **kwargs):
        raise KnownError('Method depreciated since v2.2. Use derivative or .d() instead.\n'
                         'Second derivative of class %s not implemented' % (self.__class__.__name__))
        return
    ### end ###

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


