"""
NumericalFunctionBase
Parent class for numerical functions of an arbitrary number of variables  with arbitrary names
Meaning of numerical functions:
    Numerical functions combine several aspects of functions and ndarrays.
    They are called like functions, but internally they contain a DataContainer with a grid and a variable in ndarray format.
    Because of the function-like call, the numerical function can easily direct the function to e.g. pre-calculated
    derivatives by calling the derivative of the function.
    Similar to analytical functions, numerical functions can only store one variable (and possible derivatives etc. of it)
    There are some differences between the implementation of numerical and analytical differences, see below

    Numerical functions are useful in at least two occasions:
    1) the data to be stored is gridded, but comes with a pre-calculated (gridded) derivative (or something similar)
    2) the data to be stored uses a different grid than the general calculations. The numerical functions allows access
        to this data (including interpolation) on this different grid.

    Note that there is a NumericalFunctionWrapper class that is preferred over this when the numerical function is only
    used for storage and access of already calculated data and grid

Implementation:
    In child classes, implement an __init__(dimnames, data) method.
    In this, call the parents init method with argument dimnames (i.e. NumericalFunctionbase.__init__(self, dimnames)
    It is also advised to copy parameters in data (DataContainer) to class variables in the child's init

    From the child's __init__ also calculate a grid and the data (and possibly derivatives etc.) to be stored.
    Load the grid and data to the internally managed DataContainer instance by calling self.addGrid, self.addValue etc,
    see these methods below.

    It is not necessary to implement value, derivative etc. methods in numerical functions (this is done by this parent class).

Use (same as for analytical functions, see also FunctionBase):
    In a calling script, make a function variable, say 'fun', of an implementation of FunctionBase, say 'LinearFunction'.
    The variable can be loaded with a reference to the function by typing:
        >> tempvar = LinearFunction(['x'], data)
        >> fun = tempvar.function

    Transfer fun to a DataContainer instance to access the function values by using .v(), .d() or .dd().
    Alternatively, access the function directly by using fun as a function call using named(!) arguments, e.g.
        >> value = fun(x=x)


Date: 16-07-15
Authors: Y.M. Dijkstra
"""
from .FunctionBase import FunctionBase
from src.DataContainer import DataContainer
from src.util.diagnostics import KnownError
from copy import deepcopy


class NumericalFunctionBase(FunctionBase):
    #Variables

    #Methods
    def __init__(self, dimNames):
        FunctionBase.__init__(self, dimNames)
        self.dataContainer = DataContainer()
        self.valueSize = 0
        return

    def function(self, **kwargs):
        if len([i for i in kwargs.keys() if i in self.dataContainer.v('grid', 'dimensions')]) == 0:
            return self.__setReturnReference(kwargs.get('operation'))

        # evaluate function
        try:
            returnval = self.__evaluateFunction(**kwargs)
        except FunctionEvaluationError:
            returnval = self.__setReturnReference(kwargs.get('operation'))
        return returnval

    def __evaluateFunction(self, **kwargs):
        """Overrides the function method of FunctionBase, but is very similar.
        The difference is only that FunctionBase transfers kwargs to args before calling the actual functions
        Here we keep the kwargs as the actual functions also use this.
        """
        requestSize = sum([dim in kwargs for dim in self.dimNames])  # count the number of dimensions in kwargs (this makes sure that other parameters or irrelevant dimensions are ignored)

        operation = kwargs.get('operation')
        try:
            kwargs.pop('operation')
        except:
            pass
        # direct to actual function
        if operation is None:
            returnval = self.value(**kwargs)
        elif operation == 'd':
            returnval = self.derivative(**kwargs)
        elif operation == 'n':
            returnval = -self.value(**kwargs)
        elif operation == 'dn':
            returnval = -self.derivative(**kwargs)
        else:
            raise FunctionEvaluationError
        return returnval

    def __setReturnReference(self, operation=None):
        '''No difference with FunctionBase, but required here to refer to its own functions
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

    def negfunction(self, **kwargs):
        """No difference with FunctionBase, but required here to refer to its own functions
        """
        # reset operations
        if kwargs.get('operation') == 'n':      # if the negative of a negfunction is called, return to function.
            kwargs.pop('operation')
        elif kwargs.get('operation') == 'd':
            kwargs['operation'] = 'dn'
        else:
            kwargs['operation'] = 'n'

        # evaluate
        returnval = self.function(**kwargs)
        return returnval

    def addGrid(self, gridData, gridName='grid'):
        # set own DataContainer containing grid data
        data = gridData.slice(gridName, excludeKey=True)
        # datanew = deepcopy(data)
        self.dataContainer.addData('grid', data.data)  # improper use of the DataContainer by accessing its data directly
        return

    def addValue(self, value):
        """Add a variable 'value' to the numerical function

        Parameters:
            value (ndarray) - value to be put in the internal DataContainer
        """
        self.dataContainer.addData('value', value)
        self.valueSize = len(value.shape)
        return

    def addDerivative(self, derivative, dim):
        self.dataContainer.merge({'derivative': {dim: derivative}})
        return

    ### Depreciated v2.2 [dep01] ###
    def addSecondDerivative(self, derivative, dim):
        """ Depreciated v2.2
        """
        self.dataContainer.merge({'secondDerivative': {dim: derivative}})
        return
    ### End ###

    def value(self, **kwargs):
        """Return the value of the variable in this numerical function.

        Parameters:
            kwargs (dict) - coordinates

        Returns:
            Array value using DataContainer interpolation.
        """
        return self.dataContainer.v('value', **kwargs)

    def derivative(self, **kwargs):
        """Similar to .value(). Returns the derivative uploaded to the numerical function or makes a call
        to a numerical derivation method if no derivative is uploaded.
        """
        # kwargs.pop('operation')       # obsolete
        dim = kwargs.get('dim')
        v = self.dataContainer.v('derivative', dim, **kwargs)       # try if analytical derivative is available
        if v is None:
            v = self.dataContainer.d('value', **kwargs)             # else take numerical derivative
        return v

    ### Depreciated v2.2 [dep01] ###
    def secondDerivative(self, **kwargs):
        """See .derivative(). This method does the same for the second derivative
        Depreciated 2.2 [dep01]
        """
        # kwargs.pop('operation')       # obsolete
        dim = kwargs.get('dim')
        v = self.dataContainer.v('secondDerivative', dim, **kwargs)
        if v is None:
            v = self.dataContainer.dd('value', **kwargs)
        return v

class FunctionEvaluationError(Exception):
    def __init__(self, *args):
        return
