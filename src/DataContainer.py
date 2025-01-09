"""
Class DataContainer

Data wrapper class around the dictionary. Contains a single dictionary self._data with possible sub-dictionaries,
sub-sub-dictionaries etc. The DataContainer may contain five types of data:
    1. strings
    2. lists or nested lists
    3. scalars or numpy arrays
    4. references to a function or instance method

A number of special keys exist in self._data:
    a. __derivative: dict with dimension w.r.t. to take the derivative, variable name, (sub/sub-sub variable name): data of derivative
    c. __variableOnGrid: dict with variable name, (sub/sub-sub variable name): grid name (str) to evaluate the variable on or None
        if not corresponding to a grid.

The DataContainer provides methods for adding data and accessing it, as well as accessing/computations of derivatives.

Grids:
By default all arrays/functions are evaluated on the variable 'grid'.


Please see the docstrings of the individual methods to get more information or consult the iFlow modelling framework
manual

Date: 03-02-22
Authors: Y.M. Dijkstra
"""
import types
from numbers import Number
import numpy as np
import nifty as nf
from copy import copy as cp
from src.util.diagnostics import KnownError
from src.util.interpolation import RegularGridInterpolator
from src.util.grid import convertIndexToCoordinate
from src.util import mergeDicts, derivativeOfFunction
from src.util.NFunction import NFunction
from src.util.DFunction import DFunction


class DataContainer:
    # Variables

    # Methods
    def __init__(self, *args):
        """Make a new DataContainer instance filled by dictionaries provided in the arguments

        Parameters:
            args - (optional, one or more dicts) optional dictionaries to be added to the data
        """
        self._data = {}
        self._data['__variableOnGrid'] = {}
        self._data['__derivative'] = {}
        self._data['__outputGrid'] = {}

        for i in [j for j in args if j is not None]:
            self._data.update(i)
        return

    ####################################################################################################################
    # Methods for altering, inspecting or splitting the structure of the DataContainer
    ####################################################################################################################
    def copy(self):
        """Returns a new DataContainer instance with the same data as this one.
        So the data is not copied; only the DataContainer is different.

        Returns:
            DataContainer
        """
        print('DC - method copy used. Planned to phase out use of this method')
        return DataContainer(self._data)

    def addData(self, name, data):
        """Add data with key=name to the DataContainer.
        NB. duplicate keys will be overwritten (on the lowest level if name contains subkeys).

        Parameters:
            name - (string) key
                Or (tuple/list) key, subkey, subsubkey ....
            data - (any) value, pointer etc to be stored
        """
        if isinstance(name, str):
            name = [name]

        d = self.__buildDict(name, data)
        self.merge(d)
        return

    def addDerivative(self, name, data, derivative):
        """Add a derivative with key=name. The direction and order is given by 'derivative' (e.g. x, xx, xz, z)
        NB. duplicate keys will be overwritten.

        Parameters:
            name - (string) key
                Or (tuple/list) key, subkey, subsubkey ....
            data - (any) value, pointer etc to be stored
            derivative - (string) dimension w.r.t. which to take the derivative. Order of derivative is given by the
                                  number of characters, e.g. xx is 2nd derivative w.r.t. x.
        """
        namelist = ['__derivative', derivative]
        if isinstance(name, str):
            namelist.append(name)
        else:
            namelist = namelist + list(name)
        d = self.__buildDict(namelist, data)
        self.merge(d)
        return

    def registerVariableToGrid(self, name, gridname):
        """Define that variable with 'name' is defined on the grid with 'gridname'

        Parameters:
            name - (string) key
                Or (tuple/list) key, subkey, subsubkey ....
            gridname - (strong) key to grid name
        """
        namelist = ['__variableOnGrid']
        if isinstance(name, str):
            namelist.append(name)
        else:
            namelist = namelist + list(name)
        d = self.__buildDict(namelist, gridname)
        self.merge(d)
        return

    def merge(self, container):
        """Merges a dictionary or DataContainer with the present DataContainer.
        NB. duplicate keys will be overwritten, but only on the lowest (sub-)dict level

        Parameters:
            container - (DataContainer or dict) data to be merged with this DC
        """
        if isinstance(container, dict):
            container = DataContainer(container)
        derivatives = self.slice('__derivative')
        gridsregister = self.slice('__variableOnGrid')

        allKeys = container.getAllKeys()
        derivativesKeys = derivatives.getAllKeys()
        gridsregisterKeys = gridsregister.getAllKeys()
        removeFromDer = [i for i in derivativesKeys if any([j for j in allKeys if i[2:min(len(i)-2,len(j))+2]==j[:min(len(i)-2,len(j))]])]
        removeFromGR = [i for i in gridsregisterKeys if any([j for j in allKeys if i[1:min(len(i)-1,len(j))+1]==j[1:min(len(i)-1,len(j))]])]
        for key in removeFromDer+removeFromGR:
            self.__removeKey(key, self._data)

        ## merge
        if isinstance(container, dict):
            self._data = mergeDicts(self._data, container)
        else:                                                       # assume container is a DC
            self._data = mergeDicts(self._data, container._data)

        return

    def slice(self, key, *args, excludeKey=False):
        """Returns a new DataContainer instance containing the key (+subkeys in args) from _data.
        NB. does not make a copy of the data.
        NB. does not support derivatives or alternative grids, i.e. only returns a _data class variable, leaving the
            other class variables empty

        Parameters:
            key & args (str) - keys and subkeys such as meant in .v()
            excludeKey (bool, optional) - if set to true, the key (and/or subkeys) will not be part of the new DataConatainer.
                                          This will only work if the value obtained from the keys is a dict. Else it raises an exception.
                                          Please use this option with caution. Default: False

        Returns:
            New DataContainer instance with key provided by the final element of args or by 'key' if args is empty
        """
        value = self._data
        keys = (key,)+args
        lastKey = keys[-1]
        value = self.__unwrapDictionaries(value, *keys)

        if not excludeKey:
            DC = DataContainer({lastKey: value})
        else:
            try:
                DC = DataContainer(value)
            except Exception as e:
                raise KnownError('Problem in slicing DataContainer. Result is not a dictionary.', e)

        return DC

    def getAllKeys(self):
        """Get all keys and subkeys contained in the DataContainer.
        Returns a list of tuples, where the tuples contain consecutively the key, sub-key, sub-sub-key etc.
        """
        keyList = self.__getKeysFromDict(self._data)
        return keyList

    def getKeysOf(self, key, *args):
        """Get all subkeys contained under the data with 'key' and subkeys contained in 'args'.
        The arguments key and args are not included in the returned list of keys/subkeys.
        Returns a list of subkeys, but without sub-sub-keys

        Parameters:
            key & args - (str) keys and subkeys such as meant in .v()
        """
        value = self._data[key]
        for arg in args:
            value = value[arg]
        if isinstance(value, dict):
            keyList = list(value.keys())
        else:
            keyList = []
        return keyList

    def has(self, key, *args):
        """Checks whether an entry is in this DC

        Args:
            key (str) - first key
            args (optional, str) - subkeys

        Returns: bool
        """
        subkeys = tuple([i for i in args if isinstance(i, str)])
        keys = (key,)+subkeys
        value = self.__unwrapDictionaries(self._data, *keys)

        if value is None or value == []:
            return False
        else:
            return True

    ####################################################################################################################
    # Methods for data retrieval: d, n, d, dd
    ####################################################################################################################
    def v(self, key, *args, reshape=True, copy=False, _operation=None, _dim=None, grid=None,  **coordinates):
        """Check and retrieve a value with key 'key' and optional subkeys provided in args. Specific elements of the
        data can be retrieved using either list/array indices provided in args or dimensionless positions on the grid
        axes in the named arguments 'coordinates'. In case of using indices or grid positions, the data will,
        if possible, be returned in the shape of the request.
        Example of practical use for multidimensional output:
            v(key, subkey, x=..,z=..,f=..) or v(key, subkey, .., .., ..)

        The method tries the following:
        1. Retrieve the value belonging to 'key'. Then consecutively use the optional string arguments contained in
        'args' to retrieve sub-values belonging to sub-keys. Continue until all string arguments in 'args' have been
         used.

        2. Evaluate the value found depending on the data type
            a.  dictionary
                If the value found is a dictionary structure (i.e. multiple key-value pairs). Evaluate all underlying
                values and then try to add the values. If addition is not possible, return True to indicate that the
                item exists, but is a data structure instead of a value.

                NB. this method never returns a dictionary

            b.  List/nested list
                Use numerical arguments in 'args' (one or more integers or lists of integers) to access the list and
                nested lists by the indices (like accessing an array). The whole list is returned if args are omitted.
                If the number of numerical arguments in args is less than the nesting depth of the list, the full
                sub-list is returned in the omitted dimensions.
                Call the list as v('variable', index1, index2, ...)

                NB. Lists do not support direct numerical operations and access by coordinate (see numpy arrays)

            c.  numpy array:
                Either use the numerical data in args to access the element corresponding to the indices in the array
                or use coordinates to access coordinates on a grid.

                Grid:
                    A variable is defined on grid 'grid' by default, but may be registered to another grid. This grid
                    needs to be in the self._data. The data may be requested on indices corresponding to a different
                    grid by the argument 'grid' in self.v.
                    Note that two grids:
                    1) should carry the same order of dimensions. More/less dimensions is ok.
                    2) should have the same upper and lower bounds of the axes.

                Data:
                    Dimensions of an array should follow the order prescribed by the grid dimensions.
                    For example if the registry dimensions are x, z, f, then data can have the configurations (), (x),
                    (x,z) or (x,z,f). If the data is only dependent on x and f,it  should still have dimensions (x,z,f),
                     with len(z)=1

                Calling:
                    using indices:
                    v('key', (optional) 'subkeys', .., .., ..). The dots can contain integers or 1D integer lists/arrays
                    containing the indices to return. The order of the dimensions should correspond to the grid dimensions.

                    using coordinates
                    v('key' (optional) 'subkeys', x=..., z=..., f=...), where x,z,f are the grid dimensions. If the
                    variable is e.g. only dependent on x and f, it is sufficient to call with only x=... and f=... .
                    the arguments following x,z and f should be dimensionless and may be integer or 1D integer lists/arrays.

                Returns:
                    Data is returned in the shape following the request. If the data is smaller than the requested size,
                    e.g. 1D with a 2D call, the data is extended to 2D by copying its values.
                    If a dimension is omitted, the data will be returned on the its original shape in that dimension and
                    corresponding to its original grid.

                NB. indices/coordinates are allowed to be omitted or incomplete. It is however not advised to use this
                option for retrieving data (so that calls to arrays and functions are the same). It may be used for
                checking whether a variable exists.

            d.  Reference to function or instance method:
                call the method with either grid indices or dimensionless coordinates as arguments. Calls are the same
                as for arrays. The function reference will be returned if indices/dimensions are omitted or incomplete.

            e.  Numbers:
                Evaluate operations and then reshape. Indices/coordinates are used only for the final shape. If
                indices/coordinates are used, the number is converted to an array.

            f.  String:
                Return the value. Indices/coordinates are not used.

        4. Reshape the data to the dimension requested (array, function, scalar, not list).  If the data is smaller than
        the request, its values are copied to obtain the right shape. If the data is larger than the request, data is
        returned in its original format; i.e. arrays/lists are returned in their original shape in the omitted
        dimensions and functions are returned as function references.

        Parameters:
            name - key of value in dictionary
            args - (optional, one or more str) keys of a hierarchy of sub-dictionaries (in the correct order).
                 - (optional, one or more int or 1D int lists/arrays) grid indices. Order of args should correspond to
                    order of grid dimensions.
            coordinates - (optional, one or more int or 1D int lists/arrays) dimensionless grid coordinates using argument
                        names corresponding to grid dimensions
            reshape (optional, bool) reshape result to shape of request. Default: True
            copy (optional, bool) if variable size is 1 in a dimension but request size is >1, extend values to the additional dimensions (if True) or not (if False)
            _operation (optional, 'd' or 'n'): utility of self.d() and self.n()
            _dim (optional, str): utility of self.d()
            grid (optional, str): gridname of grid to evaluate indices on.

        Except:
            KnownError - if 1. access by index: if index out of range or if too many indices are given
                            2. access by indices/coordinates while grid is missing or no interpolation is implemented for this
                               grid

        Returns:
            scalar, string, list or array belonging to key and optional subkeys on the requested indices/coordinates
            Returns None if the value could not be found.
            Returns True if a dictionary (i.e. set key-value pairs) was found and the underlying values could not be
            added
        """
        subkeys = tuple([i for i in args if isinstance(i, str)])
        keys = (key,)+subkeys
        indices = tuple([i for i in args if not isinstance(i, str)])

        if grid is None:
            grid = self.__findGridForVariable(keys, grid='grid', check=False)

        # STEP 1
        value = self._data      # use value as a running variable that is reduced step by step
        value = self.__unwrapDictionaries(value, *keys)

        # STEP 2.
        # return None if nothing was found
        if value is None or value == []:
            return None

        ################################################################################################################
        # 2a. Dictionary
        ################################################################################################################
        elif isinstance(value, dict):
            try:
                value = self.__addvalues(keys, reshape, copy, _operation, _dim, grid, *indices, **coordinates)
                if isinstance(value, dict):
                    return True
                else:
                    return value        # immediately return as further processing is not needed; .v is already called
                                        # in 'addvalues'
            except:
                return True

        ################################################################################################################
        # 2b. Lists
        ################################################################################################################
        elif isinstance(value, list) or isinstance(value, range):
            # use indices
            if indices:
                for i in indices:
                    try:
                        if isinstance(i, int):
                            value = value[i]
                        else:
                            value = (np.asarray(value)[i]).tolist()
                    except IndexError as e:
                        raise KnownError('Index out of range',e)
                    except TypeError as e:
                        raise KnownError('Tried to access a dimension that does not exist',e)

        ################################################################################################################
        # 2c. Numpy arrays or numbers
        ################################################################################################################
        elif isinstance(value, np.ndarray):
            # find grid belonging to this variable (only needed when using indices/coordinates/taking derivative)
            if (indices or coordinates) or _operation=='d':
                gridnameVariable = self.__findGridForVariable(keys, grid=grid)
            else:
                gridnameVariable = None

            # indices provided and on same grid as where data is defined
            if indices and grid==gridnameVariable:
                try:
                    # reduce indices to 0 if accessing a dimension that has only one element
                    newArgs = ()
                    for i,v in enumerate(value.shape):
                        if value.shape[i] == 1:
                           newArgs += ([0],)                # return single dimension for length-1 axes
                        elif i < len(indices):
                           newArgs += (nf.toList(indices[i]),)    # return data on requested points for existing axes
                        else:
                            newArgs += (range(0, v),)     # return data on original grid if dimension not provided
                    value = self.__evaluateOnIndices(value, *newArgs, gridname=gridnameVariable, operation=_operation, dim=_dim)       # evaluate and perform any operation on the data
                except IndexError as e:
                    raise KnownError('Index out of range', e)
                except TypeError as e:
                     raise KnownError('Tried to access a dimension that does not exist', e)

            # indices provided but on different grid as where data is defined
            elif indices:       # NB only for a regulargrid
                value = self.__evaluateOnIndices(value, gridname=gridnameVariable, operation=_operation, dim=_dim)   # first evaluate any operations on the entire array (not efficient)

                # convert indices to coordinates
                if self._data[gridnameVariable].get('gridtype') == 'Regular' and self._data[grid].get('gridtype') == 'Regular':
                    try:
                        newCoordinates = {}
                        dimsRequestGrid = self._data[grid]['dimensions']
                        # first convert all requested indices to coordinates
                        for i,v in enumerate(indices):
                            newCoordinates[dimsRequestGrid[i]] = (self._data[grid]['axis'][dimsRequestGrid[i]].flatten())[v]

                    except IndexError as e:
                        raise KnownError('Index out of range', e)
                    except TypeError as e:
                         raise KnownError('Tried to access a dimension that does not exist', e)

                    # evaluate on grid
                    interpolator = RegularGridInterpolator()
                    value = interpolator.interpolate(value, self._data[gridnameVariable], **newCoordinates)

                else:
                    # give an error message if the grid type has no data interpolation method implemented
                    raise KnownError('Access to gridded data not implemented on a grid of type %s' % (self._data[gridnameVariable]['gridtype']))

            # otherwise use coordinates
            elif coordinates:
                value = self.__evaluateOnIndices(value, gridname=gridnameVariable, operation=_operation, dim=_dim)   # first evaluate any operations on the entire array (not efficient)

                # interpolate to requested coordinates
                if self._data[gridnameVariable].get('gridtype') == 'Regular':
                    # interpolation on a regular grid
                    interpolator = RegularGridInterpolator()
                    value = interpolator.interpolate(value, self._data[gridnameVariable], **coordinates)

                else:
                    # give an error message if the grid type has no data interpolation method implemented
                    raise KnownError('Access to gridded data not implemented on a grid of type %s' % (self._data[gridnameVariable]['gridtype']))

            else:
                value = self.__evaluateOnIndices(value, gridname=gridnameVariable, operation=_operation, dim=_dim)

            # reshape
            if reshape:
                value = self.__reshape(value, *indices, gridname=grid, copyDims=copy, **coordinates)

        ################################################################################################################
        # 2d. Functions and methods
        #           try to access the function value. However if too few arguments are given, return the reference to
        #           function
        ################################################################################################################
        elif isinstance(value, types.MethodType):
            # find grid belonging to this variable (only needed when using indices/coordinates/taking derivative)
            if (indices or coordinates) or _operation=='d':
                gridnameVariable = self.__findGridForVariable(keys, grid=grid)
            else:
                gridnameVariable = None

            # check if function has dimNames attribute
            if not hasattr(value.__self__, 'dimNames'):
                raise KnownError('Function object under key %s has no attribute dimNames. Make sure that the function is part of a class that defines dimNames.\nSee iFlow manual for details.'%key)
            functiondims = value.__self__.dimNames

            # prepare coordinates to evaluate the function
            if indices:
                coordinatesCopy = convertIndexToCoordinate(self._data[grid], indices)
            elif coordinates:
                coordinatesCopy = coordinates.copy()
            else:
                coordinatesCopy = {}

            # convert called dimensions to array. i.e. function will not be called with scalar arguments
            [coordinatesCopy.update({dim:np.asarray(nf.toList(coordinatesCopy[dim]))}) for dim in functiondims if coordinatesCopy.get(dim) is not None]

            # function call with coordinates and operations
            value = self.__evaluateFunctionOnCoordinates(value, gridname=gridnameVariable, operation=_operation, dim=_dim, **coordinatesCopy)

            # reshape
            if not isinstance(value, types.MethodType):
                value = self.__reshapeFunction(value, functiondims, grid)   # before full reshape, first reshape so that shape matches the order of 'dimensions' on the grid

                if reshape:
                    value = self.__reshape(value, *indices, gridname=grid, copyDims=copy, **coordinates)

        ################################################################################################################
        # 2d. Numbers
        ################################################################################################################
        elif isinstance(value, Number):
            value = self.__evaluateOnIndices(value, operation=_operation, dim=_dim)
            if reshape:
                value = self.__reshape(value, *indices, gridname=grid, copyDims=copy, **coordinates)

        return value

    def n(self, key, *args, reshape=True, copy=False, grid='grid', **coordinates):
        """Returns the value accessed by .v() multiplied -1. Only for arrays, scalars and functions (not lists)
        See documentation of .v().
        """
        value = self.v(key, *args, reshape=reshape, copy=copy, _operation='n', grid=grid, **coordinates)
        return value

    def d(self, key, *args, reshape=True, copy=False, dim=None, grid='grid', **coordinates):
        """Returns the derivative of the value accessed by .v(). See documentation .v(). Other than .v() this raises an
        exception if no entry is found (.v returns None) or summation of subkeys fails (.v returns True).

        Parameters:
            same as in .v(). Additionally:
            dim (str) - NB not optional! Axis name along which to take the derivative. Provide multiple
                        characters to get higher derivatives. Mixed derivatives also allowed, but order
                        of derivation should not matter
        """
        # Check if dim is provided
        if dim is None:
            raise KnownError('Called .d() method without supplying the axis of derivation.'
                             '\n Please add a dim=(str) argument.')
        elif not isinstance(dim, str):
            raise KnownError('Argument dim of .d() is not a string.'
                             '\n Please add a dim=(str) argument.')

        ## 1. check if derivative exists (and can be evaluated) in __derivative
        newargs = (dim, key,) + args
        value = self.v('__derivative', *newargs, reshape=reshape, copy=copy, grid=grid, **coordinates)

        # 2. check whether value captures the derivative / captures the entire derivative; subkeys may be missing.
        # Load entire value / missing subkeys in value2 and take numerical derivative
        allkeys = self.getAllKeys()
        keys = tuple([key] + [i for i in args if isinstance(i, str)])
        indices = [i for i in args if not isinstance(i, str)]
        lenkeyset = len(keys)
        keys_der = [i[2:] for i in allkeys if (('__derivative', dim) + keys) == i[:lenkeyset+2]]
        keys_data = [i for i in allkeys if keys == i[:lenkeyset]]
        diff_keys = [i for i in keys_data if len([j for j in keys_der if j == i[:len(j)]])==0]

        # take derivative of each contribution NB. not efficient as it first takes derivative and then sums.
        # this is however easiest, since each set of subdata can be of different type/on a different grid.
        for i in diff_keys:
            val_add = self.v(*i, *indices, reshape=reshape, copy=copy, _operation='d', _dim=dim, grid=grid, **coordinates)
            if value is None:
                value = cp(val_add)         # copy function
            else:
                value = value + val_add     # NB allow exception if addition is not possible; i.e. don't return True

        return value

    ####################################################################################################################
    # Private methods
    ####################################################################################################################
    def __getKeysFromDict(self, d):
        """Utility of getAllKeys. Recursive and therefore in a separate private method
        """
        keyList =[]
        for i in d:
            if isinstance(d[i], dict):
                key = (i,)
                subkeys = self.__getKeysFromDict(d[i])
                for j in subkeys:
                    try:
                        keyList.append(key+j)
                    except:
                        keyList.append(key+(j,))
            else:
                keyList.append((i,))
        return keyList

    def __unwrapDictionaries(self, value, *keys):
        """Utility of self.v and self.slice
        Unwrap the dictionary using the key and the subkeys in the optional args.
        """
        for i, key in enumerate(keys):
            if isinstance(value, dict) and key in value:        # if the key is in the dictionary
                value = value[key]                              # step into dictionary
            else:
                value = None
        return value

    def __addvalues(self, keys, reshape, copy, operation, dim, grid, *indices, **coordinates):
        """Utility of .v. Sums all values in a dictionary and possible sub-dictionaries.
        NB1. keys is a tuple with all keys (key and subkeys)
        NB2. Does not guarantee that this is possible. An error will be thrown if summation fails.
        """
        value = None
        allkeys = self.getAllKeys()
        keys_data = [i for i in allkeys if keys == i[:len(keys)]]

        for i in keys_data:
            val_add = self.v(*i, *indices, reshape=reshape, copy=copy, _operation=operation, _dim=dim, grid=grid, **coordinates)
            if value is None:
                value = cp(val_add)         # copy function
            else:
                value = value + val_add
        return value

    def __evaluateOnIndices(self, value, *indices, gridname='grid', operation=None, dim=None):
        """Utility of .v().
        Deals with operations, such a derivation and integration on numerical data (scalar, array).
        Returns the value at the specified indices if no operation is prescribed
        """
        ############################################################################################################
        # Negation
        ############################################################################################################
        if operation == 'n':
            if indices:
                value = -value[np.ix_(*indices)]
            else:
                value = -value

        ############################################################################################################
        # Derivative
        ############################################################################################################
        elif operation == 'd':
            splitDim = self.splitDimString(dim, gridname)    # dim can contain a concatenation; split and order in order of grid dimensions
            for dir in list(set(splitDim)): # loop over all dimensions once
                if isinstance(value, Number) or value.size==1:   # return 0 for derivative of constant (either a number or array with one element)
                    value = 0.

                else:
                    order = len([i for i in splitDim if i==dir]) # collect the number of occurances of this dimension
                    if order == 1:
                        value = nf.derivative(value, dir, self.slice(gridname), *indices, gridname=gridname)
                    elif order == 2:
                        value = nf.secondDerivative(value, dir, self.slice(gridname), *indices, gridname=gridname)
                    else:
                        raise KnownError('Numerical derivatives of order %s are not implemented' % str(order))

        ############################################################################################################
        # evaluate without further operation
        ############################################################################################################
        else:
            if indices:
                value = value[np.ix_(*indices)]

        return value

    def __evaluateFunctionOnCoordinates(self, value, gridname='grid', operation=None, dim=None, **coordinates):
        """
        in case operation is not None: evaluate if all arguments for the function are provided, else return a reference
        to a function that will later compute the operation when evaluated numerically.
        """
        functiondims = value.__self__.dimNames

        ############################################################################################################
        # Negation
        ############################################################################################################
        if operation == 'n':
            if all([dim in coordinates.keys() for dim in functiondims]):
                value = - value(**coordinates)
            else:
                funobj = NFunction(value)
                value = funobj.nfunction

        ############################################################################################################
        # Derivative
        ############################################################################################################
        elif operation == 'd':
            if all([(var in coordinates.keys()) for var in functiondims]):  # i.e. check if coordinates contains all variables
                splitDim = self.splitDimString(dim, gridname)    # dim can contain a concatenation; split and order in order of grid dimensions
                for dir in list(set(splitDim)): # loop over all dimensions once
                    order = len([i for i in splitDim if i==dir]) # collect the number of occurances of this dimension
                    if order == 1:
                        value = derivativeOfFunction.derivativeOfFunction(value, dir, self.slice(gridname), gridname=gridname, epsilon = 1e-4, **coordinates)
                    elif order == 2:
                        value = derivativeOfFunction.secondDerivativeOfFunction(value, dir, self.slice(gridname), gridname=gridname, epsilon = 1e-4, **coordinates)
                    else:
                        raise KnownError('Numerical derivatives of order %s are not implemented' % str(order))
            else:
                funobj = DFunction(value, dim, self.slice(gridname), gridname='grid')
                value = funobj.dfunction

        ############################################################################################################
        # evaluate without further operation
        #   evaluate if all arguments for the function are provided, else return function reference
        ############################################################################################################
        else:
            if all([(var in coordinates.keys()) for var in functiondims]):  # i.e. check if coordinates contains all variables
                value = value(**coordinates)

        return value


    def __reshape(self, value, *indices, gridname, copyDims, **coordinates):
        """Reshape value to shape prescribed by indices or coordinates. However,
        a) excess dimensions are trimmed off
        b) if the data has more dimensions than the request, these dimensions are added

        Parameters:
            value (array, list, scalar) - data
            indices/coordinates - indices or coordinates as in self.v()
        """
        if not indices and not coordinates:
            return value

        # determine shape of request
        if indices:
            shape = [len(nf.toList(n)) for n in indices]
            shapeTruncated = [len(nf.toList(n)) for n in indices if isinstance(n, list) or isinstance(n, range) or isinstance(n, np.ndarray)]

        elif coordinates:
            shape = []
            shapeTruncated =[]
            for i, dim in enumerate(self._data[gridname]['dimensions']):
                n = coordinates.get(dim)
                if n is not None:       # i.e. if this dimension is passed in the request
                    shape += [len(nf.toList(n))]
                    if isinstance(n, list) or isinstance(n, np.ndarray) or isinstance(n, range):
                        shapeTruncated += [len(n)]
                else:                   # else, provide data on original grid.
                    try:
                        shape += [(np.asarray(value).shape)[i]]
                        shapeTruncated += [(np.asarray(value).shape)[i]]
                    except:
                        pass

        # set number of dimensions of value to that of shape
        #   first reshape to the right number of dimensions or extend the request shape if necessary
        value = np.asarray(value)
        valShape = list(value.shape)
        dlen = len(shape)-len(valShape)
        if dlen < 0:        # value has more dims than request
            shape += (valShape[dlen:])
            shapeTruncated += (valShape[dlen:])
        elif dlen > 0:
            newShape= valShape+[1]*dlen
            value = value.reshape(newShape)

        #   then recast into right shape
        copy = self._data[gridname]['copy'][:len(shape)]
        extensionMatrix = np.ones(shape)

        if not copyDims == 'all':
            for i, item in enumerate(copy):
                if item == 0 and shape[i] > 1 and value.shape[i] == 1:
                    extensionMatrix[(slice(None),)*i + (slice(1, None),)+(Ellipsis,)] = 0

        value = value*extensionMatrix
        value = value.reshape(shapeTruncated)

        return value

    def __reshapeFunction(self, value, functiondims, gridname):
        """reshape function output to be conform the order of the dimensions.
        """
        functiondims = functiondims[:]  # make a copy of the list
        dimensions = self._data[gridname]['dimensions']
        for i, dim in enumerate(functiondims):
            if dimensions[i]!=dim:
                functiondims.insert(i, 1)
                value = np.asarray(value)         # YMD bugfix 25-10-2016; previous bugfix gave problems when reshaping scalars to grids. YMD bugfix 01-06-2016; does not work for scalars
                shape = list(value.shape)
                shape.insert(i, 1)
                value = value.reshape(shape)
        return value

    def __findGridForVariable(self, keyset, grid, check=True):
        """ Find grid on which the variable with key(s) in keyset is registered. If nothing is found, fall back on the
        default grid 'grid'. Also check for availability of this grid.

        Args:
            keyset (tuple): list of key and subkeys.

        Returns:
            string name of grid
        """
        gridname = self._data['__variableOnGrid']
        for k in keyset:
            gridname = gridname.get(k)
            if gridname is None:
                gridname = grid
                break
            if not isinstance(gridname, dict):
                break

        if check:
            # some checks on the gridname and grid to provide useful error messages in case of problems
            if not isinstance(gridname, str): # raise Exception if gridname is not a string, something wrongly registered
                raise KnownError('__variableOnGrid not properly defined for variable %s: definition in '+
                                 '__variableOnGrid has more sub-dictionaries than the data itself.'%(keyset[0]))
            if self._data.get(gridname) is None:    # raise exception if grid is not available
                raise KnownError('Processing of variable %s requires grid with name %s, which does not exist'%(keyset[0], gridname))
        return gridname

    def __buildDict(self, key, data):
        d = {}
        if len(key)>1:
            d[key[0]] = self.__buildDict(key[1:], data)
        else:
            d[key[0]] = data
        return d

    def __removeKey(self,key, dictionary):
        """Utility of self.merge. Removes key from nested structure in dict.
        key is of type tuple/list
        """
        if isinstance(dictionary[key[0]], dict):
            self.__removeKey(key[1:], dictionary[key[0]])

            # check if the removal creates an empty dictionary. If so, remove it.
            if dictionary[key[0]]=={}:
                dictionary.pop(key[0])
        else:
            # dictionary.pop(key[0])
            del dictionary[key[0]]
        return

    def splitDimString(self, string, gridname):
        string_split = []
        dims = self._data[gridname]['dimensions']
        for dim in dims:
            temp = string.split(dim)
            string_split += [dim]*(len(string.split(dim))-1)
            string = ''.join(temp)      # join the split string but now without the dimensions already covered. Prevents double counting in case of dimensions with partly overlapping names (e.g. eta and t)
        if len(string):
            raise KnownError('Argument dim does not contain a concatenation of dimensions present in this grid. Left over dimensions after splitting are %s'%(string))
        return string_split
