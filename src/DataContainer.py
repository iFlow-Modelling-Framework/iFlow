"""
Class DataContainer

Data wrapping class. Contains a single dictionary with possible sub-dictionaries, sub-sub-dictionaries etc.
The DataContainer may contain five types of data:
    1. scalars or strings
    2. lists or nested lists
    3. numpy arrays
    4. references to a function or instance method

Please see the docstrings of the individual methods to get more information or consult the iFlow modelling framework
manual

Date: 05-11-15
Authors: Y.M. Dijkstra
"""
import types
from numbers import Number
import numpy as np
import nifty as nf
from src.util.diagnostics import KnownError
from src.util.interpolation import RegularGridInterpolator
from src.util.grid import convertIndexToCoordinate
from src.util import mergeDicts


class DataContainer:
    # Variables

    # Methods
    def __init__(self, *args):
        """Make a new DataContainer instance filled by dictionaries provided in the arguments

        Parameters:
            args - (optional, one or more dicts) optional dictionaries to be added to the data
        """
        self.data = {}
        for i in [j for j in args if j is not None]:
            self.data.update(i)
        return

    ####################################################################################################################
    # Methods for altering, inspecting or splitting the structure of the DataContainer
    ####################################################################################################################
    def copy(self):
        """Returns a new DataContainer instance with the same data as this one.
        So the data is not copied; only the DataContainer is different."""
        return DataContainer(self.data)

    def addData(self, name, data):
        """Add data with key=name to the DataContainer.
        NB. duplicate keys will be overwritten.

        Parameters:
            name - (string) key
            data - (any) value, pointer etc to be stored
        """
        self.data[name] = data
        return

    def merge(self, container):
        """Merges a dictionary or DataContainer with this one
        NB. duplicate keys will be overwritten, but only on the lowest (sub-)dict level

        Parameters:
            container - (DataContainer or dict) data to be merged with this container
        """
        if isinstance(container, dict):
            self.data = mergeDicts(self.data, container)
        else:
            self.data = mergeDicts(self.data, container.data)
        return

    def slice(self, key, *args, **kwargs):
        """Returns a new DataContainer instance containing the key (+subkeys in args)
        NB. does not make a copy of the data.

        Parameters:
            key & args (str) - keys and subkeys such as meant in .v()
            in kwargs:
            excludeKey (bool, optional) - if set to true, the key (and/or subkeys) will not be part of the new DataConatainer.
                                          This will only work if the value obtained from the keys is a dict. Else it raises an exception
                                          Please use this option with caution. Default: False

        Returns:
            New DataContainer instance with key provided by the final element of args or by 'key' if args is empty
        """
        value = self.data
        keys = (key,)+args
        lastKey = keys[-1]
        kwargs['dontadd'] = True    # let unwrapdictionaries know not to add data in subdicts while slicing.
        value, args, kwargs, _ = self.__unwrapDictionaries(value, key, *args, **kwargs)

        excludeKey = kwargs.get('excludeKey')
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
        keyList = self.__getKeysFromDict(self.data)
        return keyList

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

    def getKeysOf(self, key, *args):
        """Get all subkeys contained under the data with 'key' and subkeys contained in 'args'.
        The arguments key and args are not included in the returned list of keys/subkeys.
        Returns a list of subkeys, but without sub-sub-keys

        Parameters:
            key & args - (str) keys and subkeys such as meant in .v()
        """
        value = self.data[key]
        for arg in args:
            value = value[arg]
        if isinstance(value, dict):
            keyList = value.keys()
        else:
            keyList = []
        return keyList

    ####################################################################################################################
    # Methods for data retrieval: d, n, d, dd
    ####################################################################################################################
    def v(self, key, *args, **kwargs):
        """Check and retrieve a value with key 'key' and optional subkeys provided in args. Specific elements of the
        data can be retrieved using either list/array indices provided in args or dimensionless positions on the grid
        axes in the named argumenst (i.e. kwargs). In case of using indices or grid positions, the data will,
        if possible, be returned in the shape of the request.
        Example of practical use for multidimensional output:
            v(key, subkey, x=..,z=..,f=..) or v(key, subkey, .., .., ..)

        The method tries the following:
        1. Retrieve the value belonging to 'key'. Then consecutively use the optional string arguments contained in
        'args' to retrieve sub-values belonging to sub-keys. Continue untill all string arguments in 'args' have been
         used.

        2. If the value found is a dictionary structure (i.e. multiple key-value pairs). Evaluate all underlying values
        (see step 3) and then try to add the values. If addition is not possible, return True to indicate that the item
        exists, but is a data structure instead of a value.

        3. Evaluate the value found depending on the data type
            a.  List/nested list
                Use numerical arguments in 'args' (one or more integers or lists of integers) to access the list by its
                indices. The whole list is returned if args are omitted. If the number of numerical arguments in args is
                less than the dimension of the list, the full sub-list is returned in the omitted dimensions.
                Call the list as v('(optional) dicts', 'variable', index1, index2, ...)

                NB. Lists do not support direct numerical operations and access by coordinate (see numpy arrays)

            b.  numpy array:
                Either use the numerical data in args to access the element corresponding to the indices in the array
                or use kwargs to access coordinates on a grid.

                Grid:
                    The grid should be available in data as a key 'grid'. It should at least contain the 'dimensions' if
                    a call with indices (i.e. using args) is used. It should have a full grid specification if a call
                    with coordinates (i.e. kwargs) is used.

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
                    If a dimension is omitted, the data will be returned on the its original shape in that dimension.

                NB. indices/coordinates are allowed to be omitted or incomplete. It is however not advised to use this
                option for retrieving data. It may be used for checking whether a variable exists.

            c.  Reference to function or instance method:
                call the method with either grid indices or dimensionless coordinates as arguments. Calls are the same
                as for arrays. The function reference will be returned if indices/dimensions are omitted or incomplete.

            d.  Scalar/String:
                Return the value. Indices/coordinates are only used to reshape the data to the right size. This is done
                by copying the value.

        4. Reshape the data to the dimension requested (array, function, scalar, not list).  If the data is smaller than
        the request, its values are copied to obtain the right shape. If the data is larger than the request, data is
        returned in its original format; i.e. arrays/lists are returned in their original shape in the omitted
        dimensions and functions are returned as function references.

        Parameters:
            name - key of value in dictionary
            args - (optional, one or more str) keys of a hierarchy of sub-dictionaries (in the correct order).
                 - (optional, one or more int or 1D int lists/arrays) grid indices. Order of args should correspond to
                    order of grid dimensions.
            kwargs - (optional, one or more int or 1D int lists/arrays) dimensionless grid coordinates using argument
                        names corresponding to grid dimensions
                   - reshape (optional, bool) reshape result to shape of request. Default: True

        Except:
            KnownError - if 1. access by index: if index out of range or if too many indices are given
                            2. access by coordinates while grid is missing or no interpolation is implemented for this grid
                            3. method is left with a dictionary

        Returns:
            scalar, string, list or array belonging to key and optional subkeys on the requested indices/coordinates
            Returns None if the value could not be found.
            Returns True if a dictionary (i.e. set key-value pairs) was found and the underlying values could not be
            added
        """

        # preparation: remove optional argument 'reshape' from argument list and use it to set whether the data should
        # be reshaped. Default: reshape = True
        try:
            reshape = kwargs.pop('reshape')
        except:
            reshape = True

        ################################################################################################################
        ################################################################################################################
        # STEPS 1, 2
        ################################################################################################################
        ################################################################################################################
        value = self.data # use value as a running variable that is reduced step by step
        value, args, kwargs, done = self.__unwrapDictionaries(value, key, *args, **kwargs)

        # if data from sub-keys has to be summed up, the unwrapDictionaries function will internally call .v and set
        # done=True to indicate that .v does not have to be run another time. If done=True, directly return the result
        if done:
            return value

        # return None if nothing was found
        if value == []:
            return None

        # return True when a dictionary is found
        elif isinstance(value, dict):
            return True

        ################################################################################################################
        ################################################################################################################
        # STEP 3.
        ################################################################################################################
        ################################################################################################################

        ################################################################################################################
        # 3a. Lists
        ################################################################################################################
        elif isinstance(value, list) or isinstance(value, range):
            reshape = False

            # use the args as indices
            if args:
                for i in args:
                    try:
                        #value = (np.asarray(value)[i]).tolist() #removed 08-12-15, replaced by lines below
                        if isinstance(i, int):
                            value = value[i]
                        else:
                            value = (np.asarray(value)[i]).tolist()
                    except IndexError as e:
                        raise KnownError('Index out of range',e)
                    except TypeError as e:
                        raise KnownError('Tried to access a dimension that does not exist',e)

        ################################################################################################################
        # 3b. Numpy arrays
        ################################################################################################################
        elif isinstance(value, np.ndarray):
            # use the args as indices
            if args:
                try:
                    # reduce args to 0 if accessing a dimension that has only one element
                    newArgs = ()
                    for i,v in enumerate(value.shape):
                        if value.shape[i] == 1:
                           newArgs += ([0],)                # return single dimension for non-existing axes
                        elif i < len(args):
                           newArgs += (nf.toList(args[i]),)    # return data on requested points for existing axes
                        else:
                            newArgs += (range(0, v),)     # return data on original grid if dimension not provided
                    value = self.__operationsNumerical(value, *newArgs, **kwargs)       # perform any operation on the data
                except IndexError as e:
                    raise KnownError('Index out of range', e)
                except TypeError as e:
                     raise KnownError('Tried to access a dimension that does not exist', e)

            # otherwise use kwargs if not empty
            else:
                value = self.__operationsNumerical(value, **kwargs)
                if kwargs:
                    if self.data['grid'].get('gridtype') == 'Regular':
                        # interpolation on a regular grid
                        interpolator = RegularGridInterpolator()
                        value = interpolator.interpolate(value, self, **kwargs)
                    else:
                        # give an error message if the grid type has no data interpolation method implemented
                        raise KnownError('Access to gridded data not implemented on a grid of type %s' % (self.data['grid']['gridtype']))

        ################################################################################################################
        # 3c. Functions and methods
        ################################################################################################################
        elif isinstance(value, types.MethodType) or isinstance(value, types.FunctionType):
            # try to access the function value. However if too few arguments are given, return the reference to function
            coordinates = kwargs.copy()
            if args:
                coordinates.update(convertIndexToCoordinate(self.data['grid'], args))

            [coordinates.update({dim:np.asarray(nf.toList(coordinates[dim]))}) for dim in value.__self__.dimNames if coordinates.get(dim) is not None]    # convert called dimensions to array. i.e. function will not be called with scalar arguments
            functiondims = value.__self__.dimNames
            value = value(reshape=reshape, **coordinates) # function call with coordinates and operations in **coordinates
            if isinstance(value, types.MethodType) or isinstance(value, types.FunctionType):
                reshape = False
            else:
                # before full reshape, first reshape so that shape matches the order of 'dimensions'
                value = self.__reshapeFunction(value, functiondims)

        ################################################################################################################
        # 3d. Scalars, strings & other items
        ################################################################################################################
        else:
            value = self.__operationsNumerical(value, **kwargs)

        ################################################################################################################
        ################################################################################################################
        # STEP 4
        ################################################################################################################
        ################################################################################################################
        # remove operation
        try:
            kwargs.pop('operation')
        except:
            pass

        # reshape
        if reshape:
            value = self.__reshape(value, *args, **kwargs)

        return value

    def n(self, key, *args, **kwargs):
        """Returns the value accessed by .v() multiplied -1. Only for arrays, scalars and functions (not lists)
        See documentation of .v().
        """
        kwargs['operation'] = 'n'
        value = self.v(key, *args, **kwargs)
        return value

    def d(self, key, *args, **kwargs):
        """Returns the derivative of the value accessed by .v(). See documentation .v()

        Parameters:
            same as in .v(). Additionally:
            kwargs['dim'] (str, int) - NB not optional! Axis along which to take the derivative. Provide multiple
                                        characters to get higher derivatives. Mixed derivatives also allowed, but order
                                        of derivation should not matter
        """
        kwargs['operation'] = 'd'
        if kwargs.get('dim') is None:
            raise KnownError('Called .d() method without supplying the axis of derivation.'
                             '\n Please add a dim=(str or int) argument.')
        value = self.v(key, *args, **kwargs)
        return value

    ### METHOD dd DEPRECIATED FROM V2.2 (02-03-2016) [dep01]
    def dd(self, key, *args, **kwargs):
        """Returns the second derivative of the value accessed by .v(). See documentation .v().
        METHOD DEPRECIATED FROM v2.2 [dep01]

        Parameters:
            same as in .v(). Additionally:
            kwargs['dim'] (str, int) - NB not optional!
                                       axis along which to take the derivative
        """
        kwargs['operation'] = 'dd'
        if kwargs.get('dim') is None:
            raise KnownError('Called .dd() method without supplying the axis of derivation.'
                             '\n Please add a dim=(str or int) argument.')
        value = self.v(key, *args, **kwargs)
        return value
    ### END ###

    ####################################################################################################################
    # Private methods
    ####################################################################################################################
    def __unwrapDictionaries(self, value, key, *args, **kwargs):
        """Utility of self.v.
        Unwrap the dictionary using the key and the subkeys in the optional args.
        If the key/subkeys return a dictionary, it will try to add all the values in this dictionary.
        If this fails, the the method will return the found dictionary
        """
        done = False

        keyset = [key]+[i for i in args if isinstance(i, str)]
        indexset = [i for i in args if not isinstance(i, str)]
        for i, key in enumerate(keyset):
            if key in value:        # if the key is in the dictionary
                value = value[key]  # step into dictionary
            else:
                value = []

        if isinstance(value, dict) and not kwargs.get('dontadd'):
            try:
                value, done = self.__addvalues(value, *indexset, **kwargs)
            except:
                pass

        return value, indexset, kwargs, done

    def __addvalues(self, dictionary, *args, **kwargs):
        """Utility of unwrapDictionaries. Sums all values in a dictionary and possible sub-dictionaries.
         NB. Does not guarantee that this is possible. An error will be thrown if summation is impossible.
        """
        done = False
        value = None
        for dictValue in dictionary.values():
            # if the dictionary contains sub-dictionaries: further unwrap these
            if isinstance(dictValue, dict):
                dictValue, done = self.__addvalues(dictValue, *args, **kwargs)
                if value is None:
                    from copy import copy
                    value = copy(dictValue)
                else:
                    value = value + dictValue           #YMD 20-12-2016: old version 'value+= dictValue' introduced errors when summing complex numbers to a real value
            # add the values from the dictionary and subdictionaries if these contain arrays/numbers
            elif isinstance(dictValue, np.ndarray) or isinstance(dictValue, Number) or isinstance(dictValue, types.MethodType) or isinstance(dictValue, types.FunctionType):
                done = True     # indicates that .v method is no longer required anymore; return the value immediately
                dc = DataContainer({'key': dictValue})
                try:
                    dc.addData('grid', self.data['grid'])
                except:
                    pass

                if value is None:
                    from copy import copy
                    value = copy(dc.v('key', *args, **kwargs))
                else:
                    value = value + dc.v('key', *args, **kwargs) #YMD 20-12-2016: old version 'value+= dc.v(...)' introduced errors when summing complex numbers to a real value
            else:               # YMD 24-01-2018: break if there are non-numerical values in the array: cannot sum values
                value = None
                break
        if value is None:
            value = dictionary
        return value, done

    def __operationsNumerical(self, value, *args, **kwargs):
        """Utility of .v().
        Deals with operations, such a derivation and integration on numerical data (scalar, array).
        Returns the value at the specified indices if no operation is prescribed

        Parameters:
            value (scalar or ndarray) - value containing the data
            args (lists, ndarray) - indices for each dimension.
                                    NB. these should already be tailormade (e.g. excess dimensions cut off)
                                    If no args are provided, the data is returned on the original grid.
            kwargs['operation'] (string, optional) - specification of operation
                                            n: negative
                                            d: derivative
                                            dd: second derivative
            kwargs['dim'] (string, int, optional) - axis to take the operation over (if applicable).
                                                    Can be given as dimension number or dimension name.

        Exception:
            KnownError if the data cannot be accessed using the operation

        Returns:
            value with operation executed or raise an exception
        """
        if kwargs.get('operation'):
            if kwargs.get('operation') == 'n':
                if args:
                    value = -value[np.ix_(*args)]
                else:
                    value = -value
            if kwargs.get('operation') == 'd':
                dim = kwargs.get('dim')
                import numbers
                for dir in list(set(sorted(dim))): # loop over all dimensions once
                    if isinstance(value, numbers.Number):   # return 0 for derivative of constant
                        value = 0.
                    else:
                        order = len([i for i in dim if i==dir]) # collect the number of occurances of this dimension
                        if order == 1:
                            value = nf.derivative(value, dir, self.slice('grid'), *args, DERMETHOD = kwargs.get('DERMETHOD'))
                        elif order == 2:
                            value = nf.secondDerivative(value, dir, self.slice('grid'), *args)
                        else:
                            raise KnownError('Numerical derivatives of order %s are not implemented' % str(order))
            ### Depreciated since v2.2 (02-03-2016) [dep01] ###
            if kwargs.get('operation') == 'dd':
                dim = kwargs.get('dim')
                value = nf.secondDerivative(value, dim, self.slice('grid'), *args)
            ### End ###
        else:
            if args:
                value = value[np.ix_(*args)]
        return value
    ####################################################################################################################
    # Old version of operationsnumerical. Obsolete since v2.2 (02-03-16) [dep01]
    ####################################################################################################################
    # def __operationsNumerical(self, value, *args, **kwargs):
    #     """Utility of .v().
    #     Deals with operations, such a derivation and integration on numerical data (scalar, array).
    #     Returns the value at the specified indices if no operation is prescribed
    #
    #     Parameters:
    #         value (scalar or ndarray) - value containing the data
    #         args (lists, ndarray) - indices for each dimension.
    #                                 NB. these should already be tailormade (e.g. excess dimensions cut off)
    #                                 If no args are provided, the data is returned on the original grid.
    #         kwargs['operation'] (string, optional) - specification of operation
    #                                         n: negative
    #                                         d: derivative
    #                                         dd: second derivative
    #         kwargs['dim'] (string, int, optional) - axis to take the operation over (if applicable).
    #                                                 Can be given as dimension number or dimension name.
    #
    #     Exception:
    #         KnownError if the data cannot be accessed using the operation
    #
    #     Returns:
    #         value with operation executed or raise an exception
    #     """
    #     if kwargs.get('operation'):
    #         if kwargs.get('operation') == 'n':
    #             if args:
    #                 value = -value[np.ix_(*args)]
    #             else:
    #                 value = -value
    #         if kwargs.get('operation') == 'd':
    #             dim = kwargs.get('dim')
    #             value = nf.derivative(value, dim, self.slice('grid'), *args)
    #         if kwargs.get('operation') == 'dd':
    #             dim = kwargs.get('dim')
    #             value = nf.secondDerivative(value, dim, self.slice('grid'), *args)
    #     else:
    #         if args:
    #             value = value[np.ix_(*args)]
    #     return value
    ####################################################################################################################

    def __reshape(self, value, *args, **kwargs):
        """Reshape value to shape prescribed by args or kwargs. However,
        a) excess dimensions are trimmed off
        b) if the data has more dimensions than the request, these dimensions are added

        Parameters:
            value (array, list, scalar) - data
            args/kwargs - indices or coordinates as in self.v()
        """
        if not args and not kwargs:
            return value

        # determine shape of request
        if args:
            shape = [len(nf.toList(n)) for n in args]
            shapeTruncated = [len(nf.toList(n)) for n in args if isinstance(n, list) or isinstance(n, range) or isinstance(n, np.ndarray)]
        elif kwargs:
            shape = []
            shapeTruncated =[]
            for i, dim in enumerate(self.data['grid']['dimensions']):
                n = kwargs.get(dim)
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

        #   then recast into right shape (changes in v2.4)
        try:
            copy = self.data['grid']['copy'][:len(shape)]   # works in version 2.4
        except:
            if self.data.get('grid') is None:
                copy = np.ones(len(shape))
            else:
                copy = np.ones(len(self.data['grid']['dimensions']))[:len(shape)]       # backward compatibility to v2.3 [dep03]
        extensionMatrix = np.ones(shape)

        if not kwargs.get('copy') == 'all':
            for i, item in enumerate(copy):
                if item == 0 and shape[i] > 1 and value.shape[i] == 1:
                    extensionMatrix[(slice(None),)*i + (slice(1, None),)+(Ellipsis,)] = 0

        value = value*extensionMatrix
        value = value.reshape(shapeTruncated)

        return value

    def __reshapeFunction(self, value, functiondims):
        """reshape function output to be conform the order of the dimensions.
        """
        functiondims = functiondims[:]  # make a copy of the list
        dimensions = self.data['grid']['dimensions']
        for i, dim in enumerate(functiondims):
            if dimensions[i]!=dim:
                functiondims.insert(i, 1)
                value = np.asarray(value)         # YMD bugfix 25-10-2016; previous bugfix gave problems when reshaping scalars to grids. YMD bugfix 01-06-2016; does not work for scalars
                shape = list(value.shape)
                shape.insert(i, 1)
                value = value.reshape(shape)
        return value

