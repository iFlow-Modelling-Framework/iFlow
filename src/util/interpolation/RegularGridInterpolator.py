"""
Class RegularGridInterpolator
Data interpolation on a regular grid
The interpolation method can interpolate data in a list or array from a given set of coordinates.
Therefore it requires a DataContainer that contains a regular grid, i.e. variables 'dimensions', and for each dimension
in this list a 'grid' (single axis) and optional enclosure consisting of keys 'low' and 'high'

Date: 23-05-15
Authors: Y.M. Dijkstra
"""
import logging
import numpy as np
from nifty import toList
from src.util.diagnostics import KnownError
from src.util.interpolation import Intergrid


class RegularGridInterpolator:
    # Variables
    logger = logging.getLogger(__name__)

    # Methods
    def __init__(self):
        return

    def __cartesian(self, arrays):
        """Computes all permutations of numbers in the prodided 1D arrays. (~10x faster than itertools.product)
        By Stefan van der Walt, from http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

        Parameters:
            arrays: (list of 1D array-like objects)

        Returns:
            list of all permutations
        """
        arrays = [np.asarray(a) for a in arrays]
        shape = (len(x) for x in arrays)

        ix = np.indices(shape, dtype=int)
        ix = ix.reshape(len(arrays), -1).T

        result = np.zeros(ix.shape)
        for n in range(0, len(arrays)):
            result[:, n] = arrays[n][ix[:, n]]

        return result

    def interpolate(self, value, dataContainer, **kwargs):
        """Compute/find values from an array by coordinates in kwargs on a grid contained in the dataContainer.
        The method will only interpolate if the data is compatible with the grid, i.e. has dimensions equal to
        the grid axis dimension or a dimension of 1.

        access data in an array by coordinates listed in kwargs see documentation in self.v
        Steps:
        1. check if value is compatible with the grid. Else, return value unchanged
        2. retrieve axes and arguments (from kwargs) per dimension
            convert the coordinates in kwargs to points on the [0, 1] axes for all enclosed axes
        3. interpolate
        """
        dimensionList = dataContainer.v('grid', 'dimensions')  # list of dimensions in registry

        # 1. check dimensions. Stop interpolating if the dimensions are not fit
        for i, v in enumerate(value.shape):
            if i < len(dimensionList):
                dim = dimensionList[i]
                if not (v == 1 or v == dataContainer.v('grid', 'axis', dim).shape[i]):
                    raise KnownError('Tried to interpolate data that is not conform the dimensions of the grid')

        # 2. retrieve kwargs arguments and convert to numbers on the [0,1] axes if enclosures are provided
        #   2a. initialise
        axis = []
        lo = []
        hi = []
        samplePoints = []

        #   2b. prepare axes and add sample points
        for i, v in enumerate(value.shape):     # process all dimensions of the array
            if value.shape[i] == 1:                 # singular dimension
                # if dimension is a dummy dimension (i.e. length 1), do not interpolate over it.
                samplePoints.append([0])
                axis.append([0])
                lo.append(0)
                hi.append(0)
            elif i < len(dimensionList):            # dimension fits within domain of axes
                # read argument and axis
                dim = dimensionList[i]
                argument = kwargs.get(dim)
                grid = dataContainer.v('grid', 'axis', dim)
                axis.append(grid.reshape(grid.shape[i]))    # NB. only works for a single grid axis
                lo.append(axis[-1][0])
                hi.append(axis[-1][-1])

                if argument is None:
                    # if argument is not given by the user, take all points on the original grid
                    samplePoints.append(toList(axis[-1]))
                else:
                    # if argument is given (as scalar or list/array) add the points
                    samplePoints.append(toList(argument))
            else:                               # dimension beyond axes; take data on original grid
                grid = range(0, v)
                axis.append(grid)
                lo.append(axis[-1][0])
                hi.append(axis[-1][-1])
                samplePoints.append(toList(axis[-1])) # add all points to the sample points

        #   2c. rewrite sample point to all permutations
        sampleShape = [len(toList(j)) for j in samplePoints]    # shape of sample points
        samplePoints = self.__cartesian(samplePoints)

        #   2d. for each axis, check if it is enclosed. If so, convert all sample points to [0,1] on this axis
        """ # NB. COMMENTED OUT, call data using values between 0 and 1 for all axes except axes without enclosure (e.g. integer axes)
        for i, dim in enumerate(dimensionList):
            if i < len(value.shape):
                low = dataContainer.v(dim, 'low')  # lower grid enclosure (loaded here to test if it exists)
                argument = kwargs.get(dim)
                if low!=None and value.shape[i] >= 1 and (not argument == None):
                    # if axis is enclosed and is not a dummy axis and its argument is set
                    for j in range(0, len(samplePoints)):
                        newargs = {}
                        for k in range(0, len(samplePoints[j])):
                            newargs[dimensionList[k]] = samplePoints[j][k]

                        minAxis = float(dataContainer.v(dim, 'low', **newargs))            #TODO CHANGE BECAUSE IT SLOWS THE CALCULATION DOWN SIGNIFICANTLY
                        maxAxis = float(dataContainer.v(dim, 'high', **newargs))           #TODO CHANGE BECAUSE IT SLOWS THE CALCULATION DOWN SIGNIFICANTLY

                        samplePoints[j][i] = (samplePoints[j][i]-minAxis)/(maxAxis-minAxis)
        """

        # 3. interpolate
        interfunc = Intergrid(
            value,
            lo=lo, hi=hi,
            maps=axis,
            verbose=0
            )
        value = interfunc(samplePoints).reshape(sampleShape)




        return value
