"""
NumericalFunctionWrapper
Implementation of NumericalFunctionBase for an arbitrary number of dimensions with arbitrary names.
This class is useful for storage and access of an already computed variable (with its derivative or other derived quantity)
and an already available grid.

Load the variable and grid to the NumericalFunctionWrapper on instantiating. Derivatives and other derived quantities
can be 'uploaded' by invoking addDerivative or related methods.

Example:
    Let 'value' be the value of a variable to be stored and let 'DC' be a DataContainer instance containing a grid for 'value'.
    Let furthermore 'valueDerivative' be the already calculated derivative of 'value'
    Then make a numerical function wrapper by:
    >> nf = NumericalFunctionWrapper(value, DC)
    >> nf.addDerivative(valueDerivative)
    >> fun = nf.function

Date: 23-07-15
Authors: Y.M. Dijkstra
"""
from .NumericalFunctionBase import NumericalFunctionBase


class NumericalFunctionWrapper(NumericalFunctionBase):
    # Variables

    # Methods
    def __init__(self, value, gridData, gridName='grid'):
        # if grid data is a dictionary, first convert to datacontainer
        if isinstance(gridData, dict):
            from src.DataContainer import DataContainer
            dc = DataContainer()
            dc.addData(gridName, gridData)
            gridData = dc
            del dc

        # check if a grid is provided in DC data
        dimensions = gridData.v(gridName, 'dimensions')
        dimNames = [dimensions[i] for i in range(0, min(len(value.shape), len(dimensions)))]
        NumericalFunctionBase.__init__(self, dimNames)

        # add grid (with name 'grid') and value
        self.addGrid(gridData, gridName='grid')
        self.addValue(value)
        return
