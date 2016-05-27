"""
nestedListToDictionary

Date: 15-12-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
from src.util import isFloat
from src.util import isInt


def nestedListToDictionary(nestedList):
    """Convert a nested list structure to a dictionary with a similar structure. The first element of each list is taken
    as key in the dictionary. Nested lists are converted to sub-dictionaries.

    Parameters
        nestedList ((nested) list) - structure to convert
    Returns
        dictionary representing the nestedList
    """
    # strip empty lines
    d = [t for t in nestedList if t!=[]]
    
    # If argument is a sinlge entry, do not make a dictionary
    if len(d)==1:
        if len(d[0])==1:
            d = d[0][0]
            return d

    # Else: convert to dictionary
    d = {k[0]: k[1:] for k in d}

    # convert types
    for i in d:
        if any(isinstance(j, list) for j in d[i]): # check if d[i] is a nested list
            d[i] = nestedListToDictionary(d[i][0])
        else:
            for j in range(len(d[i])):
                if isInt(d[i][j]):            #check if int
                    d[i][j] = int(d[i][j])
                elif isFloat(d[i][j]):            #check if float
                    d[i][j] = float(d[i][j])
            if len(d[i])==1:                    #remove list if only one element
                d[i] = d[i][0]
    return d