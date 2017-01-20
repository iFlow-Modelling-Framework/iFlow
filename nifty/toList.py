"""If item is not a list/array wrap it in a list with this single item. However, if scalar is a list or array, do not change anything
    if item is None, then create an empty list

    Parameters:
        scalar - anything
    Returns:
        List wrapping the input item if it was not already a list/array
"""
import numpy as np


def toList(item, forceNone=False):
    if not isinstance(item, list) and not isinstance(item, np.ndarray):
        if item is None:
            if forceNone:
                item = [None]
            else:
                item = []
        else:
            item = [item]

    return item


