"""
isFloat

Date: 15-12-15
Authors: Y.M. Dijkstra, R.L. Brouwer
"""
def isInt(value):
    """Check if 'value' can be represented by a float. Return True if this is possible, else return False.
    """
    try:
        int(value)
        return True
    except ValueError:
        return False