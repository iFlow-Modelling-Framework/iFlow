"""
scalemax scales the array with the maximum value in that array

Date: 24-Nov-16
Authors: R.L. Brouwer
"""


def scalemax(value):
    if value.max() == 0:
        ret = value
    else:
        ret = value / value.max()
    return ret
