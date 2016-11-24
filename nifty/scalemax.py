"""
scalemax scales the array with the maximum value in that array

Date: 24-Nov-16
Authors: R.L. Brouwer
"""


def scalemax(value):
    return value / (value.max() * 1000.)
