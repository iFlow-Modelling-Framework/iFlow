"""
splitModuleName

Date: 23-Jul-15
Authors: Y.M. Dijkstra
"""


def splitModuleName(compositeName, separator='.'):
    """Splits a composite name or path as follows
    1) package1.package2.class => package1.package2 & class
    2) dir1/dir2/subdir => dir1.dir2 & subdir
    If separator is its default '.'
    Separator can be set to ., / or \\ to change the separation marker in the result

    If the compositeName does not contain ., \\ or //, the first return variable is empty and the second returns compositeName

    Parameters:
        compositeName (str) - name separated by dots or a path (separated by \\ or /)
        separator ('.', '/' or '\\') - separation character in end result

    Result:
        see above
    """

    compositeName.replace('\\', '.')
    compositeName.replace('/', '.')
    compositeName = compositeName.split('.')

    name = compositeName[-1]
    package = separator.join(compositeName[:-1])

    return package, name