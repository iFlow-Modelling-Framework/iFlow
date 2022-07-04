from src.util.diagnostics.KnownError import KnownError

def checkVariables(classname, *args):
        """Check if variables are set.
        For each variable this will check if the variable is different from None.
        An error is raised if the variable is None

        Parameters:
            args (tuples of two elements: (str, any)) - set of tuples of two elements.
                        The first element describes the variable name in a string, the second is the value of any type
        """
        for pair in args:
            if pair[1] is None:
                message = ("Not all required variables are given. Missing '%s' in module '%s'" % (pair[0], classname))
                raise KnownError(message)
        return