"""
mergeDicts

Date: 16-02-2022
Authors: Y.M. Dijkstra
"""


def mergeDicts(d1, d2):
        """Modifies d1 in-place to contain values from d2.  If any value in d1 is a dictionary and the corresponding
        value in d2 is also a dictionary, then merge them in-place. Overwrites values of d1 on the lowest sub-dict level
        if double values occur

        """
        for k,v2 in d2.items():
            v1 = d1.get(k)  # returns None if v1 has no value for this key
            if isinstance(v1, dict) and isinstance(v2, dict):
                mergeDicts(v1, v2)

            elif(v1.__class__.__name__=='DataContainer' and v2.__class__.__name__=='DataContainer'):
                v1.merge(v2)

            elif isinstance(v2, dict):
                d1[k] = {}
                mergeDicts(d1[k], v2)

            else:
                d1[k] = v2

        return d1
