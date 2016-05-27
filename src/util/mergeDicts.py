"""
mergeDicts

Date: 15-12-15
Authors: Edward Loper, http://stackoverflow.com/questions/10703858/python-merge-multi-level-dictionaries
"""
import collections

def mergeDicts(d1, d2):
        """Modifies d1 in-place to contain values from d2.  If any value in d1 is a dictionary and the corresponding
        value in d2 is also a dictionary, then merge them in-place. Overwrites values of d1 on the lowest sub-dict level
        if double values occur

        From Edward Loper, http://stackoverflow.com/questions/10703858/python-merge-multi-level-dictionaries
        """
        for k,v2 in d2.items():
            v1 = d1.get(k)  # returns None if v1 has no value for this key
            if ( isinstance(v1, collections.Mapping) and
                 isinstance(v2, collections.Mapping) ):
                mergeDicts(v1, v2)
            else:
                d1[k] = v2

        return d1