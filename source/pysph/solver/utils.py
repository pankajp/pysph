"""
Module contains some common functions.
"""

# standard imports
import numpy

def extract_entity_names(entity_list):
    """
    Returns the names of all entities in the give list in one string.
    """
    l = map(lambda x: x.name, entity_list)
    r = l[0]
    for n in l[1:]:
        r += '_'+n
    return r

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)
