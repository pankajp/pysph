"""
Module contains some common functions.
"""


def extract_entity_names(entity_list):
    """
    Returns the names of all entities in the give list in one string.
    """
    l = map(lambda x: x.name, entity_list)
    print l
    r = l[0]
    for n in l[1:]:
        r += '_'+n
    return r
