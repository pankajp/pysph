"""
Contains a class to hold the time step.
"""
# local imports


################################################################################
# `TimeStep` class.
################################################################################
cdef class TimeStep:
    """
    Class to hold the current timestep.

    Making this a separate class makes it easy to reference one copy of it at
    all places needing this value.
    """
    cdef public double value
