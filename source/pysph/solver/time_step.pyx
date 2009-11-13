"""
Contains a class to hold the time step.
"""

from pysph.solver.base cimport Base

################################################################################
# `TimeStep` class.
################################################################################
cdef class TimeStep(Base):
    """
    Class to hold the current timestep.

    Make this a separate class makes it easy to reference one copy of it at all
    places needing this value.

    """
    def __cinit__(self, double value=0.0, *args, **kwargs):
        """
        Constructor.
        """
        self.value = value
