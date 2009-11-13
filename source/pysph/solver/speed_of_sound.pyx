"""
Module to hold class containing the speed of sound.
"""

cdef class SpeedOfSound:
    """
    Class holding the speed of sound to be used for some simulation.

    This value will typically be used across the solver and multiple points, and
    any change in this value should be reflected at all these points. Thus one
    instance of this class is created and references of this are passed to
    different classes needing this.

    """
    def __cinit__(self, double value=1.0, *args, **kwargs):
        """
        Constructor.
        """
        self.value = value
