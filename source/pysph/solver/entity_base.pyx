"""
Classes to represent any physical entity in a simulation.
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
import types

################################################################################
# `EntityBase` class.
################################################################################
cdef class EntityBase:
    """
    Base class for any physical entity involved in a simulation.

    """
    #Defined in the .pxd file
    #cdef public int type
    #cdef public str name

    def __cinit__(self, str name ='', *args, **kwargs):
        """ C Constructor.

        Parameters:
        ----------
        name -- The name of the entity.

        Notes:
        ------
        The type of the class defaults to EntityTypes.Entity_Base

        """
        self.name = name
        self.type = EntityTypes.Entity_Base

    cpdef ParticleArray get_particle_array(self):
        """ Return the particle_array representing this entity.

        Reimplement this in the concrete entity types, which may or may not have
        particle representation of themselves.

        """
        return None

    cpdef bint is_a(self, int type):
        """ Check the type of the entity. """
        if EntityTypes.Entity_Base == type: return True

    cpdef bint is_type_included(self, list types):
        """ Returns true if the entities type or any of its
        parent types is included in the the list of types passed.

        """
        for e_type in types:
            if self.is_a(e_type):
                return True

        return False

###############################################################################
