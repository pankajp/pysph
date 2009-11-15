"""
Classes to describe solids (rigid bodies).
"""

# standard imports
import logging
logger = logging.getLogger()


# local imports
from pysph.base.point cimport Point
from pysph.solver.entity_base cimport *
from pysph.solver.entity_types cimport EntityTypes



################################################################################
# `SolidMotionType` class.
################################################################################
cdef class SolidMotionType:
    """
    Class to hold values to identify the type of motion a solid can undergo.
    """
    # meaning can move only in a straight line. 
    ConstantLinearVelocity = 0
    
    # no rotational motions, but velocity can change along a straight line.
    LinearlyAcceleration = 1
    
    # can rotate about a fixed axis with constact velocity.
    ConstantAngularVelocity = 2

    # can undergo any kind of motion.
    AngularAcceleration = 3

    def __cinit__(self, *args, **kwargs):
        """
        """
        raise SystemError, 'Do not instantiate this class'

################################################################################
# `Solid` class.
################################################################################
cdef class Solid(EntityBase):
    """
    Class to represent solids (rigid bodies).
    """
    def __cinit__(self, str name='', dict properties={}, 
                  ParticleArray particles=None, 
                  motion_type=SolidMotionType.AngularAcceleration, 
                  *args, **kwargs):
        """
        Constructor.
        """
        self.sph_particles = particles
        self.type = EntityTypes.Entity_Solid

        if self.sph_particles is None:
            self.sph_particles = ParticleArray()

        # add any default properties that must be there for solids.        
        
    cpdef bint is_a(self, int type):
        """
        Check if this entity is of the given type.
        """
        return (EntityTypes.Entity_Solid == type or
                EntityBase.is_a(self, type))

    cpdef ParticleArray get_particle_array(self):
        """
        """
        return self.sph_particles

    cpdef get_relative_velocity(self, Point pos, Point vel, Point
                                result):
        """
        Gets the relative velocity of the said particle(identified by pos and
        vel), with respect to the this solid and stores the values in result.
        """
        if self.motion_type > SolidMotionType.LinearlyAcceleration:
            msg = 'Can compute relative velocity (currently) only for \n'
            msg += 'bodies moving with constant velocity or constant '
            msg += 'linear acceleration'
            logger.warn('msg')
            return
        else:
            self._get_relative_velocity_linear_motion(pos, vel, result)

    cdef void _get_relative_velocity_linear_motion(self, Point pos, Point vel,
                                                   Point res):
        """
        Get a particle from the geometry base of the solid. Use its u,v and w as
        the linear velocity of the solid, and find the relative velocity.

        """
        pass
