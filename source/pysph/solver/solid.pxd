"""
Classes to describe solids (rigid bodies).
"""

# local imports
from pysph.base.point cimport Point

from pysph.solver.entity_base cimport *
from pysph.solver.geometry cimport GeometryBase

cdef class SolidMotionType:
    """
    Class to hold values to indentify the type of motion a solid is expected to
    undergo.
    """
    pass

cdef class Solid(EntityBase):
    """
    """
    # particle representing the solid for SPH computations.
    cdef public ParticleArray sph_particles

    # particles used to track polygons in the solids geometry.
    cdef public ParticleArray tracker_particles

    # the geometry of the solid.
    cdef public GeometryBase geometry

    # the type of motion the solid can undergo.
    cpdef public int motion_type

    # return the sph_particles particle array.
    cpdef ParticleArray get_particle_array(self)

    # find the relative velocity of the given particle(position and velocity as
    # input parametrs) with respect to the solid.
    cpdef get_relative_velocity(self, Point pos, Point vel, Point result)

    cdef void _get_relative_velocity_linear_motion(self, Point pos, Point vel,
                                                   Point res)



cdef class RigidBody(EntityBase):
    """
    Class to represent rigid bodies.

    It can be considered an improvement over the Solid class above.
    """
    cdef public ParticleArray particles
    
    cpdef ParticleArray get_particle_array(self)
