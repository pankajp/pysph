"""
Base class for all physical entities involved in the simulation.
"""
from pysph.base.carray cimport BaseArray
from pysph.base.particle_array cimport ParticleArray

from pysph.solver.base cimport Base

################################################################################
# `EntityBase` class.
################################################################################
cdef class EntityBase(Base):
    """
    Base class for any physical entity involved in a simulation.    
    """
    # properties whose value is the same for the whole entity.
    cdef public dict properties

    # unique type identifier for the entity.
    cdef public int type

    # name of the entity
    cdef public str name

    # add a property common to the whole entity
    cpdef add_property(self, str prop_name, double default_value=*)

    # check if the entity is of type etype.
    cpdef bint is_a(self, int etype)

    # function to return the set of particles representing the entity.
    cpdef ParticleArray get_particle_array(self)


################################################################################
# `Fluid` class.
################################################################################
cdef class Fluid(EntityBase):
    """
    Base class for all fluids.
    """
    cdef ParticleArray particle_array
    

################################################################################
# `Solid` class.
################################################################################
cdef class Solid(EntityBase):
    """
    Base class for all solids.
    """
    pass

# cdef class Solid(EntityBase):
#     """
#     Base class to describe solids that do not deform.

#     **Members**
#     List of possible members of this class.
    
#      - Center of mass - a point in 3d.
#      - Angular velocity - a 3-vector.
#      - linear velocity - a 3-vector - this velocity is only valid if the body is
#        undergoing pure translational motion. Otherwise the linear velocity will
#        change for each particle in the solid.                - pure rotational.
#          - some complex motion involving both translation and rotational.

#      - Variable to indicate what is the primary representation of the solid, for
#        example is the polygonal representation the main rep or is the point
#        representation the main representation. A related question, should we
#        always have a polygonal representation of the solid ? This sounds ~neat~-
#        do not have the right word for how this sounds, but sounds good. And the
#        particle representation is auxilarry, will be used by some algorithms.

#            - So, the particle representation will purely be used for SPH related
#              tasks, meaning that those will be interpolation points used in some
#              SPH interpolation. Hence they will have to be tracked by the cell
#              manager. These may appear as dests or sources in some SPH calc,
#              function.
#            - The polygonal representation (which is more logical for a solid),
#              will be used for geometric queries like:

#                  - Collision of a particle and the solid geometry.
#                  - Finding of surface normals.
#                  - Finding of tangent planes as needed by the morris97 viscosity
#                    algorithm. 
                 
#      - Functions needed of a solid object.
#          - intersection of ray and this solid.
#              - this may only be valid for solids that can have a polygonal
#                representation.
#          - 
         
#     """
#     cdef class PolygonArray polygons
#     cdef class ParticleArray particles
    
    
