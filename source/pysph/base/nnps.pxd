# local imports
from pysph.base.carray cimport LongArray, DoubleArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.polygon_array cimport PolygonArray
from pysph.base.cell cimport CellManager
from pysph.base.point cimport Point, cPoint, cPoint_distance2
from pysph.base.kernels cimport KernelBase


##############################################################################
# `Classes for nearest particle location`.
##############################################################################
cdef class NbrParticleLocatorBase:
    """Base class for all neighbor particle locators. """
    cdef public ParticleArray source
    cdef public CellManager cell_manager
    cdef public int source_index

    cdef public dict kernel_function_evaluation
    cdef public dict kernel_gradient_evaluation

    cdef public list function_cache
    cdef public list xgradient_cache
    cdef public list ygradient_cache
    cdef public list zgradient_cache

    cdef int get_nearest_particles_to_point(self, cPoint pnt, double radius,
                                            LongArray output_array, 
                                            long exclude_index=*) except -1
    cdef int _get_nearest_particles_from_cell_list(
        self, cPoint pnt, double radius, list cell_list,
        LongArray output_array, long exclude_index=*) except -1


cdef class FixedDestNbrParticleLocator(NbrParticleLocatorBase):
    """
    Particle locator, where all particle interactions have the destination
    point in a fixed particle array. This implementation assumes all particles
    to have the same interaction radius.
    """
    cdef public ParticleArray dest
    cdef readonly double radius_scale
    cdef public int dest_index
    cdef readonly str h
    cdef public DoubleArray d_h, d_x, d_y, d_z
    
    # caching support
    cdef public list particle_cache
    cdef public bint is_dirty
    cdef void update_status(self)
    cdef int update(self) except -1
    cdef int _update_cache(self) except -1
    
    cdef LongArray get_nearest_particles(self, long dest_p_index,
                                   bint exclude_self=*)
    
    cdef int get_nearest_particles_nocache(self, long dest_p_index,
                                   LongArray output_array,
                                   bint exclude_self=*) except -1

cdef class VarHNbrParticleLocator(FixedDestNbrParticleLocator):
    """
    Particle locator, where different particles can have different interaction
    radius.
    """
    cdef FixedDestNbrParticleLocator _rev_locator

##############################################################################
# `Classes for nearest polygon location`.
##############################################################################
cdef class NbrPolygonLocatorBase:
    pass

cdef class FixedDestinationNbrPolygonLocator(NbrPolygonLocatorBase):
    pass

cdef class CachedNbrPolygonLocator(FixedDestinationNbrPolygonLocator):
    pass

cdef class CachedNbrPolygonLocatorManager:
    pass

##############################################################################
# `NNPSManager` class.
##############################################################################
cdef class NNPSManager:
    """Class to manager all nnps related information."""
    cdef public CellManager cell_manager

    cdef public bint variable_h
    cdef public str h
    cdef public dict particle_locator_cache

    cdef public CachedNbrPolygonLocatorManager polygon_cache_manager

    cpdef add_interaction(self, ParticleArray source, ParticleArray dest,
                             double radius_scale)

    cpdef FixedDestNbrParticleLocator get_cached_locator(self, str source_name,
                                        str dest_name, double radius_scale)

    cpdef NbrParticleLocatorBase get_neighbor_particle_locator(self,
            ParticleArray source, ParticleArray dest=*, double radius_scale=*)

    cpdef NbrPolygonLocatorBase get_neighbor_polygon_locator(self,
            PolygonArray source, ParticleArray dest=*, double radius_scale=*)

    cdef int update(self) except -1
    
    cpdef cache_neighbors(self, KernelBase kernel)
