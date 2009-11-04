# local imports
from pysph.base.carray cimport LongArray, DoubleArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.polygon_array cimport PolygonArray
from pysph.base.cell cimport CellManager
from pysph.base.point cimport Point

cdef class CellCache:
    """
    Class to maintain list of neighboring cells for every particle in a given
    particle array.
    """
    cdef public ParticleArray p_array
    cdef public bint is_dirty
    cdef public list cache_list
    cdef public CellManager cell_manager
    cdef public bint single_layer
    cdef public double radius_scale
    
    cdef public str h
    cdef public bint variable_h

    cdef int update(self) except -1
    cdef int get_potential_cells(self, int p_index, list output_list) except -1

cdef class CellCacheManager:
    """
    Class to manage a collection of CellCaches.
    """
    cdef public dict cell_cache_dict
    cdef public bint is_dirty
    cdef public CellManager cell_manager

    cdef public str h
    cdef public bint variable_h

    cdef int update(self) except -1
    cdef void add_cache_entry(self, ParticleArray pa, double radius_scale)
    cdef CellCache get_cell_cache(self, str pa_name, double radius_scale)


################################################################################
# `Classes for nearest particle location`.
################################################################################
cdef class NbrParticleLocatorBase:
    """
    Base class for all neighbor particle locators.
    """
    cdef public ParticleArray source
    cdef public CellManager cell_manager
    cdef public int source_index

    cdef int get_nearest_particles_to_point(self, Point pnt, double radius,
                                            LongArray output_array, 
                                            long exclude_index=*) except -1
    cdef int _get_nearest_particles_from_cell_list(
        self, Point pnt, double radius, list cell_list,
        LongArray output_array, long exclude_index=*) except -1                                   


cdef class FixedDestinationNbrParticleLocator(NbrParticleLocatorBase):
    """
    Base class for all locators where query points will be from a given particle
    array (dest).

    """
    cdef public ParticleArray dest
    cdef public int dest_index
    cdef public str h
    cdef public DoubleArray d_h, d_x, d_y, d_z
    
    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array, 
                                   double radius_scale=*,
                                   bint exclude_self=*) except -1

cdef class ConstHFixedDestNbrParticleLocator(
    FixedDestinationNbrParticleLocator):
    """
    Particle locator, where all particles are assumed to have the same
    interaction radius.
    """
    
    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array,
                                   double radius_scale=*,
                                   bint exclude_self=*) except -1

cdef class VarHFixedDestNbrParticleLocator(
    FixedDestinationNbrParticleLocator):
    """
    Particle locator, where different particles can have different interaction
    radius.
    """
    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array,
                                   double radius_scale=*,
                                   bint exclude_self=*) except -1


cdef class CachedNbrParticleLocator(
    FixedDestinationNbrParticleLocator):
    """
    Base class to represent cached particle locators. The cache maintained will
    be the list of neighbor particles for every particle in dest particle array.

    """
    cdef public double radius_scale
    cdef public CellCache cell_cache
    cdef public bint caching_enabled
    cdef public list particle_cache
    cdef public bint is_dirty

    cpdef enable_caching(self)
    cpdef disable_caching(self)

    cdef void update_status(self)
    cdef int update(self) except -1

    cdef int _update_cache_using_cell_cache(self) except -1
    cdef int _update_cache(self) except -1
    cdef int _compute_nearest_particles_using_cell_cache(
        self, long dest_p_index, Point dst_pnt, double eff_radius, 
        LongArray output_array, long exclude_index=*)except -1
                                                         
                                   
cdef class ConstHCachedNbrParticleLocator(
    CachedNbrParticleLocator):
    """
    Cached locator handling particles with constant interaction radius (h).
    """
    cdef public ConstHFixedDestNbrParticleLocator _locator

    cdef int _update_cache_using_cell_cache(self) except -1
    cdef int _update_cache(self) except -1
    
cdef class VarHCachedNbrParticleLocator(
    CachedNbrParticleLocator):
    """
    Cached locator handling particles with different interaction radius (h).
    """
    cdef VarHFixedDestNbrParticleLocator _locator

cdef class CachedNbrParticleLocatorManager:
    """
    Class to manager a collection of cached locators.
    """
    cdef public CellManager cell_manager
    cdef public CellCacheManager cell_cache_manager
    cdef public bint use_cell_cache
    cdef public dict cache_dict

    cdef public bint variable_h
    cdef public str h

    cpdef enable_cell_cache_usage(self)
    cpdef disable_cell_cache_usage(self)
    
    cdef int update(self) except -1
    cpdef add_interaction(self, ParticleArray source, ParticleArray dest,
                             double radius_scale)
    cpdef CachedNbrParticleLocator get_cached_locator(self, str source_name,
                                                      str dest_name,
                                                      double radius_scale)

# ################################################################################
# # `Classes for nearest polygon location`.
# ################################################################################
cdef class NbrPolygonLocatorBase:
    pass

cdef class FixedDestinationNbrPolygonLocator(NbrPolygonLocatorBase):
    pass

cdef class CachedNbrPolygonLocator(FixedDestinationNbrPolygonLocator):
    pass

cdef class CachedNbrPolygonLocatorManager:
    pass

################################################################################
# `NNPSManager` class.
################################################################################
cdef class NNPSManager:
    """
    Class to manager all nnps related information.
    """
    cdef public bint cell_caching
    cdef public bint particle_caching
    cdef public bint polygon_caching
    cdef public CellManager cell_manager

    cdef public bint variable_h
    cdef public str h
    
    cdef public CellCacheManager cell_cache_manager
    cdef public CachedNbrParticleLocatorManager particle_cache_manager
    cdef public CachedNbrPolygonLocatorManager polygon_cache_manager
    
    
    cpdef enable_cell_caching(self)
    cpdef disable_cell_caching(self)
    
    cpdef enable_particle_caching(self)
    cpdef disable_particle_caching(self)
    cpdef enable_polygon_caching(self)
    cpdef disable_polygon_caching(self)
    
    cpdef NbrParticleLocatorBase get_neighbor_particle_locator(
        self, ParticleArray source, ParticleArray dest=*, double radius_scale=*)

    cpdef NbrPolygonLocatorBase get_neighbor_polygon_locator(
        self, PolygonArray source, ParticleArray dest=*, double radius_scale=*)

    cdef int update(self) except -1