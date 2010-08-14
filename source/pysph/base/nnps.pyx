cdef extern from 'math.h':
    cdef double fabs(double)

# logger imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.carray cimport LongArray, DoubleArray
from pysph.base.point cimport Point
from pysph.base.particle_array cimport ParticleArray
from pysph.base.cell cimport CellManager, Cell
from pysph.base.polygon_array cimport PolygonArray

cimport numpy
import numpy as np

cdef inline double square(double dx, double dy, double dz):
    return dx*dx + dy*dy + dz*dz

################################################################################
# `get_nearest_particles_brute_force` function.
################################################################################
cpdef brute_force_nnps(Point pnt, double search_radius,
                       numpy.ndarray xa, numpy.ndarray ya, numpy.ndarray za, 
                       LongArray neighbor_indices, 
                       DoubleArray neighbor_distances,
                       long exclude_index=-1):
    """
    Brute force search for neighbors, used occasionally when nnps, cell manager
    may not be available.
    """
    cdef long n = len(xa)
    cdef double r2 = search_radius*search_radius
    cdef long i
    cdef double dist
    
    for i from 0 <= i < n:
        dist = square(pnt.x - xa[i], pnt.y - ya[i], pnt.z - za[i])
        if dist <= r2 and i != exclude_index:
            neighbor_indices.append(i)
            neighbor_distances.append(dist)
    return

################################################################################
# `CellCache` class.
################################################################################
cdef class CellCache:
    """
    Class to maintain list of neighboring cells for every particle in a given
    particle array.

    **Members**

     - p_array - the particle array whose particle's near cells are to be
       cached.
     - is_dirty - indicates, if the cached data is to be recomputed.
     - cache_list - the actual cache, a list containing one list of cells for
       each particle in parray.
     - cell_manager - the cell manager to use for nearest cell queries.
     - single_layer - indicates if exactly one neighboring layer of neighboring
       cells are to be got from the cell manager.
     - radius_scale - the scale to be applied to the interaction radius of each
       particle to get the effective radius.
     - h - name of the array containing the interaction radius property of the
       particles.
     - variable_h - indicates if the particles in p_array have same/different
       interaction radii.
 
    """

    def __init__(self, CellManager cell_manager, ParticleArray p_array,
                 double radius_scale, bint variable_h=False, str h='h'):
        """
        Constructor.

        **Parameters**

         - cell_manager - the cell manager to be used for nearest cell queries.
         - p_array - the particle array, whose particle's near cells are to be
           cached.
         - radius_scale - the scale to be applied to each particles interaction
           radius to get the effective radius for caching.
         - h - the name of the array containing the interaction radius property
           of the particles.
         
        """
        cdef DoubleArray h_array
        cdef double eff_rad

        self.cell_manager = cell_manager
        self.p_array = p_array
        self.is_dirty = True
        self.cache_list = []
        self.single_layer = False
        self.radius_scale = radius_scale

        self.h = h
        self.variable_h = variable_h
        
        # now setup the single_layer variable
        if not self.variable_h:
            if cell_manager is not None and p_array is not None:
                h_array = self.p_array.get_carray(self.h)
                eff_rad = h_array.get(0)*self.radius_scale
                if fabs(cell_manager.min_cell_size-eff_rad) < 1e-09:
                    self.single_layer = True

    cdef int update(self) except -1:
        """
        Recomputes the cached data if needed.
        """
        cdef DoubleArray xa, ya, za, ha
        cdef double *x, *y, *z, *h, eff_radius
        cdef Point pnt
        cdef str xc, yc, zc
        cdef int num_particles, i
        cdef list empty_list = []
        cdef list cl

        if self.is_dirty:

            self.is_dirty = False

            # bring cell manager up-to-date.
            self.cell_manager.update()

            pnt = Point()
            
            # recompute the potential lists.
            xc = self.cell_manager.coord_x
            yc = self.cell_manager.coord_y
            zc = self.cell_manager.coord_z

            xa = self.p_array.get_carray(xc)
            ya = self.p_array.get_carray(yc)
            za = self.p_array.get_carray(zc)
            ha = self.p_array.get_carray(self.h)

            x = xa.get_data_ptr()
            y = ya.get_data_ptr()
            z = za.get_data_ptr()
            h = ha.get_data_ptr()

            num_particles = self.p_array.get_number_of_particles()

            # resize the cache list, if needed.
            if len(self.cache_list) != num_particles:
                # clear up the list and add num_particles lists
                self.cache_list[:] = empty_list
                for i from 0 <= i < num_particles:
                    self.cache_list.append([])

            for i from 0 <= i < num_particles:

                pnt.x = x[i]
                pnt.y = y[i]
                pnt.z = z[i]

                cl = self.cache_list[i]
                cl[:] = empty_list

                eff_radius = h[i]*self.radius_scale
                self.cell_manager.get_potential_cells(pnt, eff_radius, cl)
            
    cdef int get_potential_cells(self, int p_index, list output_list) except -1:
        """
        Appends the list of potential cells to output_list
        """
        cdef list pot_list

        # make sure cache data is up-to-date
        self.update()
        
        pot_list = self.cache_list[p_index]
        output_list.extend(pot_list)

        return 0

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_get_potential_cells(self, int p_index, list output_list):
        return self.get_potential_cells(p_index, output_list)

    def py_update(self):
        return self.update()


################################################################################
# `CellCacheManager` class.
################################################################################
cdef class CellCacheManager:
    """
    Manages a collection of cell caches.
    """
    def __init__(self, CellManager cell_manager=None, bint variable_h=False,
                 str h='h'):
        """
        """
        self.cell_manager = cell_manager
        self.variable_h = variable_h
        self.h = h
        self.cell_cache_dict = {}
        self.is_dirty = True

    cdef int update(self) except -1:
        """
        Sets the is_dirty bit of all the caches, if any of the particle arrays
        have changed. 
        """
        cdef CellCache cache
        cdef bint dirty = False
        cdef int i, num_caches
        cdef list cache_list
        
        num_caches = len(self.cell_cache_dict)
        cache_list = self.cell_cache_dict.values()

        for i from 0 <= i < num_caches:
            cache = cache_list[i]
            if cache.p_array.is_dirty:
                dirty = True
                break

        if dirty:
            for i from 0 <= i < num_caches:
                cache = cache_list[i]
                cache.is_dirty = True        

    cdef void add_cache_entry(self, ParticleArray pa, double radius_scale):
        """
        Add  a new entry to be cached, if it does not already exist.
        """
        cdef CellCache cache
        if self.cell_cache_dict.has_key((pa.name, radius_scale)):
            return
        else:
            cache = CellCache(self.cell_manager, pa, radius_scale,
                              self.variable_h, self.h) 
            self.cell_cache_dict[(pa.name, radius_scale)] = cache

    cdef CellCache get_cell_cache(self, str pa_name, double radius_scale):
        """
        Return the said cache, if it exists, else return None.
        """
        return self.cell_cache_dict.get((pa_name, radius_scale))

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        return self.update()
    
    def py_add_cache_entry(self, ParticleArray pa, double radius_scale):
        self.add_cache_entry(pa, radius_scale)

    def py_get_cell_cache(self, str pa_name, double radius):
        return self.get_cell_cache(pa_name, radius)
    
################################################################################
# `NbrParticleLocatorBase` class.
################################################################################
cdef class NbrParticleLocatorBase:
    """
    Base class for neighbor particle locators.

    **Members**
    
     - source - particle array in which neighbors will be searched for.
     - cell_manager - the cell manager to be used for searching.
     - source_index - index of the source particle array in the cell manager.

    """
    def __init__(self, ParticleArray source, CellManager cell_manager=None):
        """
        """
        self.source = source
        self.cell_manager = cell_manager
        self.source_index = -1

        if self.cell_manager is not None:
            # find the index of source in cell_manager.
            self.source_index = cell_manager.array_indices.get(source.name)
            if self.source_index is None:
                msg = 'Source %s does not exist'%(self.source.name)
                raise ValueError, msg
            
    cdef int get_nearest_particles_to_point(self, Point pnt, double radius,
                                            LongArray output_array, 
                                            long exclude_index=-1) except -1: 
        """
        Return indices of particles with distance < 'radius' to pnt.

        **Parameters**
        
         - pnt - the query point whose nearest neighbors are to be searched for.
         - radius - the actual radius, within which particles are to be searched
           for.
         - output_array - array to store the neighbor indices.
         - exclude_index - an index that should be excluded from the neighbors.

        **Helper Functions**
        
         - _get_nearest_particles_from_cell_list

        """
        cdef list cell_list = list()

        # make sure cell manager is updated.
        self.cell_manager.update()
        
        # get the potential cell_list from the cell manager
        self.cell_manager.get_potential_cells(pnt, radius, cell_list)
        
        # now extract the exact points from the cell_list
        self._get_nearest_particles_from_cell_list(
            pnt, radius, cell_list, output_array, exclude_index)        
        
        return 0

    cdef int _get_nearest_particles_from_cell_list(
        self, Point pnt, double radius, list cell_list, 
        LongArray output_array, long exclude_index=-1) except -1: 
        """
        """
        cdef Cell cell
        cdef list tmp_list = list()
        cdef int i
        cdef DoubleArray xa, ya, za
        cdef str xc, yc, zc
        cdef double *x, *y, *z
        cdef LongArray src_indices
        cdef Point pnt1 = Point()
        cdef long idx
        
        tmp_list.extend(cell_list)

        # get the coordinate arrays.
        xc = self.cell_manager.coord_x
        yc = self.cell_manager.coord_y
        zc = self.cell_manager.coord_z

        xa = self.source.get_carray(xc)
        ya = self.source.get_carray(yc)
        za = self.source.get_carray(zc)
        
        x = xa.get_data_ptr()
        y = ya.get_data_ptr()
        z = za.get_data_ptr()
          
        while len(tmp_list) != 0:
            cell = tmp_list.pop(0)
            # find neighbors from the cell
            src_indices = cell.index_lists[self.source_index]
            
            for i in range(src_indices.length):
                idx = src_indices.get(i)
                pnt1.x = x[idx]
                pnt1.y = y[idx]
                pnt1.z = z[idx]

                if pnt1.distance(pnt) <= radius:
                    if idx != exclude_index:
                        output_array.append(idx)
        return 0
    
    ######################################################################
    # python wrappers.
    ######################################################################
    def py_get_nearest_particles_to_point(self, Point pnt, double radius,
                                          LongArray output_array, long
                                          exclude_index=-1): 
        return self.get_nearest_particles_to_point(
            pnt, radius, output_array, exclude_index)

################################################################################
# `FixedDestinationNbrParticleLocator` class.
################################################################################
cdef class FixedDestinationNbrParticleLocator(
    NbrParticleLocatorBase):
    """
    Class to represent particle locators, where neighbor queries will be for
    particles in another particle array.

    **Members**
    
     - dest - the particle array containing particles, for whom neighbor queries
       will be made.
     - dest_index - the index of the dest particle array in the cell manager.
     - h - name of the array holding the interaction radius for each particle.
     - d_h - the array of the destination storing the interaction radii for each
       particle.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, CellManager
                 cell_manager=None, str h='h'):
        """
        Constructor.
        """

        NbrParticleLocatorBase.__init__(self, source, cell_manager)

        self.dest = dest
        self.h = h
        self.dest_index = -1

        self.d_h = None
        self.d_x = None
        self.d_y = None
        self.d_z = None
        
        if self.cell_manager is not None:
            self.dest_index = cell_manager.array_indices.get(dest.name)
            if self.dest_index is None:
                msg = 'Destination %s does not exist'%(dest.name)
                raise ValueError, msg

        if self.dest is not None:
            self.d_h = self.dest.get_carray(self.h)
            self.d_x = self.dest.get_carray(self.cell_manager.coord_x)
            self.d_y = self.dest.get_carray(self.cell_manager.coord_y)
            self.d_z = self.dest.get_carray(self.cell_manager.coord_z)


    cdef int get_nearest_particles(self, long dest_p_index, 
                                   LongArray output_array,
                                   double radius_scale=1.0,
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest. The function is implemented in derived classes.

        **Parameters**

         - dest_p_index - id of the paritcle in dest whose neighbors are to be
           found. 
         - radius_scale - the scale to be applied to the particles interaction
           to ge the effective radius of interaction.
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array. 

        """
        msg = 'FixedDestinationNbrParticleLocator::get_nearest_particles'
        raise NotImplementedError, msg

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_get_nearest_particles(self, long dest_p_index, LongArray
                                 output_array, double radius_scale=1.0, 
                                 bint exclude_self=False):
        return self.get_nearest_particles(dest_p_index, output_array,
                                          radius_scale, exclude_self)

################################################################################
# `ConstHFixedDestNbrParticleLocator` class.
################################################################################
cdef class ConstHFixedDestNbrParticleLocator(
    FixedDestinationNbrParticleLocator):
    """
    FixedDestinationNbrParticleLocator to handle queries where all particles
    will have the same interaction radius.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, CellManager
                 cell_manager=None, str h='h'):
        
        FixedDestinationNbrParticleLocator.__init__(
            self, source, dest, cell_manager, h)

    cdef int get_nearest_particles(self, long dest_p_index, 
                                   LongArray output_array, 
                                   double radius_scale=1.0,
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the paritcle in dest whose neighbors are to be
           found. 
         - radius_scale - the scale to be applied to the particles interaction
           to ge the effective radius of interaction.
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array.
        
        """
        cdef Point pnt = Point()
        cdef double eff_radius
        cdef int exclude_index = -1

        pnt.x = self.d_x.get(dest_p_index)
        pnt.y = self.d_y.get(dest_p_index)
        pnt.z = self.d_z.get(dest_p_index)
        eff_radius = self.d_h.get(dest_p_index) * radius_scale
        
        if self.source is self.dest:
            if exclude_self:
                exclude_index = dest_p_index

        return NbrParticleLocatorBase.get_nearest_particles_to_point(
            self, pnt, eff_radius, output_array, exclude_index)


################################################################################
# `VarHFixedDestNbrParticleLocator` class.
################################################################################
cdef class VarHFixedDestNbrParticleLocator(
    FixedDestinationNbrParticleLocator):
    """
    FixedDestinationNbrParticleLocator to handle queries where particles could
    have different interaction radius.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, CellManager
                 cell_manager=None, str h='h'):
        """
        Constructor.
        """
        FixedDestinationNbrParticleLocator.__init__(
            self, source, dest, cell_manager, h)

    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array,
                                   double radius_scale=1.0,
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the paritcle in dest whose neighbors are to be
           found. 
         - radius_scale - the scale to be applied to the particles interaction
           to ge the effective radius of interaction.
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array.
        
        """
        msg = 'VarHFixedDestNbrParticleLocator::get_nearest_particles'
        raise NotImplementedError, msg

################################################################################
# `CachedNbrParticleLocator` class.
################################################################################
cdef class CachedNbrParticleLocator(FixedDestinationNbrParticleLocator):
    """
    Base class for FixedDestinationNbrParticleLocator with caching.
    
    **Members**
    
     - radius_scale - the scale to be applied to all particles interaction
       radius.
     - cell_cache - a CellCache for the dest particle array.
     - caching_enabled - indicates if caching is enabled for this class.
     - particle_cache - the cache, containing one LongArray for each particle in
       dest. 
     - is_dirty - indicates if the cache contents are stale and need to be
       recomputed.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, double
                 radius_scale, CellManager cell_manager=None, CellCache
                 cell_cache=None, bint caching_enabled=False, str h='h'):

        FixedDestinationNbrParticleLocator.__init__(
            self, source, dest, cell_manager, h)

        self.radius_scale = radius_scale
        self.cell_cache = cell_cache
        self.caching_enabled = caching_enabled

        self.particle_cache = []
        self.is_dirty = True
        
    cpdef enable_caching(self):
        """
        Enable caching.
        """
        if self.caching_enabled ==  False:
            self.caching_enabled = True
            self.is_dirty = True
                 
    cpdef disable_caching(self):
        """
        Disables Caching.
        """
        self.caching_enabled = False

    cdef void update_status(self):
        """
        Updates the dirty bit.
        """
        if not self.is_dirty:
            self.is_dirty = self.source.is_dirty or self.dest.is_dirty

    cdef int _compute_nearest_particles_using_cell_cache(
        self, long dest_p_index, Point dst_pnt, double radius, LongArray
        output_array, long exclude_index=-1) except -1:
        """
        Compute the nearest particles using the cell cache.
        """
        cdef list cell_list = list()

        self.cell_cache.get_potential_cells(dest_p_index, cell_list)

        return NbrParticleLocatorBase._get_nearest_particles_from_cell_list(
            self, dst_pnt, radius, cell_list, output_array, exclude_index
            )

    cdef int update(self) except -1:
        """
        Computes contents of the cache if needed.
        """
        cdef long num_particles
        cdef int ret

        if self.is_dirty:
            self.is_dirty = False
            if self.cell_cache is not None:
                # update cell cache if we are using one.
                self.cell_cache.update()

            if self.caching_enabled:
                # make sure the cache list and number of particles are the same
                # and resize if necessary.
                num_particles = self.dest.get_number_of_particles()

                if len(self.particle_cache) != num_particles:
                    self.particle_cache[:] = []
                    for i from 0 <= i < num_particles:
                        self.particle_cache.append(LongArray())

                if self.cell_cache is not None:
                    return self._update_cache_using_cell_cache()
                else:
                    return self._update_cache()

        return 0

    cdef int _update_cache(self) except -1:
        """
        Updates the cache, without using a cell cache.
        Implement this in derived classes.
        """
        msg = 'CachedNbrParticleLocator::_update_cache'
        raise NotImplementedError, msg


    cdef int _update_cache_using_cell_cache(self) except -1:
        """
        Updates the cache using the cell cache.
        Implement this in derived classes.
        """
        msg = 'CachedNbrParticleLocator::_update_cache_using_cell_cache'
        raise NotImplementedError, msg

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        self.update()

    def py_update_status(self):
        self.update_status()


################################################################################
# `ConstHCachedNbrParticleLocator` class.
################################################################################
cdef class ConstHCachedNbrParticleLocator(
    CachedNbrParticleLocator):
    """
    Cached locator handling particles with constant interaction radius (h).

    **Members**
    
     - _locator - a ConstHFixedDestNbrParticleLocator, to do the actual
       computation.

    """
    def __init__(self, ParticleArray source, ParticleArray dest, double
                 radius_scale, CellManager cell_manager=None, 
                 CellCache cell_cache=None, bint caching_enabled=False,
                 str h = 'h'):
        """
        Constructor.
        """
        
        CachedNbrParticleLocator.__init__(self, source, dest, radius_scale,
                                          cell_manager, cell_cache,
                                          caching_enabled, h)
        self._locator = ConstHFixedDestNbrParticleLocator(source, dest,
                                                          cell_manager, h)

    
    cdef int get_nearest_particles(self, long dest_p_index, LongArray
                                   output_array, double radius_scale=1.0, 
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the paritcle in dest whose neighbors are to be
           found. 
         - radius_scale - the scale to be applied to the particles interaction
           to ge the effective radius of interaction. This parameter will be
           unused, if caching is enabled or cell cache is used.
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array.

        """
        cdef LongArray index_array
        cdef int self_id = -1
        cdef int i
        cdef LongArray to_remove
        cdef long *data
        cdef Point pnt
        cdef long exclude_index = -1
        cdef double eff_radius

        to_remove = LongArray(1)

        # update internal data.
        self.update()

        if self.caching_enabled:
            # return data from cache.
            index_array = self.particle_cache[dest_p_index]

            output_array.resize(index_array.length)
            output_array.set_data(index_array.get_npy_array())
            data = output_array.get_data_ptr()
            
            if self.dest is self.source:
                if exclude_self:
                    to_remove.set(0, -1)
                    # remove dest_p_index if its present
                    for i from 0 <= i < output_array.length:
                        if data[i] == dest_p_index:
                            to_remove.set(0, i)
                            break
                    if to_remove.get(0) != -1:
                        output_array.remove(to_remove.get_npy_array())
        else:
            if self.cell_cache is None:
                # cell cache is also not present, just call the base class
                # method to get nearest neighbors.
                return self._locator.get_nearest_particles(
                    dest_p_index, output_array, self.radius_scale, exclude_self)
            else:
                
                pnt = Point()

                pnt.x = self.d_x.get(dest_p_index)
                pnt.y = self.d_y.get(dest_p_index)
                pnt.z = self.d_z.get(dest_p_index)

                eff_radius = self.d_h.get(dest_p_index)*self.radius_scale
                
                if self.source is self.dest:
                    if exclude_self:
                        exclude_index = dest_p_index
                                
                return self._compute_nearest_particles_using_cell_cache(
                    dest_p_index, pnt, eff_radius, output_array, exclude_index)
        return 0

    cdef int _update_cache(self) except -1:
        """
        Update the particle cache without using the cell cache.
        """
        cdef long num_particles, i
        cdef LongArray index_cache

        num_particles = self.dest.get_number_of_particles()

        for i from 0 <= i < num_particles:
            index_cache = self.particle_cache[i]
            index_cache.reset()

            self._locator.get_nearest_particles(
                i, index_cache, self.radius_scale, False)

        return 0

    cdef int _update_cache_using_cell_cache(self) except -1:
        """
        Update the particle cache using the cell cache.
        """
        cdef long num_particles, i
        cdef LongArray index_cache
        cdef double *x, *y, *z, *h, eff_radius
        cdef list cell_list = []
        cdef list empty_list = []
        cdef Point pnt = Point()
        cdef int ret = 0
        
        num_particles = self.dest.get_number_of_particles()
    
        x = self.d_x.get_data_ptr()
        y = self.d_y.get_data_ptr()
        z = self.d_z.get_data_ptr()
        h = self.d_h.get_data_ptr()
        
        for i from 0 <= i < num_particles:
            index_cache = self.particle_cache[i]
            index_cache.reset()

            pnt.x = x[i]
            pnt.y = y[i]
            pnt.z = z[i]

            eff_radius = h[i]*self.radius_scale

            cell_list[:] = empty_list
            
            self._compute_nearest_particles_using_cell_cache(
                i, pnt, eff_radius, index_cache, -1)

        return ret

################################################################################
# `VarHCachedNbrParticleLocator` class.
################################################################################
cdef class VarHCachedNbrParticleLocator(
    CachedNbrParticleLocator):
    """
    Cached locator handling particles with constant interaction radius (h).

    **Members**
    
     - _locator - a VarHFixedDestNbrParticleLocator, to do the actual
       computation.

    **Todo**
    
     - implement various functions.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, double
                 radius_scale, CellManager cell_manager=None, CellCache
                 cell_cache=None, bint caching_enabled=False, str h='h'):
        
        CachedNbrParticleLocator.__init__(self, source, dest, radius_scale,
                                          cell_manager, cell_cache, 
                                          caching_enabled, h)

        self._locator = VarHFixedDestNbrParticleLocator(
            source, dest, cell_manager, h)

################################################################################
# `CachedNbrParticleLocatorManager` class.
################################################################################
cdef class CachedNbrParticleLocatorManager:
    """
    Class to manage a collection of CachedParticleLocators.
    """
    def __init__(self, CellManager cell_manager=None, 
                 CellCacheManager cell_cache_manager=None,
                 bint use_cell_cache=False, bint variable_h=False,
                 str h='h'):
        """
        """
        self.cell_manager = cell_manager
        self.cell_cache_manager = cell_cache_manager
        self.use_cell_cache = use_cell_cache
        self.cache_dict = dict()

        self.variable_h = variable_h
        self.h = h

    cpdef enable_cell_cache_usage(self):
        """
        """
        self.use_cell_cache = True

        # for every cache maintained, if it does not have a cell cache
        # already, add a cell cache.
        cdef list cache_list
        cdef CellCache c_cache
        cdef CachedNbrParticleLocator p_cache
        cdef int num_caches, i
        cache_list = self.cache_dict.values()
        num_caches = len(cache_list)

        for i from 0 <= i < num_caches:
            p_cache = cache_list[i]
            if p_cache.cell_cache is None:
                self.cell_cache_manager.add_cache_entry(p_cache.dest,
                                                        p_cache.radius_scale)
                p_cache.cell_cache =  self.cell_cache_manager.get_cell_cache(
                    p_cache.dest.name, p_cache.radius_scale)
        
    cpdef disable_cell_cache_usage(self):
        """
        """
        cdef list cache_list
        cdef CellCache c_cache
        cdef CachedNbrParticleLocator p_cache
        cdef int num_caches, i

        self.use_cell_cache = False

        # disable cell caching for all caches. the entries however are not
        # removed from the cell cache manager
        cache_list = self.cache_dict.values()
        num_caches = len(cache_list)

        for i from 0 <= i < num_caches:
            p_cache = cache_list[i]
            p_cache.cell_cache = None

    cdef int update(self) except -1:
        """
        Calls update_status on each of the CachedParticleLocators being maintained.
        """
        cdef int i, num_caches
        cdef list cache_list = self.cache_dict.values()
        num_caches = len(cache_list)
        cdef CachedNbrParticleLocator loc

        for i from 0 <= i < num_caches:
            loc = cache_list[i]
            loc.update_status()

    cpdef add_interaction(self, ParticleArray source, ParticleArray dest,
                          double radius_scale):
        """
        Check if this interaction was already there in the internal dict, if
        yes, enable caching for that entry.

        """
        cdef tuple t = (source.name, dest.name, radius_scale)
        cdef CachedNbrParticleLocator loc
        cdef CellCache cell_cache=None
        
        loc = self.cache_dict.get(t)
        if loc is None:
            if self.use_cell_cache:
                # add an entry for this dest and radius_scale into
                # cell_cache_manager.
                self.cell_cache_manager.add_cache_entry(dest, radius_scale)
                cell_cache = self.cell_cache_manager.get_cell_cache(dest.name,
                                                                    radius_scale)
                if cell_cache is None:
                    raise RuntimeError, 'unable to add cell cache'
                
            # create a new cache object and insert into dict.
            if not self.variable_h:
                loc = ConstHCachedNbrParticleLocator(source, dest, radius_scale,
                                                     cell_manager=self.cell_manager,
                                                     cell_cache=cell_cache,
                                                     h=self.h)
            else:
                loc = VarHCachedNbrParticleLocator(source, dest, radius_scale, 
                                                   cell_manager=self.cell_manager,
                                                   cell_cache=cell_cache,
                                                   h=self.h)
            self.cache_dict[t] = loc
        else:
            # meaning this interaction was already present.
            # just enable caching for it.
            loc.enable_caching()
            

    cpdef CachedNbrParticleLocator get_cached_locator(self, str source_name,
                                                      str dest_name, double
                                                      radius):
        """
        Returns a cached object if it exists, else returns None.
        """
        return self.cache_dict.get((source_name, dest_name, radius))

    def py_update(self):
        self.update()
                                                         
    
################################################################################
# `NNPSManager` class.
################################################################################
cdef class NNPSManager:
    """
    Class to provide management of all nearest neighbor search related
    functionality.

    **Members**
    
     - cell_caching - indicates if cell caching is to be enabled.
     - particle_caching - indicates if particle caching is to be enabled.
     - polygon_caching - indicates if polygon caching is enabled.
     - cell_manager - CellManager to compute nearest cells.
     - variable_h - indicates if variable-h computations are needed.

    """
    def __init__(self, CellManager cell_manager=None, bint cell_caching=False,
                 bint particle_caching=True, polygon_caching=True, bint
                 variable_h=False, str h='h'):
        """
        Construct.
        """
        self.cell_caching = cell_caching
        self.particle_caching = particle_caching
        self.polygon_caching = polygon_caching
        self.cell_manager = cell_manager
        self.variable_h = variable_h
        self.h = h

        self.cell_cache_manager = CellCacheManager(
            self.cell_manager, variable_h=self.variable_h, h=h )
        
        self.particle_cache_manager = CachedNbrParticleLocatorManager(
            self.cell_manager, self.cell_cache_manager, self.cell_caching,
            self.variable_h, self.h)

        self.polygon_cache_manager = CachedNbrPolygonLocatorManager(
            self.cell_manager, self.cell_cache_manager, self.cell_caching)
        
    cpdef enable_cell_caching(self):
        """
        Enables cell caching.
        """
        self.cell_caching = True
        self.particle_cache_manager.enable_cell_cache_usage()
        #self.polygon_cache_manager.enable_cell_cache_usage()
        return 0

    cpdef disable_cell_caching(self):
        """
        Disables cell caching.
        """
        self.cell_caching = False
        self.particle_cache_manager.disable_cell_cache_usage()
        #self.polygon_cache_manager.disable_cell_cache_usage()
        return 0

    cpdef enable_particle_caching(self):
        """
        Enables particle caching.
        """
        self.particle_caching = True

    cpdef disable_particle_caching(self):
        """
        Disables particle caching.
        """
        self.particle_caching = False

    cpdef enable_polygon_caching(self):
        """
        Enables polygon caching.
        """
        self.polygon_caching = True

    cpdef disable_polygon_caching(self):
        """
        Disables polygon caching.
        """
        self.polygon_caching = False

    cpdef NbrParticleLocatorBase get_neighbor_particle_locator(
        self, ParticleArray source, ParticleArray dest=None, double
        radius_scale=-1.0):
        """
        Returns an appropriate neighbor particle locator.
        """
        cdef CachedNbrParticleLocator loc
        
        if dest is None:
            return NbrParticleLocatorBase(source, self.cell_manager)
        else:
            if radius_scale < 0.0 or not self.particle_caching:
                if self.variable_h:
                    return VarHFixedDestNbrParticleLocator(
                        source, dest, self.cell_manager, self.h)
                else:
                    return ConstHFixedDestNbrParticleLocator(
                        source, dest, self.cell_manager, self.h)
            else:
                self.particle_cache_manager.add_interaction(
                    source, dest, radius_scale)
                loc = self.particle_cache_manager.get_cached_locator(
                    source.name, dest.name, radius_scale)
                return loc
            
    cpdef NbrPolygonLocatorBase get_neighbor_polygon_locator(
        self, PolygonArray source, ParticleArray dest=None, double radius_scale=-1.0):
        """
        Returns an appropriate neighbor polygon locator.
        """
        msg = 'NNPSManager::get_neighbor_polygon_locator'
        raise NotImplementedError, msg

    cdef int update(self) except -1:
        """
        """
        # update the status of the caches.
        self.cell_cache_manager.update()
        self.particle_cache_manager.update()

        # update the status of the cell manager.
        # this may or may not update the cell manager. That depends on how the
        # cell managers update_status is implemented.
        self.cell_manager.update_status()

        #self.polygon_cache_manager.update()

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        return self.update()
    
