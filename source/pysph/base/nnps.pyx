"""Module to implement neighboring particle searches 

** Usage **

 - Create a `ParticleArray` and `CellManager` to bin the particles into cells

::
    
    x = numpy.array([-0.5, -0.5, 0.5, 0.5, 1.5, 2.5, 2.5])
    y = numpy.array([2.5, -0.5, 1.5, 0.5, 0.5, 0.5, -0.5])
    z = numpy.array([0., 0, 0, 0, 0, 0, 0])
    h = numpy.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    parr1 = ParticleArray(name='parr1', **{'x':{'data':x}, 'y':{'data':y}, 'z':
                                            {'data':z}, 'h':{'data':h}})
    parr2 = ParticleArray(name='parr2', **{'x':{'data':y}, 'y':{'data':z}, 'z':
                                            {'data':x}, 'h':{'data':h}})
    
    cell_mgr = CellManager(arrays_to_bin=[parr1,parr2], min_cell_size=1.,
                               max_cell_size=2.0)
    
 - Now create an `NNPSManager` which will give us neighbor particle locators
   Select variable_h option if the h of particles not constant

::
    
    nnps_mgr = NNPSManager(cell_mgr, variable_h=True)

 - The `NNPSManager` gives locators where the influence of particles in the 
   source array need to be calculated on the particles in dest array.
   radius_scale is the scaling factor of `h` (kernel support radius)

::
    
    nbrl = nm.get_neighbor_particle_locator(source=parrs1, dest=parrs1,
                                            radius_scale=1.0)

 - Now the locator can be used to locate the neighboring particles by
   specifying the index of the dest particle and passing an output argument
   `output_array` to which the neighbor particles' indices are appended.
   `exclude_self` argument can be specified in case source and dest arrays are
   same and the dest particle itself is to be excluded from the neighbors
   
::
    
    output_array = LongArray()
    nbrl.get_nearest_particles(2, output_array)

"""

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

###############################################################################
# `get_nearest_particles_brute_force` function.
###############################################################################
cpdef brute_force_nnps(Point pnt, double search_radius,
                       numpy.ndarray xa, numpy.ndarray ya, numpy.ndarray za,
                       LongArray neighbor_indices,
                       DoubleArray neighbor_distances,
                       long exclude_index=-1):
    """
    Brute force search for neighbors, can be used when nnps, cell manager
    may not be available.
    """
    cdef long n = len(xa)
    cdef double r2 = search_radius*search_radius
    cdef long i
    cdef double dist
    
    for i in range(n):
        dist = square(pnt.x - xa[i], pnt.y - ya[i], pnt.z - za[i])
        if dist <= r2 and i != exclude_index:
            neighbor_indices.append(i)
            neighbor_distances.append(dist)
    return

###############################################################################
# `get_adjacency_matrix` and related functions for testing and verification.
###############################################################################
cpdef numpy.ndarray get_distance_matrix(numpy.ndarray xa, numpy.ndarray ya,
                                        numpy.ndarray za):
    """Returns the distance matrix from between the coordinates specified """
    cdef numpy.ndarray[ndim=1,dtype=numpy.float64_t] x = xa
    cdef numpy.ndarray[ndim=1,dtype=numpy.float64_t] y = ya
    cdef numpy.ndarray[ndim=1,dtype=numpy.float64_t] z = za
    cdef long i, j, N
    cdef double dist
    N = len(xa)
    cdef numpy.ndarray[ndim=2,dtype=numpy.float64_t] ret = np.zeros((N,N))
    
    for i in range(N):
        for j in range(i, N):
            dist = square(x[j] - x[i], y[j] - y[i], z[j] - z[i])
            ret[i,j] = ret[j,i] = dist
    return np.sqrt(ret)

cpdef numpy.ndarray get_distance_matrix_pa(ParticleArray parray):
    """Returns the distance matrix of particles in parray. """
    return get_distance_matrix(parray.get('x'), parray.get('y'),
                               parray.get('z'))

cpdef numpy.ndarray get_adjacency_matrix(numpy.ndarray xa, numpy.ndarray ya,
                        numpy.ndarray za, numpy.ndarray ha, radius_scale=1.0):
    """Returns the dense adjacency effect matrix from i^th row to j^th col """
    dist = get_distance_matrix(xa, ya, za)
    return dist < ha[:,None] * radius_scale

cpdef numpy.ndarray get_adjacency_matrix_pa(ParticleArray parray,
                                            double radius_scale=1.0):
    """Returns the dense adjacency matrix of particles in parray. """
    return get_adjacency_matrix(parray.get('x'), parray.get('y'),
                    parray.get('z'), parray.get('h'), radius_scale)

###############################################################################
# `NbrParticleLocatorBase` class.
###############################################################################
cdef class NbrParticleLocatorBase:
    """
    Base class for neighbor particle locators.

    **Members**
    
     - source - particle array in which neighbors will be searched for.
     - cell_manager - the cell manager to be used for searching.
     - source_index - index of the source particle array in the cell manager.

    """
    def __init__(self, ParticleArray source, CellManager cell_manager=None):
        """Constructor:
        
        ** Parameters **
        
        - source - `ParticleArray` in which neighbors will be searched for.
        - cell_manager - `CellManager` which bins the source particle array
            into cells
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
         - radius - the radius within which particles are to be searched for.
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
        self._get_nearest_particles_from_cell_list(pnt, radius, cell_list,
                                       output_array, exclude_index)
        
        return 0

    cdef int _get_nearest_particles_from_cell_list(self, Point pnt,
                    double radius, list cell_list,
                    LongArray output_array, long exclude_index=-1) except -1: 
        """ Get the neighboring particles to `pnt` from the `cell_list` """
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
        """Return indices of particles with distance < 'radius' to pnt.

        **Parameters**
        
         - pnt - the query point whose nearest neighbors are to be searched for.
         - radius - the radius within which particles are to be searched for.
         - output_array - array to store the neighbor indices.
         - exclude_index - an index that should be excluded from the neighbors.
         
        """
        return self.get_nearest_particles_to_point(
            pnt, radius, output_array, exclude_index)

###############################################################################
# `FixedDestNbrParticleLocator` class.
###############################################################################
cdef class FixedDestNbrParticleLocator(NbrParticleLocatorBase):
    """Class to represent particle locators, where neighbor queries will be for
    particles in another particle array and where all particles are assumed to
    have the same interaction radius

    **Members**
    
     - dest - the particle array containing particles for whom neighbor queries
       will be made.
     - radius_scale - the scale to be applied to the particles interaction
     - dest_index - the index of the dest particle array in the cell manager.
     - h - name of the array holding the interaction radius for each particle.
     - d_h, d_x,d_y,d_z - the arrays of the destination storing the interaction
       radii and position x,y,z for each particle.
     - caching_enabled - indicates if caching is enabled for this class.
     - particle_cache - the cache, containing one LongArray for each particle
       in dest. 
     - is_dirty - indicates if the cache contents are stale and need to be
       recomputed.
    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 double radius_scale, CellManager cell_manager=None,
                 bint caching_enabled=False, str h='h'):
        NbrParticleLocatorBase.__init__(self, source, cell_manager)
        self.dest = dest
        self.h = h
        self.radius_scale = radius_scale
        self.dest_index = -1
        
        self.caching_enabled = caching_enabled
        self.particle_cache = []
        self.is_dirty = True

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
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the particle in dest whose neighbors are to be
           found. 
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array.
        
        """
        cdef LongArray index_array
        cdef int self_id = -1
        cdef int i
        cdef LongArray to_remove
        cdef long *data
        cdef long exclude_index = -1

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
                    for i in range(output_array.length):
                        if data[i] == dest_p_index:
                            to_remove.set(0, i)
                            break
                    if to_remove.get(0) != -1:
                        output_array.remove(to_remove.get_npy_array())
        else:
            # dont use cache, get neighbors directly
            return self.get_nearest_particles_nocache(dest_p_index, 
                                   output_array, exclude_self)
        
        return 0
    
    cdef int get_nearest_particles_nocache(self, long dest_p_index, 
                                   LongArray output_array,
                                   bint exclude_self=False) except -1:
        """Get the nearest particles without using cache (direct computation)
        
        This is a convenience internal method and works only for constant h
        """
        cdef Point pnt = Point()
        cdef long exclude_index = -1
        pnt.x = self.d_x.get(dest_p_index)
        pnt.y = self.d_y.get(dest_p_index)
        pnt.z = self.d_z.get(dest_p_index)
        
        cdef double eff_radius = self.d_h.get(dest_p_index) * self.radius_scale
        
        if self.source is self.dest:
            if exclude_self:
                exclude_index = dest_p_index

        return NbrParticleLocatorBase.get_nearest_particles_to_point(
        self, pnt, eff_radius, output_array, exclude_index)

    cpdef enable_caching(self):
        """Enable caching."""
        if self.caching_enabled ==  False:
            self.caching_enabled = True
            self.is_dirty = True

    cpdef disable_caching(self):
        """Disables Caching."""
        self.caching_enabled = False

    cdef void update_status(self):
        """Updates the dirty flag."""
        if not self.is_dirty:
            self.is_dirty = self.source.is_dirty or self.dest.is_dirty

    cdef int update(self) except -1:
        """Computes contents of the cache if needed."""
        cdef long num_particles, i
        cdef int ret = 0

        if self.is_dirty:

            if self.caching_enabled:
                # make sure the cache list and number of particles are the same
                # and resize if necessary.
                num_particles = self.dest.get_number_of_particles()

                if len(self.particle_cache) != num_particles:
                    self.particle_cache[:] = []
                    for i in range(num_particles):
                        self.particle_cache.append(LongArray())

                ret = self._update_cache()
            self.is_dirty = False

        return ret

    cdef int _update_cache(self) except -1:
        """Update the particle cache. """
        cdef long num_particles, i
        cdef LongArray index_cache

        num_particles = self.dest.get_number_of_particles()

        for i in range(num_particles):
            index_cache = self.particle_cache[i]
            index_cache.reset()

            self.get_nearest_particles_nocache(
                i, index_cache, False)

        return 0

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        """Computes contents of the cache if needed."""
        self.update()

    def py_update_status(self):
        """Updates the dirty flag."""
        self.update_status()

    def py_get_nearest_particles(self, long dest_p_index, LongArray
                                 output_array, bint exclude_self=False):
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the particle in dest whose neighbors are to be
           found. 
         - output_array - array to store the neighbor indices into.
         - exclude_self - indicates if dest_p_index be excluded from
           output_array.
        
        """
        return self.get_nearest_particles(dest_p_index, output_array,
                                          exclude_self)


###############################################################################
# `VarHNbrParticleLocator` class.
###############################################################################
cdef class VarHNbrParticleLocator(FixedDestNbrParticleLocator):
    """
    Neighbor Particle Locator to handle queries where particles may
    have different interaction radii.
    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 double radius_scale, CellManager cell_manager=None,
                 str h='h'):
        FixedDestNbrParticleLocator.__init__(
            self, source, dest, radius_scale, cell_manager, h)
        
        self._rev_locator = FixedDestNbrParticleLocator(
            dest, source, radius_scale, cell_manager, h)
        
        self.update()

    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array,
                                   bint exclude_self=False) except -1:
        """
        Gets particles in source, that are nearest to the particle dest_p_index
        in dest.

        **Parameters**

         - dest_p_index - id of the particle in dest whose neighbors are to be
           found.
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

        # return data from cache.
        index_array = self.particle_cache[dest_p_index]

        output_array.resize(index_array.length)
        output_array.set_data(index_array.get_npy_array())
        data = output_array.get_data_ptr()
        
        if self.dest is self.source:
            if exclude_self:
                to_remove.set(0, -1)
                # remove dest_p_index if its present
                for i in range(output_array.length):
                    if data[i] == dest_p_index:
                        to_remove.set(0, i)
                        break
                if to_remove.get(0) != -1:
                    output_array.remove(to_remove.get_npy_array())
        return 0
    
    cdef int update(self) except -1:
        """Computes contents of the cache if needed. """
        self._rev_locator.update()
        return FixedDestNbrParticleLocator.update(self)
    
    cdef int _update_cache(self) except -1:
        """Update the particle cache without using the cell cache. """
        cdef long num_particles, i, j, k
        cdef LongArray index_cache
        cdef LongArray rev_cache = LongArray()
        
        num_d_particles = self.dest.get_number_of_particles()
        num_s_particles = self.source.get_number_of_particles()
        
        for i in range(num_d_particles):
            index_cache = self.particle_cache[i]
            index_cache.reset()
            
            FixedDestNbrParticleLocator.get_nearest_particles_nocache(self,
                i, index_cache, False)
        
        self._rev_locator._update_cache()
        
        for j in range(num_s_particles):
            rev_cache.reset()
            self._rev_locator.get_nearest_particles_nocache(j,
                                            rev_cache, False)
            for k in range(rev_cache.length):
                index_cache = self.particle_cache[rev_cache[k]]
                if j not in index_cache:
                    index_cache.append(j)
        return 0

###############################################################################
# `NNPSManager` class.
###############################################################################
cdef class NNPSManager:
    """
    Class to provide management of all nearest neighbor search related
    functionality.

    **Members**
    
     - cell_manager - CellManager to compute nearest cells.
     - variable_h - indicates if variable-h computations are needed.

    """
    def __init__(self, CellManager cell_manager=None,
                 bint variable_h=False, str h='h'):
        self.cell_manager = cell_manager
        self.variable_h = variable_h
        self.h = h
        
        self.particle_locator_cache = dict()
        
        self.polygon_cache_manager = CachedNbrPolygonLocatorManager(
            self.cell_manager)
    
    cpdef NbrParticleLocatorBase get_neighbor_particle_locator(
        self, ParticleArray source, ParticleArray dest=None, double
        radius_scale=1.0):
        """Returns an appropriate neighbor particle locator. """
        if dest is None:
            return NbrParticleLocatorBase(source, self.cell_manager)
        else:
            self.add_interaction(source, dest, radius_scale)
            return self.get_cached_locator(source.name, dest.name, radius_scale)
    
    cpdef NbrPolygonLocatorBase get_neighbor_polygon_locator(self,
            PolygonArray source, ParticleArray dest=None,
            double radius_scale=1.0):
        """Returns an appropriate neighbor polygon locator. """
        msg = 'NNPSManager::get_neighbor_polygon_locator'
        raise NotImplementedError, msg

    cdef int update(self) except -1:
        """ Update the status of the neighbor locators. """
        # update the status of the caches.
        # Calls update_status on each CachedParticleLocators being maintained.
        cdef int i, num_caches
        cdef list cache_list = self.particle_locator_cache.values()
        num_caches = len(cache_list)
        cdef FixedDestNbrParticleLocator loc

        for i in range(num_caches):
            loc = cache_list[i]
            loc.update_status()

        # update the status of the cell manager.
        # this may or may not update the cell manager. That depends on how the
        # cell managers update_status is implemented.
        self.cell_manager.update_status()

        #self.polygon_cache_manager.update()
    
    cpdef add_interaction(self, ParticleArray source, ParticleArray dest,
                          double radius_scale):
        """
        Check if this interaction already exists in the internal dict, if
        yes, enable caching for that entry
        """
        cdef tuple t = (source.name, dest.name, radius_scale)
        cdef FixedDestNbrParticleLocator loc
        
        loc = self.particle_locator_cache.get(t)
        if loc is None:
            # create a new cache object and insert into dict.
            if not self.variable_h:
                loc = FixedDestNbrParticleLocator(source, dest, radius_scale,
                                   cell_manager=self.cell_manager, h=self.h)
            else:
                loc = VarHNbrParticleLocator(source, dest, radius_scale,
                                 cell_manager=self.cell_manager, h=self.h)
            self.particle_locator_cache[t] = loc
        else:
            # meaning this interaction was already present.
            # just enable caching for it.
            loc.enable_caching()
    
    cpdef FixedDestNbrParticleLocator get_cached_locator(self,
                str source_name, str dest_name, double radius):
        """Returns a cached object if it exists, else returns None."""
        return self.particle_locator_cache.get((source_name,dest_name,radius))
    
    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        """ Update the status of the neighbor locators. """
        return self.update()

