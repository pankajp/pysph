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
    cdef double sqrt(double)

from pysph.base.point cimport Point_new, Point_distance2, Point_distance

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
    """ Perform a brute force search for the nearest neighbors """ 
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
    Nearest neighbor locator for a source particle array. 
    The main use is to return nearest particles for a query point.

    Members:
    --------
    
    source -- particle array in which neighbors will be searched for.
    cell_manager -- cell manager for the source
    source_index -- index of the source particle array in the cell manager.

    """
    def __init__(self, ParticleArray source, CellManager cell_manager=None):
        """ Constructor
        
        Parameters:
        -----------        

        source -- `ParticleArray` in which neighbors will be searched for.
        cell_manager - `CellManager` which bins the source into cells.

        Warning:
        -------
        An error is raised if the source is not available in the cell manager 

        """
        self.source = source
        self.cell_manager = cell_manager
        self.source_index = -1

        self.particle_neighbors = {}
        self.kernel_function_evaluation = {}
        self.kernel_gradient_evaluation = {}

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
        Return source indices that are within a certain radius from a point.

        Parameters:        
        
        pnt -- the query point whose nearest neighbors are to be searched for.
        radius -- the radius within which particles are to be searched for.
        output_array -- array to store the neighbor indices.
        exclude_index -- an index that should be excluded from the neighbors.

        Helper Functions:
        -----------------
        
         _get_nearest_particles_from_cell_list

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
        """ Extract nearest neighbors from a cell_list
        
        Parameters:
        -----------
        pnt -- query point for searching
        radius -- radius of search
        output_array -- output parameter. indices are appended to it.

        Algorithm:
        ----------
        for each cell in the cell list
            find cell indices of `source` particle array in that cell
            compute the norm of distance
            append to output_array if norm <= radius

        Notes:
        ------
        A cell has a list called `index_lists` which stores particle indices
        for a particular array. Since the `source` array is assumed to be 
        present in cell's `arrays_to_bin` with index `source_index`, we 
        can access particles for this source in the cell.

        `source_index` is setup at construction of this class.

        """
        cdef Cell cell
        cdef int i
        cdef DoubleArray xa, ya, za
        cdef str xc, yc, zc
        cdef double *x, *y, *z
        cdef LongArray src_indices
        cdef Point pnt1 = Point_new(0,0,0)
        cdef long idx
        
        # get the coordinate arrays.
        xc = self.cell_manager.coord_x
        yc = self.cell_manager.coord_y
        zc = self.cell_manager.coord_z

        xa = self.source.get_carray(xc)
        ya = self.source.get_carray(yc)
        za = self.source.get_carray(zc)
        
        x = xa.data
        y = ya.data
        z = za.data
        
        if exclude_index >= 0:
            for cell in cell_list:
                # find neighbors from the cell
                src_indices = cell.index_lists[self.source_index]
                
                for i in range(src_indices.length):
                    idx = src_indices.data[i]
                    pnt1.x = x[idx]
                    pnt1.y = y[idx]
                    pnt1.z = z[idx]
    
                    if Point_distance2(pnt1, pnt) <= radius*radius:
                        if idx != exclude_index:
                            output_array.append(idx)
        else:
            # no exclude_index
            for cell in cell_list:
                # find neighbors from the cell
                src_indices = cell.index_lists[self.source_index]
                
                for i in range(src_indices.length):
                    idx = src_indices.data[i]
                    pnt1.x = x[idx]
                    pnt1.y = y[idx]
                    pnt1.z = z[idx]
    
                    if Point_distance2(pnt1, pnt) <= radius*radius:
                        output_array.append(idx)
        return 0
    
    ######################################################################
    # python wrappers.
    ######################################################################
    def py_get_nearest_particles_to_point(self, Point pnt, double radius,
                                          LongArray output_array, long
                                          exclude_index=-1):
        """
        Return source indices that are within a certain radius from a point.

        Parameters:        
        
        pnt -- the query point whose nearest neighbors are to be searched for.
        radius -- the radius within which particles are to be searched for.
        output_array -- array to store the neighbor indices.
        exclude_index -- an index that should be excluded from the neighbors.

        """

        return self.get_nearest_particles_to_point(
            pnt, radius, output_array, exclude_index)

###############################################################################
# `FixedDestNbrParticleLocator` class.
###############################################################################
cdef class FixedDestNbrParticleLocator(NbrParticleLocatorBase):
    """
    Nearest neighbor locator between a destination and a source particle
    array. The query points will be the destination particle positions.
    It is assumed that the particles have constant smoothing length.

    Data Members:
    -------------    
    dest - the query point particle array
    radius_scale - kernel support radius factor `kfac`
    dest_index - destination index in the cell manager
    h -- destination interaction radius
    d_h, d_x,d_y,d_z -- destination particles properties
    caching_enabled -- flag to turn caching on 
    particle_cache -- a cache object. one LongArray for each particle in dest
    is_dirty -- flag to recompute the cache

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 double radius_scale, CellManager cell_manager=None,
                 bint caching_enabled=False, str h='h'):
        """ Constructor
        
        Parameters:
        -----------
        source -- the source particle array
        dest -- the destination particle array
        radius_scale -- the kernel radius support `kfac`
        
        Notes:
        ------
        The destination is assumed to be in the cell manager if provided.
        This is used to set the `dest_index` attribute. 
        
        """

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
        Get the nearest source particles to particles in `dest_p_index`

        Parameters:
        -----------

        dest_p_index -- index of query destination particle
        output_array -- array to store the neighbor indices into.
        exclude_self -- indicates if dest_p_index be excluded from
                        output_array.

        Algorithm:
        ----------
        if caching is enabled:
            return cache data after removing self if exclude_self is true
        else:
           call get_nearest_particles_nocache to get neighbors directly
        
        """
        cdef LongArray index_array
        cdef int self_id = -1
        cdef int i
        cdef LongArray to_remove
        cdef long *data
        cdef long exclude_index = -1

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
                    to_remove = LongArray()
                    # cached array have self index as first value
                    to_remove.append(0)
                    output_array.remove(to_remove.get_npy_array())
        else:
            # dont use cache, get neighbors directly
            return self.get_nearest_particles_nocache(dest_p_index, 
                                   output_array, exclude_self)
        
        return 0
    
    cdef int get_nearest_particles_nocache(self, long dest_p_index, 
                                   LongArray output_array,
                                   bint exclude_self=False) except -1:
        """ Direct computation of the neighbor search
        
        Algorithm:
        ----------
        calculate the effective radius of search
        call parent class to get the neighbors from source

        Notes:
        ------
        The parent class NbrParticleLocatorBase provides functions for 
        returning a list of neighbors given a point and a radius.
        This function is called with point determined by `dest_p_index` in
        the destination particle manager.

        This is a convenience internal method and works only for constant h

        """
        cdef Point pnt = Point_new(0,0,0)
        cdef long exclude_index = -1
        pnt.x = self.d_x.data[dest_p_index]
        pnt.y = self.d_y.data[dest_p_index]
        pnt.z = self.d_z.data[dest_p_index]
        
        cdef double eff_radius = self.d_h.data[dest_p_index] * self.radius_scale
        
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
        """ Computes contents of the cache 

        Algorithm:
        ----------
        if cache is to be recomputed
            if caching is enabled
                resize particle cache if required
                call _update_cache to populate the cache contents

        """
        cdef long num_particles, i
        cdef int ret = 0

        if self.is_dirty:

            if self.caching_enabled:
                # make sure the cache list and number of particles are the same
                # and resize if necessary.
                num_particles = self.dest.get_number_of_particles()

                if len(self.particle_cache) != num_particles:
                    self.particle_cache = []
                    for i in range(num_particles):
                        self.particle_cache.append(LongArray())

                ret = self._update_cache()
            self.is_dirty = False

        return ret

    cdef int _update_cache(self) except -1:
        """Update `particle_cache` by appending to each LongArray """
        cdef long num_particles, i, j
        cdef LongArray index_cache

        num_particles = self.dest.get_number_of_particles()

        for i in range(num_particles):
            index_cache = self.particle_cache[i]
            index_cache.reset()

            self.get_nearest_particles_nocache(
                i, index_cache, False)
            
            # keep particle at the first index for possibly faster exclude_self
            if self.source == self.dest:
                for j in range(index_cache.length):
                    if i == index_cache[j]:
                        # swap first index with i
                        index_cache[j] = index_cache[0]
                        index_cache[0] = i
                        break
            
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
        """ Constructor:
        
        Parameters:
        -----------
        source -- the source particle array 
        dest -- the destination particle array
        radius_scale -- kernel support radius `kfac`

        Notes:
        ------
        Subclassed from FixedDestNbrParticleLocator, the constructor of which
        is called. Remember this was for fixed smoothing length neighbor 
        queries.
        
        A reversed FixedDestNbrParticleLocator is also created where the 
        roles of the source and destination are reversed.

        """
        FixedDestNbrParticleLocator.__init__(
            self, source, dest, radius_scale, cell_manager, h)
        
        self._rev_locator = FixedDestNbrParticleLocator(
            dest, source, radius_scale, cell_manager, h)
        
        self.update()

    cdef int get_nearest_particles(self, long dest_p_index,
                                   LongArray output_array,
                                   bint exclude_self=False) except -1:
        """
        Get neighbors from source to `dest_p_index` particle in dest

        Parameters:
        -----------

        dest_p_index -- query destination particle index
        output_array -- array to store the neighbor indices into.
        exclude_self -- exclude dest_p_index if true
        
        """

        cdef LongArray index_array
        cdef int self_id = -1
        cdef int i
        cdef LongArray to_remove
        cdef long *data
        cdef long exclude_index = -1

        # update internal data.
        self.update()

        # return data from cache.
        index_array = self.particle_cache[dest_p_index]

        output_array.resize(index_array.length)
        output_array.set_data(index_array.get_npy_array())
        data = output_array.data
        if self.dest is self.source:
            if exclude_self:
                to_remove = LongArray()
                # remove dest_p_index if its present
                to_remove.append(0)
                output_array.remove(to_remove.get_npy_array())
        return 0
    
    cdef int update(self) except -1:
        """Computes contents of the cache. 
        
        Notes:
        ------
        Caches for both NeighborLocators are updated as before.
        The magic happens because _update_cache is different now.

        """
        self._rev_locator.update()
        return FixedDestNbrParticleLocator.update(self)
    
    cdef int _update_cache(self) except -1:
        """Update the particle cache without using the cell cache.

        Algorithm:
        ----------
        update index_cache as before finding source particle ids
        for each source particle
            get nearest particles from rev_loc -> rev_cache
            for each index in rev_cache
                retreive from the cache the destination index_cache
                add source to index_cache if not present

        
        Notes:
        ------
        index_cache refers to the cache to beused
        rev_cache is used only internally like _rev_loc                    
        
        """
        cdef long num_d_particles, num_s_particles, i, j, k
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
                index_cache = self.particle_cache[rev_cache.data[k]]
                if index_cache.index(j) < 0:
                    index_cache.append(j)
        return 0

###############################################################################
# `NNPSManager` class.
###############################################################################
cdef class NNPSManager:
    """ Class to provide management of all nearest neighbor search related
    functionality.

    Implemented are a single source neighbor locator as well as neighbor
    locators between particle arrays with both fixed smoothing length and
    variable smoothing length.

    Members:
    --------
    cell_manager -- CellManager to compute nearest cells.
    variable_h -- indicates if variable-h computations are needed.
    particle_locator_cache -- cache object for source destination interactions

    """

    #Defined in the .pxd file
    #cdef public CellManager cell_manager
    #cdef public bint variable_h
    #cdef public str h
    #cdef public dict particle_locator_cache

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
        """Returns an appropriate neighbor particle locator. 

        Notes:
        ------
        if no destination is specified return a base locator
        else check if the cache already has a locator between two arrays
        and return that locator if it exists
        
        """
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
        """ Update the status of the neighbor locators. 
        
        Algorithm:
        ----------
        for all NbrParticleLocators being maintained
            call it's update_status (set is_dirty if src or dst is_dirty)
        call cell_manager's update_status (set if any pa is_dirty)

        Notes:
        ------
        This is the most important function as on proper setup,
        the solver will call this function. This in turn will calculate
        the locator caches if required thus providing us with up to date
        neighbor locators.

        """

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
        """ Add an interaction between a source and desti particle array

        Algorithm:
        ----------
        check in the cache if it already exists
        if not
            create an appropriate locator and add to the cache
        else
            explicitly enable caching for the locator (it exists)

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

    def cache_neighbors(self, kernel):
        cdef dict locator_cache = self.particle_locator_cache
        cdef NbrParticleLocatorBase loc
        cdef ParticleArray dest, source
        cdef long np, i, j, nnbrs
        cdef double h, w

        cdef DoubleArray xd, yd, zd, xs, ys, zs, hs, hd

        cdef Point _dst, _src, grad

        for loc in locator_cache.values():
            dest = loc.dest
            source = loc.source
            
            np = dest.get_number_of_particles()

            loc.particle_neighbors = {}
            loc.kernel_function_evaluation = {}
            loc.kernel_gradient_evaluation = {}

            xd = dest.get_carray('x')
            yd = dest.get_carray('y')
            zd = dest.get_carray('z')
            hd = dest.get_carray('h')
          
            xs = source.get_carray('x')
            ys = source.get_carray('y')
            zs = source.get_carray('z')
            hs = source.get_carray('h')

            for i in range(np):
                nbrs = LongArray()
                loc.py_get_nearest_particles(i, nbrs, exclude_self=False)
                loc.particle_neighbors[i] = nbrs

                _dst = Point(xd.data[i], yd.data[i], zd.data[i])

                if nbrs.length > 0:
                    loc.kernel_function_evaluation[i] = {}
                    loc.kernel_gradient_evaluation[i] = {}

                    # compute the kenrel evaluations
                
                    for j in range(nbrs.length):
                        s_id = nbrs.data[j]
                        
                        _src = Point(xs.data[s_id],ys.data[s_id],zs.data[s_id])

                        h = 0.5 * (hd[i] + hs[s_id])
                        
                        grad = Point()

                        w = kernel.py_function(_dst, _src, h)
                        kernel.py_gradient(_dst, _src, h, grad)
                        
                        loc.kernel_function_evaluation[i][s_id] = w
                        loc.kernel_gradient_evaluation[i][s_id] = grad

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        """ Update the status of the neighbor locators. """
        return self.update()
