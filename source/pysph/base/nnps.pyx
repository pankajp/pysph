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
    
    output_array = nbrl.get_nearest_particles(2)

"""

cdef extern from 'math.h':
    cdef double fabs(double)
    cdef double sqrt(double)

from pysph.base.point cimport cPoint_new, cPoint_distance2, cPoint_distance

# logger imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.carray cimport LongArray, DoubleArray
from pysph.base.point cimport Point
from pysph.base.particle_array cimport ParticleArray
from pysph.base.cell cimport CellManager, Cell, find_cell_id
from pysph.base.polygon_array cimport PolygonArray

cimport numpy
import numpy as np

from cpython.list cimport *
from cpython.dict cimport *

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
# `NeighborLocatorType` class.
###############################################################################
class NeighborLocatorType:
    """ An Empty class to emulate an Enum for the neighbor locator types """

    SPHNeighborLocator = 0
    NSquareNeighborLocator = 1
    DSMCNeighborLocator = 2

    def __init__(self):
        raise SystemError, 'Do not instantiate the EntityTypes class'



###############################################################################
# `NbrParticleLocatorBase` class.
###############################################################################
cdef class NbrParticleLocatorBase:

    # Defined in the .pxd file
    # cdef public CellManager cell_manager
    # cdef public ParticleArray source
    # cdef public int source_index

    """
    Nearest neighbor locator for a source particle array. 
    The main use is to return nearest particles for a query point.

    Members:
    --------

    cell_manager -- CellManager to manage the binning
    source -- particle array in which neighbors will be searched for.
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
        self.cell_manager = cell_manager
        self.source = source
        self.source_index = -1

        if self.cell_manager is not None:

            self.source_index = cell_manager.array_indices.get(source.name)
            if self.source_index is None:
                msg = 'Source %s does not exist'%(self.source.name)
                raise ValueError, msg

    cdef int get_nearest_particles_to_point(
        self, cPoint pnt, double radius, LongArray output_array, 
        long exclude_index=-1) except -1:
        
        if self.locator_type == NeighborLocatorType.SPHNeighborLocator:
            return self.get_nearest_particles_to_point_sph(pnt, radius,
                                                           output_array,
                                                           exclude_index)

        if self.locator_type == NeighborLocatorType.NSquareNeighborLocator:
            return self.get_nearest_particles_to_point_all(pnt, output_array,
                                                           exclude_index)

            
    cdef int get_nearest_particles_to_point_sph(
        self, cPoint pnt, double radius, LongArray output_array, 
        long exclude_index=-1) except -1: 

        """ Get the nearest particles to a given point

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

        # make sure cell manager is updated. That is, perform the binning

        self.cell_manager.update()
        
        # get the potential cell_list from the cell manager

        self.cell_manager.get_potential_cells(pnt, radius, cell_list)
        
        # now extract the exact points from the cell_list

        self._get_nearest_particles_from_cell_list(pnt, radius, cell_list,
                                                   output_array, exclude_index)
        return 0

    cdef int get_nearest_particles_to_point_all(
        self, cPoint pnt, LongArray output_array, 
        long exclude_index=-1) except -1:

        cdef long nnbrs = self.source.get_number_of_particles()
        cdef long i

        output_array.resize(nnbrs)

        for i in range(nnbrs):
            output_array.data[i] = i

        if exclude_index >= 0:
            output_array.data[exclude_index], output_array.data[nnbrs-1] = \
                                              output_array.data[nnbrs-1], \
                                              output_array.data[exclude_index]
            
            output_array.resize(nnbrs - 1)

        return 0

    cdef int get_nearest_particles_to_point_dsmc(
        self, cPoint pnt, LongArray output_array,
        long exclude_index=-1) except -1:

        pass

    cdef int _get_nearest_particles_from_cell_list(
        self, cPoint pnt,
        double radius, list cell_list, LongArray output_array,
        long exclude_index=-1) except -1:
        
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
        cdef DoubleArray xa, ya, za
        cdef LongArray src_indices
        cdef Cell cell

        cdef str xc, yc, zc
        cdef long idx
        cdef int i, j

        cdef cPoint src, dst

        cdef double radius2 = radius*radius

        cdef int ncells = PyList_Size( cell_list )
        
        xc = self.cell_manager.coord_x
        yc = self.cell_manager.coord_y
        zc = self.cell_manager.coord_z

        xa = self.source.get_carray(xc)
        ya = self.source.get_carray(yc)
        za = self.source.get_carray(zc)
        
        dst.x = pnt.x; dst.y = pnt.y; dst.z = pnt.z
        
        if exclude_index >= 0:
            for j in range( ncells ):
                cell = cell_list[j]
                src_indices = cell.index_lists[self.source_index]
                
                for i in range(src_indices.length):
                    idx = src_indices.data[i]

                    src.x = xa.data[idx]
                    src.y = ya.data[idx]
                    src.z = za.data[idx]

                    if cPoint_distance2(src, dst) < radius2:
                        if idx != exclude_index:
                            output_array.append(idx)
    
        else:
            for j in range( ncells ):
                cell = cell_list[j]
                src_indices = cell.index_lists[self.source_index]
                
                for i in range(src_indices.length):
                    idx = src_indices.data[i]

                    src.x = xa.data[idx]
                    src.y = ya.data[idx]
                    src.z = za.data[idx]

                    if cPoint_distance2(src, dst) < radius2:
                        output_array.append(idx)
                    
        return 0

    cpdef set_locator_type(self, int locator_type):
        self.locator_type = locator_type
    
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
            pnt.data, radius, output_array, exclude_index)

###############################################################################
# `FixedDestNbrParticleLocator` class.
###############################################################################
cdef class FixedDestNbrParticleLocator(NbrParticleLocatorBase):

    # Defined in the .pxd file
    # cdef public ParticleArray dest
    # cdef readonly double radius_scale
    # cdef public int dest_index
    # cdef public DoubleArray d_h, d_x, d_y, d_z


    """
    Nearest neighbor locator between a destination and a source particle
    array. The query points will be the destination particle positions.
    It is assumed that the particles have constant smoothing length.

    Data Members:
    -------------    
    dest - The ParticleArray which provides as the query points.
    radius_scale - kernel support radius factor `kfac`
    dest_index - destination index in the cell manager

    d_h, d_x,d_y,d_z -- destination particles properties
    particle_cache -- a cache object. one LongArray for each particle in dest
    is_dirty -- flag to recompute the cache

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 double radius_scale, CellManager cell_manager=None,
                 str h='h'):
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
        self.radius_scale = radius_scale
        self.h = h

        self.dest_index = -1
                
        self.particle_cache = list()
        self.is_dirty = True

        self.d_h = None
        self.d_x = None
        self.d_y = None
        self.d_z = None

        # find the dest index in the CellManager
        
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

    cdef LongArray get_nearest_particles(self, long dest_p_index, 
                                   bint exclude_self=False):
        """
        Get the nearest source particles to particles in `dest_p_index`

        Parameters:
        -----------

        dest_p_index -- index of query destination particle
        exclude_self -- indicates if dest_p_index be excluded from
                        output_array.

        Algorithm:
        ----------
        return cache data after removing self if exclude_self is true
        
        Note:
        -----
        
        If exclude_self is True then a copy of the internal cached array is
        returned after removing last element, else the cached array is returned.
        Do **NOT** modify the returned array
        exclude_self argument is deprecated (lot slower)
        The last index in returned array is guaranteed to be self particle if
        src and dest arrays are same. This can be used to exclude it.
        
        """
        cdef LongArray index_array, output_array
        cdef int i
        cdef long *data
        cdef cPoint pnt

        if not self.locator_type == NeighborLocatorType.NSquareNeighborLocator:
        
            # update internal data. Calculate the particle cache
        
            self.update()
        
            # return data from cache.

            output_array = index_array = self.particle_cache[dest_p_index]

            if self.dest is self.source:
                if exclude_self:
                    # cached array has self index as last value
                    output_array = LongArray(index_array.length-1)
                    # TODO: can use memcpy
                    for i in range(index_array.length-1):
                        output_array[i] = index_array.data[i]

        if self.locator_type == NeighborLocatorType.NSquareNeighborLocator:
            output_array = LongArray()
            index_array = LongArray()

            pnt.x = self.d_x.data[dest_p_index]
            pnt.y = self.d_y.data[dest_p_index]
            pnt.z = self.d_z.data[dest_p_index]

            if (self.dest is self.source) and (exclude_self):
                self.get_nearest_particles_to_point_all(
                    pnt, output_array, dest_p_index)
            else:
                self.get_nearest_particles_to_point_all(
                    pnt, output_array, -1)
        
        return output_array
    
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
        cdef long exclude_index = -1
        cdef cPoint pnt
        
        pnt.x = self.d_x.data[dest_p_index]
        pnt.y = self.d_y.data[dest_p_index]
        pnt.z = self.d_z.data[dest_p_index]
        
        cdef double eff_radius = self.d_h.data[dest_p_index] * self.radius_scale
        
        if self.source is self.dest:
            if exclude_self:
                exclude_index = dest_p_index

        return NbrParticleLocatorBase.get_nearest_particles_to_point(
            self, pnt, eff_radius, output_array, exclude_index)

    cdef void update_status(self):
        """ Set to dirty if either source or destination is dirty."""
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

            # resize the cache if necessary
            
            num_particles = self.dest.get_number_of_particles()
            if len(self.particle_cache) != num_particles:
                self.particle_cache = []
                for i in range(num_particles):
                    PyList_Append( self.particle_cache, LongArray() )

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
            
            # keep particle at the last index for faster exclude_self
            if self.source == self.dest:
                for j in range(index_cache.length-1):
                    if i == index_cache[j]:
                        # swap last index with i
                        index_cache[j] = index_cache[index_cache.length-1]
                        index_cache[index_cache.length-1] = i
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

    def py_get_nearest_particles(self, long dest_p_index,
                                 bint exclude_self=False):

        """ Return nearese neighbors for a particle

        Parameters:
        ------------

        dest_p_index -- id of the particle in dest whose neighbors are to be
        found. 

        exclude_self -- indicates if dest_p_index be excluded from
        output_array.
        
        """
        return self.get_nearest_particles(dest_p_index, exclude_self)


###############################################################################
# `VarHNbrParticleLocator` class.
###############################################################################
cdef class VarHNbrParticleLocator(FixedDestNbrParticleLocator):

    # Defined in the .pxd file
    # cdef FixedDestNbrParticleLocator _rev_locator

    """ A particle locator in which the interaction radii for
    particles may be varying and symmetric interactions are
    required. That is, if particle a is a neighbor of particle b then
    particle b must be a neighbor for particle a.
    
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

    cdef int update(self) except -1:
        """ Compute contents of the cache. 
        
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
                retrieve from the cache the destination index_cache
                add source to index_cache if not present

        
        Notes:
        ------
        index_cache refers to the cache to be used
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
            rev_cache = self._rev_locator.particle_cache[j]

            for k in range(rev_cache.length):
                index_cache = self.particle_cache[rev_cache.data[k]]
                if index_cache.index(j) < 0:
                    index_cache.append(j)
        return 0

###############################################################################
# `NNPSManager` class.
###############################################################################
cdef class NNPSManager:

    # Defined in the .pxd file
    # cdef public CellManager cell_manager
    # cdef public dict particle_locator_cache
    # cdef public bint variable_h

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

    def __init__(self, CellManager cell_manager=None,
                 bint variable_h=False, str h='h',
                 int locator_type=NeighborLocatorType.SPHNeighborLocator):
        self.cell_manager = cell_manager
        self.variable_h = variable_h
        self.h = h
        self.locator_type = locator_type
        
        self.particle_locator_cache = dict()
        
        self.polygon_cache_manager = CachedNbrPolygonLocatorManager(
            self.cell_manager)
    
    cpdef NbrParticleLocatorBase get_neighbor_particle_locator(
        self, ParticleArray source, ParticleArray dest=None, double
        radius_scale=1.0):
        """ Return an appropriate neighbor particle locator. 

        Notes:
        ------
        if no destination is specified return a base locator
        else check if the cache already has a locator between two arrays
        and return that locator if it exists
        
        """
        if dest is None:
            loc = NbrParticleLocatorBase(source, self.cell_manager)
            loc.set_locator_type(self.locator_type)
            return loc
        else:
            self.add_interaction(source, dest, radius_scale)
            return self.get_cached_locator(source.name, dest.name, radius_scale)
    
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

        # Calls update_status on each CachedParticleLocators being maintained.

        cdef FixedDestNbrParticleLocator loc
        cdef int i, num_caches
        
        #cdef list cache_list = self.particle_locator_cache.values()

        cdef list cache_list = PyDict_Values( self.particle_locator_cache )
        num_caches = PyList_Size( cache_list )

        for i in range(num_caches):
            loc = cache_list[i]
            loc.update_status()

        # update the cell manager
        
        self.cell_manager.update_status()

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
            if not self.variable_h:
                loc = FixedDestNbrParticleLocator(source, dest, radius_scale,
                                   cell_manager=self.cell_manager, h=self.h)

                loc.set_locator_type(self.locator_type)
                
            else:
                loc = VarHNbrParticleLocator(source, dest, radius_scale,
                                 cell_manager=self.cell_manager, h=self.h)

                loc.set_locator_type(self.locator_type)
                
            self.particle_locator_cache[t] = loc
    
    cpdef FixedDestNbrParticleLocator get_cached_locator(self,
                str source_name, str dest_name, double radius):
        """Returns a cached object if it exists, else returns None."""
        return self.particle_locator_cache.get((source_name,dest_name,radius))

    cpdef NbrPolygonLocatorBase get_neighbor_polygon_locator(
        self,PolygonArray source, ParticleArray dest=None,
        double radius_scale=1.0):

        """Returns an appropriate neighbor polygon locator. """
        msg = 'NNPSManager::get_neighbor_polygon_locator'
        raise NotImplementedError, msg

    ######################################################################
    # python wrappers.
    ######################################################################
    def py_update(self):
        """ Update the status of the neighbor locators. """
        return self.update()
