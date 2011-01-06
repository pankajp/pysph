# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.point cimport *
from pysph.base.carray cimport *
from pysph.base.particle_array cimport ParticleArray

# python c-api imports
from cpython cimport *

cdef extern from 'math.h':
    int abs(int)

cdef extern from 'limits.h':
    cdef int INT_MAX
    cdef double ceil(double)
    cdef double floor(double)
    cdef double fabs(double)


# numpy import
cimport numpy
import numpy

def INT_INF():
    return INT_MAX

def py_real_to_int(real_val, step):
    return real_to_int(real_val, step)

def py_find_cell_id(origin, pnt, cell_size):
    return find_cell_id(origin, pnt, cell_size)

cdef inline int real_to_int(double real_val, double step):
    """ Return the bin index to which the given position belongs.

    Parameters:
    -----------
    val -- The coordinate location to bin
    step -- the bin size    

    Example:
    --------
    real_val = 1.5, step = 1.0 --> ret_val = 1

    real_val = -0.5, step = 1.0 --> real_val = -1
    
    """
    cdef int ret_val
    if real_val < 0.0:
        ret_val =  <int>(real_val/step) - 1
    else:
        ret_val =  <int>(real_val/step)

    return ret_val

cdef inline IntPoint find_cell_id(Point origin, Point pnt, double cell_size):
    """ Find the cell index for the corresponding point 

    Parameters:
    -----------
    origin -- a cell's origin
    pnt -- the point for which the index is sought
    cell_size -- the cell size to use
    id -- output parameter holding the cell index 

    Algorithm:
    ----------
    performs a box sort based on the point, cell origin and cell size

    Notes:
    ------
    Uses the function  `real_to_int`
    
    """
    return IntPoint_new(real_to_int(pnt.x-origin.x, cell_size),
                        real_to_int(pnt.y-origin.y, cell_size),
                        real_to_int(pnt.z-origin.z, cell_size))

def py_construct_immediate_neighbor_list(cell_id, neighbor_list,
                                         include_self=True, distance=1):
    """ Construct a list of cell ids neighboring the given cell."""
    construct_immediate_neighbor_list(cell_id, neighbor_list, include_self,
                                      distance)

cdef inline void construct_immediate_neighbor_list(IntPoint cell_id, list
               neighbor_list, bint include_self=True, int distance=1): 
    """Return the 27 nearest neighbors for a given cell when distance = 1"""
    cdef list cell_list = []
    cdef int i,j,k
    for i in range(-distance, distance+1):
        for j in range(-distance, distance+1):
            for k in range(-distance, distance+1):
                cell_list.append(IntPoint(cell_id.x+i, cell_id.y+j,
                                          cell_id.z+k))
    if not include_self:
        cell_list.remove(cell_id)
    neighbor_list.extend(cell_list)

def py_construct_face_neighbor_list(cell_id, neighbor_list, include_self=True):
    """ Construct a list of cell ids, which share a face(3d) or 
    edge(2d) with the given cell_id.
    """
    construct_face_neighbor_list(cell_id, neighbor_list, include_self)

cdef inline construct_face_neighbor_list(IntPoint cell_id, list neighbor_list,
                                         bint include_self=True, int
                                         dimension=3):
    """ Construct a list of cell ids, which share a face(3d) or 
    edge(2d) with the given cell_id.

    """
    if include_self:
        neighbor_list.append(cell_id)
    
    # face neighbors in x dimension.
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y, cell_id.z))

    if dimension < 3:
        neighbor_list.append(IntPoint(cell_id.x, cell_id.y+1, cell_id.z))
        neighbor_list.append(IntPoint(cell_id.x, cell_id.y-1, cell_id.z))

    if dimension == 3:
        neighbor_list.append(IntPoint(cell_id.x, cell_id.y, cell_id.z+1))
        neighbor_list.append(IntPoint(cell_id.x, cell_id.y, cell_id.z-1))
                                      
def py_cell_encloses_sphere(IntPoint id, Point world_origin, double cell_size,
                            Point pnt, double radius):
    """Check if sphere of `radius` center 'pnt' is enclosed by a cell."""
    return cell_encloses_sphere(id, world_origin, cell_size, pnt, radius)

cdef inline bint cell_encloses_sphere(IntPoint id, Point world_origin,
                                double cell_size, Point pnt, double radius):
    """
    Checks if sphere of `radius` centered at 'pnt' is completely 
    enclosed by a cell.
    
    Parameters:
    -----------
    id -- id of the cell.
    world_origin -- origin with respect to which this 'id' was calculated.
    cell_size -- size of the sides of the cells.
    pnt -- center of sphere.
    radius -- radius of sphere.

    Algorithm:
    ----------

    for each vertex of the cell
        if distance of vertex from pnt is less than or equal to radius
            return false

     return true

    """

    cdef double distance
    cdef Point cell_vertex = Point()
    cdef int i,j,k
    
    # find the first point of the cell.
    cell_vertex.x = world_origin.x + id.x*cell_size
    cell_vertex.y = world_origin.y + id.y*cell_size
    cell_vertex.z = world_origin.z + id.z*cell_size
    
    for i in range(0,2):
        for j in range(0,2):
            for k in range(0,2):
                cell_vertex.set(world_origin.x + (id.x+i)*cell_size,
                                world_origin.y + (id.y+j)*cell_size,
                                world_origin.z + (id.z+k)*cell_size,
                                )
                distance = cell_vertex.euclidean(pnt)
                if distance > radius:
                    # make sure its not very close.
                    if fabs(distance-radius) < 1e-09:
                        return False
    return True

###############################################################################
# `Cell` class.
###############################################################################
cdef class Cell:
    """
    The Cell class is a generalization of the standard bins used for box sort.
    A Cell maintains an index_list of particle indices corresponding to 
    particle arrays.

    Data Attributes:
    ----------------
    IntPoint id -- a unique identifier for the cell
    double cell_size --  extents of the cell
    list arrays_to_bin -- particle arrays to be binned within this cell.
    list index_list -- mapping of particle indices contained to arrays_to_bin
    int jump_tolerance -- limit the particle's movement 

    Notes:
    ------
    The cell defines an origin with respect to which it has an extent 
    defined to be it's `size`. Indices of particles corresponding to 
    ParticleArrays in `arrays_to_bin` are maintained in `index_lists`, one
    for each particle array.
    When a cell's update method is called, the cell checks for particles 
    that have possibly escaped it's limits and creates a new cell id for 
    those partices, while removing the corresponding indices from the 
    appropriate list in `index_lists`. This new cell data can then 
    be used in any which way one wants.    

    """

    #Defined in the .pxd file 
    #cdef public IntPoint id
    #cdef public double cell_size
    #cdef public CellManager cell_manager
    #cdef public list arrays_to_bin
    #cdef public Point origin

    #cdef public int jump_tolerance
    #cdef public list index_lists

    def __init__(self, IntPoint id, CellManager cell_manager=None, double
                 cell_size=0.1, int jump_tolerance=1):

        self.id = IntPoint()

        self.id.x = id.x
        self.id.y = id.y
        self.id.z = id.z

        self.cell_size = cell_size

        self.coord_x = 'x'
        self.coord_y = 'y'
        self.coord_z = 'z'

        self.jump_tolerance = jump_tolerance

        self.arrays_to_bin = []

        self.origin = Point(0., 0., 0.)
        
        self.set_cell_manager(cell_manager)
    
    def __str__(self):
        return 'Cell(id=%s,size=%g,np=%d)' %(self.id, self.cell_size,
                    self.get_number_of_particles())
    
    def __repr__(self):
        return 'Cell(id=%s,size=%g,np=%d)' %(self.id, self.cell_size,
                    self.get_number_of_particles())

    cpdef set_cell_manager(self, CellManager cell_manager):
        self.cell_manager = cell_manager

        if self.index_lists is None:
            self.index_lists = list()

        if self.cell_manager is None:
            self.arrays_to_bin[:] = []
            self.index_lists[:] =[]
        else:
            self.arrays_to_bin[:] = self.cell_manager.arrays_to_bin
            self.coord_x = self.cell_manager.coord_x
            self.coord_y = self.cell_manager.coord_y
            self.coord_z = self.cell_manager.coord_z
            self.origin.x = self.cell_manager.origin.x
            self.origin.y = self.cell_manager.origin.y
            self.origin.z = self.cell_manager.origin.z
            self._init_index_lists()
                         
    cpdef get_centroid(self, Point centroid):
        """Returns the centroid of this cell in 'centroid'.

        Notes:
        ------
        The centroid in any coordinate direction is defined to be the
        origin plus half the cell size in that direction

        """
        centroid.x = self.origin.x + (<double>self.id.x + 0.5)*self.cell_size
        centroid.y = self.origin.y + (<double>self.id.y + 0.5)*self.cell_size
        centroid.z = self.origin.z + (<double>self.id.z + 0.5)*self.cell_size

    cpdef Cell get_new_sibling(self, IntPoint id):
        """Return a new cell with the given id

        Notes:
        ------
        The new cell size is taken to be the same as this cell's size

        """
        cdef Cell cell = Cell(id=id, cell_manager=self.cell_manager,
                              cell_size=self.cell_size,
                              jump_tolerance=self.jump_tolerance)
        return cell

    cpdef int update(self, dict data) except -1:
        """
        Locate particles that have escaped this Cell.
        
        Parameters
        ----------
         
        data -- output parameter. A dictionary to store cell ids and
                particles that have escaped this cell
        
        Algorithm::
        -----------
        
        for pa in arrays_to_bin
            if pa is dirty # particles have moved
                for those particles that are in this Cell
                    if particle has escaped 
                        mark this particle for removal from this Cell
                        find the cell to which it has escaped
                        create a new entry for that cell if required
                        add this particle's index to the newly created cell
                        
                remove marked particles 

        """
        cdef int i,j
        cdef int num_arrays = len(self.arrays_to_bin)
        cdef ParticleArray parray
        cdef DoubleArray xa, ya, za
        cdef double *x, *y, *z
        cdef LongArray index_array, index_array1
        cdef LongArray to_remove = LongArray()
        cdef long *indices
        cdef long num_particles
        cdef Point pnt = Point()
        cdef IntPoint id
        cdef Cell cell
        cdef IntPoint pdiff
        cdef str msg
        
        for i in range(num_arrays):
            
            parray = self.arrays_to_bin[i]
            
            # check if parray has been modified.
            if parray.is_dirty == False:
                continue

            num_particles = parray.get_number_of_particles()

            xa = parray.get_carray(self.coord_x)
            ya = parray.get_carray(self.coord_y)
            za = parray.get_carray(self.coord_z)
            
            x = xa.get_data_ptr()
            y = ya.get_data_ptr()
            z = za.get_data_ptr()

            index_array = self.index_lists[i]
            indices = index_array.get_data_ptr()

            to_remove.reset()
            
            for j from 0 <= j < index_array.length:
                # check if this particle is stale information.
                # probably removed by an outflow ?
                if indices[j] >= num_particles:
                    to_remove.append(j)
                else:
                    pnt.x = x[indices[j]]
                    pnt.y = y[indices[j]]
                    pnt.z = z[indices[j]]

                    # find the cell containing this point

                    id = find_cell_id(self.origin, pnt, self.cell_size)
                    
                    if id.is_equal(self.id):
                        continue

                    to_remove.append(j)
                    
                    # has the particle moved too far??

                    pdiff = self.id.diff(id)
                    self.check_jump_tolerance(id)

                    #create new cell if id doesn't exist and add particles

                    cell = data.get(id)
                    if cell is None:
                        cell = self.get_new_sibling(id)
                        data[id.copy()] = cell
                        
                    cell_index_array = cell.index_lists[i]
                    cell_index_array.append(indices[j])

            # now remove all escaped and invalid particles.

            index_array.remove(to_remove.get_npy_array())

        return 0

    cpdef long get_number_of_particles(self):
        """ Return the total number of particles in the Cell 

        Note:
        -----
        This is the cumulative sum of particle arrays if multiple 
        arrays exist. 
        
        """
        cdef int i, num_arrays
        cdef long num_particles = 0
        cdef LongArray arr

        num_arrays = len(self.index_lists)
        
        for i in range(num_arrays):
            arr = self.index_lists[i]
            num_particles += arr.length
        
        return num_particles

    cpdef bint is_empty(self):
        if self.get_number_of_particles() == 0:
            return True
        else:
            return False

    cpdef int add_particles(self, Cell cell) except -1:
        """ Add particle indices from a given cell

        Parameters:
        -----------
        cell -- cell from which to add particle indices.

        Notes:
        ------
        The input cell is assumed to have the same set of particle arrays. 
        This is why the check_particle_id is called which would raise an 
        error if the particle id is negative or more than the number of 
        particles in the particle array.

        """
        cdef int i, j, id
        cdef int num_arrays, num_particles
        cdef LongArray dest_array
        cdef LongArray source_array
        cdef ParticleArray parr
        
        num_arrays = len(self.arrays_to_bin)
        
        for i in range(num_arrays):
            parr = self.arrays_to_bin[i]
            num_particles = parr.get_number_of_particles()
            source_array = cell.index_lists[i]
            dest_array = self.index_lists[i]
            for j from 0 <= j < source_array.length:
                id = source_array.get(j)
                self.check_particle_id(id, num_particles)
                dest_array.append(id)

        return 0

    cpdef int clear(self) except -1:
        """Clear the index_lists"""
        self.index_lists[:] = []   
        return 0

    cpdef insert_particles(self, int parray_id, LongArray indices):
        """
        Insert particle indices of the parray given by "parray_id" from the
        array "indices" into the cell. 

        Parameters:
        -----------

        parray_id -- id of the particle array in `arrays_to_bin`
        indices -- particle ids to extend the array

        """
        cdef LongArray index_array = self.index_lists[parray_id]
        index_array.extend(indices.get_npy_array())
        
    cpdef get_particle_ids(self, list particle_id_list):
        """
        Get the particle indices corresponding to arrays_to_bin  

        Parameters:
        -----------
        particle_id_list - output parameter. Particle ids will be appended.

        """

        cdef int num_arrays = 0
        cdef int i = 0
        cdef LongArray source, dest

        num_arrays = len(self.arrays_to_bin)

        if len(particle_id_list) == 0:
            # create num_arrays LongArrays
            for i in range(num_arrays):
                particle_id_list.append(LongArray())
        
        for i in range(num_arrays):
            dest = particle_id_list[i]
            source = self.index_lists[i]
            dest.extend(source.get_npy_array())

    cpdef get_particle_counts_ids(self, list particle_id_list,
                                  LongArray particle_counts):
        """
        Finds the indices of particles for each particle array in arrays_to_bin
        contained within this cell. Also returns the number of particles ids
        that were added in this call for each of the arrays.

        **Parameters**

            - particle_id_list - output parameter, should contain one LongArray
            for each array in arrays_to_bin. Particle ids will be appended 
            to the LongArrays.

            - counts - output parameter, A LongArray with one value per array in
            arrays_to_bin. Each entry will contain the number of particle ids
            that were appended to each of the arrays in particle_id_list in this
            call to get_particle_counts_ids.
        """
        cdef int num_arrays = 0
        cdef int i = 0
        cdef LongArray source, dest

        num_arrays = len(self.arrays_to_bin)

        if len(particle_id_list) == 0:
            for i in range(num_arrays):
                particle_id_list.append(LongArray(0))
                
        if particle_counts.length == 0:
            particle_counts.resize(num_arrays)
            # set the values to zero
            particle_counts._npy_array[:] = 0

        for i in range(num_arrays):
            dest = particle_id_list[i]
            source = self.index_lists[i]
            dest.extend(source.get_npy_array())
            particle_counts.data[i] += source.length
     
    cpdef clear_indices(self, int parray_id):
        """Clear the particles ids of parray_id stored in the cells."""
        cdef LongArray index_array = self.index_lists[parray_id]
        index_array.reset()

    cdef _init_index_lists(self):
        """initialize the index_lists.

        An empty LongArray is created for each ParticleArray in 
        `arrays_to_bin`

        """
        cdef int i
        cdef int num_arrays

        if self.cell_manager is None:
            return

        num_arrays = len(self.arrays_to_bin)
        self.index_lists[:] = []
        
        for i in range(num_arrays):
            self.index_lists.append(LongArray())

    def check_jump_tolerance(self, IntPoint id):
        """ Check if the particle has moved more than the jump tolerance """

        pdiff = self.id.diff(id)                    
        if (abs(pdiff.x) > self.jump_tolerance or abs(pdiff.y) >
            self.jump_tolerance or abs(pdiff.z) >
            self.jump_tolerance):
            
            msg = 'Particle moved by more than one cell width\n'
            #msg += 'Point (%f, %f, %f)\n'%(pnt.x, pnt.y, pnt.z)
            msg += 'self id : (%d, %d, %d)\n'%(self.id.x, self.id.y,
                                               self.id.z)
            msg += 'new id  : (%d, %d, %d)\n'%(id.x, id.y, id.z)
            msg +='J Tolerance is : %s, %d\n'%(self, 
                                               self.jump_tolerance)
            raise RuntimeError, msg

    def check_particle_id(self, int id, int num_particles):
        """ Sanity check on the particle index """
        if id >= num_particles or id < 0:
            msg = 'trying to add invalid particle\n'
            msg += 'num_particles : %d\n'%(num_particles)
            msg += 'pid : %d\n'%(id)
            raise RuntimeError, msg

    def py_add_particles(self, Cell cell):
        self.add_particles(cell)
        
    def py_update(self, dict data):
        self.update(data)

    def py_get_number_of_particles(self):
        return self.get_number_of_particles()

    def py_is_empty(self):
        return self.is_empty()

    def py_get_centroid(self, Point pnt):
        self.get_centroid(pnt)

    def py_clear(self):
        self.clear()

###############################################################################


###############################################################################
# `CellManager` class.
###############################################################################
cdef class CellManager:
    """Cell Manager class"""
    # FIXME:
    # 1. Simple API function to add an array to bin _BEFORE_ the cell manager
    # has been initialized.
    #
    # 2. Provision to add new array to bin  _AFTER_ the cell manager has once
    # been initialized. Will require some dirty changes to all cells that have
    # already been created. Whether this feature is required can be thought over
    # again. 

    #Defined in the .pxd file
    #cdef public Point origin
    #cdef public double cell_size
    #cdef public bint is_dirty    
    #cdef public dict array_indices
    #cdef public list arrays_to_bin
    #cdef public dict cells_dict
    #cdef public double min_cell_size, max_cell_size
    #cdef public int jump_tolerance
    #cdef public bint initialized    
    #cdef public str coord_x, coord_y, coord_z

    def __init__(self, list arrays_to_bin=[], double min_cell_size=-1.0,
                 double max_cell_size=0.5, Point origin=Point(0,0,0),
                 bint initialize=True, double max_radius_scale=2.0):
        
        self.origin = Point()

        self.origin.x = origin.x
        self.origin.y = origin.y
        self.origin.z = origin.z

        self.max_radius_scale = max_radius_scale
        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size

        self.array_indices = dict()
        self.arrays_to_bin = list()
        self.arrays_to_bin[:] = arrays_to_bin
        
        self.cells_dict = dict()
        
        self.coord_x = 'x'
        self.coord_y = 'y'
        self.coord_z = 'z'

        self.jump_tolerance = 1
        self.cell_size = 0
        
        self.is_dirty = True

        self.initialized = False

        if initialize == True:
            self.initialize()

    cpdef set_jump_tolerance(self, int jump_tolerance):
        """Sets the jump tolerance value of the cells."""
        cdef int i
        cdef Cell cell
        self.jump_tolerance = jump_tolerance

        if len(self.cells_dict) == 0:
            logger.debug('Cells dict empty')
            return

        num_cells = len(self.cells_dict)
        cdef list cells_list = self.cells_dict.values()
        
        for i in range(num_cells):
            cell = cells_list[i]
            cell.jump_tolerance = jump_tolerance

    cpdef int update(self) except -1:
        """Update the cell manager if particles have changed (is_dirty)"""
        cdef int i, num_arrays
        cdef ParticleArray parray
        
        if self.is_dirty:

            # update the cells.

            self.cells_update()
        
            # delete empty cells if any.

            self.delete_empty_cells()

            # reset the dirty bit of all particle arrays.

            num_arrays = len(self.arrays_to_bin)
            for i in range(num_arrays):
                parray = self.arrays_to_bin[i]
                parray.set_dirty(False)

            self.is_dirty = False

        return 0
    
    
    cpdef int cells_update(self) except -1:
        """Update particle information.
        
        Algorithm:
        ----------
        All escaping particles (in newly created cells) are got from 
        all the cells
        
        For all new cells returned, if some already exist, the data is
        merged with them.
        
        If a newly created cell does not exists, it is added.
        
        Notes:
        ------
        Called from update()
        
        """

        cdef int i
        cdef int num_cells = len(self.cells_dict)
        cdef dict collected_data
        cdef Cell smaller_cell, smaller_cell_1
        cdef list cell_list
        cdef list smaller_cell_list = self.cells_dict.values()

        collected_data = dict()
        
        # collect all escaped particles from cells.
        for i in range(num_cells):
            smaller_cell = smaller_cell_list[i]
            smaller_cell.update(collected_data)

        cell_list = collected_data.values()
        num_cells = len(cell_list)

        for i in range(num_cells):
            smaller_cell = cell_list[i]
            
            smaller_cell_1 = self.cells_dict.get(smaller_cell.id)
            if smaller_cell_1 is None:
                # meaning there does not exist any cell for the
                # region occupied by smaller_cell, so we just
                # add 'smaller_cell' to the cell list
                self.cells_dict[smaller_cell.id] = smaller_cell
            else:
                smaller_cell_1.add_particles(smaller_cell)

        return 0
    
    cpdef add_array_to_bin(self, ParticleArray parr):
        """add arrays to the CellManager (before initialization)"""
        if self.initialized == True:
            msg = 'Cannot add array to bin\n'
            msg +='cell manager already initialized'
            logger.error(msg)
            raise SystemError, msg
        else:
            if self.arrays_to_bin.count(parr) == 0:
                self.arrays_to_bin.append(parr)
                if parr.name == '':
                    logger.warn('particle array (%s) name not set'%(parr))
        
    cpdef int update_status(self):
        """Updates the is_dirty flag to indicate is an update is required.

        Algorithm:
        ----------
        if any particle array's is_dirty bit is True
            set cell manager's is_dirty bit to True

        Notes:
        ------
        Any module that may modify the particle data, should call the cell
        managers update_status function.

        """
        cdef int i,  num_arrays
        cdef ParticleArray parray
        
        num_arrays = len(self.arrays_to_bin)
        
        for i in range(num_arrays):
            parray = self.arrays_to_bin[i]
            if parray.is_dirty:
                self.is_dirty = True
                break

        return 0

    def set_dirty(self, bint value):
        """Sets/Resets the dirty flag."""
        self.is_dirty = value

    cpdef initialize(self):
        """
        Initialize the cell manager.
    
        Algorithm::
        -----------
        clear current data
        build the cell structure

    	Warning:
        ------
        All previously stored data will be cleared

    	"""
        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        # clear current data.

        self.clear()

        # setup some data structures.

        self._rebuild_array_indices()

        # setup the cells dict

        self._setup_cells_dict()
        
        # recompute cell sizes.

        self.compute_cell_size(self.min_cell_size, self.max_cell_size)
        
        # build cell.

        self._build_cell()

        # update

        self.update()

        # now reset the jump tolerance back to 1
        # we do not want particles to jump across
        # multiple cells.
        self._reset_jump_tolerance()

        self.initialized = True

    cpdef clear(self):
        """Clear the dictionary `cells_dict`."""
        self.cells_dict.clear()

    cpdef _setup_cells_dict(self):
        """ Create an empty dict for `cells_dict`"""
        self.cells_dict = dict()

    cpdef double compute_cell_size(self, double min_size, double max_size):
        # TODO: compute size depending on some variation of 'h'
        """
        # TODO: compute size depending on some variation of 'h'
        
    	Parameters:
        -----------
        min_size - smallest cell size needed.
        max_size - largest cell size needed.

        Algorithm::
        if min_size <= 0: choose 2*max_radius_scale*min_h
        else: choose min_size
        
        This is very simplistic method to find the cell sizes, derived solvers
        may want to use something more sophisticated or probably set the cell
        sizes manually.
        """
        # TODO: implement
        if min_size <= 0:
            min_h, max_h = self._compute_minmax_h()
            # arbitrarily set min_size as some multiple of min_h
            min_size = 2 * self.max_radius_scale * min_h
        if min_size <= 0:
            # default min_size, this should *NOT* happen
            min_size = 1.0
        self.cell_size = min_size
        logger.info('using cell size of %f'%(min_size))
        return self.cell_size
    
    def _compute_minmax_h(self):
        """
        Find the minimum 'h' value from all particle arrays of all entities.
        returns -1, -1 if computation fails such as no particles
        """
        cdef double min_h = 1e100, max_h = -1.0
        cdef bint size_computed = False
        for parr in self.arrays_to_bin:
            if parr is None or parr.get_number_of_particles() == 0:
                continue
            h = parr.h
            if h is None:
                continue
            min_h = min(numpy.min(h), min_h)
            max_h = max(numpy.max(h), max_h)
            size_computed = True

        if size_computed == False:
            logger.info('No particles found - using default cell sizes')
            return -1, -1
        
        return min_h, max_h
    
    cpdef _build_cell(self):
        """ Build the base cell and add it to `cells_dict`
        
        Algorithm::
        -----------
        create a cell at origon
        set the jump_tolerance to infinity
        add all particle indices to this cell

    	"""

        cdef int i, num_arrays, num_particles
        cdef ParticleArray parry
        cdef numpy.ndarray index_arr_source
        cdef LongArray index_arr
        cdef Cell cell
 
        # create a leaf cell with all particles.

        cell = Cell(id=IntPoint(0, 0, 0), cell_manager=self,
                             cell_size=self.cell_size,
                             jump_tolerance=INT_MAX)
        
        # now add all particles of all arrays to this cell.

        num_arrays = len(cell.arrays_to_bin)
        for i in range(num_arrays):
            parray = cell.arrays_to_bin[i]
            num_particles = parray.get_number_of_particles()

            index_arr_source = numpy.arange(num_particles, dtype=numpy.long)
            index_arr = cell.index_lists[i]
            index_arr.resize(num_particles)
            index_arr.set_data(index_arr_source)
        
        # now add a cell at the origin (contains all the particles)

        self.cells_dict.clear()
        self.cells_dict[IntPoint(0, 0, 0)] = cell
        
    cpdef _rebuild_array_indices(self):
        """ Reset the mapping from array name to array in arrays_to_bin """

        cdef int i 
        cdef int num_arrays = len(self.arrays_to_bin)
        cdef ParticleArray parr

        self.array_indices.clear()

        for i in range(num_arrays):
            parr  = self.arrays_to_bin[i]
            self.array_indices[parr.name] = i

    cdef int get_potential_cells(self, Point pnt, double radius,
                                 list cell_list) except -1:
        """
        Gets cell that will potentially contain neighbors the the given point.
    
    	Parameters:
        -----------
         
        pnt -- the point whose neighbors are to be found.
        radius -- the radius within which neighbors are to be found.
        cell_list -- output parameter, where potential cells are appended.

        Algorithm::
        -----------
        find the cell id to which the point belongs
        construct an immediate neighbor id list
        return cells corresponding to these cell ids
        
    	"""
        cdef IntPoint cell_id = IntPoint()
        cdef IntPoint id
        cdef list neighbor_list = list()
        cdef int i

        cell_id = find_cell_id(self.origin, pnt, self.cell_size)

        # construct ids of all neighbors around cell_id, this
        # will include the cell also.
        construct_immediate_neighbor_list(cell_id, neighbor_list,
                                          True, 
                                          <int>ceil(radius/self.cell_size))

        for i in range(len(neighbor_list)):
            id = neighbor_list[i]
            cell = self.cells_dict.get(id)
            if cell is not None:
                cell_list.append(cell)
        
        return 0

    cdef int _get_cells_within_radius(self, Point pnt, double radius,
                                        list cell_list) except -1:
        """
        Finds all cells within the given radius of pnt.

    	**Parameters**
         
         - pnt - point around which cells are to be searched for.
         - radius - search radius
         - cell_list - output parameter to add the found cells to.

        **Algorithm**::
        
         find the 6 farthest points (one in each direction from given point)
         
         find the cell ids corresponding to those points.
         
         enumerate all cell ids between the enclosing cells, checking if the
         cells are valid and adding them to the cell list.

    	**Notes**

    	**Helper Functions**

    	**Issues**

    	"""
        cdef int max_x, min_x, i
        cdef int max_y, min_y, j
        cdef int max_z, min_z, k

        cdef IntPoint max_cell = IntPoint()
        cdef IntPoint min_cell = IntPoint()
        cdef IntPoint diff
        cdef IntPoint id = IntPoint()
        cdef Point tmp_pt = Point()
        cdef IntPoint curr_id = IntPoint()

        cdef Cell cell

        # find the cell within which this point is located.
        curr_id = find_cell_id(self.origin, pnt, self.cell_size)

        tmp_pt.x = pnt.x - radius
        tmp_pt.y = pnt.y - radius
        tmp_pt.z = pnt.z - radius

        min_cell = find_cell_id(self.origin, tmp_pt, self.cell_size)

        tmp_pt.x = pnt.x + radius
        tmp_pt.y = pnt.y + radius
        tmp_pt.z = pnt.z + radius

        max_cell = find_cell_id(self.origin, tmp_pt, self.cell_size)

        diff = max_cell.diff(min_cell)
        diff.x += 1
        diff.y += 1
        diff.z += 1

        for i in range(diff.x):
            for j from 0 <= j < diff.y:
                for k from 0 <= k < diff.z:
                    id.x = min_cell.x + i
                    id.y = min_cell.y + j
                    id.z = min_cell.z + k

                    cell = self.cells_dict.get(id)
                    if cell is not None:
                        cell_list.append(cell)        
        return 0

    cdef void _reset_jump_tolerance(self):
        """Resets the jump tolerance of all cells to 1."""
        cdef list cells_list
        cdef Cell cell
        cdef int i, num_cells

        self.jump_tolerance = 1

        if len(self.cells_dict) == 0:
            return

        num_cells = len(self.cells_dict)
        cells_list = self.cells_dict.values()
        
        for i in range(len(self.cells_dict)):
            cell = cells_list[i]
            cell.jump_tolerance = 1       
    
    cpdef insert_particles(self, int parray_id, LongArray indices):
        """Insert particles"""
        cdef ParticleArray parray = self.arrays_to_bin[parray_id]
        cdef dict particles_for_cells = dict()
        cdef int num_particles = parray.get_number_of_particles()
        cdef int i
        cdef DoubleArray x, y, z
        cdef double cell_size = self.cell_size
        cdef IntPoint id = IntPoint()
        cdef Point pnt = Point()
        cdef LongArray cell_indices, la
        cdef Cell cell
        cdef IntPoint cid
        
        x = parray.get_carray(self.coord_x)
        y = parray.get_carray(self.coord_y)
        z = parray.get_carray(self.coord_z)
        
        for i in range(indices.length):
            if indices.data[i] >= num_particles:
                # invalid particle being added.
                # raise error and exit
                msg = 'Particle %d does not exist'%(indices.data[i])
                logger.error(msg)
                raise ValueError, msg
            
            pnt.x = x.data[indices.data[i]]
            pnt.y = y.data[indices.data[i]]
            pnt.z = z.data[indices.data[i]]

            # find the cell to which this particle belongs to 
            id = find_cell_id(self.origin, pnt, cell_size)
            if PyDict_Contains(particles_for_cells, id) == 1:
                cell_indices = <LongArray>PyDict_GetItem(particles_for_cells, id)
            else:
                cell_indices = LongArray()
                particles_for_cells[id] = cell_indices

            cell_indices.append(indices.data[i])

        for cid, la in particles_for_cells.iteritems():
            if PyDict_Contains(self.cells_dict, cid):
                cell = <Cell>PyDict_GetItem(self.cells_dict, cid)
            else:
                # create a cell with the given id.
                cell = self.get_new_cell(cid)
                self.cells_dict[cid] = cell

            cell.insert_particles(parray_id, la)
    
    cpdef Cell get_new_cell(self, IntPoint id):
        """Create and return a new cell. """
        return Cell(id=id, cell_manager=self, 
                            cell_size=self.cell_size,
                            jump_tolerance=self.jump_tolerance)
    
    cpdef int delete_empty_cells(self) except -1:
        '''delete empty cells'''
        cdef int num_cells = len(self.cells_dict)
        cdef int i
        cdef Cell cell
        cdef list cell_list = self.cells_dict.values()

        for i in range(num_cells):
            cell = cell_list[i]
            if cell.get_number_of_particles() == 0:
                self.cells_dict.pop(cell.id)

        return 0
   
    # python functions for each corresponding cython function for testing.
    def py_update(self):
        return self.update()

    def py_update_status(self):
        return self.update_status()

    def py_initialize(self):
        self.initialize()

    def py_clear(self):
        self.clear()

    def py_compute_cell_size(self, double min_size, double max_size):
        return self.compute_cell_size(min_size, max_size)

    def py_build_cell(self):
        self._build_cell()

    def py_rebuild_array_indices(self):
        self._rebuild_array_indices()

    def py_setup_cells_dict(self):
        self._setup_cells_dict()

    def py_get_potential_cells(self, Point pnt, double radius, list cell_list):
        self.get_potential_cells(pnt, radius, cell_list)

    def py_reset_jump_tolerance(self):
        self._reset_jump_tolerance()

    def py_update_cells_dict(self):
        self.update_cells_dict()
    
    def py_get_number_of_particles(self):
        return self.get_number_of_particles()

