
# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.point cimport *
from pysph.base.carray cimport *
from pysph.base.particle_array cimport ParticleArray

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

def py_find_cell_id(origin, pnt, cell_size, outpoint):
    find_cell_id(origin, pnt, cell_size, outpoint)

def py_find_hierarchy_level_for_radius(radius, min_cell_size, max_cell_size,
                                       cell_size_step, num_levels):
    return find_hierarchy_level_for_radius(radius, min_cell_size, max_cell_size,
                                       cell_size_step, num_levels)

cdef public int find_hierarchy_level_for_radius(double radius, double min_cell_size, double max_cell_size,
                                         double cell_size_step, int num_levels):
        """
        Find hierarchy level to search in given interaction radius.
        """
        cdef double temp1, temp2, diff
        if num_levels == 1:
            if radius > min_cell_size:
                return 1
            else:
                return 0
        
        diff = radius - min_cell_size

        if diff < 0:
            return 0

        temp1 = diff/cell_size_step
        temp2 = floor(temp1)

        if temp2 > num_levels:
            return num_levels
        else:
            if (temp1-temp2) < 10e-09:
                return <int>temp2
            else:
                return <int>temp2 + 1

cdef inline int real_to_int(double real_val, double step):
    """
    """
    cdef int ret_val
    if real_val < 0.0:
        ret_val =  <int>(real_val/step) - 1
    else:
        ret_val =  <int>(real_val/step)

    return ret_val

cdef inline void find_cell_id(Point origin, Point pnt, double cell_size,
                              IntPoint id):
    id.x = real_to_int(pnt.x-origin.x, cell_size)
    id.y = real_to_int(pnt.y-origin.y, cell_size)
    id.z = real_to_int(pnt.z-origin.z, cell_size)

cdef inline void construct_immediate_neighbor_list(IntPoint cell_id, list
                                                   neighbor_list, bint include_self=True): 
    """
    Construct a list of cell ids neighboring the given cell.
    """
    if include_self:
        neighbor_list.append(cell_id)
    
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y+1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y-1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y+1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y+1, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y-1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x+1, cell_id.y-1, cell_id.z-1))

    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y+1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y-1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y+1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y+1, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y-1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x-1, cell_id.y-1, cell_id.z-1))

    neighbor_list.append(IntPoint(cell_id.x, cell_id.y+1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y+1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y+1, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y-1, cell_id.z))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y-1, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y-1, cell_id.z-1))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y, cell_id.z+1))
    neighbor_list.append(IntPoint(cell_id.x, cell_id.y, cell_id.z-1))

def py_construct_immediate_neighbor_list(cell_id, neighbor_list,
                                         include_self=True):
    construct_immediate_neighbor_list(cell_id,
                                      neighbor_list,
                                      include_self) 
                                      
def py_cell_encloses_sphere(IntPoint id, Point world_origin, double cell_size,
                            Point pnt, double radius):
    """
    """
    return cell_encloses_sphere(id, world_origin, cell_size, pnt, radius)

cdef inline bint cell_encloses_sphere(IntPoint id, Point world_origin, double cell_size,
                              Point pnt, double radius):
    """
    Checks if point 'pnt' is completely enclosed by a cell.
    
    **Parameters**
    
     - id - id of the cell.
     - world_origin - origin with respect to which this 'id' was calculated.
     - cell_size - size of the sides of the cells.
     - pnt - center of sphere.
     - radius - radius of sphere.

    **Algorithm**::
    
     for each vertex of the cell
         if distance of vertex from pnt is less than or equal to radius
             return false

     return true

    """
    cdef double distance
    cdef Point cell_vertex = Point()

    # find the first point of the cell.
    cell_vertex.x = world_origin.x + id.x*cell_size
    cell_vertex.y = world_origin.y + id.y*cell_size
    cell_vertex.z = world_origin.z + id.z*cell_size
    
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False
    
    cell_vertex.x += cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.z += cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.x -= cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.y += cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.z -= cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.x += cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    cell_vertex.z += cell_size
    distance = cell_vertex.euclidean(pnt)
    if distance > radius:
        # make sure its not very close.
        if fabs(distance-radius) < 1e-09:
            return False

    return True

cdef class Cell:
    """
    The cell base class.
    """
    def __init__(self, IntPoint id, CellManager cell_manager=None, double
                 cell_size=0.1, int level=1, str coord_x='x', str coord_y='y',
                 str coord_z='z', *args, **kwargs):

        self.id = IntPoint()

        self.id.x = id.x
        self.id.y = id.y
        self.id.z = id.z

        self.cell_size = cell_size

        self.level = level

        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.jump_tolerance = 1

        self.arrays_to_bin = []

        self.origin = Point(0., 0., 0.)
        
        self.set_cell_manager(cell_manager)

    cpdef set_cell_manager(self, CellManager cell_manager):

        self.cell_manager = cell_manager
        if cell_manager is not None:
            self.arrays_to_bin[:] = cell_manager.arrays_to_bin
            self.coord_x = cell_manager.coord_x
            self.coord_y = cell_manager.coord_y
            self.coord_z = cell_manager.coord_z
            self.origin.x = cell_manager.origin.x
            self.origin.y = cell_manager.origin.y
            self.origin.z = cell_manager.origin.z
                         
    cpdef get_centroid(self, Point centroid):
        """
        Returns the centroid of this cell in 'centroid'.
    	"""
        centroid.x = self.origin.x + (<double>self.id.x + 0.5)*self.cell_size
        centroid.y = self.origin.y + (<double>self.id.y + 0.5)*self.cell_size
        centroid.z = self.origin.z + (<double>self.id.z + 0.5)*self.cell_size

    cpdef Cell get_new_sibling(self, IntPoint id):
        """
        Create a new cell and return.
        """
        cdef Cell cell = Cell(id, self.cell_manager, self.cell_size, self.level)
        return cell

    cpdef int update(self, dict data) except -1:
        """
        """
        raise NotImplementedError, 'Cell::update called'

    cpdef long get_number_of_particles(self):
        """
        """
        raise NotImplementedError, 'Cell::get_number_of_particles called'

    cpdef bint is_empty(self):
        """
    	"""
        raise NotImplementedError, 'Cell::is_empty called'

    cpdef int add_particles(self, Cell cell) except -1:
        """
        """
        raise NotImplementedError, 'Cell::add_particles'

    
    cpdef int update_cell_manager_hierarchy_list(self) except -1:
        """
        """
        raise NotImplementedError, 'Cell::update_cell_manager_hierarchy_list'

    cpdef int clear(self) except -1:
        """
        """
        raise NotImplementedError, 'Cell::clear'

    cpdef int delete_empty_cells(self) except -1:
        """
        """
        raise NotImplementedError, 'Cell::delete_empty_cells'

    cpdef double get_child_size(self):
        """
        Return the size of the child cell.
        """
        return self.cell_manager.cell_sizes.data[self.level-1]

    cpdef insert_particles(self, int parray_id, LongArray indices):
        """
        Does a top-down insertion of particles into the hierarchy.
        """
        cdef ParticleArray parray = self.cell_manager.arrays_to_bin[parray_id]
        cdef dict particles_for_children = dict()
        cdef int num_particles = parray.get_number_of_particles()
        cdef int i
        cdef DoubleArray x, y, z
        cdef DoubleArray cell_sizes = self.cell_manager.cell_sizes
        cdef double child_size = self.get_child_size()
        cdef IntPoint id = IntPoint()
        cdef Point pnt = Point()
        cdef LongArray child_indices, la
        cdef Cell child
        cdef IntPoint cid
        
        x = parray.get_carray(self.coord_x)
        y = parray.get_carray(self.coord_y)
        z = parray.get_carray(self.coord_z)
        
        for i from 0 <= i < indices.length:
            if indices.data[i] >= num_particles:
                # invalid particle being added.
                # raise error and exit
                msg = 'Particle %d does not exist'%(indices.data[i])
                logger.error(msg)
                raise ValueError, msg
            
            pnt.x = x.data[indices.data[i]]
            pnt.y = y.data[indices.data[i]]
            pnt.z = z.data[indices.data[i]]

            # find the cell at the lower level, to which this particle belongs
            # to. 
            find_cell_id(self.origin, pnt, child_size, id)
            child_indices = particles_for_children.get(id)

            if child_indices is None:
                child_indices = LongArray()
                particles_for_children[id.copy()] = child_indices

            child_indices.append(indices.data[i])

        num_children = len(particles_for_children)
        
        for cid, la in particles_for_children.iteritems():
            child = self.cell_dict.get(cid)

            if child is None:
                # create a child with the given id.
                child = self.get_new_child(cid)
                self.cell_dict[cid.copy()] = child

            child.insert_particles(parray_id, la)        

    cpdef get_particle_ids(self, list particle_id_list):
        """
        Finds the indices of particle ids for each particle array in
        arrays_to_bin.

        **Parameters**

            - particle_id_list - output parameter, should contain one 
              LongArray for each array in arrays_to_bin. Particle ids will be
              appended to the LongArrays

        """
        raise NotImplementedError, 'Cell::get_particle_ids'
     
    cpdef Cell get_new_child(self, IntPoint id):
        """
        Create a new child node depending on this nodes level in the hierarchy.
        """
        cdef int num_levels = self.cell_manager.num_levels
        cdef DoubleArray cell_sizes = self.cell_manager.cell_sizes
        
        if (num_levels - self.level) == num_levels-1:
            return LeafCell(id=id, cell_manager=self.cell_manager,
                            cell_size=cell_sizes.data[self.level-1],
                            level=self.level-1,
                            jump_tolerance=1)
        else:
            return NonLeafCell(id=id, cell_manager=self.cell_manager,
                               cell_size=cell_sizes.data[self.level-1],
                               level=self.level-1)

    cpdef clear_indices(self, int parray_id):
        """
        Clear the particles ids of parray_id stored in the hierarchy.
        """
        raise NotImplementedError, 'Cell::clear_indices'

    cpdef bint is_leaf(self):
        """
        """
        return False

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

    def py_delete_empty_cells(self):
        self.delete_empty_cells()

    def py_update_cell_manager_hierarchy_list(self):
        self.update_cell_manager_hierarchy_list()
        
    def py_clear(self):
        self.clear()

    def py_is_leaf(self):
        return self.is_leaf()

################################################################################
# `LeafCell` class.
################################################################################
cdef class LeafCell(Cell):
    """
    """
    def __init__(self, IntPoint id, CellManager cell_manager=None, double
                  cell_size=0.1, int level=0, int jump_tolerance=1): 
        Cell.__init__(self, id=id, cell_manager=cell_manager,
                      cell_size=cell_size, level=level)
        
        self.jump_tolerance = jump_tolerance

    cpdef Cell get_new_sibling(self, IntPoint id):
        """
        Create a new LeafCell and return.
        """
        cdef LeafCell cell = LeafCell(id=id, cell_manager=self.cell_manager,
                                      cell_size=self.cell_size, 
                                      level=self.level,
                                      jump_tolerance=self.jump_tolerance)
        return <Cell>cell

    cpdef Cell get_new_child(self, IntPoint id):
        """
        Raises an error if this is called.
        """
        msg='Cannot create child node for LeafCell'
        logger.error(msg)
        raise SystemError, msg

    cpdef bint is_leaf(self):
        """
        """
        return True

    cpdef double get_child_size(self):
        """
        Return the size of the child cell.
        """
        msg = 'get_child_size called on leaf node'
        logger.error(msg)
        raise SystemError, msg

    cdef _init_index_lists(self):
        """
        """
        cdef int i
        cdef int num_arrays

        if self.cell_manager is None:
            return

        num_arrays = len(self.arrays_to_bin)
        self.index_lists[:] = []
        
        for i from 0 <= i < num_arrays:
            self.index_lists.append(LongArray())

    cpdef get_particle_ids(self, list particle_id_list):
        """
        Finds the indices of particle ids for each particle array in
        arrays_to_bin.

        **Parameters**
        
            - particle_id_list - output parameter, should contain one 
              LongArray for each array in arrays_to_bin. Particle ids will be
              appended to the LongArrays
        """
        cdef int num_arrays = 0
        cdef int i = 0
        cdef LongArray source, dest

        num_arrays = len(self.arrays_to_bin)

        if len(particle_id_list) == 0:
            # create num_arrays LongArrays
            for i from 0 <= i < num_arrays:
                particle_id_list.append(LongArray())
        
        for i from 0 <= i < num_arrays:
            dest = particle_id_list[i]
            source = self.index_lists[i]
            dest.extend(source.get_npy_array())

    cpdef set_cell_manager(self, CellManager cell_manager):
        """
        """
        self.cell_manager = cell_manager

        if self.index_lists is None:
            self.index_lists = list()

        if self.cell_manager is None:
            self.arrays_to_bin[:] = []
            self.index_lists[:] =[]
            self.coord_x = 'x'
            self.coord_y = 'y'
            self.coord_z = 'z'
        else:
            self.arrays_to_bin[:] = self.cell_manager.arrays_to_bin
            self.coord_x = self.cell_manager.coord_x
            self.coord_y = self.cell_manager.coord_y
            self.coord_z = self.cell_manager.coord_z
            self.origin.x = self.cell_manager.origin.x
            self.origin.y = self.cell_manager.origin.y
            self.origin.z = self.cell_manager.origin.z
            self._init_index_lists()

    cpdef int add_particles(self, Cell cell1) except -1:
        """
        Add particle indices in cell to the current cell.
        
        **Parameters**
         
         - cell - a leaf cell from which to add the new particle indices.

        **Notes**
         
         - the input cell should have the same set of particle arrays.

        """
        cdef int i, j, id
        cdef int num_arrays, num_particles, num_indices
        cdef LongArray dest_array
        cdef LongArray source_array
        cdef NonLeafCell cell = <NonLeafCell>cell1
        cdef ParticleArray parr
        
        if not cell.id.is_equal(self.id):
            raise RuntimeError, 'invalid cell as input'

        num_arrays = len(self.arrays_to_bin)

        for i from 0 <= i < num_arrays:
            parr = self.arrays_to_bin[i]
            num_particles = parr.get_number_of_particles()
            source_array = cell.index_lists[i]
            dest_array = self.index_lists[i]
            for j from 0 <= j < source_array.length:
                id = source_array.get(j)
                if id >= num_particles or id < 0:
                    msg = 'trying to add invalid particle\n'
                    msg += 'num_particles : %d\n'%(num_particles)
                    msg += 'pid : %d\n'%(id)
                    raise RuntimeError, msg
                else:
                    dest_array.append(id)

        return 0
            
    cpdef int update(self, dict data) except -1:
        """
        Finds particles that have escaped this cell.
    
	**Parameters**
         
         - data - output parameters, to store cell ids and particles that have
           moved to those cells.

        **Algorithm**::
        
         for each array in arrays_to_bin
             if array is dirty
                 for each particle of that array that is in this cell
                     check if the particle is still there in this cell
                     if not 
                         find the new cell this particles has moved to.
                         if an entry for that cell does not exist in data
                             create a leaf cell for that cell.
                         
                         add this particle entry to the appropriate array in the
                         newly created cell.

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
        cdef IntPoint id = IntPoint()
        cdef LeafCell cell
        cdef IntPoint pdiff
        cdef str msg
        
        for i from 0 <= i < num_arrays:
            
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
                    find_cell_id(self.origin, pnt, self.cell_size, id)
                    
                    if id.is_equal(self.id):
                        continue

                    to_remove.append(j)
                    
                    pdiff = self.id.diff(id)

                    # make sure particles have not moved more than 
                    # the permitted jump_tolerance.
                    if (abs(pdiff.x) > self.jump_tolerance or abs(pdiff.y) >
                        self.jump_tolerance or abs(pdiff.z) >
                        self.jump_tolerance):
                        
                        msg = 'Particle moved by more than one cell width\n'
                        msg += 'Point (%f, %f, %f)\n'%(pnt.x, pnt.y, pnt.z)
                        msg += 'self id : (%d, %d, %d)\n'%(self.id.x, self.id.y,
                                                         self.id.z)
                        msg += 'new id  : (%d, %d, %d)\n'%(id.x, id.y, id.z)
                        msg += 'Jump tolerance is : %s, %d\n'%(self, self.jump_tolerance)
                        raise RuntimeError, msg

                    # add this particle to the particles that are to be removed.
                    cell = data.get(id)
                    if cell is None:
                        # create a leaf cell and add it to dict.
                        # cell = LeafCell(id, self.cell_manager, self.cell_size,
                        #                 self.level, self.jump_tolerance)
                        cell = self.get_new_sibling(id)
                        data[id.copy()] = cell
                        
                    index_array1 = cell.index_lists[i]
                    index_array1.append(indices[j])

            # now remove all escaped and invalid particles.
            index_array.remove(to_remove.get_npy_array())

        return 0

    cpdef insert_particles(self, int parray_id, LongArray indices):
        """
        Insert the particles in indices into the leaf cell.
        """
        cdef LongArray index_array = self.index_lists[parray_id]
        index_array.extend(indices.get_npy_array())

    cpdef clear_indices(self, int parray_id):
        """
        Clear the particles ids of parray_id stored in the hierarchy.
        """
        cdef LongArray index_array = self.index_lists[parray_id]
        index_array.reset()

    cpdef bint is_empty(self):
        """
        """
        if self.get_number_of_particles() == 0:
            return True
        else:
            return False

    cpdef long get_number_of_particles(self):
        """
        """
        cdef int i, num_arrays
        cdef long num_particles = 0
        cdef LongArray arr

        num_arrays = len(self.index_lists)
        
        for i from 0 <= i < num_arrays:
            arr = self.index_lists[i]
            num_particles += arr.length
        
        return num_particles

    cpdef int update_cell_manager_hierarchy_list(self) except -1:
        """
        """
        cdef CellManager cell_manager = self.cell_manager
        cdef str msg
        cdef dict hierarchy_dict
        if cell_manager is None:
            return 0

        if self.level >= len(cell_manager.hierarchy_list):
            msg = 'invalid cell level'
            raise RuntimeError, msg

        hierarchy_dict = cell_manager.hierarchy_list[self.level]
        hierarchy_dict[self.id.copy()] = self

        return 0

    cpdef int clear(self) except -1:
        """
        """
        self.index_lists[:] = []   
        return 0

    cpdef int delete_empty_cells(self) except -1:
        """
        """
        return 0

################################################################################
# `NonLeafCell` class.
################################################################################
cdef class NonLeafCell(Cell):
    """
    """
    def __init__(self, IntPoint id, CellManager cell_manager=None, double
                  cell_size=0.1, int level=1):
        Cell.__init__(self, id=id, cell_manager=cell_manager,
                      cell_size=cell_size, level=level)

        self.cell_dict = {}

    cpdef Cell get_new_sibling(self, IntPoint id):
        """
        Create and return a new cell at the same level.
        """
        cdef NonLeafCell cell = NonLeafCell(id=id,
                                            cell_manager=self.cell_manager,
                                            cell_size=self.cell_size,
                                            level=self.level)
        return <Cell>cell
                                            
    cpdef int add_particles(self, Cell cell) except -1:
        """
        Add particles from given cell, into this cell.
    
	**Parameters**
         
         - cell - the cell from which particles are to be added.

        **Algorithm**::
         
         for every smaller_cell in 'cell'
            if smaller_cell exists in self.cell_dict
                call update on the smaller cell that already exists, with
                smaller_cell as parameter.
            else
                add smaller_cell into self.cell_dict   
         
        **Notes**
         
         - we KNOW that 'cell' is a NonLeafCell.

    	"""
        cdef int num_cells
        cdef int i
        cdef list smaller_cell_list = cell.cell_dict.values()
        cdef Cell smaller_cell, curr_smaller_cell
        num_cells = len(smaller_cell_list)

        for i from 0 <= i < num_cells:
            smaller_cell = smaller_cell_list[i]

            curr_smaller_cell = self.cell_dict.get(smaller_cell.id)

            if curr_smaller_cell is None:
                self.cell_dict[smaller_cell.id.copy()] = smaller_cell
            else:
                curr_smaller_cell.add_particles(smaller_cell)

        return 0

    cpdef get_particle_ids(self, list particle_id_list):
        """
        Finds the indices of particle ids for each particle array in
        arrays_to_bin.

        **Parameters**
        
            - particle_id_list - output parameter, should contain one 
              LongArray for each array in arrays_to_bin. Particle ids will be
              appended to the LongArrays
        """
        cdef int i, num_children, num_arrays
        cdef list children = self.cell_dict.values()
        cdef Cell cell
        num_children = len(children)

        if len(particle_id_list) == 0:
            num_arrays = len(self.arrays_to_bin)
            # create num_arrays LongArrays
            for i from 0 <= i < num_arrays:
                particle_id_list.append(LongArray())
        
        for i from 0 <= i < num_children:
            cell = children[i]
            cell.get_particle_ids(particle_id_list)        

    cpdef clear_indices(self, int parray_id):
        """
        Clear the particles ids of parray_id stored in the hierarchy.
        """
        cdef int i, num_children
        cdef list child_list = self.cell_dict.values()
        num_children = len(child_list)
        cdef Cell child

        for i from 0 <= i < num_children:
            child = child_list[i]
            child.clear_indices(parray_id)
    
    cpdef int update(self, dict data) except -1:
        """
        Update the particles under this cell, returning all
        particles that have created a new cell outside the
        bounds of this cell.

	**Parameters**
        
         - data - output parameter in which all newly created cells at this
           level are to be returned.

        **Algorithm**::
        
         for every cell in this cell's cell_list
             perform an update and collect the data about escaping cells.
             
         for every cell in the collected data
             check if cell is in this cells bounds
                 if yes
                     if cell is not already there
                         create a cell with appropriate id
                     update the cell with new data
                 if not
                     Create a new cell(at this level in hierarchy)
                     add the smaller cell into this cell
                     and store the larger cell in output dict

    	**Issues**
        
         - Not sure when do we delete cells.

    	"""
        cdef int i
        cdef int num_cells = len(self.cell_dict)
        cdef dict output_data, collected_data
        cdef Cell smaller_cell, smaller_cell_1
        cdef NonLeafCell cell
        cdef list cell_list
        cdef Point centroid = Point()
        cdef IntPoint cell_id = IntPoint()
        cdef list smaller_cell_list = self.cell_dict.values()

        output_data = data
        collected_data = dict()
        # collect all escaped partiles 
        # from cells down in the hierarchy.
        for i from 0 <= i < num_cells:
            smaller_cell = smaller_cell_list[i]
            smaller_cell.update(collected_data)

        cell_list = collected_data.values()
        num_cells = len(cell_list)

        for i from 0 <= i < num_cells:
            smaller_cell = cell_list[i]
            
            # get the cell centroid
            smaller_cell.get_centroid(centroid)
            # find the larger cell within which it lies.
            find_cell_id(self.origin, centroid, self.cell_size, cell_id)
            
            # if it lies in this cell, add the new particles to the
            # cells in the hierarchy below.
            if cell_id.is_equal(self.id):
                smaller_cell_1 = self.cell_dict.get(smaller_cell.id)
                if smaller_cell_1 is None:
                    # meaning there does not exist any cell for the
                    # region occupied by smaller_cell, so we just
                    # add 'smaller_cell' to the cell list
                    self.cell_dict[smaller_cell.id.copy()] = smaller_cell
                else:
                    smaller_cell_1.add_particles(smaller_cell)
            else:
                # meaning smaller_cell is not inside this cell
                cell = output_data.get(cell_id)
                if cell is None:
                    # Create a non-leaf node
                    # cell = NonLeafCell(cell_id, self.cell_manager,
                    #                    self.cell_size, self.level)
                    cell = self.get_new_sibling(cell_id)
                    output_data[cell_id.copy()] = cell

                    # add the escaped smaller cell to this cell.
                    cell.cell_dict[smaller_cell.id.copy()] = smaller_cell
                else:
                    smaller_cell_1 = cell.cell_dict.get(smaller_cell.id)
                    if smaller_cell_1 is None:
                        cell.cell_dict[smaller_cell.id.copy()] = smaller_cell
                    else:
                        smaller_cell_1.add_particles(smaller_cell)

        return 0

    cpdef int update_cell_manager_hierarchy_list(self) except -1:
        """
        """
        cdef CellManager cell_manager = self.cell_manager
        cdef str msg
        cdef dict hierarchy_dict
        cdef Cell cell
        cdef int num_cells, i

        cdef list cell_list = self.cell_dict.values()

        num_cells = len(self.cell_dict)

        if cell_manager is None:
            return 0

        if self.level >= len(cell_manager.hierarchy_list):
            msg = 'invalid cell level'
            raise RuntimeError, msg

        hierarchy_dict = cell_manager.hierarchy_list[self.level]
        hierarchy_dict[self.id.copy()] = self

        for i from 0 <= i < num_cells:
            cell = cell_list[i]
            cell.update_cell_manager_hierarchy_list()   
        
        return 0

    cpdef int clear(self) except -1:
        """
        Clear internal information.
        """
        cdef int i, num_cells
        num_cells = len(self.cell_dict)
        cdef list cell_list = self.cell_dict.values()
        cdef Cell cell

        for i from 0 <= i < num_cells:
            cell = cell_list[i]
            cell.clear()

        self.cell_dict.clear()

        return 0

    cpdef int delete_empty_cells(self) except -1:
        """
        Delete any empty cells underneath this cell.
    
        **Algorithm**::
        
         Call delete_empty_cells for every smaller_cell contained in this cell.
         
         find the number of particles for every smaller cell.

         if number of particles is 0, remove that cell from the cell_dict         

    	**Issues**
        
         - too many passes down the cell hierarchy.

    	"""
        cdef int num_cells = len(self.cell_dict)
        cdef int i
        cdef Cell cell
        cdef list cell_list = self.cell_dict.values()

        for i from 0 <= i < num_cells:
            cell = cell_list[i]
            cell.delete_empty_cells()

            if cell.get_number_of_particles() == 0:
                self.cell_dict.pop(cell.id)

        return 0

    cpdef long get_number_of_particles(self):
        """
        Return the number of particles.
        """
        cdef long num_particles = 0
        cdef Cell cell
        cdef int num_cells = len(self.cell_dict)
        cdef int i
        cdef cell_list = self.cell_dict.values()

        for i from 0 <= i < num_cells:
            cell = cell_list[i]
            num_particles += cell.get_number_of_particles()

        return num_particles        

################################################################################
# `RootCell` class.
################################################################################
cdef class RootCell(NonLeafCell):
    """
    """
    def __init__(self, CellManager cell_manager=None, double
                  cell_size=0.1):
        NonLeafCell.__init__(self, id=IntPoint(0, 0, 0),
                             cell_manager=cell_manager, cell_size=cell_size,
                             level=-1)
        pass

    cpdef Cell get_new_sibling(self, IntPoint id):
        """
        Create and return a new cell at the same level as this.
        """
        cdef RootCell cell = RootCell(cell_manager=self.cell_manager,
                                      cell_size=self.cell_size)
        return <Cell>cell

    cpdef Cell get_new_child(self, IntPoint id):
        """
        Create and return a new cell at one level lower in the cell hierarchy.
        """
        cdef int num_levels = self.cell_manager.num_levels
        cdef DoubleArray cell_sizes = self.cell_manager.cell_sizes
        if num_levels == 1:
            return LeafCell(id=id, cell_manager=self.cell_manager, 
                            cell_size=cell_sizes.data[0],
                            level=0,
                            jump_tolerance=self.cell_manager.jump_tolerance)
        else:
            return NonLeafCell(id=id, cell_manager=self.cell_manager,
                               cell_size=cell_sizes.data[num_levels-1],
                               level=num_levels-1)

    cpdef int update(self, dict data) except -1:
        """
        Update particle information.
    
	**Parameters**
        
         - data - should be None. The root cell does not fill up data for anyone.

        **Algorithm**::
        
         All escaping particles (in newly created cells) are got from levels
         underneath the root.
         
         For all new cells returned, if some already exist, the data is merged
         with them.

         If a newly created cell does not exists, it is created as a child.

    	**Notes**
         
	**Helper Functions**

    	**Issues**

    	"""
        cdef int i
        cdef int num_cells = len(self.cell_dict)
        cdef dict collected_data
        cdef Cell smaller_cell, smaller_cell_1
        cdef NonLeafCell cell
        cdef list cell_list
        cdef list smaller_cell_list = self.cell_dict.values()

        collected_data = dict()

        # collect all escaped partiles 
        # from cells down in the hierarchy.
        for i from 0 <= i < num_cells:
            smaller_cell = smaller_cell_list[i]
            smaller_cell.update(collected_data)

        cell_list = collected_data.values()
        num_cells = len(cell_list)

        for i from 0 <= i < num_cells:
            smaller_cell = cell_list[i]
            
            smaller_cell_1 = self.cell_dict.get(smaller_cell.id)
            if smaller_cell_1 is None:
                # meaning there does not exist any cell for the
                # region occupied by smaller_cell, so we just
                # add 'smaller_cell' to the cell list
                self.cell_dict[smaller_cell.id.copy()] = smaller_cell
            else:
                smaller_cell_1.add_particles(smaller_cell)

        return 0

################################################################################
# `CellManager` class.
################################################################################
cdef class CellManager:
    """
    """
    # FIXME:
    # 1. Simple API function to add an array to bin _BEFORE_ the cell manager
    # has been initialized.
    #
    # 2. Provision to add new array to bin  _AFTER_ the cell manager has once
    # been initialized. Will require some dirty changes to all cells that have
    # already been created. Whether this feature is required can be thought over
    # again. 

    def __init__(self, list arrays_to_bin=[], object particle_manager=None, double
                  min_cell_size=0.1, double max_cell_size=0.5, Point
                  origin=Point(0., 0., 0), int num_levels=1, str coord_x='x',
                  str coord_y='y', str coord_z='z', bint initialize=True):
        
        self.particle_manager = particle_manager
        self.origin = Point()

        self.origin.x = origin.x
        self.origin.y = origin.y
        self.origin.z = origin.z

        self.min_cell_size = min_cell_size
        self.max_cell_size = max_cell_size
        self.num_levels = num_levels

        self.array_indices = dict()
        self.arrays_to_bin = list()
        self.arrays_to_bin[:] = arrays_to_bin
        
        self.hierarchy_list = list()
        
        self.coord_x = coord_x
        self.coord_y = coord_y
        self.coord_z = coord_z

        self.jump_tolerance = 1
        self.cell_sizes = DoubleArray(0)
        
        self.is_dirty = True

        # create the root cell.
        self.root_cell = RootCell(cell_manager=self)

        self.initialized = False

        if initialize == True:
            self.initialize()

    def set_jump_tolerance(self, int jump_tolerance):
        """
        Sets the jump tolerance value of the cells.
        """
        self.jump_tolerance = jump_tolerance

        if len(self.hierarchy_list) == 0:
            logger.debug('Hierarchy list empty')
            return

        leaf_dict = self.hierarchy_list[0]
        num_leaves = len(leaf_dict)
        leaf_list = leaf_dict.values()
        
        for i from 0 <= i < num_leaves:
            leaf_cell = leaf_list[i]
            leaf_cell.jump_tolerance = jump_tolerance

    cpdef int update(self) except -1:
        """
        Update the cell manager.
        """
        cdef int i, num_arrays
        cdef ParticleArray parray
                
        if self.is_dirty:

            # update the cells.
            self.root_cell.update(None)
        
            # delete empty cells if any.
            self.root_cell.delete_empty_cells()

            # update the cell hierarchy_list.
            self.update_cell_hierarchy_list()

            # reset the dirty bit of all particle arrays.
            num_arrays = len(self.arrays_to_bin)
        
            for i from 0 <= i < num_arrays:
                parray = self.arrays_to_bin[i]
                parray.set_dirty(False)

            self.is_dirty = False

        return 0

    cpdef add_array_to_bin(self, ParticleArray parr):
        """
        """
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
        """
        Updates the is_dirty flag to indicate is an update is required.

        Any module that may modify the particle data, should call the cell
        managers update_status function.
        """
        cdef int i,  num_arrays
        cdef ParticleArray parray
        
        num_arrays = len(self.arrays_to_bin)
        
        for i from 0 <= i < num_arrays:
            parray = self.arrays_to_bin[i]
            if parray.is_dirty:
                self.is_dirty = True
                break

    def set_dirty(self, bint value):
        """
        Sets/Resets the dirty bit.
        """
        self.is_dirty = value

    cpdef initialize(self):
        """
        Initialize the cell manager.
    
        **Algorithm**::
        
         clear current data
         
         find the cell sizes of each level.
         
         make a hierarchy, with each level containing one cell of that
         heirarchy.

         call update on the root cell.

    	**Notes**
        
         - previous data will be cleared.

    	**Issues**
         
         - FIXME: the updation of the hierarchy list could be done along with the 
           cells update functions, instead of the
           update_cell_manager_hierarchy_list function.
           
    	"""

        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        # clear current data.
        self.clear()

        # setup some data structures.
        self._rebuild_array_indices()

        # setup the hierarhy list
        self._setup_hierarchy_list()
        
        self.root_cell.level = self.num_levels

        # recompute cell sizes.
        self.compute_cell_sizes(self.min_cell_size, self.max_cell_size,
                                self.num_levels, self.cell_sizes)
        
        # build a base hierarchy.
        self._build_base_hierarchy()

        # update
        self.update()

        # now reset the jump tolerance back to 1
        # we do not want particles to jump across
        # multiple cells.
        self._reset_jump_tolerance()        

        self.initialized = True

    cpdef clear(self):
        """
        Clears all information in the cell manager.
        """
        cdef int i

        # clear the hierarchy dict.
        self.hierarchy_list[:] = []

        if self.root_cell is not None:
            self.root_cell.clear()

    cpdef _setup_hierarchy_list(self):
        """
        """
        cdef int i
        
        for i from 0 <= i < self.num_levels+1:
            self.hierarchy_list.append(dict())

    cpdef compute_cell_sizes(self, double min_size, double max_size, int
                             num_levels, DoubleArray arr):
        """
        Get the cell sizes for each level requested.
    
	**Parameters**
        
         - min_size - smallest cell size needed.
         - max_size - largest cell size needed.
         - num_levels - number of levels needed (including the max and min
           levels)
         - arr - a double array where the cell sizes will be stored.

        **Algorithm**::
         
         divide the range betweeen max and min cell sizes into (num_level-2)
         equal parts.

    	**Notes**
        
         - arr[0] will have the min cell size.
         - arr[num_levels-1] will have the max cell size.
        
	**Helper Functions**

    	**Issues**

        """
        cdef double delta
        cdef int i

        if num_levels == 1:
            arr.resize(1)
            arr.set(0, min_size)
            self.cell_size_step = 0
        elif num_levels == 2:
            arr.resize(2)
            arr.set(0, min_size)
            arr.set(1, max_size)
            self.cell_size_step = max_size-min_size
        else:
            delta = (max_size-min_size)/<double>(num_levels-1.)
            self.cell_size_step = delta
            arr.resize(num_levels)
            arr.set(0, min_size)
            arr.set(num_levels-1, max_size)
            
            for i from 1 <= i < num_levels-1:
                arr.set(i, min_size + i*delta)

    cpdef _build_base_hierarchy(self):
        """
        Create a hierarchy of cells containing one cell from each level in the
        hierarchy.
    
        **Algorithm**::
        
         create a root cell
         
         Starting from the origin, and the highest level(one below the root)
         create cells of decreasing sizes.
         The leaf cell should be created with a jump_tolerance of infinity.
         
    	**Notes**

	**Helper Functions**

    	**Issues**

    	"""
        cdef int i, num_arrays, num_particles
        cdef RootCell root_cell = self.root_cell
        cdef list cell_list = []
        cdef ParticleArray parry
        cdef numpy.ndarray index_arr_source
        cdef LongArray index_arr
        cdef NonLeafCell inter_cell
        cdef LeafCell leaf_cell
        cdef double leaf_size = self.cell_sizes.get(0)
 
        # create a leaf cell with all particles.
        leaf_cell = LeafCell(id=IntPoint(0, 0, 0), cell_manager=self,
                             cell_size=leaf_size, level=0,
                             jump_tolerance=INT_MAX)
        num_arrays = len(leaf_cell.arrays_to_bin)
        # now add all particles of all arrays to this cell.
        for i from 0 <= i < num_arrays:
            parray = leaf_cell.arrays_to_bin[i]
            num_particles = parray.get_number_of_particles()

            index_arr_source = numpy.arange(num_particles, dtype=numpy.long)
            index_arr = leaf_cell.index_lists[i]
            index_arr.resize(num_particles)
            index_arr.set_data(index_arr_source)
       
        cell_list.append(leaf_cell)
        # for each intermediate level in the hierarchy create a NonLeafCell
        for i from 1 <= i < self.num_levels:
            inter_cell = NonLeafCell(IntPoint(0, 0, 0), self,
                                     self.cell_sizes.get(i), i)
            # add the previous level cell to this cell
            inter_cell.cell_dict[IntPoint(0, 0, 0)] = cell_list[i-1]
            cell_list.append(inter_cell)

        # now add the top level cell to the roots cell list
        root_cell.clear()
        root_cell.cell_dict[IntPoint(0, 0, 0)] = cell_list[self.num_levels-1]
        
        # build the hierarchy list also
        self.update_cell_hierarchy_list()
        
    cpdef update_cell_hierarchy_list(self):
        """
        Update the lists containing references to cells from each 
        level in the hierarchy.
        """
        
        # clear the current data
        cdef int i
        cdef dict d

        for i from 0 <= i < self.num_levels:
            d = self.hierarchy_list[i]
            d.clear()

        self.root_cell.update_cell_manager_hierarchy_list()
        
    cpdef _rebuild_array_indices(self):
        """
        Rebuild the mapping from array name to position in arrays_to_bin list.
        """
        cdef int i 
        cdef int num_arrays = len(self.arrays_to_bin)
        cdef ParticleArray parr

        self.array_indices.clear()

        for i from 0 <= i < num_arrays:
            parr  = self.arrays_to_bin[i]
            self.array_indices[parr.name] = i

    cdef int get_potential_cells(self, Point pnt, double radius, list cell_list,
                                 bint single_layer=True) except -1:
        """
        Gets cell that will potentially contain neighbors the the given point.
    
	**Parameters**
         
         - pnt - the point whose neighbors are to be found.
         - radius - the radius within which neighbors are to be found.
         - cell_list - output parameter, where potential cells are appended.
         - single_layer - indicates if exactly one layer around the cell
           containging 'pnt' should be returned.

        **Algorithm**::
        
         level <- find level in hierarchy to search for cells, this depends on
         the interaction 'radius'
         
         if single_layer is False:
             level <- level - 1
         
         cell_id <- get cell containing point.

         if single_layer is True
             _get_one_layer_of_cells()
         else
             _get_cells_within_radius()
             
    	**Notes**

	**Helper Functions**
        
         - _get_one_layer_of_cells
         - _get_cells_within_radius

    	**Issues**

    	"""
        cdef int level
        
        level = find_hierarchy_level_for_radius(radius, self.min_cell_size,
                                                self.max_cell_size,
                                                self.cell_size_step,
                                                self.num_levels)

        if single_layer == False:
            if level > 0:
                level = level - 1
            self._get_cells_within_radius(pnt, radius, level, cell_list)
        else:
            self._get_one_layer_of_cells(pnt, radius, level, cell_list)
        
        return 0

    cdef int _get_cells_within_radius(self, Point pnt, double radius, int level, list
                                      cell_list) except -1:
        """
        Finds all cells within the given radius of pnt.

	**Parameters**
         
         - pnt - point around which cells are to be searched for.
         - radius - search radius
         - level - level in the hierarchy to search for cells.
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
        cdef double level_size

        cdef IntPoint max_cell = IntPoint()
        cdef IntPoint min_cell = IntPoint()
        cdef IntPoint diff
        cdef IntPoint id = IntPoint()
        cdef Point tmp_pt = Point()
        cdef IntPoint curr_id = IntPoint()

        cdef Cell cell
        cdef dict current_level

        if level == self.num_levels:
            cell_list.append(self.root_cell)
            return 0

        # get the current cells list and size.
        level_size = self.cell_sizes.get(level)
        current_level = self.hierarchy_list[level]

        # find the cell within which this point is located.
        find_cell_id(self.origin, pnt, level_size, curr_id)

        # check if the search sphere lies completely within 
        # current cell, if yes, we just return current cell
        # if it exists.
#         if cell_encloses_sphere(curr_id, self.origin, level_size, pnt, radius) == True:
#             cell = current_level.get(curr_id)
#             if cell is not None:
#                 cell_list.append(cell)
#                 return 0

        tmp_pt.x = pnt.x - radius
        tmp_pt.y = pnt.y - radius
        tmp_pt.z = pnt.z - radius

        find_cell_id(self.origin, tmp_pt, level_size, min_cell)

        tmp_pt.x = pnt.x + radius
        tmp_pt.y = pnt.y + radius
        tmp_pt.z = pnt.z + radius

        find_cell_id(self.origin, tmp_pt, level_size, max_cell)

        diff = max_cell.diff(min_cell)
        diff.x += 1
        diff.y += 1
        diff.z += 1

        for i from 0 <= i < diff.x:
            for j from 0 <= j < diff.y:
                for k from 0 <= k < diff.z:
                    id.x = min_cell.x + i
                    id.y = min_cell.y + j
                    id.z = min_cell.z + k

                    cell = current_level.get(id)
                    if cell is not None:
                        cell_list.append(cell)        
        return 0

    cdef int _get_one_layer_of_cells(self, Point pnt, double radius, int level, list
                                     cell_list) except -1:
        """
        """
        cdef IntPoint cell_id = IntPoint()
        cdef IntPoint id
        cdef double level_size
        cdef list neighbor_list = list()
        cdef int i
        cdef dict current_level

        if level == self.num_levels:
            cell_list.append(self.root_cell)
            return 0

        current_level = self.hierarchy_list[level]

        level_size = self.cell_sizes.get(level)
        
        find_cell_id(self.origin, pnt, level_size, cell_id)

        # construct ids of all neighbors around cell_id, this
        # will include the cell also.
        construct_immediate_neighbor_list(cell_id, neighbor_list)

        for i from 0 <= i < 27:
            id = neighbor_list[i]
            cell = current_level.get(id)
            if cell is not None:
                cell_list.append(cell)
        
        return 0    

    cdef void _reset_jump_tolerance(self):
        """
        Resets the jump tolerance of all leaf cells to 1.
        """
        cdef dict leaf_dict
        cdef list leaf_list
        cdef LeafCell leaf_cell
        cdef int i, num_leaves

        self.jump_tolerance = 1

        if len(self.hierarchy_list) == 0:
            return

        leaf_dict = self.hierarchy_list[0]
        num_leaves = len(leaf_dict)
        leaf_list = leaf_dict.values()
        
        for i from 0 <= i < num_leaves:
            leaf_cell = leaf_list[i]
            leaf_cell.jump_tolerance = 1       
 
    # python functions for each corresponding cython function for testing purposes.
    def py_update(self):
        return self.update()

    def py_update_status(self):
        return self.update_status()

    def py_initialize(self):
        self.initialize()

    def py_clear(self):
        self.clear()

    def py_compute_cell_sizes(self, double min_size, double max_size, int num_levels, DoubleArray arr):
        self.compute_cell_sizes(min_size, max_size, num_levels, arr)

    def py_build_base_hierarchy(self):
        self._build_base_hierarchy()

    def py_rebuild_array_indices(self):
        self._rebuild_array_indices()

    def py_setup_hierarchy_list(self):
        self._setup_hierarchy_list()

    def py_get_potential_cells(self, Point pnt, double radius, list cell_list,
                               bint single_layer=True):
        self.get_potential_cells(pnt, radius, cell_list, single_layer)

    def py_reset_jump_tolerance(self):
        self._reset_jump_tolerance()

    def py_update_cell_hierarchy_list(self):
        self.update_cell_hierarchy_list()
    
