from pysph.base.point cimport *
from pysph.base.carray cimport *

from pysph.base.particle_array cimport ParticleArray
# forward declaration for CellManager
cdef class CellManager

cdef inline int real_to_int(double val, double step)
cdef inline void find_cell_id(Point origin, Point pnt, double cell_size, IntPoint id)
cdef inline void construct_immediate_neighbor_list(IntPoint pnt, list
                                                   neighbor_list, bint include_self=*)
cdef inline int find_hierarchy_level_for_radius(double radius, double
                                                       min_cell_size, double
                                                       max_cell_size, double
                                                       cell_size_step, int num_levels)

cdef inline bint cell_encloses_sphere(IntPoint id, Point world_origin, double
                                      cell_size, Point pnt, double radius)
cdef class Cell:
    # Member variables.
    cdef public IntPoint id
    cdef public double cell_size
    cdef public CellManager cell_manager
    cdef public list arrays_to_bin
    
    cdef public str coord_x
    cdef public str coord_y
    cdef public str coord_z
    cdef public Point origin

    cdef public int level

    cdef public int jump_tolerance
    
    # Member functions.
    cpdef int add_particles(self, Cell cell) except -1
    cpdef int update(self, dict data) except -1
    cpdef long get_number_of_particles(self)
    cpdef bint is_empty(self)
    cpdef get_centroid(self, Point pnt)
    cpdef int delete_empty_cells(self) except -1

    cpdef set_cell_manager(self, CellManager cell_manager)
    cpdef int update_cell_manager_hierarchy_list(self) except -1
    cpdef int clear(self) except -1
    cpdef bint is_leaf(self)
    cpdef Cell get_new_sibling(self, IntPoint id)
    cpdef get_particle_ids(self, list particle_id_list)
    cpdef Cell get_new_child(self, IntPoint id)
    cpdef insert_particles(self, int parray_id, LongArray indices)
    cpdef clear_indices(self, int parray_id)
    cpdef double get_child_size(self)

cdef class LeafCell(Cell):
    """
    A Leaf cell class.
    """
    cdef public list index_lists
    cdef _init_index_lists(self)
    cpdef set_cell_manager(self, CellManager cell_manager)
    cpdef Cell get_new_sibling(self, IntPoint id)

cdef class NonLeafCell(Cell):
    """
    Class to represent any cell above the leaf cell.
    """
    cdef public dict cell_dict
    cpdef Cell get_new_sibling(self, IntPoint id)

cdef class RootCell(NonLeafCell):
    """
    Class to represent a root cell in a cell hierarchy.
    """
    pass

cdef class CellManager:
    """
    Class to manager all cells.
    """
    cdef public Point origin
    cdef public int num_levels
    cdef public DoubleArray cell_sizes
    cdef public double cell_size_step
    cdef public object particle_manager
    cdef public bint is_dirty    
    cdef public dict array_indices
    cdef public list arrays_to_bin
    cdef public list hierarchy_list
    cdef public double min_cell_size, max_cell_size
    cdef public int jump_tolerance

    cdef public bint initialized
    
    cdef public str coord_x, coord_y, coord_z

    cdef public RootCell root_cell
    
    cpdef int update(self) except -1
    cpdef int update_status(self) except -1
    cpdef initialize(self)
    cpdef clear(self)
    cpdef add_array_to_bin(self, ParticleArray parr)

    cpdef compute_cell_sizes(self, double, double, int, DoubleArray)
    cpdef update_cell_hierarchy_list(self)
    cpdef _build_base_hierarchy(self)
    cpdef _rebuild_array_indices(self)
    cpdef _setup_hierarchy_list(self)
    
    cdef int get_potential_cells(self, Point pnt, double radius, list cell_list,
                                 bint single_layer=*) except -1

    cdef int _get_cells_within_radius(self, Point pnt, double radius, int level, list
                                      cell_list) except -1
    cdef int _get_one_layer_of_cells(self, Point pnt, double radius, int level, list
                                     cell_list) except -1
    cdef void _reset_jump_tolerance(self)

