from pysph.base.point cimport *
from pysph.base.carray cimport *

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
    cdef int add_particles(self, Cell cell) except -1
    cdef int update(self, dict data) except -1
    cdef long get_number_of_particles(self)
    cdef bint is_empty(self)
    cdef void get_centroid(self, Point pnt)
    cdef void delete_empty_cells(self)

    cpdef set_cell_manager(self, CellManager cell_manager)
    cdef void update_cell_manager_hierarchy_list(self)
    cdef void clear(self)
    cdef bint is_leaf(self)

cdef class LeafCell(Cell):
    """
    A Leaf cell class.
    """
    cdef public list index_lists
    cdef _init_index_lists(self)
    cpdef set_cell_manager(self, CellManager cell_manager)

cdef class NonLeafCell(Cell):
    """
    Class to represent any cell above the leaf cell.
    """
    cdef public dict cell_dict    

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
    
    cdef public str coord_x, coord_y, coord_z

    cdef public RootCell root_cell
    
    cdef int update(self) except -1
    cdef int update_status(self) except -1
    cdef initialize(self)
    cdef void clear(self)

    cdef void compute_cell_sizes(self, double, double, int, DoubleArray)
    cdef void update_cell_hierarchy_list(self)
    cdef void _build_base_hierarchy(self) except *
    cdef void _rebuild_array_indices(self)
    cdef void _setup_hierarchy_list(self)
    cdef int get_potential_cells(self, Point pnt, double radius, list cell_list,
                                 bint single_layer=*) except -1

    cdef int _get_cells_within_radius(self, Point pnt, double radius, int level, list
                                      cell_list) except -1
    cdef int _get_one_layer_of_cells(self, Point pnt, double radius, int level, list
                                     cell_list) except -1
    cdef void _reset_jump_tolerance(self)
