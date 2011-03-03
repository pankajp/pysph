from pysph.base.point cimport *
from pysph.base.carray cimport *

from pysph.base.particle_array cimport ParticleArray

# forward declaration for CellManager
cdef class CellManager

cdef inline int real_to_int(double val, double step)
cdef inline cIntPoint find_cell_id(cPoint pnt, double cell_size)
cdef inline void construct_immediate_neighbor_list(cIntPoint pnt, list
               neighbor_list, bint include_self=*, int distance=*)

cdef inline bint cell_encloses_sphere(IntPoint id,
                          double cell_size, cPoint pnt, double radius)

cdef class Cell:
    # Member variables.
    cdef public IntPoint id
    cdef public double cell_size
    cdef public CellManager cell_manager
    cdef public list arrays_to_bin
    
    cdef readonly str coord_x
    cdef readonly str coord_y
    cdef readonly str coord_z

    cdef public int jump_tolerance
    cdef public list index_lists
    
    # Member functions.
    cpdef int add_particles(self, Cell cell) except -1
    cpdef int update(self, dict data) except -1
    cpdef long get_number_of_particles(self)
    cpdef bint is_empty(self)
    cpdef get_centroid(self, Point pnt)

    cpdef set_cell_manager(self, CellManager cell_manager)
    
    cpdef int clear(self) except -1
    cpdef Cell get_new_sibling(self, IntPoint id)
    cpdef get_particle_ids(self, list particle_id_list)
    cpdef get_particle_counts_ids(self, list particle_list, LongArray particle_counts)
    cpdef insert_particles(self, int parray_id, LongArray indices)
    cpdef clear_indices(self, int parray_id)

    cdef _init_index_lists(self)
    cpdef set_cell_manager(self, CellManager cell_manager)
    cpdef Cell get_new_sibling(self, IntPoint id)


cdef class CellManager:
    """
    Class to manager all cells.
    """
    cdef public double cell_size
    cdef public bint is_dirty    
    cdef public dict array_indices
    cdef public list arrays_to_bin
    cdef public dict cells_dict
    cdef public double min_cell_size, max_cell_size
    cdef public double max_radius_scale
    cdef public int jump_tolerance

    cdef public bint initialized
    
    cdef public str coord_x, coord_y, coord_z

    cpdef int update(self) except -1
    cpdef int update_status(self) except -1
    cpdef initialize(self)
    cpdef clear(self)
    cpdef add_array_to_bin(self, ParticleArray parr)

    cpdef double compute_cell_size(self, double min_size=*, double max_size=*)
    
    # carried over from RootCell
    cpdef int cells_update(self) except -1
    
    cpdef _build_cell(self)
    cpdef _rebuild_array_indices(self)
    cpdef _setup_cells_dict(self)
    cpdef set_jump_tolerance(self, int jump_tolerance)
    cdef check_jump_tolerance(self, cIntPoint myid, cIntPoint newid)
    cpdef list delete_empty_cells(self)
    cpdef insert_particles(self, int parray_id, LongArray indices)
    cpdef Cell get_new_cell(self, IntPoint id)
    
    cdef int get_potential_cells(self, cPoint pnt, double radius, list cell_list) except -1

    cdef int _get_cells_within_radius(self, cPoint pnt, double radius,
                                      list cell_list) except -1
    cdef void _reset_jump_tolerance(self)
    cpdef long get_number_of_particles(self)
