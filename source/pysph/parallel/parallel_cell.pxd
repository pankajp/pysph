"""
Declaration file for cython parallel cell module.
"""


from pysph.base.point cimport Point, IntPoint
from pysph.base.cell cimport CellManager, Cell

# forward declarations.
cdef class ParallelCellManager(CellManager)

###############################################################################
# `ProcessorMap` class.
###############################################################################
cdef class ProcessorMap:
     cdef public ParallelCellManager cell_manager
     cdef public Point origin
     cdef public local_block_map
     cdef public dict block_map
     cdef public list nbr_procs
     cdef public int pid
     cdef public double block_size
     cdef public dict cell_map

     cpdef merge(self, ProcessorMap proc_map)
     cpdef find_region_neighbors(self)

###############################################################################
# `ParallelCellManager` class.
###############################################################################
cdef class ParallelCellManager(CellManager):
    cdef public object solver 
    cdef public int dimension
    cdef public list glb_bounds_min, glb_bounds_max
    cdef public list local_bounds_min, local_bounds_max
    cdef public double glb_min_h, glb_max_h
    cdef public double local_min_h, local_max_h
    cdef public int factor
    cdef public int pid

    cdef public object parallel_controller, pc
    cdef public load_balancer
    cdef public ProcessorMap proc_map
    cdef public bint load_balancing

    cpdef glb_update_proc_map(self)
    cpdef remove_remote_particles(self)

    cdef public bint initial_redistribution_done
    cdef public dict adjacent_remote_cells
    cdef public dict remote_particle_indices
    cdef public dict nbr_cell_info
    cdef public dict new_particles_for_neighbors
    cdef public dict new_region_particles
    cdef public dict new_cells_added
    cdef public list adjacent_processors

    cdef public dict neighbor_share_data

    #cdef public ParallelCellManager cell_manager
    cpdef find_adjacent_remote_cells(self)
    cpdef update_cell_neighbor_information(self)
    cpdef bin_particles_top_down(self)
    cpdef bin_particles(self)
    cpdef create_new_particle_copies(self, dict cell_dict)
    cpdef assign_new_blocks(self, dict new_block_dict, dict new_particles)
    cpdef dict _resolve_conflicts(self, dict data)
    cpdef exchange_crossing_particles_with_neighbors(self, dict remote_cells,
                                                     dict particles)
    cpdef update_remote_particle_properties(self, list props=*)
    cpdef exchange_neighbor_particles(self)
    cpdef add_entering_particles_from_neighbors(self, dict new_particles)
    cpdef add_local_particles_to_parray(self, dict particle_data)
    cdef list get_communication_data(self, int num_arrays, list cell_list)

    cpdef Cell get_new_cell_for_copy(self, IntPoint id, int pid)
    cpdef dict _get_cell_data_for_neighbor(self, list cell_list, list props=*)
    
cdef class ParallelCellInfo:
    cdef public Cell cell
    cdef ParallelCellManager cell_manager
    cdef public dict neighbor_cell_pids
    cdef public dict remote_pid_cell_count
    cdef public int num_remote_neighbors
    cdef public int num_local_neighbors
    
cdef class ParallelCell(Cell):
    cdef public int pid
    cdef public ParallelCellInfo parallel_cell_info

cdef class ParallelCellRemoteCopy(ParallelCell):
    cdef public list particle_start_indices
    cdef public list particle_end_indices
    cdef public int num_particles
