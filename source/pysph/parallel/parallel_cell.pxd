"""
Declaration file for cython parallel cell module.
"""


from pysph.base.point cimport Point, IntPoint
from pysph.base.cell cimport CellManager, Cell
cimport mpi4py.MPI as MPI


cpdef dict share_data(int mypid, list send_procs, object data,
                      MPI.Comm comm, int tag=*, bint multi=*,
                      list recv_procs=*)

# forward declarations.
cdef class ParallelCellManager(CellManager)

###############################################################################
# `ProcessorMap` class.
###############################################################################
cdef class ProcessorMap:
     cdef public Point origin
     cdef public dict local_block_map
     cdef public dict block_map
     cdef public dict load_per_proc
     cdef public list nbr_procs
     cdef public int pid
     cdef public double block_size
     cdef public dict cell_map
     cdef public dict conflicts

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
    cdef dict trf_particles

    cdef double block_size
    cdef double min_block_size

    cdef public object parallel_controller
    cdef public load_balancer
    cdef public ProcessorMap proc_map
    cdef public bint load_balancing

    cpdef glb_update_proc_map(self)
    cpdef remove_remote_particles(self)

    cdef public bint initial_redistribution_done
    cdef public dict remote_particle_indices

    #cdef public ParallelCellManager cell_manager
    cpdef compute_block_size(self, double block_size)
    cpdef update_cell_neighbor_information(self)
    cpdef rebin_particles(self)
    cpdef bin_particles(self)
    cpdef create_new_particle_copies(self, dict blocks_dict_to_copy,
                                     bint mark_src_remote=*, bint local_only=*)
    cpdef mark_crossing_particles(self, dict remote_block_dict)
    cpdef assign_new_blocks(self, dict new_block_dict)
    cpdef dict _resolve_conflicts(self, dict data)
    cpdef exchange_crossing_particles_with_neighbors(self, dict block_particles)
    cpdef update_remote_particle_properties(self, list props=*)
    cpdef exchange_neighbor_particles(self)
    cpdef add_entering_particles_from_neighbors(self, dict new_particles)
    cpdef add_local_particles_to_parray(self, dict particle_data)
    cpdef update_remote_particle_properties(self, list props=*)
    cpdef exchange_neighbor_particles(self)
    cpdef transfer_blocks_to_procs(self, dict procs_blocks,
                                   bint mark_remote=*, list recv_procs=*)
    cdef list get_communication_data(self, int num_arrays, list cell_list)

    cpdef dict _get_cell_data_for_neighbor(self, list cell_list, list props=*)
    cpdef list get_cells_in_block(self, IntPoint bid)
    cpdef list get_particle_indices_in_block(self, IntPoint bid)
    cpdef list get_particles_in_block(self, IntPoint bid)
    
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
