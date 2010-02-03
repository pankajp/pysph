# cython: profile=True
"""
Declaration file for cython parallel cell module.
"""

from pysph.solver.base cimport Base
from pysph.base.point cimport Point, IntPoint
from pysph.base.cell cimport CellManager, RootCell, Cell, LeafCell, NonLeafCell

# forward declarations.
cdef class ParallelCellManager(CellManager)
cdef class ProcessorMap(Base)

################################################################################
# `ProcessorMap` class.
################################################################################
cdef class ProcessorMap(Base):
     cdef public ParallelCellManager cell_manager
     cdef public Point origin
     cdef public dict local_p_map
     cdef public dict p_map
     cdef public list nbr_procs
     cdef public int pid
     cdef public double bin_size

     cpdef merge(self, ProcessorMap proc_map)
     cpdef find_region_neighbors(self)


################################################################################
# `ParallelCellManager` class.
################################################################################
cdef class ParallelCellManager(CellManager):
    cdef public object solver 
    cdef public int dimension
    cdef public list glb_bounds_min, glb_bounds_max
    cdef public double glb_min_h, glb_max_h
    cdef public double max_radius_scale
    cdef public int pid, parallel_cell_level

    cdef public object parallel_controller, pc
    cdef public load_balancer
    cdef public ProcessorMap proc_map
    cdef public bool load_balancing

    cpdef glb_update_proc_map(self)
    cpdef remove_remote_particles(self)


#rom pysph.parallel.parallel_controller cimport ParallelController
cdef class ParallelRootCell(RootCell):
    cdef public bint initial_redistribution_done
    cdef public dict adjacent_remote_cells
    cdef public dict nbr_cell_info
    cdef public dict new_particles_for_neighbors
    cdef public dict new_region_particles
    cdef public dict new_cells_added
    cdef public list adjacent_processors
    #cdef public ParallelCellManager cell_manager
    cdef public int pid
    cdef public object parallel_controller

    cpdef Cell get_new_child_for_copy(self, IntPoint id, int pid)
    cpdef find_adjacent_remote_cells(self)
    cpdef update_cell_neighbor_information(self)
    cpdef bin_particles_top_down(self)
    cpdef bin_particles(self)
    cpdef create_new_particle_copies(self, dict cell_dict)
    cpdef assign_new_cells(self, dict new_cell_dict, dict new_particles)
    cpdef dict _resolve_conflicts(self, dict data)
    cpdef exchange_crossing_particles_with_neighbors(self, dict remote_cells,
                                                     dict particles)
    cpdef add_entering_particles_from_neighbors(self, dict new_particles)
    cpdef add_local_particles_to_parray(self, dict particle_list)
    cpdef update_remote_particle_properties(self, list props=*)
    cpdef exchange_neighbor_particles(self)
    cpdef dict _get_cell_data_for_neighbor(self, list cell_list, list props=*)
    
cdef class ParallelCellInfo(Base):
    cdef public Cell cell
    cdef public ParallelRootCell root_cell
    cdef public dict neighbor_cell_pids
    cdef public dict remote_pid_cell_count
    cdef public int num_remote_neighbors
    cdef public int num_local_neighbors
    
cdef class ParallelLeafCell(LeafCell):
    cdef public int pid
    cdef public ParallelCellInfo parallel_cell_info

cdef class ParallelNonLeafCell(NonLeafCell):
    cdef public int pid
    cdef public ParallelCellInfo parallel_cell_info

cdef class ParallelLeafCellRemoteCopy(ParallelLeafCell):
    cdef public list particle_start_indices
    cdef public list particle_end_indices
    cdef public int num_particles

cdef class ParallelNonLeafCellRemoteCopy(ParallelNonLeafCell):
    cdef public list particle_start_indices
    cdef public list particle_end_indices
    cdef public int num_particles
