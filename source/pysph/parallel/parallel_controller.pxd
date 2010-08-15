"""
Declarations for the parallel_controller module
"""

# mpi imports
#cimport mpi4py.MPI as MPI
from mpi4py cimport MPI

# local imports

#from pysph.parallel.parallel_cell cimport ParallelCellManager

cdef class ParallelController:
    """
    """
    cdef public object solver
    cdef public object  cell_manager
    cdef public MPI.Comm comm
    cdef public int num_procs
    cdef public list children_proc_ranks
    cdef public int parent_rank
    cdef public bint setup_control_tree_done
    cdef public int rank
    cdef public int l_child_rank, r_child_rank
    

    cpdef get_glb_min_max(self, dict local_min_dict, dict local_max_dict)
    

