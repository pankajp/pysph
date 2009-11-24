"""
Component to update the NNPS.
"""

# local imports
from pysph.base.nnps cimport NNPSManager
from pysph.solver.solver_base cimport SolverComponent

cdef class NNPSUpdater(SolverComponent):
    """
    Component to update the nnps every step.
    """
    cdef public NNPSManager nnps_manager


