"""
Module containing integrators that perform XSPH velocity correction prior to
position stepping.
"""

# local imports
from pysph.solver.integrator_base cimport *

cdef class EulerXSPHIntegrator(Integrator):
    """
    Class to perform position stepping by applying xsph velocity correction.
    """
    pass
