"""
Module containing integrators that perform XSPH velocity correction prior to
position stepping.
"""

# local imports
from pysph.solver.integrator_base cimport *
from pysph.solver.runge_kutta_integrator cimport *

cdef class EulerXSPHIntegrator(Integrator):
    """
    Class to perform position stepping by applying xsph velocity correction.
    """
    pass

cdef class RK2XSPHIntegrator(RK2Integrator):
    """
    Class to perform position stepping by applying xsph velocity correction.
    """
    pass
