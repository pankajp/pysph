"""
Component to implement XSPH velocity correction.
"""

from pysph.sph.sph_func cimport *
from pysph.solver.sph_component cimport *
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.integrator_base cimport *

cdef class XSPHFunction3D(SPHFunctionParticle3D):
    """
    SPH function to compute xpsh velocity correction for 3d particles.
    """
    pass

cdef class XSPHVelocityComponent(SPHComponent):
    """
    Component to compute velocity correction using the XSPH method.
    """
    cdef public double epsilon

cdef class EulerXSPHPositionStepper(ODEStepper):
    """
    Position stepper with XSPH correction.
    """
    cdef public double epsilon
    cdef public XSPHVelocityComponent xsph_component

