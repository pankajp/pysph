"""
Components to compute forces at boundaries.
"""

# local imports
from pysph.base.kernels cimport KernelBase
from pysph.sph.sph_func cimport SPHFunctionParticle3D
from pysph.solver.speed_of_sound cimport SpeedOfSound
from pysph.solver.sph_component cimport SPHComponent


cdef class RepulsiveBoundaryKernel(KernelBase):
    """
    """
    pass

cdef class SPHRepulsiveBoundaryFunction(SPHFunctionParticle3D):
    """
    Class to compute iteraction of boundary and non-boundary particles.
    """
    cdef public SpeedOfSound speed_of_sound


cdef class SPHRepulsiveBoundaryForceComponent(SPHComponent):
    """
    SPH Component to compute repulsive forces from solid boundaries.
    """
    cdef public SpeedOfSound speed_of_sound
