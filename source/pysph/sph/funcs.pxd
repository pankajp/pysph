"""
SPH functions for density and related computation.
"""

# local imports
from pysph.sph.sph_func cimport SPHFunctionParticle3D
from pysph.base.carray cimport DoubleArray

from pysph.base.point cimport Point
from pysph.base.kernels cimport MultidimensionalKernel
from pysph.base.particle_array cimport ParticleArray


cdef class SPH(SPHFunctionParticle3D):
    """
    Simple interpolation function for 3D cases.
    """
    cdef str prop_name
    cdef DoubleArray s_prop, d_prop

cdef class SPHRho(SPHFunctionParticle3D):
    """
    SPH function to compute density for 3d particles.
    """
    pass

cdef class SPHDensityRate(SPHFunctionParticle3D):
    """
    SPH function tom compute density rate for 3d particles.
    """
    pass


cdef class SPHPressureGradient(SPHFunctionParticle3D):
    """
    SPH function to compute pressure gradient.
    """
    cdef str pressure
    cdef DoubleArray s_pressure, d_pressure
