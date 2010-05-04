"""
Contains various components to compute pressure gradients.
"""

# pysph.base imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.kernels cimport MultidimensionalKernel

#pysph.sph imports
from pysph.sph.funcs cimport SPHPressureGradient, SPHRho
from pysph.sph.calc cimport SPHCalc

#pysph.solver imports
from entity_types cimport EntityTypes
from sph_component cimport SPHSourceDestMode, SPHComponent
from fluid cimport Fluid


cdef class SPHPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    cdef int compute(self) except -1
    
cdef class SPHSummationDensityComponent(SPHComponent):
    """ Summation density """
    
    cdef int compute(self) except -1
