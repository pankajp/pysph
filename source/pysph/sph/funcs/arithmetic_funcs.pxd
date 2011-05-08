from pysph.sph.sph_func cimport SPHFunction
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport DoubleArray
from pysph.base.kernels cimport KernelBase

cdef class PropertyGet(SPHFunction):
    cdef list prop_names, d_props

cdef class PropertyAdd(SPHFunction):
    cdef list prop_names, d_props
    cdef int num_props
    cdef public double constant

cdef class PropertyNeg(SPHFunction):
    cdef list prop_names, d_props

cdef class PropertyMul(SPHFunction):
    cdef list prop_names, d_props
    cdef int num_props
    cdef public double constant

cdef class PropertyInv(SPHFunction):
    cdef list prop_names, d_props

