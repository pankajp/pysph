
cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport DoubleArray, IntArray
from pysph.base.point cimport Point

from pysph.base.kernels cimport MultidimensionalKernel


################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle:
    cdef public ParticleArray source, dest
    cdef public str h, m, rho, p, e, x, y, z, u, v, w
    cdef public str tmpx, tmpy, tmpz, type
        
    cdef public DoubleArray s_h, s_m, s_rho, d_h, d_m, d_rho
    cdef public DoubleArray s_x, s_y, s_z, d_x, d_y, d_z
    cdef public DoubleArray s_u, s_v, s_w, d_u, d_v, d_w
    cdef public DoubleArray s_p, s_e, d_p, d_e

    cdef public Point _src
    cdef public Point _dst
    
    cdef public str name, id

    cdef public bint kernel_gradient_correction	

    cpdef setup_arrays(self)

    cpdef int output_fields(self) except -1

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr)

################################################################################
# `SPHFunctionPoint` class.
################################################################################
cdef class SPHFunctionPoint:
    cdef public ParticleArray array
    cdef public str h, m, rho, p, e, x, y, z, u, v, w
    cdef public str tmpx, tmpy, tmpz
        
    cdef public DoubleArray s_h, s_m, s_rho, d_h, d_m, d_rho
    cdef public DoubleArray s_p, s_e, d_p, d_e
    cdef public DoubleArray s_x, s_y, s_z, d_x, d_y, d_z
    cdef public DoubleArray s_u, s_v, s_w, d_u, d_v, d_w

    cdef public Point _src, _dst
    
    cpdef setup_arrays(self)
    cpdef int output_fields(self) except -1

    cdef void eval(self, Point pnt, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr)

    cpdef py_eval(self, Point pnt, int dest_pid, 
                  MultidimensionalKernel kernel, numpy.ndarray
                  nr, numpy.ndarray dnr)

##############################################################################
