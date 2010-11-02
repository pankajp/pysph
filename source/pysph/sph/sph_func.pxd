
cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport DoubleArray, IntArray
from pysph.base.point cimport Point

from pysph.base.kernels cimport KernelBase


################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle:
    cdef public ParticleArray source, dest
    cdef public str h, m, rho, p, e, x, y, z, u, v, w
    cdef public str tmpx, tmpy, tmpz, type
    cdef public str cs
        
    cdef public DoubleArray s_h, s_m, s_rho, d_h, d_m, d_rho
    cdef public DoubleArray s_x, s_y, s_z, d_x, d_y, d_z
    cdef public DoubleArray s_u, s_v, s_w, d_u, d_v, d_w
    cdef public DoubleArray s_p, s_e, d_p, d_e	
    cdef public DoubleArray s_cs, d_cs

    #rkpm first order correction terms
    
    cdef public DoubleArray rkpm_beta1, rkpm_beta2, rkpm_beta3
    cdef public DoubleArray rkpm_alpha, rkpm_dalphadx, rkpm_dalphady
    cdef public DoubleArray rkpm_dbeta1dx, rkpm_dbeta1dy
    cdef public DoubleArray rkpm_dbeta2dx, rkpm_dbeta2dy

    #bonnet and lok correction terms ONLY FOR THE DESTINATION!!!
    cdef public DoubleArray bl_l11, bl_l12, bl_l13, bl_l22, bl_l23, bl_l33
    
    cdef public Point _src
    cdef public Point _dst
    
    cdef public str name, id

    cdef public bint bonnet_and_lok_correction
    cdef public bint rkpm_first_order_correction

    cpdef setup_arrays(self)

    cpdef int output_fields(self) except -1

    cdef void eval(self, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr)

    cdef double rkpm_first_order_kernel_correction(self, int dest_pid)

    cdef double rkpm_first_order_gradient_correction(self, int dest_pid)

    cdef double bonnet_and_lok_gradient_correction(self, int dest_pid,
                                                   Point grad)

################################################################################
# `SPHFunctionPoint` class.
################################################################################
cdef class SPHFunctionPoint:
    cdef public ParticleArray source
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
                   KernelBase kernel, double *nr, double *dnr)

    cpdef py_eval(self, Point pnt, int dest_pid, 
                  KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr)

##############################################################################
