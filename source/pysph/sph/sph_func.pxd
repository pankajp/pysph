
cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport DoubleArray
from pysph.base.point cimport Point

from pysph.base.kernelbase cimport KernelBase



################################################################################
# `MakeCoords` class.
################################################################################
cdef class MakeCoords:
    pass

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle:
    cdef public ParticleArray source, dest
    cdef public str h, mass, rho
    cdef public DoubleArray s_h, d_h
    cdef public DoubleArray s_mass, d_mass    
    cdef public DoubleArray s_rho, d_rho

    cdef public Point _pnt1
    cdef public Point _pnt2

    cpdef setup_arrays(self)

    cpdef int output_fields(self) except -1
    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr)
    cpdef py_eval(self, int source_pid, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr)

cdef class SPHFuncParticleUser(SPHFunctionParticle):
    """
    User defined SPHFunctionParticle.
    """
    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr)
    cpdef py_eval(self, int source_pid, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr)

################################################################################
# `SPHFunctionPoint` class.
################################################################################
cdef class SPHFunctionPoint:
    cdef public ParticleArray source
    cdef public str h, mass, rho
    cdef public DoubleArray s_h, s_mass, s_rho

    cdef public Point _pnt1, _pnt2
    
    cpdef setup_arrays(self)
    cpdef int output_fields(self) except -1

    cdef void eval(self, Point pnt, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr) 
    cpdef py_eval(self, Point pnt, int dest_pid, KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr)

################################################################################
# `SPHFunctionParticle1D` class.
################################################################################ 
cdef class SPHFunctionParticle1D(SPHFunctionParticle):
    """
    1-D SPH function.
    """
    cdef public DoubleArray s_x, d_x
    cdef public DoubleArray s_velx, d_velx
    cdef public str coord_x, velx
    
    cpdef setup_arrays(self)

################################################################################
# `SPHFunctionPoint1D` class.
################################################################################
cdef class SPHFunctionPoint1D(SPHFunctionPoint):
    """
    1-D SPH function at random points.
    """
    cdef public DoubleArray s_x
    cdef public DoubleArray s_velx
    cdef public str coord_x, velx
    
    cpdef setup_arrays(self)

################################################################################
# `SPHFunctionParticle2D` class.
################################################################################
cdef class SPHFunctionParticle2D(SPHFunctionParticle):
    """
    2-D SPH function.
    """
    cdef public DoubleArray s_x, d_x
    cdef public DoubleArray s_y, d_y

    cdef public DoubleArray s_velx, d_velx
    cdef public DoubleArray s_vely, d_vely

    cdef public str coord_x, coord_y, velx, vely

    cpdef setup_arrays(self)

################################################################################
# `SPHFunctionPoint2D` class.
################################################################################
cdef class SPHFunctionPoint2D(SPHFunctionPoint):
    """
    2-D SPH function for random points.
    """
    cdef public DoubleArray s_x, s_y
    cdef public DoubleArray s_velx, s_vely
    cdef public str coord_x, coord_y, velx, vely

    cpdef setup_arrays(self)

################################################################################
# `SPHFunctionParticle3D` class.
################################################################################
cdef class SPHFunctionParticle3D(SPHFunctionParticle):
    """
    3-D SPH function.
    """
    cdef public DoubleArray s_x, d_x
    cdef public DoubleArray s_y, d_y
    cdef public DoubleArray s_z, d_z

    cdef public DoubleArray s_velx, d_velx
    cdef public DoubleArray s_vely, d_vely
    cdef public DoubleArray s_velz, d_velz

    cdef public str coord_x, coord_y, coord_z, velx, vely, velz
    
    cpdef setup_arrays(self)

################################################################################
# `SPHFunctionPoint3D` class.
################################################################################    
cdef class SPHFunctionPoint3D(SPHFunctionPoint):
    """
    3-D SPH function at random points.
    """
    cdef public DoubleArray s_x
    cdef public DoubleArray s_y
    cdef public DoubleArray s_z

    cdef public DoubleArray s_velx
    cdef public DoubleArray s_vely
    cdef public DoubleArray s_velz

    cdef public str coord_x, coord_y, coord_z, velx, vely, velz

    cpdef setup_arrays(self)


# some inline convenience functions.
cdef inline void make_coords_1d(DoubleArray x, Point pnt, int pid)
cdef inline void make_coords_2d(DoubleArray x, DoubleArray y, Point pnt, int pid)
cdef inline void make_coords_3d(DoubleArray x, DoubleArray y, DoubleArray z, Point
                           pnt, int pid)
