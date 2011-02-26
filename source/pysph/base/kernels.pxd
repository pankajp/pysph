""" Definitions for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunal.r.puri@gmail.com>
#Copyright (c) 2010, Prabhu Ramachandran

from pysph.base.point cimport Point, cPoint_sub, cPoint, cPoint_new, \
            cPoint_norm, cPoint_scale
from pysph.base.carray cimport DoubleArray

cimport numpy

##############################################################################
#`KernelBase`
##############################################################################
cdef class KernelBase:
    cdef readonly int dim
    cdef readonly double fac

    cdef public DoubleArray smoothing
    cdef public DoubleArray distances
    cdef public DoubleArray function_cache
    cdef public DoubleArray gradient_cache

    cdef public bint has_constant_h
    cdef public double constant_h
    cdef public double distances_dx

    cdef double function(self, cPoint pa, cPoint pb, double h)
    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h)
    cdef double _fac(self, double h)
    cpdef double radius(self)
    cpdef int dimension(self)
    cpdef double __gradient(self, Point pa, Point pb, double h)

    cdef interpolate_function(self, double rab)
    cdef interpolate_gradients(self, double rab)

##############################################################################
# `Poly6Kernel` class.
##############################################################################
cdef class Poly6Kernel(KernelBase):
    pass

##############################################################################
#`CubicSplineKernel`
##############################################################################
cdef class CubicSplineKernel(KernelBase):
    pass

##############################################################################
#`QuinticSplineKernel`
##############################################################################
cdef class QuinticSplineKernel(KernelBase):
    pass

##############################################################################
#`WendlandQuinticSplineKernel`
##############################################################################
cdef class WendlandQuinticSplineKernel(KernelBase):
    pass

##############################################################################
#`HarmonicKernel`
##############################################################################
cdef class HarmonicKernel(KernelBase):
    cdef public int n
    cdef double facs[10]

##############################################################################
#'Gaussian Kernel'
##############################################################################
cdef class GaussianKernel(KernelBase):
    pass

##############################################################################
#`M6SplineKernel`
##############################################################################
cdef class M6SplineKernel(KernelBase):
    pass

##############################################################################
#'W8 Kernel'
##############################################################################
cdef class W8Kernel(KernelBase):
    pass

##############################################################################
#`W10Kernel`
##############################################################################
cdef class W10Kernel(KernelBase):
    pass

cdef class RepulsiveBoundaryKernel(KernelBase):
    pass
