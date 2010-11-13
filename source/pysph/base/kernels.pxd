""" Definitions for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Kunal Puri

from pysph.base.point cimport Point, Point_sub

##############################################################################
#`KernelBase`
##############################################################################
cdef class KernelBase:
    cdef readonly int dim
    cdef readonly double fac

    cdef double function(self, Point pa, Point pb, double h)
    cdef void gradient(self, Point pa, Point pb, double h, Point result)
    cdef double laplacian(self, Point pa, Point pb, double h)
    cdef double _fac(self, double h)
    cpdef double radius(self)
    cpdef int dimension(self)

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
    cdef public dict facs
    #cdef public Point r_tmp


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
