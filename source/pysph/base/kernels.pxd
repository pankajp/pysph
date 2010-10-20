""" Definitions for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Kunal Puri

from pysph.base.point cimport Point, Point_sub

##############################################################################
#`MultidimensionalKernel`
##############################################################################
cdef class MultidimensionalKernel:
    cdef public int dim

    cdef double function(self, Point pa, Point pb, double h)
    cdef void gradient(self, Point pa, Point pb, double h, Point result)
    cdef double laplacian(self, Point pa, Point pb, double h)
    cdef double _fac(self, double h)
    cpdef double radius(self)
    cpdef int dimension(self)

##############################################################################
#`CubicSplineKernel`
##############################################################################
cdef class CubicSplineKernel(MultidimensionalKernel):
    pass

##############################################################################
#`HarmonicKernel`
##############################################################################
cdef class HarmonicKernel(MultidimensionalKernel):
    cdef public int n
    cdef public dict facs


##############################################################################
#'Gaussian Kernel'
##############################################################################
cdef class GaussianKernel(MultidimensionalKernel):
    pass

##############################################################################
#`M6SplineKernel`
##############################################################################
cdef class M6SplineKernel(MultidimensionalKernel):
    pass

##############################################################################
#'W8 Kernel'
##############################################################################
cdef class W8Kernel(MultidimensionalKernel):
    pass

##############################################################################
#`W10Kernel`
##############################################################################
cdef class W10Kernel(MultidimensionalKernel):
    pass

cdef class RepulsiveBoundaryKernel(MultidimensionalKernel):
    """
    """
    pass
