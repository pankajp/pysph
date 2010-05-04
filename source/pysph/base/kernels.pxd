""" Definitions for the SPH kernels defined in kernels.pyx
"""

#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Kunal Puri

from pysph.base.point cimport Point

##############################################################################
#`MultidimensionalKernel`
##############################################################################
cdef class MultidimensionalKernel:
    cdef public int dim

    cdef double function(self, Point p1, Point p2, double h)
    cdef void gradient(self, Point p1, Point p2, double h, Point result)
    cdef double laplacian(self, Point p1, Point p2, double h)
    cdef double _fac(self, double h)
    cpdef double radius(self)
    cpdef int dimension(self)

##############################################################################
#`CubicSplineKernel`
##############################################################################
cdef class CubicSplineKernel(MultidimensionalKernel):
    pass
