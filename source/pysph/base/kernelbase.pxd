"""
Contains the base class for all kernels.
"""

# local imports
from pysph.base.point cimport Point

cdef class KernelBase:
    """
    Base class for all kernels.
    """
    cdef double function(self, Point p1, Point p2, double h)
    cdef void gradient(self, Point p1, Point p2, double h, Point result)
    cdef double laplacian(self, Point p1, Point p2, double h)
    cpdef double radius(self)
    cpdef int dimension(self)

cdef class Kernel1D(KernelBase):
    """
    Base class for 1-d kernels.
    """
    pass

cdef class Kernel2D(KernelBase):
    """
    Base class for 2-d kernels.
    """
    pass

cdef class Kernel3D(KernelBase):
    """
    Base class for 3-d kernels.
    """
    pass
