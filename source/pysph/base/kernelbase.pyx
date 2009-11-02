"""
Contains the base class for all kernels.

1-d kernels derive from Kernel1D and are found in kernels1d.pyx
2-d kernels derive from Kernel2D and are found in kernels2d.pyx
3-d kernels derive from Kernel3D and are found in kernels3d.pyx

"""

# local imports
from pysph.base.point cimport Point


################################################################################
# `KernelBase` class.
################################################################################
cdef class KernelBase:
    """
    Base class for all kernels.
    """
    cdef double function(self, Point p1, Point p2, double h):
        """        
        """
        raise NotImplementedError, 'KernelBase::function'

    cdef void gradient(self, Point p1, Point p2, double h, Point result):
        """
        """
        raise NotImplementedError, 'KernelBase::gradient'

    cdef double laplacian(self, Point p1, Point p2, double h):
        """
        """
        raise NotImplementedError, 'KernelBase::laplacian'

    cpdef double radius(self):
        """
        """
        raise NotImplementedError, 'KernelBase::radius'

    cpdef int dimension(self):
        """
        """
        raise NotImplementedError, 'KernelBase::dimension'

    def py_function(self, Point p1, Point p2, double h):
        return self.function(p1, p2, h)

    def py_gradient(self, Point p1, Point p2, double h, Point result):
        self.gradient(p1, p2, h, result)

    def py_laplacian(self, Point p1, Point p2, double h):
        return self.laplacian(p1, p2, h)

################################################################################
# `Kernel1D` class.
################################################################################
cdef class Kernel1D(KernelBase):
    """
    Base class for 1-d kernels.
    """
    cpdef int dimension(self):
        """
        """
        return 1

################################################################################
# `Kernel2D` class.
################################################################################ 
cdef class Kernel2D(KernelBase):
    """
    Base class for 2-d kernels.
    """
    cpdef int dimension(self):
        """
        """
        return 2

################################################################################
# `Kernel3D` class.
################################################################################ 
cdef class Kernel3D(KernelBase):
    """
    Base class for 3-d kernels.
    """
    cpdef int dimension(self):
        """
        """
        return 3
