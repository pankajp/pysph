"""
Various 3d kernels declarations.
"""

from pysph.base.kernelbase cimport Kernel3D

cdef class Poly6Kernel3D(Kernel3D):
    """
    This class represents a polynomial kernel with support 1.0
    """
    pass

cdef class CubicSpline3D(Kernel3D):
    """
    Cubic spline kernel from Mon05
    """
    pass


