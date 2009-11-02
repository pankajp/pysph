"""
Various 2D kernels.
"""

cimport numpy

from pysph.base.point cimport Point
from pysph.base.kernelbase cimport Kernel2D


from math import pi

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double fabs(double)


cdef:
    double PI = pi
    double SQRT_1_PI = 1.0/sqrt(PI)
    double infty = numpy.inf

################################################################################
# `CubicSpline2D` class.
################################################################################
cdef class CubicSpline2D(Kernel2D):
    """
    Cubic spline kernel for use in 2D computations.
    The z coordinates are expected to contain 0 values.
    If they do have some values the behaviour is unexpected.

    The kernel is similar to the 3D case but with different
    normalizing constants.
    """
    cdef double function(self, Point p1, Point p2, double h):
        """
        """
        cdef Point r = p2-p1
        cdef double l2_norm_r = sqrt(r.norm())
        cdef double q = l2_norm_r/h
        cdef double ret = 0.0
        cdef double factor = 1.0/(h**2.0)
        factor *= 5.0/(14.0*PI)

        if 0.0 <= q <= 1.0:
            ret = (2.0-q)**3.0 - 4.0*(1.0-q)**3.0
            ret *= factor
        elif 1.0 <= q <= 2.0:
            ret = (2.0-q)**3.0
            ret *= factor
        else:
            return 0.0

        return ret

    cdef void gradient(self, Point p1, Point p2, double h, Point grad):
        """
        """
        cdef Point r = p1-p2
        cdef double l2_norm_r = sqrt(r.norm())
        cdef double q = l2_norm_r/h
        cdef double factor = (-5.0)/(14.0*PI*(h**3.0)*l2_norm_r)
        cdef double temp = 0.0
        
        if l2_norm_r < 1e-12:
            grad.x = grad.y = grad.z = 0.0
            return

        if 0.0 <= q <= 1.0:
            temp = 3.0*((2.0-q)**2.0) - 12.0*((1.0-q)**2.0)
        elif 1.0 <= q <= 2.0:
            temp = 3.0*((2.0-q)**2.0)

        grad.x = temp*factor*r.x
        grad.y = temp*factor*r.y
        grad.z = 0.0

    cpdef double radius(self):
        return 2.0

