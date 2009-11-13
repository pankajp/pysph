"""
Various 3d kernels.
"""

cimport numpy
import numpy

from pysph.base.point cimport Point
from pysph.base.kernelbase cimport Kernel3D

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
# `Poly6Kernel3D` class.
################################################################################
cdef class Poly6Kernel3D(Kernel3D):
    """
    This class represents a polynomial kernel with support 1.0
    from mueller et. al 2003
    """
    cdef double function(self, Point p1, Point p2, double h):
        """
        """
        cdef Point r = p2-p1
        cdef double mag_sqr_r   = r.norm()
        cdef double ret = 0.0
        if sqrt(mag_sqr_r) > h:
            ret = 0.0
        else:
            ret = (h**2.0 - mag_sqr_r)**3.0
            ret *= (315.0)/(64.0*PI*(h**9.0))
        return ret

    cdef void gradient(self, Point p1, Point p2, double h, Point grad):
        """
        """
        cdef Point r = p1-p2
        cdef double part = 0.0
        cdef double mag_square_r = r.norm()
        cdef double const1 = 315.0/(64.0*PI*(h**9))
        part = -6.0*const1*((h**2 - mag_square_r)**2)
        grad.x = r.x * part
        grad.y = r.y * part
        grad.z = r.z * part

    cdef double laplacian(self, Point p1, Point p2, double h):
        """
        """
        cdef Point r = p2-p1
        cdef double mag_square_r = r.norm()
        cdef double h_sqr = h*h
        cdef double const1 = 315.0/(64.0*PI*(h**9))
        cdef double ret = (-6.0)*const1*(h_sqr-mag_square_r)
        ret = ret * (3.0*h_sqr - 7.0*mag_square_r)
        return ret
            
    cpdef double radius(self):
        """
        """
        return 1.0

################################################################################
# `CubicSpline3D` class.
################################################################################
cdef class CubicSpline3D(Kernel3D):
    """
    Cubic Spline kernel from Monaghan05.
    ** References **
    1. Smoothed Particle Hydrodynamics, Rep. Prog. Phys. 68 (2005), 1703-1759
    """
    cdef double function(self, Point p1, Point p2, double h):
        """
        """
        cdef Point r = p2-p1
        cdef double l2_norm_r = sqrt(r.norm())
        cdef double q = l2_norm_r/h
        cdef double ret = 0.0
        cdef double factor = 1.0/(h**3.0)
        factor /= 4.0*PI

        if 0.0 <= q <= 1.0:
            ret  =(2.0-q)**3.0  - 4.0*(1.0-q)**3.0
            ret *= factor
        elif 1.0 <= q <=2.0:
            ret  = (2.0-q)**3
            ret  *= factor
        else:
            return 0.0

        return ret
    
    cdef void gradient(self, Point p1, Point p2, double h, Point grad):
        """
        """
        cdef Point r = p1-p2
        cdef double l2_norm_r = sqrt(r.norm())
        cdef double q = l2_norm_r/h
        cdef double factor = (-1.0)/(4.0*PI*(h**4.0)*l2_norm_r)
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
        grad.z = temp*factor*r.z

    cdef double laplacian(self, Point p1, Point p2, double h):
        """
        """
        raise NotImplementedError('laplacian for the CubicSpline3D kernel not implemented')

    cpdef double radius(self):
        return 2.0
