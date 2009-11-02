"""
Various SPH 1-d kernels.
"""

# local imports
from pysph.base.point cimport Point
from pysph.base.kernelbase cimport Kernel1D

from math import pi

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double fabs(double)

cdef:
    double PI = pi
    double SQRT_1_PI = 1.0/sqrt(PI)

cdef inline double sign(double x):
    if x > 0.0:
        return 1.0
    else:
        return -1.0

################################################################################
# `Lucy1D` class.
################################################################################ 
cdef class Lucy1D(Kernel1D):
    """
    This class represents Lucy's (1977) kernel.
    """
    cdef double function(self, Point p1, Point p2, double h):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double q = fabs(p2.x-p1.x)/h
        cdef double tmp = 1.0 - q
        if (q > 1.0):
            return 0.0
        else:
            return (1.25*(1.0 + 3.0*q)*tmp*tmp*tmp)/h

    cdef void gradient(self, Point p1, Point p2, double h, Point result):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double q = (p2.x-p1.x)/h
        cdef double qm = fabs(q)
        cdef double tmp = 1.0 - qm
        if (qm > 1.0):
            result.x = 0.0
        else:
            result.x = 15.0/(h*h)*q*tmp*tmp

    cpdef double radius(self):
        """Return the radius of influence (k) of a given kernel, i.e. h*k is
        the region of influence of the kernel.
        """
        return 1.0

################################################################################
# `Gaussian1D` class.
################################################################################ 
cdef class Gaussian1D(Kernel1D):
    """
    This class represents a 1D Gaussian kernel.
    """
    cdef double function(self, Point p1, Point p2, double h):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double q = fabs(p2.x-p1.x)/h
        if (q > 3.0):
            return 0.0
        else:
            return SQRT_1_PI*exp(-q*q)/h

    cdef void gradient(self, Point p1, Point p2, double h, Point result):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double q = (p2.x-p1.x)/h
        if (fabs(q) > 3.0):
            result.x =  0.0
        else:
            result.x = 2.0*SQRT_1_PI*q*exp(-q*q)/(h*h)

    cpdef double radius(self):
        """Return the radius of influence (k) of a given kernel, i.e. h*k is
        the region of influence of the kernel.
        """
        return 3.0

################################################################################
# `Cubic1D` class.
################################################################################ 
cdef class Cubic1D(Kernel1D):

    """Cubic B-spline kernel in 1D."""

    cdef double function(self, Point p1, Point p2, double h):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double q = fabs(p2.x-p1.x)/h
        cdef double tmp = 2 - q
        if (q > 2.0):
            return 0.0
        elif (q > 1.0):
            return tmp*tmp*tmp/(6.0*h)
        else:
            return (2.0/3.0 - q*q*(1.0 - 0.5*q))/h

    cdef void gradient(self, Point p1, Point p2, double h, Point result):
        """The function evaluation of the kernel.  Note that we assume
        that the first argument `p1` is the location where the kernel is
        being evaluated centered at `p2`.
        """
        cdef double qm = fabs((p1.x-p2.x)/h)
        cdef double sgn = sign(p1.x-p2.x)
        cdef double tmp = 2.0 - qm
        if (qm > 2.0):
            result.x =  0.0
        elif (qm > 1.0):
            result.x =  -0.5*sgn*tmp*tmp/(h*h)
        else:
            result.x =  (1.5*qm - 2.0)*qm*sgn/(h*h)

    cpdef double radius(self):
        """Return the radius of influence (k) of a given kernel, i.e. h*k is
        the region of influence of the kernel.
        """
        return 2.0
