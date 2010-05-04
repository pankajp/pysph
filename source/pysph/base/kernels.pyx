""" 
Module to implement various SPH kernels in multiple dimensions 
"""
#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Kunal Puri

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double fabs(double)

cimport numpy 
import numpy
cdef:
    double PI = numpy.pi
    double SQRT_1_PI = 1.0/sqrt(PI)
    double infty = numpy.inf

##############################################################################
#`MultidimensionalKernel`
##############################################################################
cdef class MultidimensionalKernel:
    """ A base class that handles multiple dimensions. """

    #Defined in the .pxd file
    #cdef public int dim

    def __init__(self, dim = 1):
        """ Constructor for the Cubic Spline Kernel

        Parameters:
        -----------
        dim -- The dimensionality for the kernel. Defaults to 1D
        
        """
        self.dim = dim

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
        return -1

    cdef double _fac(self, double h):
        raise NotImplementedError, 'KernelBase::_fac'

    def py_function(self, Point p1, Point p2, double h):
        return self.function(p1, p2, h)

    ##########################################################################
    # Functions used for testing.
    ##########################################################################
    def _function(self, x=None, y=None, z=None):
        """ Wrapper for the function evaluation amenable to a
        call by scipy's quad for testing.
        
        Parameters:
        -----------
        kwargs -- (x, y, z): Point `p1` of evaluation for the kernel.

        """
        h = 0.01
        p1 = Point()
        dim = 0

        if x is not None: setattr(p1, 'x', x); dim += 1
        if y is not None: setattr(p1, 'y', y); dim += 1
        if z is not None: setattr(p1, 'z', z); dim += 1

        msg = 'Point dimension = %d, Kernel dimension = %d'%(dim, self.dim)
        assert dim == self.dim, 'Incompatible dimensions' + msg

        p2 = Point(0,0,0)
        return self.function(p1, p2, h)

    def _gradient(self, x=None, y=None, z=None, res=None):
        """ Wrapper for the gradient evaluation amenable to a call
        by scipy's quad for testing. 

        Parameters:
        -----------
        x -- The `x` location for the point of evaluation.
        y -- The `y` location for the point of evaluation.
        z -- The `z` location for the point of evaluation.
        res -- Point type for the result.

        """
        h = 0.01
        p1 = Point()
        dim = 0

        assert res is not None, 'Nowhere to add the solution!'
        if x is not None: setattr(p1, 'x', x); dim += 1
        if y is not None: setattr(p1, 'y', y); dim += 1
        if z is not None: setattr(p1, 'z', z); dim += 1

        msg = 'Point dimension = %d, Kernel dimension = %d'%(dim, self.dim)
        assert dim == self.dim, 'Incompatible dimensions' + msg
        
        grad = Point()
        self.gradient(p1, Point(), h, grad)

        res = res + grad
        return res.x, res.y, res.z
    
    def _gradient1D(self, x, grad):
        vec = self._gradient(x, None, None, grad)
        return vec[0] 

    def _gradient2D(self, x, y, grad, i = 0):
        vec = self._gradient(x, y, None, grad)
        return vec[i]
    
    def _gradient3D(self, x, y, z, grad, i = 0):
        vec = self._gradient(x, y, z, grad)
        return vec[i]
        
    def py_fac(self, h):
        return self._fac(h)
##############################################################################


##############################################################################
#`CubicSplineKernel`
##############################################################################
cdef class CubicSplineKernel(MultidimensionalKernel):
    """ Cubic Spline Kernel as described in the paper: "Smoothed
    Particle Hydrodynamics ", J.J. Monaghan, Annual Review of
    Astronomy and Astrophysics, 1992, Vol 30, pp 543-574.

    """
    cdef double function(self, Point p1, Point p2, double h):
        """ Evaluate the strength of the kernel centered at `p2` at
        the point `p1`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = p1 - p2
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val
        
        if q > 2.0:
            val = 0.0
        elif 1.0 < q <= 2.0:
            val = 0.25 * (2 - q) * (2 - q) * (2 - q)
        else:
            val = 1 - 1.5 * (q*q) + 0.75 * (q*q*q)

        return val * fac

    cdef void gradient(self, Point p1, Point p2, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `p2`, at the
        point `p1`.

        """
        cdef double fac = self._fac(h)
        cdef Point r = p1 - p2
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val
        
        if q > 1e-12:
            if q > 2.0:
                val = 0.0

            elif 1.0 <= q <= 2.0:
                val = -0.75 * (2-q) * (2-q)/(h * rab)

            else:
                val = 3.0/(h*h)*(0.75*q - 1)
        else:
            val = 0.0

        grad.x = -r.x * (val * fac)
        grad.y = -r.y * (val * fac)
        grad.z = -r.z * (val * fac)

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        elif dim == 1: return (2./3./h)
        elif dim == 2: return 10/(7*PI)/(h*h)
        elif dim == 3: return 1./(PI)/(h*h*h)

    cpdef double radius(self):
        return 2
##############################################################################
