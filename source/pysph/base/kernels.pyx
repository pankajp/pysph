#cython: cdivision=True
""" 
Module to implement various SPH kernels in multiple dimensions 
"""
#Author: Kunal Puri <kunalp@aero.iitb.ac.in>
#Copyright (c) 2010, Prabhu Ramachandran

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double fabs(double)
    double sin(double)
    double cos(double)
    double pow(double x, double y)
    int floor(double)
    double fmod(double, double)

cimport numpy 
import numpy

from pysph.base.point cimport cPoint_length, cPoint_distance, cPoint_distance2

cdef:
    double PI = numpy.pi
    double SQRT_1_PI = 1.0/sqrt(PI)
    double infty = numpy.inf

cdef inline double h_dim(double h, int dim):
    if dim == 1:
        return 1/h
    elif dim == 2:
        return 1/(h*h)
    else:
        return 1/(h*h*h)

##############################################################################
#`KernelBase`
##############################################################################
cdef class KernelBase:
    """ A base class that handles kernels in multiple dimensions. """

    #Defined in the .pxd file
    #cdef readonly int dim

    def __init__(self, int dim=1, double constant_h=-1):
        """ Constructor interface for multidimensional kernels

        Parameters:
        -----------
        dim -- The dimensionality for the kernel. Defaults to 1D
        
        """
        self.dim = dim
        self.fac = self._fac(1.0)

        self.smoothing = DoubleArray()

        self.distances = DoubleArray()
        self.function_cache = DoubleArray()
        self.gradient_cache = DoubleArray()

        n = 1000001

        self.distances_dx = -1

        self.has_constant_h = False
        self.constant_h = constant_h
        if constant_h > 0:
            self.init_cache(n)

    cdef double function(self, cPoint pa, cPoint pb, double h):
        """        
        """
        raise NotImplementedError, 'KernelBase::function'

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """
        """
        raise NotImplementedError, 'KernelBase::gradient'

    cpdef double radius(self):
        """
        """
        raise NotImplementedError, 'KernelBase::radius'

    cpdef int dimension(self):
        """
        """
        return self.dim

    cdef double _fac(self, double h):
        raise NotImplementedError, 'KernelBase::_fac'

    def py_function(self, Point pa, Point pb, double h):
        return self.function(pa.data, pb.data, h)

    def py_gradient(self, Point pa, Point pb, double h, Point grad):
        grad.set_from_cPoint(self.gradient(pa.data, pb.data, h))

    cpdef double __gradient(self, Point pa, Point pb, double h):
        raise NotImplementedError, 'KernelBase::__gradient'

    def init_cache(self, n=10001):

        cdef int i
        cdef double w, grad
        cdef double h = self.constant_h
        cdef numpy.ndarray h_arr = numpy.linspace(0,2*h,n)

        self.distances.resize(n)
        self.distances.set_data(h_arr)

        self.distances_dx = h_arr[1] - h_arr[0]
        
        self.function_cache.resize(n)
        self.gradient_cache.resize(n)

        cdef numpy.ndarray[ndim=1,dtype=numpy.float] fc = numpy.zeros(n, dtype=float)
        cdef numpy.ndarray[ndim=1,dtype=numpy.float] gc = numpy.zeros(n, dtype=float)

        cdef cPoint pa = cPoint(0,0,0)
        cdef cPoint pb = cPoint(0,0,0)

        for i in range(n):
            pb.x = self.distances[i]
            w = self.py_function(pa, pb, h)
            grad = self.__gradient(pa, pb, h)

            fc[i] = w
            gc[i] = grad

        self.function_cache.set_data(fc)
        self.gradient_cache.set_data(gc)

        self.has_constant_h = True

    cdef interpolate_function(self, double rab):

        cdef double dx = self.distances_dx
        cdef int index_low, index_high

        cdef double* fc = self.function_cache.get_data_ptr()

        if rab > 2*self.constant_h:
            return 0.0

        else:
            index_low = floor(rab/dx)
            index_high = index_low + 1

            slope = (fc[index_high] - fc[index_low])/dx

            # y = mx + c

            return slope * fmod(rab,dx) + fc[index_low]
            
            #print "Numpy interpolation ", numpy.interp(rab, self.distances,
            #self.function_cache)
            
            #print "This interpolation ", slope*fmod(rab,dx) + fc[index_low]

    cdef interpolate_gradients(self, double rab):

        cdef double dx = self.distances_dx
        cdef int index_low, index_high

        cdef double* gc = self.gradient_cache.get_data_ptr()

        if rab > 2 * self.constant_h:
            return 0.0
        else:
            index_low = floor(rab/dx)
            index_high = index_low + 1

            slope = (gc[index_high] - gc[index_low])/dx

            # y = mx + c

            return slope * fmod(rab,dx) + gc[index_low]

            #print "Numpy interpolation ", numpy.interp(rab, self.distances,
            #                                           self.gradient_cache)
            #print "This interpolation ", slope*fmod(rab,dx) + gc[index_low]

    ##########################################################################
    # Functions used for testing.
    ##########################################################################
    def _function(self, x=None, y=None, z=None):
        """ Wrapper for the function evaluation amenable to a
        call by scipy's quad for testing.
        
        Parameters:
        -----------
        kwargs -- (x, y, z): Point `p2` of evaluation for the kernel.

        """
        h = 0.01
        cdef cPoint p2 = cPoint(0,0,0)
        dim = 0

        if x is not None: p2.x = x; dim += 1
        if y is not None: p2.y = y; dim += 1
        if z is not None: p2.z = z; dim += 1

        msg = 'Point dimension = %d, Kernel dimension = %d'%(dim, self.dim)
        assert dim == self.dim, 'Incompatible dimensions' + msg

        return self.function(cPoint(0,0,0), p2, h)

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
        cdef cPoint p1 = cPoint(0,0,0)
        dim = 0

        assert res is not None, 'Nowhere to add the solution!'
        if x is not None: p1.x = x; dim += 1
        if y is not None: p1.y = y; dim += 1
        if z is not None: p1.z = z; dim += 1

        msg = 'Point dimension = %d, Kernel dimension = %d'%(dim, self.dim)
        assert dim == self.dim, 'Incompatible dimensions' + msg
        
        cdef cPoint grad = self.gradient(cPoint_new(0,0,0), p1, h)

        res = res + grad
        return res.x, res.y, res.z
    
    def _gradient1D(self, x, grad):
        vec = self._gradient(x, None, None, grad)
        return vec[0]

    def _gradient2D(self, x, y, grad, i=0):
        vec = self._gradient(x, y, None, grad)
        return vec[i]
    
    def _gradient3D(self, x, y, z, grad, i=0):
        vec = self._gradient(x, y, z, grad)
        return vec[i]
        
    def py_fac(self, h):
        return self._fac(h)

##############################################################################
# `Poly6Kernel` class.
##############################################################################
cdef class Poly6Kernel(KernelBase):
    """
    This class represents a polynomial kernel with support 1.0
    from mueller et. al 2003
    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        """
        cdef double mag_sqr_r = cPoint_distance2(pa, pb)
        cdef double ret = 0.0
        if mag_sqr_r > h*h:
            return 0.0
        ret = (h*h - mag_sqr_r)**3
        ret /= h**6
        return ret * h_dim(h, self.dim) * self.fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef cPoint grad
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double part = 0.0
        cdef double mag_square_r = cPoint_norm(r)
        cdef double fac
        if mag_square_r <= h*h:
            fac = h_dim(h, self.dim) * self.fac
            part = -6.0*fac*((h**2 - mag_square_r)**2)/h**6
        grad.x = r.x * part
        grad.y = r.y * part
        grad.z = r.z * part
        return grad

    cdef double _fac(self, double h):
        if self.dim == 3:
            return 315.0/(64.0*PI*h*h*h)
        else:
            raise NotImplementedError, 'Poly6Kernel in %d dim'%self.dim
    
    cpdef double radius(self):
        return 1.0

##############################################################################
# DummyKernel` class.
##############################################################################
cdef class DummyKernel(KernelBase):
    cdef double function(self, cPoint pa, cPoint pb, double h):
        return 1.0

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        cdef cPoint grad
        return grad

    cdef double _fac(self, double h):
        return 1.0
    
    cpdef double radius(self):
        return 1.0
    
##############################################################################
#`CubicSplineKernel`
##############################################################################
cdef class CubicSplineKernel(KernelBase):
    """ Cubic Spline Kernel as described in the paper: "Smoothed
    Particle Hydrodynamics ", J.J. Monaghan, Annual Review of
    Astronomy and Astrophysics, 1992, Vol 30, pp 543-574.

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        
        """
        cdef double rab = sqrt((pa.x-pb.x)*(pa.x-pb.x)+
                              (pa.y-pb.y)*(pa.y-pb.y) + 
                              (pa.z-pb.z)*(pa.z-pb.z))

        if self.has_constant_h:
            return self.interpolate_function(rab)

        cdef double q = rab/h
        cdef double val
        cdef double fac = h_dim(h, self.dim) * self.fac
        
        if q > 2.0:
            val = 0.0
        elif q > 1.0:
            val = 0.25 * (2 - q) * (2 - q) * (2 - q)
        else:
            val = 1 - 1.5 * (q*q) * (1 - 0.5 * q)

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef cPoint grad
        cdef double rab = sqrt((pa.x-pb.x)*(pa.x-pb.x)+
                               (pa.y-pb.y)*(pa.y-pb.y) + 
                               (pa.z-pb.z)*(pa.z-pb.z))

        cdef double rx, ry, rz
        cdef double q = rab/h
        cdef double val = 0.0
        cdef double wgrad
        
        fac = h_dim(h, self.dim)*self.fac
        
        rx = pa.x - pb.x; ry = pa.y - pb.y; rz = pa.z - pb.z

        if self.has_constant_h:
            wgrad = self.interpolate_gradients(rab)

            grad.x = wgrad*rx
            grad.y = wgrad*ry
            grad.z = wgrad*rz
            
        else:
        
            if q > 2.0:
                pass
            elif q >= 1.0:
                val = -0.75 * (2-q) * (2-q)/(h * rab)
            elif q > 1e-14:
                val = 3.0*(0.75*q - 1)/(h*h)

            grad.x = rx * (val * fac)
            grad.y = ry * (val * fac)
            grad.z = rz * (val * fac)
        return grad

    cpdef double __gradient(self, Point pa, Point pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef double rab = sqrt((pa.x-pb.x)*(pa.x-pb.x)+
                              (pa.y-pb.y)*(pa.y-pb.y) + 
                              (pa.z-pb.z)*(pa.z-pb.z))

        cdef double q = rab/h
        cdef double val = 0.0
        
        fac = h_dim(h, self.dim)*self.fac
        
        if q > 2.0:
            pass
        elif q >= 1.0:
            val = -0.75 * (2-q) * (2-q)/(h * rab)
        elif q > 1e-14:
            val = 3.0*(0.75*q - 1)/(h*h)

        return val * fac

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:  return  (2./3.)/h
        elif dim == 2: return 10/(7*PI)/(h*h)
        elif dim == 3: return 1./PI/(h*h*h)
        else: raise ValueError

    cpdef double radius(self):
        return 2
##############################################################################


##############################################################################
#`QuinticSplineKernel`
##############################################################################
cdef class QuinticSplineKernel(KernelBase):
    """ The Quintic spline kernel defined in 

    "Modelling Low Reynolds Number Incompressible Flows Using SPH",
    Joseph P. Morris, Patrik J. Fox and Yi Zhu, Journal of Computational
    Physics, 136, 214-226

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val

        cdef double tmp1 = (3-q)
        cdef double tmp2 = (2-q)
        cdef double tmp3 = (1-q)
      
        if q > 3.0:
            val = 0.0
        else:
            tmp1 *= (tmp1 * tmp1 * tmp1 * tmp1)
            
            if q > 2.0:
                val = tmp1
            else:
                tmp2 *= (tmp2 * tmp2 * tmp2 * tmp2)
                val = tmp1 - 6 * tmp2
                if q > 1.0:
                    pass    
                else:
                    tmp3 *= (tmp3 * tmp3 * tmp3 * tmp3)
                    val += tmp3

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef cPoint grad
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val
        cdef double power

        cdef double tmp1 = (3-q)
        cdef double tmp2 = (2-q)
        cdef double tmp3 = (1-q)

        if rab < 1e-16:
            val = 0.0
        else:
            fac *= 1./(h*rab)
            if q > 3.0:
                val = 0.0
            else:
                tmp1 *= (tmp1 * tmp1 * tmp1)
            
                if q > 2.0:
                    val = 5*tmp1
                else:
                    tmp2 *= (tmp2 * tmp2 * tmp2)
                    val = 5*tmp1 - 30 * tmp2
                    if q > 1.0:
                        pass    
                    else:
                        tmp3 *= (tmp3 * tmp3 * tmp3)
                        val += 75*tmp3

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:
            raise NotImplementedError
        elif dim == 2:
            return 7./(478*PI*h*h)
        elif dim == 3:
            raise NotImplementedError
        else:
            raise ValueError

    cpdef double radius(self):
        return 3


##############################################################################
#`WendlandQuinticSplineKernel`
##############################################################################
cdef class WendlandQuinticSplineKernel(KernelBase):
    """ The Quintic spline kernel defined in 

    "Modelling Low Reynolds Number Incompressible Flows Using SPH",
    Joseph P. Morris, Patrik J. Fox and Yi Zhu, Journal of Computational
    Physics, 136, 214-226

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val

        cdef double tmp = (1-0.5*q)
      
        if q > 2.0:
            val = 0.0
        else:
            tmp *= (tmp * tmp * tmp)
            tmp *= (2*q + 1)
            
            val = tmp

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef cPoint grad
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val
        cdef double power

        cdef double tmp = (1-0.5*q)

        if rab < 1e-16:
            val = 0.0
        else:
            fac *= 1./(h*rab)

            if q > 2.0:
                val = 0
            else:
                tmp *= (tmp * tmp)
                val = tmp * -5*q

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:
            raise NotImplementedError
        elif dim == 2:
            return (7./4*PI*h*h)
        elif dim == 3:
            raise NotImplementedError
        else:
            return (21.0/16.0*PI*h*h)
            raise ValueError

    cpdef double radius(self):
        return 2

##############################################################################
#`HarmonicKernel`
##############################################################################
cdef class HarmonicKernel(KernelBase):
    """ The one parameter family of kernel defined in the paper:
    "A one parameter family of interpolating kernels for smoothed 
    particle hydrodynamics studies. ", R.M. Cabezon et al., JCP (2008).

    """
    def __init__(self, dim=1, n=3):
        """ Initialize the kernel with dimension and index `n` """
        self.dim = dim
        self.n = n
        if dim == 1:
            self.facs[:] = [0.424095, 0.553818, 0.660203, 0.752215,
                        0.834354, 0.909205, 0.978402, 1.043052,
                        1.103944, 1.161662]
        elif dim == 2:
            self.facs[:] = [0.196350, 0.322194, 0.450733, 0.580312,
                        0.710379, 0.840710, 0.971197, 1.101785,
                        1.232440, 1.363143]
        elif dim == 3:
            self.facs[:] = [0.098175, 0.196350, 0.317878, 0.458918,
                        0.617013, 0.790450, 0.977949, 1.178511,
                        1.391322, 1.615708]

    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        
        """
        cdef double fac = self.facs[self.n -1]*(h**(-self.dim))
        #cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = sqrt((pa.x - pb.x)**2 + (pa.y - pb.y)**2 + (pa.z - pb.z)**2)
        cdef double q = rab/h
        cdef double val, tmp
        cdef double n = self.n
        
        if q > 2.0:
            val = 0.0
        elif 0 < q <= 2.0:
            tmp = 0.5*PI*q
            val = sin(tmp)/tmp
            val = val ** n
        else:
            val = 1

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef cPoint grad
        cdef double fac = self.facs[self.n -1]*(h**(-self.dim))
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val, tmp1, tmp2, tmp3, tmp4, power
        cdef double n = self.n

        if q > 1e-12:

            if 0.0 < q <= 2.0:
                tmp1 = 0.5*q*PI
                tmp2 = sin(tmp1)
                tmp3 = cos(tmp1)
                tmp4 = tmp2/tmp1

                if n == 1:
                    power = 1.0
                else:
                    power = tmp4 ** (n-1)

                val = n/(h*q*rab) * power * (tmp3 - tmp4)
            else:
                val = 0.0
        else:
            val = 0.0

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        return self.facs[self.n -1]/(h**self.dim)

    cpdef double radius(self):
        return 2
#############################################################################

##############################################################################
#`M6 Spline Kernel`
##############################################################################
cdef class M6SplineKernel(KernelBase):
    """Quintic M6 polynomial spline kernel with a support radius of 2.
    Cabezón, Rubén M., Domingo García-Senz, and Antonio Relaño.
    “A one-parameter family of interpolating kernels for smoothed particle
    hydrodynamics studies.” Journal of Computational Physics 227, no. 19
    (October 1, 2008): 8523-8540
    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val
        
        if 0 <= q <=2.0/3:
            val = (((2-q)**5)-6*(((4.0/3)-q)**5)+15*(((2.0/3)-q)**5))

        elif 2.0/3 < q <= 4.0/3:
            val = (((2-q)**5)-6*(((4.0/3)-q)**5))

        elif 4.0/3 < q <=2.0:
            val = ((2-q)**5)

        else:
            val = 0

        return val*fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val
        
        if 0 < q <= 2.0/3:
            val = -(5.0*((2.0-q)**4)-30*(((4.0/3)-q)**4)+75.0*(((2.0/3)-q)**4))

        elif 2.0/3 < q <= 4.0/3:
            val = -(5.0*((2.0-q)**4)-30.0*(((4.0/3)-q)**4))

        elif 4.0/3 < q <=2:
            val = -(5.0*(2.0-q)**4)

        else:
            val = 0

        return cPoint_scale(r, val * fac)

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        elif dim == 1: return (243.0/(2560.0*h))
        elif dim == 2: return (15309.0/(61184*PI*h*h))
        elif dim == 3: return (2187.0/(10240*PI*h*h*h))

    cpdef double radius(self):
        return 2
#############################################################################

##############################################################################
#`GaussianKernel`
##############################################################################
cdef class GaussianKernel(KernelBase):
    """ Gaussian  Kernel

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val

        val = fac*exp(-q*q)

        return val

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef cPoint grad
        cdef double rab = sqrt((pa.x-pb.x)*(pa.x-pb.x)+
                              (pa.y-pb.y)*(pa.y-pb.y) + 
                              (pa.z-pb.z)*(pa.z-pb.z))

        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double q = rab/h
        cdef double val = 0.0

        if q > 1e-14:
            val = -2*q*exp(-q*q)/(rab*h)
        
        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        elif dim == 1: return (1.0/((PI**.5)*h))
        elif dim == 2: return (1.0/(((PI**.5)*h)**2))
        elif dim == 3: return (1.0/(((PI**.5)*h)**3))

    cpdef double radius(self):
        return 3.0
##############################################################################


##############################################################################
#`W8Kernel`
##############################################################################
cdef class W8Kernel(KernelBase):
    """ W8 Kernel as described in the International Journal for Numerical 
    Methods in Engineering: "Truncation error in mesh-free particle methods", 
    N. J. Quinlan, M. Basa and M. Lastiwka
    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.
        
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val
        cdef double *a = [0.603764, -0.580823, 0.209206, -0.0334338, 0.002]
        cdef int i

        if q > 2.0:
            val = 0.0

        elif q == 0.0:
            val = a[0]

        else:
            for i in range(5):
                val += a[i]*(q**(2*i))

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef cPoint grad
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val
        cdef double *a = [-0.580823, 0.209206, -0.0334338, 0.002]
        cdef int i
        
        if q > 2.0 or q == 0.0:
            val = 0.0

        else:
            for i in range(4):
                val += 2*(i)*a[i]*(q)**(2*(i)-1)
     

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:
            return (1.0/h)
        elif dim == 2:
            return 0.6348800317791645948695695182/(h*h)
        elif dim == 3:
            return 0.42193070061027009917782110312/(h**3)
        else:
            raise ValueError

    cpdef double radius(self):
        return 2

##############################################################################


##############################################################################
#`W10Kernel`
##############################################################################
cdef class W10Kernel(KernelBase):
    """W10 Kernel

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef double rab = cPoint_distance(pa, pb)
        cdef double q = rab/h
        cdef double val, power
        cdef int i
        cdef double * coeffs=[0.676758,-0.845947,0.422974,
                          -0.105743,0.0132179,-0.000660896]
        if q > 2.0:
            val = 0.0
        else:
            val=0
            power=0
            for i in range(6):
                val+=coeffs[i]*(q**(power))
                power+=2

        return val * fac

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef cPoint grad
        cdef double fac = self.fac * h_dim(h, self.dim)
        cdef cPoint r = cPoint_sub(pa, pb)
        cdef double rab = cPoint_length(r)
        cdef double q = rab/h
        cdef double val
        cdef double power
        cdef int i
        cdef double * coeffs=[0.676758,-0.845947,0.422974,
                          -0.105743,0.0132179,-0.000660896]
        if q > 2.0:
            val = 0.0
        else:
            val=0
            power=2
            for i in range(1,6):
                val+=coeffs[i]*power*(q**(power-1))
                power+=2.0

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
        return grad

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:
            return (1.0/h)
        elif dim == 2:
            return 0.70546843480700509585727594599/(h*h)
        elif dim == 3:
            return 0.51716118669363224608159742165/(h**3)
        else:
            raise ValueError

    cpdef double radius(self):
        return 2


################################################################################
# `RepulsiveBoundaryKernel` class.
################################################################################
cdef class RepulsiveBoundaryKernel(KernelBase):
    """
    Kernel class to be used for boundary SPH summations.

    This is a kernel used for boundary force
    calculation in the Becker07.

    NOTES:
    ------
    -- The function and gradient computations are very 
       similar. We __do not__ expect the function to be 
       used anywhere. Only the gradient should be used.
       The function is implemented for debugging purposes,
       the gradient is kept for performance.
    -- __DO NOT__ put this kernel in the kernel_list in the 
       test_kernels3d module. It will fail, as it is not
       meant to be a generic SPH kernel.
    -- __DO NOT__ use this kernel for anything else, you
       may not get expected results.

    References:
    -----------
    1. [becker07] Weakly Compressible SPH for free surface flows.

    """
    cdef double function(self, cPoint pa, cPoint pb, double h):
        """
        """
        cdef double dist = cPoint_distance(pa, pb)
        cdef double q = dist/h
        cdef double temp = 0.0

        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        return temp

    cdef cPoint gradient(self, cPoint pa, cPoint pb, double h):
        """
        """
        cdef cPoint grad
        cdef double dist = cPoint_distance(pa, pb)
        cdef double q = dist/h
        cdef double temp = 0.0
        
        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        temp *= 0.02
        temp /= dist*dist

        grad.x = temp*(pa.x-pb.x)
        grad.y = temp*(pa.y-pb.y)
        grad.z = temp*(pa.z-pb.z)
        return grad

    cpdef double radius(self):
        return 2.0

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        else: return (1.0/h)



