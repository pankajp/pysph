# cython: profile=True
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
        """ Constructor interface for multidimensional kernels

        Parameters:
        -----------
        dim -- The dimensionality for the kernel. Defaults to 1D
        
        """
        self.dim = dim

    cdef double function(self, Point pa, Point pb, double h):
        """        
        """
        raise NotImplementedError, 'KernelBase::function'

    cdef void gradient(self, Point pa, Point pb, double h, Point result):
        """
        """
        raise NotImplementedError, 'KernelBase::gradient'

    cdef double laplacian(self, Point pa, Point pb, double h):
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

    def py_function(self, Point pa, Point pb, double h):
        return self.function(pa, pb, h)

    def py_gradient(self, Point pa, Point pb, double h, Point grad):
        self.gradient(pa, pb, h, grad)

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
        p2 = Point()
        dim = 0

        if x is not None: setattr(p2, 'x', x); dim += 1
        if y is not None: setattr(p2, 'y', y); dim += 1
        if z is not None: setattr(p2, 'z', z); dim += 1

        msg = 'Point dimension = %d, Kernel dimension = %d'%(dim, self.dim)
        assert dim == self.dim, 'Incompatible dimensions' + msg

        p1 = Point(0,0,0)
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
        self.gradient(Point(), p1, h, grad)

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
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val
        
        if q > 2.0:
            val = 0.0
        elif q > 1.0:
            val = 0.25 * (2 - q) * (2 - q) * (2 - q)
        else:
            val = 1 - 1.5 * (q*q) * (1 - 0.5 * q)

        return val * fac

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val = 0.0
        
        if q > 2.0:
            pass
        elif q >= 1.0:
            val = -0.75 * (2-q) * (2-q)/(h * rab)
        elif q > 1e-14:
            val = 3.0*(0.75*q - 1)/(h*h)

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
    
    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val
        
        if q > 1e-12:
            if q > 2.0:
                val = 0.0
            elif 1.0 <= q <= 2.0:
                val = -3*(q-2)*(q-1) / (h**2*q)
            else:
                val = 9*(q-1)/h**2
        else:
            val = 1.0
        
        return val * fac
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1: return (2./3.)/h
        elif dim == 2: return 10/(7*PI * h*h)
        elif dim == 3: return 1./(PI * h*h*h)
        else: raise ValueError

    cpdef double radius(self):
        return 2
##############################################################################


##############################################################################
#`QuinticSplineKernel`
##############################################################################
cdef class QuinticSplineKernel(MultidimensionalKernel):
    """ The Quintic spline kernel defined in 

    "Modelling Low Reynolds Number Incompressible Flows Using SPH",
    Joseph P. Morris, Patrik J. Fox and Yi Zhu, Journal of Computational
    Physics, 136, 214-226

    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        raise NotImplementedError
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim == 1:
            raise NotImplementedError
        elif dim == 2:
            return 7./(478*PI)
        elif dim == 3:
            raise NotImplementedError
        else:
            raise ValueError

    cpdef double radius(self):
        return 3


##############################################################################
#`WendlandQuinticSplineKernel`
##############################################################################
cdef class WendlandQuinticSplineKernel(MultidimensionalKernel):
    """ The Quintic spline kernel defined in 

    "Modelling Low Reynolds Number Incompressible Flows Using SPH",
    Joseph P. Morris, Patrik J. Fox and Yi Zhu, Journal of Computational
    Physics, 136, 214-226

    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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
                val = -5*q*tmp            

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)

    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        raise NotImplementedError
    
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
            raise ValueError

    cpdef double radius(self):
        return 2

##############################################################################
#`HarmonicKernel`
##############################################################################
cdef class HarmonicKernel(MultidimensionalKernel):
    """ The one parameter family of kernel defined in the paper:
    "A one parameter family of interpolating kernels for smoothed 
    particle hydrodynamics studies. ", R.M. Cabezon et al., JCP (2008).

    """
    def __init__(self, dim = 1, n = 3):
        """ Initialize the kernel with dimension and index `n` """
        self.dim = dim
        self.n = n

        self.facs = {1:[0.424095, 0.553818, 0.660203, 0.752215,
                        0.834354, 0.909205, 0.978402, 1.043052,
                        1.103944, 1.161662],
                     
                     2:[0.196350, 0.322194, 0.450733, 0.580312,
                        0.710379, 0.840710, 0.971197, 1.101785,
                        1.232440, 1.363143],
                     
                     3:[0.098175, 0.196350, 0.317878, 0.458918,
                        0.617013, 0.790450, 0.977949, 1.178511,
                        1.391322, 1.615708]
                     }

    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

                val = n/(h*q*rab) * power * (tmp3 - tmp2/tmp1)
            else:
                val = 0.0
        else:
            val = 0.0

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
    
    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val, tmp
        cdef double n = self.n
        cdef double k = PI*q/2.0
        cdef double sk = sin(k)
        cdef double skk = sk / k
        
        if q > 2.0:
            val = 0.0
        elif q > 1e-12:
            tmp = (skk**(n-2)*n)/(h*q)**2
            val = (cos(k)-skk)**2*(n-1) - sk**2
            val *= tmp
        else:
            val = -PI**2/(4*h**2)
        return val * fac

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        cdef int n = self.n
        
        return self.facs[self.dim][self.n -1]/(h**dim)

    cpdef double radius(self):
        return 2
#############################################################################

##############################################################################
#`M6 Spline Kernel`
##############################################################################

cdef class M6SplineKernel(MultidimensionalKernel):
    """
    
    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)

    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val
        
        if 0 <= q <= 2.0/3:
            val = -(900*q**3-1200*q**2+320)/(3*h**2)

        elif 2.0/3 < q <= 4.0/3:
            val = (10*(405*q**4-1620*q**3+2160*q**2-1008*q+80))/(27*h**2*q)

        elif 4.0/3 < q <=2:
            val = -(10*(q-2)**3*(3*q-2))/(h**2*q)
        
        return val * fac
    
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
cdef class GaussianKernel(MultidimensionalKernel):
    """ Gaussian  Kernel

    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val

        val = fac*exp(-q*q)

        return val

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val

        val = -2*q*exp(-q*q)
        
        grad.x = r.x * (val * fac)
        grad.y = r.y * (val * fac)
        grad.z = r.z * (val * fac)
    
    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double q2 = q*q
        
        return fac * (4*q2-6) * exp(-q2) / h**2
            
    
    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        elif dim == 1: return (1.0/((PI**.5)*h))
        elif dim == 2: return (1.0/(((PI**.5)*h)**2))
        elif dim == 3: return (1.0/(((PI**.5)*h)**3))

    cpdef double radius(self):
        return 2
##############################################################################


##############################################################################
#`W8Kernel`
##############################################################################
cdef class W8Kernel(MultidimensionalKernel):
    """ W8 Kernel as described in the International Journal for Numerical 
    Methods in Engineering: "Truncation error in mesh-free particle methods", 
    N. J. Quinlan, M. Basa and M. Lastiwka
    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.

        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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
    
    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val = 0.0
        cdef double *a = [0.209206, -0.0334338, 0.002]
        cdef int i
        
        if q < 2.0:
            for i in range(3):
                val += a[i] * (i+2)*(i+3) * q**(2*i)
        else:
            return 0.0
        val /= h**2
        
        return val * fac
        

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
cdef class W10Kernel(MultidimensionalKernel):
    """W10 Kernel

    """
    cdef double function(self, Point pa, Point pb, double h):
        """ Evaluate the strength of the kernel centered at `pa` at
        the point `pb`.         
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """Evaluate the gradient of the kernel centered at `pa`, at the
        point `pb`.
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
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

    cdef double laplacian(self, Point pa, Point pb, double h):
        """Evaluate the laplacina of the kernel centered at `pa`, at the
        point `pb`
        """
        cdef double fac = self._fac(h)
        cdef Point r = Point_sub(pa, pb)
        cdef double rab = r.length()
        cdef double q = rab/h
        cdef double val = 0.0
        cdef int i
        cdef double *a = [0.422974, -0.105743, 0.0132179, -0.000660896]
        
        if q < 2.0:
            for i in range(4):
                val += a[i] * (i+2)*(i+3) * q**(2*i)
        else:
            return 0.0
        val /= h**2
        
        return val * fac
    
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
cdef class RepulsiveBoundaryKernel(MultidimensionalKernel):
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
    cdef double function(self, Point pa, Point pb, double h):
        """
        """
        cdef Point dir = Point_sub(pa, pb)
        cdef double dist = sqrt(dir.norm())
        cdef double q = dist/h
        cdef double temp = 0.0

        dir.x = dir.x/dist
        dir.y = dir.y/dist
        dir.z = dir.z/dist

        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        return temp

    cdef void gradient(self, Point pa, Point pb, double h, Point grad):
        """
        """
        cdef Point dir = Point_sub(pa, pb)
        cdef double dist = sqrt(dir.norm())
        cdef double q = dist/h
        cdef double temp = 0.0

        dir.x = dir.x/dist
        dir.y = dir.y/dist
        dir.z = dir.z/dist
        
        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        temp *= 0.02
        temp /= dist

        grad.x = (temp*dir.x)
        grad.y = (temp*dir.y)
        grad.z = (temp*dir.z)

    cdef double laplacian(self, Point pa, Point pb, double h):
        """
        """
        raise NotImplementedError
    
    cpdef double radius(self):
        return 2.0

    cdef double _fac(self, double h):
        """ Return the normalizing factor given the smoothing length. """
        cdef int dim = self.dim
        if dim > 3:
            raise ValueError
        else: return (1.0/h)



