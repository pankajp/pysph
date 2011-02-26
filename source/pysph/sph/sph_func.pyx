#include for malloc
from libc.stdlib cimport *

cimport numpy

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle:
    """
    Base class to represent an interaction between two particles from two
    possibly different particle arrays. 

    This class requires access to particle properties of possibly two different
    entities. Since there is no particle class having all properties at one
    place, names used for various properties and arrays corresponding to those
    properties are stored in this class for fast access to property values both
    at the source and destination.
    
    This class contains names, and arrays of common properties that will be
    needed for any particle-particle interaction computation. The data within
    these arrays, can be used as *array.data[pid]*, where pid in the particle
    index, "data" is the actual c-pointer to the data.

    All source arrays are prefixed with a "s_". All destination arrays are
    prefixed by a "d_". For example the mass property of the source will be in
    the s_m array.

    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, hks = False, *args, **kwargs):
        """
        Constructor.
        """
        self.name = ""
        self.id = ""
        self.tag = ""
        self.source = source
        self.dest = dest

        #Default properties
        self.x = 'x'
        self.y = 'y'
        self.z = 'z'
        self.u = 'u'
        self.v = 'v'
        self.w = 'w'
        self.m = 'm'
        self.rho = 'rho'
        self.h = 'h'
        self.p = 'p'
        self.e = 'e'
        self.cs = 'cs'

        self.s_x = None
        self.s_y = None
        self.s_z = None
        self.s_u = None
        self.s_v = None
        self.s_w = None
        self.s_m = None
        self.s_rho = None
        self.s_h = None
        self.s_p = None
        self.s_e = None

        self.d_x = None
        self.d_y = None
        self.d_z = None
        self.d_u = None
        self.d_v = None
        self.d_w = None
        self.d_m = None
        self.d_rho = None
        self.d_h = None
        self.d_p = None
        self.d_e = None

        #kernel correction of Bonnet and Lok

        self.bonnet_and_lok_correction = False

        #flag for the rkpm first order kernel correction

        self.rkpm_first_order_correction = False

        # kernel function and gradient evaulation

        self.kernel_function_evaluation = {}
        self.kernel_gradient_evaluation = {}

        # type of kernel symmetrization

        self.hks = hks

        self.function_cache = []
        self.xgradient_cache = []
        self.ygradient_cache = []
        self.zgradient_cache = []

        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """ Gets the various property arrays from the particle arrays. """
        self.s_x = self.source.get_carray(self.x)
        self.s_y = self.source.get_carray(self.y)
        self.s_z = self.source.get_carray(self.z)
        self.s_u = self.source.get_carray(self.u)
        self.s_v = self.source.get_carray(self.v)
        self.s_w = self.source.get_carray(self.w)
        self.s_h = self.source.get_carray(self.h)
        self.s_m = self.source.get_carray(self.m)
        self.s_rho = self.source.get_carray(self.rho)
        self.s_p = self.source.get_carray(self.p)
        self.s_e = self.source.get_carray(self.e)
        self.s_cs = self.source.get_carray(self.cs)

        self.d_x = self.dest.get_carray(self.x)
        self.d_y = self.dest.get_carray(self.y)
        self.d_z = self.dest.get_carray(self.z)
        self.d_u = self.dest.get_carray(self.u)
        self.d_v = self.dest.get_carray(self.v)
        self.d_w = self.dest.get_carray(self.w)
        self.d_h = self.dest.get_carray(self.h)
        self.d_m = self.dest.get_carray(self.m)
        self.d_rho = self.dest.get_carray(self.rho)
        self.d_p = self.dest.get_carray(self.p)
        self.d_e = self.dest.get_carray(self.e)
        self.d_cs = self.dest.get_carray(self.cs)

    cdef void eval(self, int k, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):

        """ Computes the contribution of particle at source_pid on particle at
        dest_pid.

        """
        raise NotImplementedError, 'SPHFunctionParticle::eval'

    cpdef int output_fields(self) except -1:
        raise NotImplementedError, 'SPHFunctionParticle::output_fields'
    
    cpdef setup_iter_data(self):
        """ setup operations performed in each iteration """
        pass
    
    cdef double rkpm_first_order_kernel_correction(self, int dest_pid):
        """ Return the first order correction term for an interaction """

        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1*rab.x + beta2*rab.y)

    cdef double rkpm_first_order_gradient_correction(self, int dest_pid):
        """ Return the first order correction term for an interaction """
        
        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1*rab.x + beta2*rab.y)

    cdef double bonnet_and_lok_gradient_correction(self, int dest_pid,
                                                   cPoint grad):
        """ Correct the gradient of the kernel """

        cdef double x, y, z

        cdef double l11, l12, l13, l21, l22, l23, l31, l32, l33

        l11 = self.bl_l11.data[dest_pid]
        l12 = self.bl_l12.data[dest_pid]
        l13 = self.bl_l13.data[dest_pid]
        l22 = self.bl_l22.data[dest_pid]
        l23 = self.bl_l23.data[dest_pid]
        l33 = self.bl_l33.data[dest_pid]

        l21 = self.bl_l12.data[dest_pid]
        l31 = self.bl_l13.data[dest_pid]
        l32 = self.bl_l23.data[dest_pid]

        x = grad.x; y = grad.y; z = grad.z

        grad.x = l11*x + l12*y + l13*z
        
        grad.y = l21*x + l22*y + l23*z

        grad.z = l31*x + l32*y + l33*z        

    def py_eval(self, int k, int source_pid, int dest_pid,
                KernelBase kernel):

        cdef double nr[3], dnr[3]

        nr[0] = 0.0; nr[1] = 0.0; nr[2] = 0.0
        dnr[0] = 0.0; dnr[1] = 0.0; dnr[2] = 0.0
        self.eval(k, source_pid,  dest_pid, kernel, &nr[0], &dnr[0])

        return nr[0], dnr[0]

################################################################################
# `SPHFunctionPoint` class.
################################################################################
cdef class SPHFunctionPoint:
    """
    Base class to compute the contribution of an SPH particle, on a point in
    space.

    The class is designed on similar lines to the SPHFunctionParticle class,
    except that destination point, can be any random point. Thus no dest
    particle array is necessary here. The eval interfaces in the derived classes
    also have a different signature than that of the eval interfaces of classes
    derived from SPHFunctionParticle.

    """
    def __init__(self, ParticleArray array, bint setup_arrays=True, 
                 *args, **kwargs):

        self.source = array
        
        #Default properties
        self.m = 'm'
        self.rho = 'rho'
        self.h = 'h'
        self.p = 'p'
        self.e = 'e'
        self.x = 'x'
        self.y = 'y'
        self.z = 'z'
        self.u = 'u'
        self.v = 'v'
        self.w = 'w'

        self.s_m = None
        self.s_h = None
        self.s_rho = None
        self.s_p = None
        self.s_e = None
        self.s_x = None
        self.s_y = None
        self.s_z = None
        self.s_u = None
        self.s_v = None
        self.s_w = None

        self.d_m = None
        self.d_h = None
        self.d_rho = None
        self.d_p = None
        self.d_e = None
        self.d_x = None
        self.d_y = None
        self.d_z = None
        self.d_u = None
        self.d_v = None
        self.d_w = None

        if setup_arrays:
            self.setup_arrays()
    
    cpdef setup_arrays(self):
        """
        """
        self.s_x = self.source.get_carray(self.x)
        self.s_y = self.source.get_carray(self.y)
        self.s_z = self.source.get_carray(self.z)
        self.s_u = self.source.get_carray(self.u)
        self.s_v = self.source.get_carray(self.v)
        self.s_w = self.source.get_carray(self.w)
        self.s_h = self.source.get_carray(self.h)
        self.s_m = self.source.get_carray(self.m)
        self.s_rho = self.source.get_carray(self.rho)
        self.s_p = self.source.get_carray(self.p)
        self.s_e = self.source.get_carray(self.e)


    cdef void eval(self, cPoint pnt, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """
        Computes the contribution of particle at source_pid on point pnt.

        **Parameters**

         - pnt - the point at which some quatity is to be interpolated.
         - source_pid - the neighbor whose contribution is to be computed.
         - kernel - the kernel to be used.
         - nr - memory location to store the numerator of the result.
         - dnr - memory location to store the denominator of the result.

        """
        raise NotImplementedError, 'SPHFunctionPoint::eval'

    cpdef py_eval(self, Point pnt, int dest_pid, 
                  KernelBase kernel, numpy.ndarray
                  nr, numpy.ndarray dnr):
        """
        Python wrapper for the eval function, to be used in tests.
        """
        cdef double *_nr
        cdef double *_dnr
        cdef int i
        
        _nr = <double*>malloc(sizeof(double)*self.output_fields())
        _dnr = <double*>malloc(sizeof(double)*self.output_fields())

        self.eval(pnt.data, dest_pid, kernel, _nr, _dnr)

        for i in range(self.output_fields()):
            nr[i] += _nr[i]
            dnr[i] += _dnr[i]

        free(<void*>_nr)
        free(<void*>_dnr)

    cpdef int output_fields(self) except -1:
        """
        Returns the number of output fields, this SPHFunctionPoint will write
        to. This does not depend on the dimension of the simulation, it just
        indicates, the size of the arrays, dnr and nr that need to be passed to
        the eval function.
        """
        raise NotImplementedError, 'SPHFunctionPoint::output_fields'
