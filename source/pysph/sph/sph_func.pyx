#include for malloc
from libc.stdlib cimport * 
cimport numpy
import numpy

def get_all_funcs():
    ''' function to gather all implemented funcs in pysph.sph.funcs package '''
    import os
    import pysph.sph.funcs as funcs_pkg
    funcs = {}
    for funcs_dir in funcs_pkg.__path__:
        search_modules = [os.path.splitext(i) for i in os.listdir(funcs_dir)]
        pass
        search_modules = [i[0] for i in search_modules if i[1][:3]=='.py']
        for mod_name in search_modules:
            mod_name = 'pysph.sph.funcs.'+mod_name
            mod = __import__(mod_name, fromlist=True)
            for name,value in mod.__dict__.iteritems():
                if type(value) == type and issubclass(value, SPHFunction) and (
                        not name.startswith('SPHFunction')):
                    funcs['%s.%s'%(mod_name,name)] = value
    return funcs


class Function(object):
    """ Class that defines sph function (sph.funcs) and its parameter values

    **Methods**

    - get_func -- Return a particular instance of SPHFunctionParticle
      with an appropriate source and destination particle array
    
    - get_func_class --  get the class for which func will be created
    
    **Example**

    The sph function MonaghanArtificialVsicosity (defined in
    sph.funcs.viscosity_funcs) requires the parameter values 'alpha',
    'beta', 'gamma' and 'eta' to define the artificial viscosity. This
    function may be created as:

    - avisc = Function(MonaghanArtificialVsicosity, hks=False, alpha, beta ..)
    - avisc_func = avisc.get_funcs(source, dest)
    
    or as an alternative may also be created as follows:
    
    - avisc = MonaghanArtificialVsicosity.withargs(hks=False, alpha, beta, ..)
    - avisc_func = avisc.get_funcs(source, dest)
    
    Function provides a convenient way to create funcs between multiple
    source and destination particle arrays with specified
    parameter values

    """
    def __init__(self, sph_func, *args, **kwargs):
        """ Constructor

        Parameters:
        -----------

        sph_func -- the SPHFunction class type
        *args, **kwargs -- optional positional and keyword arguments

        """

        self.sph_func = sph_func
        self.args = args
        self.kwargs = kwargs
    
    def get_func_class(self):
        """ get the class for which func will be created """
        return self.sph_func
    
    def get_func(self, source, dest):
        """ Return a SPHFunctionParticle instance with source and dest """
        func = self.sph_func(source, dest, *self.args, **self.kwargs)
        return func

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunction:
    """ Base class to represent an operation on a particle array.

    This class requires access to particle properties of a ParticleArray.
    Since there is no particle class having all properties at one place,
    names used for various properties and arrays corresponding to those
    properties are stored in this class for fast access to property values both
    at the source and destination.
    
    This class contains names, and arrays of common properties that will be
    needed for an operation. The data within
    these arrays, can be used as *array.data[pid]*, where pid in the particle
    index, "data" is the actual c-pointer to the data.

    All arrays are prefixed with a "s_". Destination arrays prefixed by "d_"
    are an alias for the same array prefixed with "s_". For example the mass
    property of the source will be in the s_m array which is same as d_m.
    This is not true for subclasses of :class:`SPHFunctionParticle` which
    can have different source and destination pairs

    """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 bint setup_arrays=True, *args, **kwargs):
        """ dest argument is unused. self.dest is set to source """
        self.name = ""
        self.id = ""
        self.tag = ""
        self.source = source
        self.dest = source

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
        
        self.num_outputs = 3

        self.src_reads = ['x','y','z','h','m', 'rho']
        self.dst_reads = ['x','y','z','h','tag']

        self.kernel = None

        self.cl_kernel_src_file = ''
        self.cl_kernel = object()
        self.cl_program = object()
        self.context = object()

        self.cl_args = []
        self.cl_args_name = []

        self.global_sizes = (self.dest.get_number_of_particles(), 1, 1)
        self.local_sizes = (1,1,1)
        
        if setup_arrays:
            self.setup_arrays()
    
    # convenience methods to be able to use class instead of Function object
    # for default construction (testing) and class.withargs(*) as an alias for
    # explicitly creating Function objects
    @classmethod
    def withargs(cls, *args, **kwargs):
        """ Return a :class:`Function` object for this class with arguments """
        return Function(cls, *args, **kwargs)
    
    @classmethod
    def get_func(cls, source, dest):
        """ Construct an instance of this class with default arguments
        This method enables this class to act like a :class:`Function`
        instance

        """        
        return Function(cls).get_func(source, dest)
    
    @classmethod
    def get_func_class(cls):
        return cls
    
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
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        """ Evaluate the store the results in the output arrays """
        cdef double result[3]
        cdef int i
        
        # get the tag array pointer
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()

        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                self.eval_single(i, kernel, result)
                output1.data[i] += result[0]
                output2.data[i] += result[1]
                output3.data[i] += result[2]
            else:
                output1.data[i] = output2.data[i] = output3.data[i] = 0

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        """ Evaluate the function on a single dest particle
        
        Implement this in a subclass to do the actual computation

        """
        raise NotImplementedError, 'SPHFunction.eval_single()'
    	
    cpdef int output_fields(self) except - 1:
        return self.num_outputs
    
    cpdef setup_iter_data(self):
        """ setup operations performed in each iteration
        
        Override this in a subclass to do any operations at every iteration
        """
        pass

    def setup_cl(self, object program, object context):
        """ OpenCL setup for the function.

        You may determine the OpenCL kernel launch parameters from within
        this function

        Currently we're using the default:

        global_sizes = (ndp, 1, 1)
        local_sizes = (1, 1, 1)

        """
        self.cl_program = program
        self.context = context

    def set_cl_kernel_args(self):

        self.cl_args_name = []
        self.cl_args = []
        
        # setup the  sph kernel args
        nbrs = numpy.int32(self.source.get_number_of_particles())
        self.cl_args.append(nbrs)
        self.cl_args_name.append('int const nbrs')

        if self.kernel is not None:        
            kernel_type = numpy.int32(self.kernel.get_type())
            dim = numpy.int32(self.kernel.dim)   

            self.cl_args.append(kernel_type)
            self.cl_args_name.append('int const kernel_type')

            self.cl_args.append(dim)
            self.cl_args_name.append('int const dim')
        
        for prop in self.dst_reads:
            self.cl_args.append(self.dest.get_cl_buffer(prop))

            if not prop == "tag":
                self.cl_args_name.append('__global REAL* d_%s'%(prop))
            else:
                self.cl_args_name.append('__global int* d_tag')

        for prop in self.src_reads:
            self.cl_args.append(self.source.get_cl_buffer(prop))
            self.cl_args_name.append('__global REAL* s_%s'%(prop))

        # append the output buffer. 
        self.cl_args.append( self.dest.get_cl_buffer('tmpx') )
        self.cl_args_name.append('__global REAL* tmpx')

        self.cl_args.append( self.dest.get_cl_buffer('tmpy') )
        self.cl_args_name.append('__global REAL* tmpy')

        self.cl_args.append( self.dest.get_cl_buffer('tmpz') )
        self.cl_args_name.append('__global REAL* tmpz')

        self._set_extra_cl_args()

    def _set_extra_cl_args(self):
        raise NotImplementedError("SPHFunction _set_extra_cl_args!")
        
    def set_cl_program(self, object program):
        self.cl_program = program

    def get_cl_workgroup_code(self):
        return """unsigned int work_dim = get_work_dim();
    unsigned int dest_id = get_gid(work_dim); """        

################################################################################
# `SPHFunctionParticle` class.
################################################################################
cdef class SPHFunctionParticle(SPHFunction):
    """ Base class to represent an interaction between two particles from two
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
                 bint setup_arrays=True, hks=False,
                 exclude_self=False,
                 FixedDestNbrParticleLocator nbr_locator=None,
                 *args, **kwargs):

        SPHFunction.__init__(self, source, setup_arrays=False, *args, **kwargs)

        self.dest = dest
        self.exclude_self = exclude_self
        
        #kernel correction of Bonnet and Lok
        self.bonnet_and_lok_correction = False

        #flag for the rkpm first order kernel correction
        self.rkpm_first_order_correction = False

        # type of kernel symmetrization
        self.hks = hks

        if setup_arrays:
            self.setup_arrays()

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        """ Computes contribution of all neighbors on particle at dest_pid """
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length

        if self.exclude_self:
            if self.src is self.dest:
                # this works because nbrs has self particle in last position
                nnbrs -= 1
        
        result[0] = result[1] = result[2] = 0.0
        for j in range(nnbrs):
            self.eval_nbr(nbrs.data[j], dest_pid, kernel, result)
    
    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                   KernelBase kernel, double * result):
        """ Computes contribution of particle at source_pid on dest_pid
        
        Implement this in a subclass to do the actual computation

        """
        raise NotImplementedError, 'SPHFunctionParticle.eval_nbr()'

    cdef double rkpm_first_order_kernel_correction(self, size_t dest_pid):
        """ Return the first order correction term for an interaction """

        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1 * rab.x + beta2 * rab.y)

    cdef double rkpm_first_order_gradient_correction(self, size_t dest_pid):
        """ Return the first order correction term for an interaction """
        
        cdef double beta1, beta2, alpha
        cdef cPoint rab = cPoint_sub(self._dst, self._src)
        
        beta1 = self.d_beta1.data[dest_pid]
        beta2 = self.d_beta2.data[dest_pid]
        alpha = self.d_alpha.data[dest_pid]

        return alpha * (1.0 + beta1 * rab.x + beta2 * rab.y)

    cdef double bonnet_and_lok_gradient_correction(self, size_t dest_pid,
                                                   cPoint * grad):
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

        grad.x = l11 * x + l12 * y + l13 * z
        grad.y = l21 * x + l22 * y + l23 * z
        grad.z = l31 * x + l32 * y + l33 * z        

################################################################################
# `CSPHFunctionParticle` class.
################################################################################
cdef class CSPHFunctionParticle(SPHFunctionParticle):
    """ `SPHFunctionParticle` class for use of corrected SPH (CSPH) operations
    
    In this case numerator and denominator are computed for each neighbor
    particle and finally the numerator is divided with the denominator.
    A more efficient way if multiple such funcs are needed may be to do these
    operations in separate funcs so the results can be reused
    """

    cdef void eval_single(self, size_t dest_pid, KernelBase kernel,
                          double * result):
        """ Computes contribution of all neighbors on particle at dest_pid """
        cdef double dnr[3] # denominator
        cdef LongArray nbrs = self.nbr_locator.get_nearest_particles(dest_pid)
        cdef size_t nnbrs = nbrs.length
        if self.exclude_self:
            if self.src is self.dest:
                # this works because nbrs has self particle in last position
                nnbrs -= 1
        
        result[0] = result[1] = result[2] = 0.0
        dnr[0] = dnr[1] = dnr[2] = 0.0
        for j in range(nnbrs):
            self.eval_nbr_csph(nbrs.data[j], dest_pid, kernel, result, dnr)
        
        for m in range(3):
            if dnr[m] != 0.0:
                result[m] /= dnr[m]
    
    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double * result, double * dnr):
        """ Compute influence when denominator is separately affected by nbrs
        
        This is used in cases such as CSPH where the summation if weighted
        by the kernel sum of all the neighboring particles
        """
        raise NotImplementedError, 'CSPHFunctionParticle.evaleval_nbr_csph()'
