"""
General purpose code for SPH computations.

This module provides the SPHCalc class, which does the actual SPH summation.
    
"""

#include "stdlib.pxd"
from libc.stdlib cimport *

cimport numpy
import numpy

from os import path

# logging import
import logging
logger=logging.getLogger()

# local imports
from pysph.base.particle_array cimport ParticleArray, LocalReal, Dummy
from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator
from pysph.base.nnps cimport NbrParticleLocatorBase

from pysph.sph.sph_func cimport SPHFunction
from pysph.sph.funcs.basic_funcs cimport BonnetAndLokKernelGradientCorrectionTerms,\
    FirstOrderCorrectionMatrix, FirstOrderCorrectionTermAlpha, \
    FirstOrderCorrectionMatrixGradient, FirstOrderCorrectionVectorGradient


from pysph.base.carray cimport IntArray, DoubleArray


from pysph.solver.cl_utils import HAS_CL, get_cl_include
if HAS_CL:
    import pyopencl as cl
    from pyopencl.array import vec

cdef int log_level = logger.level

###############################################################################
# `SPHCalc` class.
###############################################################################
cdef class SPHCalc:
    """ A general purpose summation object
    
    Members:
    --------
    source -- the source particle array
    dest -- the destination particles array
    func -- the function to use between the source and destination
    nbr_loc -- a list of neighbor locator for each (source, dest)
    kernel -- the kernel to use
    nnps_manager -- the NNPSManager for the neighbor locator

    Notes:
    ------
    We assume that one particle array is being used in the simulation and
    hence, the source and destination, as well as the func's source and 
    destination are the same particle array.
    
    The particle array should define a flag `bflag` to indicate if a particle
    is a solid of fluid particle. The function can in turn distinguish 
    between these particles and decide whether to include the influence 
    or not.

    The base class and the subclass `SPHFluidSolid` assume that the 
    interaction is to be considered between both types of particles.
    Use the `SPHFluid` and `SPHSolid` subclasses to perform interactions 
    with fluids and solids respectively.
    
    """

    #Defined in the .pxd file
    #cdef public ParticleArray dest, source
    #cdef public list funcs
    #cdef public list nbr_locators
    #cdef public NNPSManager nnps_manager
    #cdef public KernelBase kernel
    #cdef public Particles particles
    #cdef public LongArray nbrs

    def __cinit__(self, particles, list sources, ParticleArray dest,
                  KernelBase kernel, list funcs,
                  list updates, integrates=False, dnum=0, nbr_info=True,
                  str id = "", bint kernel_gradient_correction=False,
                  kernel_correction=-1, int dim = 1, str snum=""):

        self.nbr_info = nbr_info
        self.particles = particles
        self.sources = sources
        self.nsrcs = len(sources)
        self.dest = dest
        self.nbr_locators = []

        self.nnps_manager = particles.nnps_manager

        self.funcs = funcs
        self.kernel = kernel

        self.nbrs = LongArray()

        self.integrates = integrates
        self.updates = updates
        self.nupdates = len(updates)

        self.kernel_correction = kernel_correction

        self.dnum = dnum
        self.id = id

        self.dim = dim
        self.snum = snum

        self.correction_manager = None

        self.tag = ""

        self.src_reads = []
        self.dst_reads = []
        self.initial_props = []
        self.dst_writes = {}

        self.context = object()
        self.queue = object()
        self.cl_kernel = object()
        self.cl_kernel_src_file = ''

        self.check_internals()
        self.setup_internals()

    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

        logger.info("SPHCalc:check_internals: calc %s"%(self.id))

        if (len(self.sources) == 0 or self.nnps_manager is None
            or self.dest is None or self.kernel is None or len(self.funcs)
            == 0):
            logger.warn('invalid input to setup_internals')
            logger.info('sources : %s'%(self.sources))
            logger.info('nnps_manager : %s'%(self.nnps_manager))
            logger.info('dest : %s'%(self.dest))
            logger.info('kernel : %s'%(self.kernel))
            logger.info('sph_funcs : %s'%(self.funcs)) 
            
            return

        # we need one sph_func for each source.

        if len(self.funcs) != len(self.sources):
            msg = 'One sph function is needed per source'
            raise ValueError, msg

        # ensure that all the funcs are of the same class and have same tag

        funcs = self.funcs
        for i in range(len(self.funcs)-1):
            if type(funcs[i]) != type(funcs[i+1]):
                msg = 'All sph_funcs should be of same type'
                raise ValueError, msg
            if funcs[i].tag != funcs[i+1].tag:
                msg = "All functions should have the same tag"
                raise ValueError, msg
        #check that the function src and dsts are the same as the calc's

        for i in range(len(self.funcs)):
            if funcs[i].source != self.sources[i]:
                msg = 'SPHFunction.source not same as'
                msg += ' SPHCalc.sources[%d]'%(i)
                raise ValueError, msg

            # not valid for SPHFunction
            #if funcs[i].dest != self.dest:
            #    msg = 'SPHFunction.dest not same as'
            #    msg += ' SPHCalc.dest'
            #    raise ValueError, msg

        func = self.funcs[0]

        self.src_reads = func.src_reads
        self.dst_reads = func.dst_reads
        self.cl_kernel_src_file = func.cl_kernel_src_file

        src = path.join( path.abspath('.'), 'funcs/' )
        src = path.join( src, func.cl_kernel_src_file )

        if not path.isfile(src):
            #raise RuntimeWarning, "Kernel file does not exist!"
            pass

        self.cl_kernel_src_file = src

    cdef setup_internals(self):
        """ Set the update update arrays and neighbor locators """

        cdef FixedDestNbrParticleLocator loc
        cdef SPHFunction func
        cdef int nsrcs = self.nsrcs
        cdef ParticleArray src
        cdef int i

        self.nbr_locators[:] = []

        # set the calc's tag from the function tags. Check ensures all are same
        self.tag = self.funcs[0].tag

        # set the neighbor locators
        for i in range(nsrcs):
            src = self.sources[i]
            func = self.funcs[i]

            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())
            func.nbr_locator = loc

            logger.info("""SPHCalc:setup_internals: calc %s using 
                        locator (src: %s) (dst: %s) %s """
                        %(self.id, src.name, self.dest.name, loc))
            
            self.nbr_locators.append(loc)

    cpdef sph(self, str output_array1=None, str output_array2=None, 
              str output_array3=None, bint exclude_self=False): 
        """
        """
        if output_array1 is None: output_array1 = 'tmpx'
        if output_array2 is None: output_array2 = 'tmpy'
        if output_array3 is None: output_array3 = 'tmpz'

        cdef DoubleArray output1 = self.dest.get_carray(output_array1)
        cdef DoubleArray output2 = self.dest.get_carray(output_array2)
        cdef DoubleArray output3 = self.dest.get_carray(output_array3)

        self.reset_output_arrays(output1, output2, output3)
        self.sph_array(output1, output2, output3, exclude_self)

    cpdef sph_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                     output3, bint exclude_self=False):
        """
        Similar to the sph1 function, except that this can handle
        SPHFunction that compute 3 output fields.

        **Parameters**
        
         - output1 - the array to store the first output component.
         - output2 - the array to store the second output component.
         - output3 - the array to store the third output component.
         - exclude_self - indicates if each particle itself should be left out
           of the computations.

        """

        cdef SPHFunction func

        if self.kernel_correction != -1 and self.nbr_info:
            self.correction_manager.set_correction_terms(self)
        
        for func in self.funcs:
            func.nbr_locator = self.nnps_manager.get_neighbor_particle_locator(
                func.source, self.dest, self.kernel.radius())

            func.eval(self.kernel, output1, output2, output3)

    cdef reset_output_arrays(self, DoubleArray output1, DoubleArray output2,
                             DoubleArray output3):

        cdef int i
        for i in range(output1.length):
            output1.data[i] = 0.0
            output2.data[i] = 0.0
            output3.data[i] = 0.0

#############################################################################


class CL_SPHCalc(object):
    """ OpenCL aware SPHCalc """

    def __init__(self, calc, context):
        """ Constructor

        Parameters:
        -----------

        calc -- An SPHCalc instance that is used to create the CL version

        context -- An OpenCL context.

        """

        self.calc = calc
        self.context = context

        self.devices = context.devices

        # create a command queue with the first device on the context 

        self.queue = cl.CommandQueue(context, self.devices[0])

        self.setupCL()

    def setupCL(self):
        """ Setup the CL related stuff """

        self.setup_program()

        self.setup_buffers()        

    def setup_program(self):
        
        prog_src_file = self.calc.cl_kernel_src_file

        prog_src_file = open(prog_src_file).read()

        build_options = get_cl_include()

        self.prog = cl.Program(self.context, prog_src_file).build(
            build_options)

    def setup_buffers(self):

        mf = cl.mem_flags
        ctx = self.context

        dst = self.calc.dest
        self.np = np = dst.get_number_of_particles()

        # set the host particle array
        self.host_pa = host_pa = numpy.zeros(shape=(np,),dtype=vec.float16)

        for i in range(np):

            host_pa[i][0] = dst.x[i]
            host_pa[i][1] = dst.y[i]
            host_pa[i][2] = dst.z[i]
            host_pa[i][3] = dst.u[i]
            host_pa[i][4] = dst.v[i]
            host_pa[i][5] = dst.w[i]
            host_pa[i][6] = dst.h[i]
            host_pa[i][7] = dst.m[i]
            host_pa[i][8] = dst.rho[i]
            host_pa[i][9] = dst.p[i]
            host_pa[i][10] = dst.e[i]
            host_pa[i][11] = dst.cs[i]
            
            host_pa[i][12] = dst.tmpx[i]
            host_pa[i][13] = dst.tmpy[i]
            host_pa[i][14] = dst.tmpz[i]
            host_pa[i][15] = dst.x[i]
        
        self.host_tag = host_tag = numpy.ones(shape=(np,), dtype=numpy.int)

        self.host_kernel_type = numpy.ones(shape=(1,), dtype=numpy.int)
        self.host_dim = numpy.ones(shape=(1,), dtype=numpy.int)
        self.host_np = numpy.ones(shape=(1,), dtype=numpy.int) * self.np

        self.host_result = host_result = numpy.zeros(shape=(np,),
                                                     dtype=numpy.float32)

        # allocate the device buffers
        self.device_pa = devica_pa = cl.Buffer(ctx,
                                               mf.READ_ONLY | mf.COPY_HOST_PTR,
                                               hostbuf=host_pa)

        self.device_tag = device_tag = cl.Buffer(ctx,
                                                 mf.READ_ONLY |mf.COPY_HOST_PTR,
                                                 hostbuf=host_tag)

        self.device_kernel_type = device_tag = cl.Buffer(
            ctx, mf.READ_ONLY |mf.COPY_HOST_PTR,
            hostbuf=self.host_kernel_type)

        self.device_dim = device_tag = cl.Buffer(
            ctx, mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=self.host_dim)

        self.device_np = device_tag = cl.Buffer(
            ctx, mf.READ_ONLY |mf.COPY_HOST_PTR, hostbuf=self.host_np)

        self.device_result = device_result = cl.Buffer(ctx,
                                                       mf.WRITE_ONLY,
                                                       host_result.nbytes)

    def cl_sph(self):
        """ Evaluate the contribution from the sources on the
        destinations using OpenCL.

        Particles are represented as float16 arrays and are assumed to
        be defined in ParticleArray itselt as a result of a call to
        ParticleArray's `setupCL` function with a CommandQueue as
        argument.

        Thus device buffer representaions for the ParticleArray are
        obtained as:

        pa.pa_buf_device
        pa.pa_tag_device

        pa_buf_device is a float16 array with the following implicit
        ordering of variables

        0:x, 1:y, 2:z, 3:u, 4:v, 5:w, 6:h, 7:m, 8:rho, 9:p, 10:e, 11:cs
        12:tmpx, 13:tmpy, 14:tmpz, 15:x

        pa_tag_device is an int2 array with the following ordering
        0:tag, 1:idx

        Since an SPH function operates between a source and
        destination ParticleArray, we need to pass in the
        corresponding buffers.

        Output is computed in tmpx, tmpy and tmpz by default and since
        they are defined as components 12, 13, and 14 respectively,
        they need not be explicitly passed.

        The functions implemented in OpenCL have the signature:

        __kernel void function(__global float16* dst, __global float16* src,
                               __global int2* dst_tag, __global int* np,
                               __global_int* kernel_type, __global_int* dim)

        I think this should suffice for the evaluation of the function
        and hence the contribution from a source to a destination.
        
        """

        self.prog.CL_SPHRho(self.queue, (self.np,1,1), (1,1,1),
                            self.device_pa, self.device_pa, self.device_tag,
                            self.device_result, self.device_kernel_type,
                            self.device_dim, self.device_np)

        cl.enqueue_read_buffer(self.queue, self.device_result, self.host_result)

        np = self.calc.dest.get_number_of_particles()

        host_array = self.device_pa.get_host_array((np,), vec.float16)

        print host_array[1][12]

        print self.host_result
