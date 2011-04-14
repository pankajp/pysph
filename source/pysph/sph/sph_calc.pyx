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
import pysph.solver.api as solver

from os import path

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
        
        src = solver.get_pysph_root()
        src = path.join(src, 'sph/funcs/' + func.cl_kernel_src_file)

        if not path.isfile(src):
            raise RuntimeWarning, "Kernel file does not exist!"

        self.cl_kernel_src_file = src
        self.cl_kernel_function_name = func.cl_kernel_function_name

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

        func = self.funcs[0]

        self.src_reads = func.src_reads
        self.dst_reads = func.dst_reads

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

class CLCalc(SPHCalc):
    """ OpenCL aware SPHCalc """

    def set_context(self, context):
        self.context = context
        self.devices = context.devices

        # create a command queue with the first device on the context 
        self.queue = cl.CommandQueue(context, self.devices[0])

        self.setupCL()

    def setupCL(self):
        """ Setup the CL related stuff """

        self.setup_program()

        # set up the device buffers for the srcs and dest

        self.dest.setupCL(self.queue)

        for src in self.sources:
            src.setupCL(self.queue)

    def setup_program(self):
        """ Setup the OpenCL function used by this Calc
        
        The calc computes the interation on a single destination from
        a list of sources, using the same function.

        A call to this function sets up the OpenCL kernel which
        encapsulates the SPHFunction function.

        The source for the OpenCL kernel is referenced from the
        SPHFunction
        
        """
        
        prog_src_file = self.cl_kernel_src_file

        prog_src_file = open(prog_src_file).read()

        build_options = get_cl_include()

        self.prog = cl.Program(self.context, prog_src_file).build(
            build_options)
        
    def sph(self):
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
        
        # set the tmpx, tmpy and tmpz arrays to 0

        self.reset_output_arrays()

        dst = self.dest
        npd = dst.get_number_of_particles()

        mf = cl.mem_flags

        sph_kernel_type = cl.array.Array(self.queue, (1,),numpy.int32)
        sph_kernel_type.set(numpy.array([1,], numpy.int32), self.queue)

        dim = cl.array.Array(self.queue, (1,), dtype=numpy.int32)
        dim.set(numpy.array([self.kernel.dim,], numpy.int32), self.queue)

        for cl_kernel in self.prog.all_kernels():
            if cl_kernel.function_name == self.cl_kernel_function_name:
                break

        for i in range(self.nsrcs):
            src = self.sources[i]

            np = cl.array.Array(self.queue, (1,), numpy.int32)
            np.set(numpy.array([src.get_number_of_particles()], numpy.int32))

            cl_kernel(self.queue, (npd, 1, 1), (1,1,1),
                      dst.pa_buf_device, src.pa_buf_device,
                      dst.pa_tag_device, np.data, sph_kernel_type.data,
                      dim.data)

    def reset_output_arrays(self):
        """ Reset the dst tmpx, tmpy and tmpz arrays to 0

        Since multiple functions contribute to the same LHS value, the
        OpenCL kernel code increments the result to tmpx, tmpy and tmpz.
        To avoid unplesant behavior, we set these variables to 0 before
        any function is called.
        
        """

        if not self.dest.cl_setup_done:
            raise RuntimeWarning, "CL not setup on destination array!"

        npd = self.dest.get_number_of_particles()
        self.prog.set_tmp_to_zero(self.queue, (npd,1,1), (1,1,1),
                                  self.dest.pa_buf_device)

#############################################################################
