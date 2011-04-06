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

from pysph.solver.cl_utils import HAS_CL
if HAS_CL:
    import pyopencl as cl

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
        self.dst_writes = []
        self.initial_props = []

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

    def setupCL(self, context, queue, options):
        self.context = context
        self.queue = queue

        cl_kernel_src = open(self.cl_kernel_src_file, 'r').read()

        program = cl.Program(self.context, cl_kernel_src).build(options=options)
        self.cl_kernel = program.all_kernels()[0]
            
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
