"""
General purpose code for SPH computations.

This module provides the SPHBase class, which does the actual SPH summation.
    
"""

#include "stdlib.pxd"
from libc.stdlib cimport *

cimport numpy
import numpy

# logging import
import logging
logger=logging.getLogger()

# local imports
from pysph.base.particle_array cimport ParticleArray, LocalReal, Dummy
from pysph.base.point cimport Point
from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator
from pysph.base.nnps cimport NbrParticleLocatorBase

from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.sph.funcs.basic_funcs cimport BonnetAndLokKernelGradientCorrectionTerms,\
    FirstOrderCorrectionMatrix, FirstOrderCorrectionTermAlpha, \
    FirstOrderCorrectionMatrixGradient, FirstOrderCorrectionVectorGradient


from pysph.base.carray cimport IntArray, DoubleArray

cdef int log_level = logger.level

###############################################################################
# `SPHBase` class.
###############################################################################
cdef class SPHBase:
    """ A general purpose summation object
    
    Members:
    --------
    source -- the source particle array
    dest -- the destination particles array
    func -- the function to use between the source and destination
    nbr_loc -- a list of neighbor locator for each (source, dest)
    kernel -- the multi-dimensional kernel to use
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

        """ Constructor """

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

        self.check_internals()
        self.setup_internals()

    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

        logger.info("SPHBase:check_internals: calc %s"%(self.id))

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
                msg = 'SPHFunctionParticle.source not same as'
                msg += ' SPHBase.sources[%d]'%(i)
                raise ValueError, msg

            if funcs[i].dest != self.dest:
                msg = 'SPHFunctionParticle.dest not same as'
                msg += ' SPHBase.dest'
                raise ValueError, msg

    cdef setup_internals(self):
        """ Set the update update arrays and neighbor locators """

        cdef FixedDestNbrParticleLocator loc
        cdef SPHFunctionParticle func
        cdef int nsrcs = self.nsrcs
        cdef ParticleArray src
        cdef int i

        self.nbr_locators[:] = []

        #set the calc's tag from the function tags. Check ensures all are same

        self.tag = self.funcs[0].tag

        # set the neighbor locators

        for i in range(nsrcs):
            src = self.sources[i]
            func = self.funcs[i]

            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())

            func.kernel_function_evaluation = loc.kernel_function_evaluation
            func.kernel_gradient_evaluation = loc.kernel_gradient_evaluation

            logger.info("""SPHBase:setup_internals: calc %s using 
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
        
        self.sph_array(output1, output2, output3, exclude_self)

    cpdef sph_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                     output3, bint exclude_self=False):
        """
        Similar to the sph1 function, except that this can handle
        SPHFunctionParticle that compute 3 output fields.

        **Parameters**
        
         - output1 - the array to store the first output component.
         - output2 - the array to store the second output component.
         - output3 - the array to store the third output component.
         - exclude_self - indicates if each particle itself should be left out
           of the computations.

        """

        cdef size_t np = self.dest.get_number_of_particles()
        cdef long dest_pid, source_pid
        cdef double nr[3], dnr[3]
        cdef size_t i
        cdef SPHFunctionParticle func
        global log_level
        log_level = logger.level

        # get the tag array pointer

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        if self.kernel_correction != -1 and self.nbr_info:
            self.correction_manager.set_correction_terms(self)
        
        for func in self.funcs:
            func.setup_iter_data()

        # loop over all particles

        for i from 0 <= i < np:

            dnr[0] = dnr[1] = dnr[2] = 0.0
            nr[0] = nr[1] = nr[2] = 0.0

            if tag[i] == LocalReal:

                self.eval(i, &nr[0], &dnr[0], exclude_self)

            if dnr[0] == 0.0:
                output1.data[i] = nr[0]
            else:
                output1.data[i] = nr[0]/dnr[0]

            if dnr[1] == 0.0:
                output2.data[i] = nr[1]
            else:
                output2.data[i] = nr[1]/dnr[1]

            if dnr[2] == 0.0:
                output3.data[i] = nr[2]
            else:
                output3.data[i] = nr[2]/dnr[2]

    cdef eval(self, size_t i, double* nr, double* dnr,
              bint exclude_self):
        raise NotImplementedError, 'SPHBase::eval'                       

#############################################################################

cdef class SPHCalc(SPHBase):
    
    cdef eval(self, size_t i, double* nr, double* dnr, 
              bint exclude_self):
    
        cdef ParticleArray src, pae
        cdef SPHFunctionParticle func
        cdef FixedDestNbrParticleLocator loc
        cdef size_t k
        cdef int j, nnbrs
        cdef LongArray nbrs

        for j in range(self.nsrcs):
            
            src = self.sources[j]
            func = self.funcs[j]
            loc  = self.nbr_locators[j]

            nbrs = loc.get_nearest_particles(i)
            nnbrs = nbrs.length
            if exclude_self:
                if src is self.dest:
                    # this works because nbrs has self particle in last position
                    nnbrs -= 1

            for k in range(nnbrs):
                func.eval(nbrs.data[k], i, self.kernel, &nr[0], &dnr[0])

            if log_level < 30:

                logger.info("""SPHCalc:eval: calc %s, dest %s, source %s"""
                            %(self.id, self.dest.name, src.name))

                logger.info("SPHCalc:eval Neighbor indices for particle %d %s"
                            %(i, nbrs.get_npy_array()))

                pae = src.extract_particles(nbrs, ['idx'])
                logger.info("""SPHCalc:eval: Neighbors for particle %d : %s"""
                            %(i, pae.get('idx')))

#############################################################################

cdef class SPHEquation(SPHBase):

    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

        logger.info("SPHEquation:check_internals: calc %s"%(self.id))

        if (self.nnps_manager is None or self.dest is None \
                or self.kernel is None or len(self.funcs)  == 0):
            logger.warn('invalid input to setup_internals')
            logger.info('sources : %s'%(self.sources))
            logger.info('nnps_manager : %s'%(self.nnps_manager))
            logger.info('dest : %s'%(self.dest))
            logger.info('kernel : %s'%(self.kernel))
            logger.info('sph_funcs : %s'%(self.funcs)) 
            
            return

        # ensure that all the funcs are of the same class

        funcs = self.funcs
        for i in range(len(self.funcs)-1):
            if type(funcs[i]) != type(funcs[i+1]):
                msg = 'All sph_funcs should be of same type'
                raise ValueError, msg
            
        # check that the function src and dsts are the same as the calc's

            if funcs[i].dest != self.dest:
                msg = 'SPHFunctionParticle.dest not same as'
                msg += ' SPHBase.dest'
                raise ValueError, msg

    cdef eval(self, size_t i, double* nr, double* dnr, 
              bint exclude_self):
    
        cdef SPHFunctionParticle func = self.funcs[0]
        func.eval(-1, i, self.kernel, &nr[0], &dnr[0])

#############################################################################
