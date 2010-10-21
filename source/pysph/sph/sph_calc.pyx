"""
General purpose code for SPH computations.

This module provdies the SPHBase class, which does the actual SPH summation.
    
"""

#include "stdlib.pxd"
from libc.stdlib cimport *

cimport numpy

# logging import
import logging
logger=logging.getLogger()

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport Point
from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator
from pysph.base.nnps cimport NbrParticleLocatorBase

from pysph.base.particle_tags cimport LocalReal, Dummy
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.base.carray cimport IntArray, DoubleArray

###############################################################################
# `SPHBase` class.
###############################################################################
cdef class SPHBase:
    """ A general purpose summation object
    
    Members:
    --------
    source -- the source particlel array
    dest -- the destination particles array
    func -- the function to use betwenn the source and destination
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
    between these particles and deciide wehter to include the influence 
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
    #cdef public MultidimensionalKernel kernel
    #cdef public Particles particles
    #cdef public LongArray nbrs

    def __cinit__(self, particles, list sources, ParticleArray dest,
                  MultidimensionalKernel kernel, list funcs,
                  list updates, integrates=False, dnum=0, nbr_info=True,
                  str id = ""):

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

        self.check_internals()
        self.setup_internals()

        self.dnum = dnum
        self.id = id

    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

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

        # ensure that all the funcs are of the same class

        funcs = self.funcs
        for i in range(len(self.funcs)-1):
            if type(funcs[i]) != type(funcs[i+1]):
                msg = 'All sph_funcs should be of same type'
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
        cdef int nsrcs = self.nsrcs
        cdef ParticleArray src
        cdef int i

        self.nbr_locators[:] = []

        #set the neighbor locators

        for i in range(nsrcs):
            src = self.sources[i]
            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())
            logger.info('Using locator : %s, %s, %s'%(src.name, 
                                                     self.dest.name, loc))
            self.nbr_locators.append(loc)
            
    cpdef sph(self, str output_array1=None, str output_array2=None, 
              str output_array3=None, bint exclude_self=False): 
        """
        """
        if not output_array1: output_array1 = 'tmpx'
        if not output_array2: output_array2 = 'tmpy'
        if not output_array3: output_array3 = 'tmpz'            

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

        #get the tag array pointer

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        #loop over all particles

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

        cdef SPHFunctionParticle func
        cdef FixedDestNbrParticleLocator loc
        cdef size_t k, s_idx
        cdef LongArray nbrs = self.nbrs
        cdef int j

        for j in range(self.nsrcs):
            
            func = self.funcs[j]
            loc  = self.nbr_locators[j]

            nbrs.reset()
            loc.get_nearest_particles(i, nbrs, exclude_self)

            msg = """Number of neighbors for particle %d of dest %d, from
                  source %d are %d """ %(i, self.dnum, j, nbrs.length)

            for k from 0 <= k < self.nbrs.length:
                s_idx = self.nbrs.get(k)
                func.eval(s_idx, i, self.kernel, &nr[0], &dnr[0])

#############################################################################

cdef class SPHEquation(SPHBase):

    cpdef check_internals(self):
        """ Check for inconsistencies and set the neighbor locator. """

        # check if the data is sane.

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
            
        #check that the function src and dsts are the same as the calc's

            if funcs[i].dest != self.dest:
                msg = 'SPHFunctionParticle.dest not same as'
                msg += ' SPHBase.dest'
                raise ValueError, msg

    cdef eval(self, size_t i, double* nr, double* dnr, 
              bint exclude_self):
    
        cdef SPHFunctionParticle func = self.funcs[0]
        func.eval(-1, i, self.kernel, &nr[0], &dnr[0])

#############################################################################
