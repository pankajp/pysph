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
from pysph.sph.funcs.basic_funcs cimport KernelGradientCorrectionTerms,\
    FirstOrderKernelCorrectionTermsForBeta, \
    FirstOrderKernelCorrectionTermsForAlpha

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
                  str id = "", bint kernel_gradient_correction=False,
                  bint first_order_kernel_correction=False,
                  int dim = 1):

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

        self.dnum = dnum
        self.id = id

        self.kernel_gradient_correction = kernel_gradient_correction
        self.first_order_kernel_correction=first_order_kernel_correction

        self.dim = dim

        self.check_internals()
        self.setup_internals()

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
        cdef SPHFunctionParticle func
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


        #set the kernel correction arrays if reqired

        if self.first_order_kernel_correction:

            if not self.dest.properties.has_key('beta1'):
                self.dest.add_property({'name':"beta1"})
            
            if not self.dest.properties.has_key('beta2'):
                self.dest.add_property({'name':"beta2"})

            if not self.dest.properties.has_key('beta3'):
                self.dest.add_property({'name':"beta3"})
            
            if not self.dest.properties.has_key('alpha'):
                self.dest.add_property({'name':"alpha"})

            for i in range(nsrcs):
                func = self.funcs[i]
                func.d_beta1 = self.dest.get_carray("beta1")
                func.d_beta2 = self.dest.get_carray("beta2")
                func.d_beta3 = self.dest.get_carray("beta3")
                func.d_alpha = self.dest.get_carray("alpha")
            
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

        #evaluate kernel correction terms if requested
        if self.first_order_kernel_correction and self.nbr_info:
            self.evaluate_first_order_kernel_correction_terms(exclude_self)

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

    cdef evaluate_kgc_terms(self, size_t i, int j):
        """ Evaluate the kernel gradient correction terms """
        
        cdef SPHFunctionParticle kgc
        cdef double m[3], l[3]
        cdef double a, b, d, fac
        cdef DoubleArray l11, l12, l22

        cdef ParticleArray src = self.sources[j]
        cdef SPHFunctionParticle func = self.sph_funcs[j]

        #initialize the arrays
        m[0] = m[1] = m[2] = 0.0
        l[0] = l[1] = l[2] = 0.0

        #Add the matrix arrays to the dest if it does not exist
        
        if not self.dest.properties.has_key("l1l"):
            self.dest.add_property({"name":"l11"})
            
        if not self.dest.properties.has_key("l12"):
            self.dest.add_property({"name":"l12"})

        if not self.dest.properties.has_key("l22"):
            self.dest.add_property({"name":"l22"})
                
        #Get the matrix arrays

        l11 = self.dest.get_carray("l11")
        l12 = self.dest.get_carray("l12")
        l22 = self.dest.get_carray("l22")

        #set the kernel gradient correction function
                
        kgc = KernelGradientCorrectionTerms(source=src, dest=self.dest)
                
        #evaluate the kernel gradient correction for the particle i

        for k from 0 <= k < self.nbrs.length:
            s_idx = self.nbrs.get(k)
            kgc.eval(s_idx, i, self.kernel, &m[0], &l[0])
                    
        #get the coefficients of the matrix
                    
        a = m[0]; b = m[1]; d = m[2]
        
        fac = a*d - b*b

        #prevent a divide by zero if the source is the dest

        if not fac < 1e-16:
            fac = 1./fac 
        else:
            func.kernel_gradient_correction = False
                    
        #set the coefficients of the inverted matrix
                
        l11.data[i] = fac * d
        l12.data[i] = -fac * b
        l22.data[i] = fac * a

    cdef evaluate_first_order_kernel_correction_terms(self, 
                                                      bint exclude_self=False):
        """ Evaluate the kernel correction terms """
        
        cdef SPHFunctionParticle fbeta, falpha, func
        cdef double m[3], l[3], aj, bj
        cdef double a, b, d, det, l11, l12, l22, b1, b2, b3

        cdef size_t np = self.dest.get_number_of_particles()
        cdef long dest_pid, source_pid
        cdef size_t i, k, s_idx
        cdef int j

        cdef ParticleArray src
        cdef FixedDestNbrParticleLocator loc

        cdef DoubleArray beta1, beta2, beta3, alpha

        cdef LongArray nbrs = self.nbrs

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        beta1 = self.dest.get_carray("beta1")
        beta2 = self.dest.get_carray("beta2")
        beta3 = self.dest.get_carray("beta3")
        alpha = self.dest.get_carray("alpha")

        for i from 0 <= i < np:

            if tag[i] == LocalReal:

                m[0] = m[1] = m[2] = 0.0
                l[0] = l[1] = l[2] = 0.0

                #evaluate the beta terms for particle i
                for j in range(self.nsrcs):
            
                    func = self.funcs[j]
                    loc  = self.nbr_locators[j]
                    src = self.sources[j]

                    nbrs.reset()
                    loc.get_nearest_particles(i, nbrs, exclude_self)
  
                    #set the kernel gradient correction function
                    
                    fbeta=FirstOrderKernelCorrectionTermsForBeta(
                        source=src, dest=self.dest)

                    for k from 0 <= k < self.nbrs.length:
                        s_idx = self.nbrs.get(k)
                        fbeta.eval(s_idx, i, self.kernel, &m[0], &l[0])

                #get the coefficients of the matrix
                
                a = m[0]; b = m[1]; d = m[2]
                b1 = l[0]; b2 = l[1]; b3 = l[2]

                if self.dim == 1:
                    d = 1.0
                
                det = a*d - b*b

                #prevent a divide by zero if the source is the dest

                if not (-1e-15 < det < 1e-15):
                    det = 1./det 
                else:
                    func.first_order_kernel_correction = False
                        
                #set the coefficients of the inverted matrix
            
                l11 = det*d; l12 = det * -b; l22 = det * a

                #set the beta vector for particle i

                beta1.data[i] = l11*b1 + l12*b2
                beta2.data[i] = l12*b1 + l22*b2
        
                aj = 0.0; bj = 0.0
                #evaluate the alpha terms for particle i
                for j in range(self.nsrcs):
            
                    func = self.funcs[j]
                    loc  = self.nbr_locators[j]
                    src = self.sources[j]
                
                    nbrs.reset()
                    loc.get_nearest_particles(i, nbrs, exclude_self)
  
                    #set the alpha correction function

                    falpha=FirstOrderKernelCorrectionTermsForAlpha(
                        source=src, dest=self.dest)
                
                    for k from 0 <= k < nbrs.length:
                        s_idx = self.nbrs.get(k)
                        falpha.eval(s_idx, i, self.kernel, &aj, &bj)
    
                #prevent a divide by zero if the source is the dest
                if -1e-15 < aj < 1e-15:
                    alpha.data[i] = 1.0
                else:
                    alpha.data[i] = 1./(aj)

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

            #evaluate the kernel gradient correction for the particle i
            if self.kernel_gradient_correction:
                func.kernel_gradient_correction = True
                self.evaluate_kgc_terms(i, j)

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
