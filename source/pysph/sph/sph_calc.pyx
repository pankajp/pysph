"""
General purpose code for SPH computations.

This module provdies the SPHBase class, which does the actual SPH summation.
    
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

        #set the neighbor locators

        for i in range(nsrcs):
            src = self.sources[i]
            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())

            logger.info("""SPHBase:setup_internals: calc %s using 
                        locator (src: %s) (dst: %s) %s """
                        %(self.id, src.name, self.dest.name, loc))
            
            self.nbr_locators.append(loc)

        #set the kernel correction arrays if reqired

        if self.rkpm_first_order_correction:

            if not self.dest.properties.has_key('rkpm_beta1'):
                self.dest.add_property({'name':"rkpm_beta1"})
            
            if not self.dest.properties.has_key('rkpm_beta2'):
                self.dest.add_property({'name':"rkpm_beta2"})

            if not self.dest.properties.has_key('rkpm_beta3'):
                self.dest.add_property({'name':"rkpm_beta3"})
            
            if not self.dest.properties.has_key('rkpm_alpha'):
                self.dest.add_property({'name':"rkpm_alpha"})
                
            if not self.dest.properties.has_key("rkpm_dalphadx"):
                self.dest.add_property({"name":"rkpm_dalphadx"})

            if not self.dest.properties.has_key("rkpm_dalphady"):
                self.dest.add_property({"name":"rkpm_dalphady"})

            if not self.dest.properties.has_key("rkpm_dbeta1dx"):
                self.dest.add_property({"name":"rkpm_dbeta1dx"})

            if not self.dest.properties.has_key("rkpm_dbeta1dy"):
                self.dest.add_property({"name":"rkpm_dbeta1dy"})

            if not self.dest.properties.has_key("rkpm_beta2dx"):
                self.dest.add_property({"name":"rkpm_dbeta2dy"})

            if not self.dest.properties.has_key("rkpm_dbeta2dy"):
                self.dest.add_property({"name":"rkpm_dbeta2dy"})

            for i in range(nsrcs):
                func = self.funcs[i]
                func.rkpm_first_order_correction = True
                func.d_rkpm_beta1 = self.dest.get_carray("rkpm_beta1")
                func.d_rkpm_beta2 = self.dest.get_carray("rkpm_beta2")
                func.d_rkpm_beta3 = self.dest.get_carray("rkpm_beta3")
                func.d_rkpm_alpha = self.dest.get_carray("rkpm_alpha")
                func.d_rkpm_beta1 = self.dest.get_carray("rkpm_dalphadx")
                func.d_rkpm_beta2 = self.dest.get_carray("rkpm_dalphady")
                func.d_rkpm_beta3 = self.dest.get_carray("rkpm_dbeta1dx")
                func.d_rkpm_alpha = self.dest.get_carray("rkpm_dbeta1dy")
                func.d_rkpm_beta3 = self.dest.get_carray("rkpm_dbeta2dx")
                func.d_rkpm_alpha = self.dest.get_carray("rkpm_dbeta2dy")
            
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

        #get the tag array pointer

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        if self.kernel_correction != -1 and self.nbr_info:
            self.correction_manager.set_correction_terms(self)

        #logger.info("""SPHBase:sph: calc %s looping over all destination 
        #               particles """%(self.id))

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

    cdef evaluate_rkpm_first_order_correction_terms(
        self,  bint exclude_self=False):

        """ Evaluate the kernel correction terms """
        
        cdef SPHFunctionParticle fbeta, falpha, fmatgrad, fvecgrad, func

        cdef double mat[3], rhs[3]
        cdef double matg1[3], matg2[3]
        cdef double rhs1[3], rhs2[3]
        cdef double aj[3], bj[3]

        cdef double a, b, d, l11, l12, l22

        cdef double b1, b2, b3
        cdef double det, one_by_det
        cdef double dadx, dady, dbdx, dbdy, dddx, dddy
        cdef double db1dx, db1dy, db2dx, db2dy
        cdef double ddetdx, ddetdy
        cdef tmpdx, tmpdy

        cdef size_t np = self.dest.get_number_of_particles()
        cdef long dest_pid, source_pid
        cdef size_t i, k, s_idx
        cdef int j

        cdef ParticleArray src
        cdef FixedDestNbrParticleLocator loc

        cdef DoubleArray beta1, beta2, alpha
        cdef DoubleArray dbeta1dx, dbeta1dy, dbeta2dx, dbeta2dy
        cdef DoubleArray dalphadx, dalphady

        cdef LongArray nbrs = self.nbrs

        cdef LongArray tag_arr = self.dest.get_carray('tag')
        cdef long* tag = tag_arr.get_data_ptr()

        beta1 = self.dest.get_carray("rkpm_beta1")
        beta2 = self.dest.get_carray("rkpm_beta2")
        alpha = self.dest.get_carray("rkpm_alpha")
        
        dbeta1dx = self.dest.get_carray("rkpm_dbeta1dx")
        dbeta1dy = self.dest.get_carray("rkpm_dbeta1dy")

        dbeta2dx = self.dest.get_carray("rkpm_dbeta2dx")
        dbeta2dy = self.dest.get_carray("rkpm_dbeta2dy")

        dalphadx = self.dest.get_carray("rkpm_dalphadx")
        dalphady = self.dest.get_carray("rkpm_dalphady")

        for i from 0 <= i < np:

            if tag[i] == LocalReal:

                mat[0] = mat[1] = mat[2] = 0.0

                rhs[0] = rhs[1] = rhs[2] = 0.0

                matg1[0] = matg1[1] = matg1[2] = 0.0

                matg2[0] = matg2[1] = matg2[2] = 0.0

                rhs1[0] = rhs1[1] = rhs1[2] = 0.0

                rhs2[0] = rhs2[1] = rhs2[2] = 0.0

                aj[0] = aj[1] = aj[2] = 0.0

                bj[0] = bj[1] = bj[2] = 0.0

                #evaluate the beta terms for particle i
                for j in range(self.nsrcs):
            
                    func = self.funcs[j]
                    loc  = self.nbr_locators[j]
                    src = self.sources[j]

                    nbrs.reset()
                    loc.get_nearest_particles(i, nbrs, exclude_self)
  
                    #set the kernel gradient correction function
                    
                    fbeta=FirstOrderCorrectionMatrix(
                        source=src, dest=self.dest)

                    fmatgrad = FirstOrderCorrectionMatrixGradient(
                        source=src, dest=self.dest)

                    fvecgrad = FirstOrderCorrectionVectorGradient(
                        source=src, dest=self.dest)

                    for k from 0 <= k < self.nbrs.length:
                        s_idx = self.nbrs.get(k)

                        fbeta.eval(s_idx, i, self.kernel, &mat[0], &rhs[0])
                        
                        fmatgrad.eval(s_idx,i,self.kernel,&matg1[0],&matg2[0])

                        fvecgrad.eval(s_idx, i, self.kernel, &rhs1[0], &rhs2[0])

                #get the coefficients of the matrix to invert
                
                a = mat[0]; b = mat[1]; d = mat[2]
                b1 = rhs[0]; b2 = rhs[1]; b3 = rhs[2]

                if self.dim == 1:
                    d = 1.0
                
                det = a*d - b*b

                #prevent a divide by zero if the source is the dest

                if not (-1e-15 < det < 1e-15):
                    one_by_det = 1./det
                    
                    #set the coefficients of the inverted matrix
            
                    l11 = det*d; l12 = det * -b; l22 = det * a

                    #set the beta vector for particle i

                    beta1.data[i] = l11*b1 + l12*b2
                    beta2.data[i] = l12*b1 + l22*b2
                    
                    dadx = matg1[0]; dady = matg1[1]; dbdx = matg1[2]
                    dbdy = matg2[0]; dddx = matg2[1]; dddy = matg2[2]
                    
                    db1dx = rhs1[0]; db1dy = rhs1[1]
                    db2dx = rhs1[2]; db2dy = rhs2[0]

                    ddetdx = a*dddx + d*dadx - 2*b*dbdx
                    ddetdy = a*dddy + d*dady - 2*b*dbdy

                    #evaluate dbeta1dx
                    tmpdx = det*(d*db1dx + b1*dddx - b*db2dx - b2*dbdx) -\
                        ddetdx*(d*b1 - b*b2)
                    
                    tmpdx *= (one_by_det*one_by_det)
                    dbeta1dx.data[i] = tmpdx

                    #evaluate dbeta1dy
                    tmpdy = det*(d*db1dy + b1*dddy - b*db2dy - b2*dbdy) -\
                        ddetdy*(d*b1 - b*b2)

                    tmpdy *= (one_by_det*one_by_det)                    
                    dbeta1dy.data[i] = tmpdy

                    #evaluate dbeta2dx
                    tmpdx = det*(-b*db1dx - b1*dbdx + a*db2dx + b2*dadx) -\
                        ddetdx*(a*b2 - b*b1)
                    
                    tmpdx *= (one_by_det*one_by_det)
                    dbeta2dx.data[i] = tmpdx

                    #evaluate dbeta2dy
                    tmpdy = det*(-b*db1dy - b1*dbdy + a*db2dy + b2*dady) -\
                        ddetdx*(a*b2 - b*b1)
                    
                    tmpdy *= (one_by_det*one_by_det)
                    dbeta2dy.data[i] = tmpdx                    
                   
                    for j in range(self.nsrcs):
            
                        func = self.funcs[j]
                        loc  = self.nbr_locators[j]
                        src = self.sources[j]
                
                        nbrs.reset()
                        loc.get_nearest_particles(i, nbrs, exclude_self)
  
                        #set the alpha correction function

                        falpha=FirstOrderCorrectionTermAlpha(
                            source=src, dest=self.dest)
                
                        for k from 0 <= k < nbrs.length:
                            s_idx = self.nbrs.get(k)
                            falpha.eval(s_idx, i, self.kernel, &aj[0], &bj[0])
                            
                    #prevent a divide by zero if the source is the dest
                    if -1e-15 < aj[0] < 1e-15:
                        alpha.data[i] = 1.0
                    else:
                        alpha.data[i] = 1./(aj[0])
                        dalphadx.data[i] = aj[1]/(aj[0]*aj[0])
                        dalphady.data[i] = aj[2]/(aj[0]*aj[0])

                else:
                    func.rkpm_first_order_correction = False
                        

#############################################################################

cdef class SPHCalc(SPHBase):
    
    cdef eval(self, size_t i, double* nr, double* dnr, 
              bint exclude_self):
    
        cdef ParticleArray src, pae
        cdef SPHFunctionParticle func
        cdef FixedDestNbrParticleLocator loc
        cdef size_t k, s_idx
        cdef LongArray nbrs = self.nbrs
        cdef int j

        for j in range(self.nsrcs):
            
            src = self.sources[j]
            func = self.funcs[j]
            loc  = self.nbr_locators[j]

            nbrs.reset()
            loc.get_nearest_particles(i, nbrs, exclude_self)

            if logger.level < 30:

                logger.info("""SPHCalc:eval: calc %s, dest %s, source %s"""
                            %(self.id, self.dest.name, src.name))

                logger.info("SPHCalc:eval Neighbor indices for particle %d %s"
                            %(i, nbrs.get_npy_array()))

                pae = src.extract_particles(nbrs, ['idx'])
                logger.info("""SPHCalc:eval: Neighbors for particle %d : %s"""
                            %(i, pae.get('idx')))

            for k from 0 <= k < self.nbrs.length:
                s_idx = self.nbrs.data[k]
                func.eval(s_idx, i, self.kernel, &nr[0], &dnr[0])

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
