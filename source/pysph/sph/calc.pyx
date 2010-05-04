"""
General purpose code for SPH computations.

This module provdies the SPHBase class, which does the actual SPH summation.
     
"""

include "stdlib.pxd"

cimport numpy
import numpy

################################################################################
# `SPHCalc` class.
################################################################################
cdef class SPHCalc:
    """ SPHBase is a general purpose class for SPH calculations.
    
    Used in conjunction with SPHComponent to evaluate the influence of
    source particle arrays on a single destination. Look at the .pxd
    file to see the variables defined.

    """    
    #Defined in the .pxd file
    #cdef public ParticleArray dest
    #cdef public list srcs
    #cdef public list nbr_locators
    #cdef public list sph_funcs
    #cdef public MultidimensionalKernel kernel
    #cdef public str h

    def __cinit__(self, list srcs=[], ParticleArray dest=None, 
                  MultidimensionalKernel kernel=None, 
                  list sph_funcs=[], str h='h',  bint setup_internals=True,
                  *args, **kwargs): 
        """ Constructor. """

        self.nbr_locators = []
        self.srcs = srcs
        self.sph_funcs = sph_funcs
        
        self.dest = dest
        self.kernel = kernel
        self.dim = kernel.dim

        self.h = h

        if setup_internals:
            self.setup_nnps()

    cpdef setup_nnps(self):
        """ Check and prepare the data to perform as required."""

        self._check()

        # create the neighbor locators
        cdef NNPS nps
        self.nbr_locators = []
        for src in self.srcs:
            bin_size = numpy.max(src.get(self.h))
            nps = NNPS(src)
            nps.update(bin_size)
            self.nbr_locators.append(nps)

    def _check(self):
        """ Check the data for consistency. """
        
        # Sanity check. 
        nsrc = len(self.srcs)
        nfuncs = len(self.sph_funcs)
        kernel = self.kernel
        dst = self.dest
        
        # we need one sph_func for each source.
        if nfuncs != nsrc:
            msg = 'One sph function is needed per source'
            raise ValueError, msg

        sph_funcs = self.sph_funcs

        #The sph_func's must be alike.
        for i in range(nfuncs - 1):
            if type(sph_funcs[i]) != type(sph_funcs[i+1]):
                msg = 'All sphfuncs should be of same type'
                raise ValueError, msg
            
        #The source of the sph_func must match that of the list.
        for i in range(nfuncs):
            if sph_funcs[i].source != self.srcs[i]:
                msg = 'SPHFunctionParticle.source not same as'
                msg += ' SPHBase.sources[%d]'%(i)
                raise ValueError, msg

            #The destination of the sph_func must match the destnation.
            if sph_funcs[i].dest != self.dest:
                msg = 'SPHFunctionParticle.dest not same as'
                msg += ' SPHBase.dest'
                raise ValueError, msg
        

    cpdef sph(self, list outputs, bint exclude_self=False):
        assert len(outputs)== 3
        
        cdef DoubleArray op1 = self.dest.get_carray(outputs[0])
        cdef DoubleArray op2 = self.dest.get_carray(outputs[1])
        cdef DoubleArray op3 = self.dest.get_carray(outputs[2])
        
        self.sph_array(op1, op2, op3, exclude_self)

    cdef sph_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                   output3, bint exclude_self=False):
        """
        """
        cdef long dest_pid, source_pid
        cdef LongArray nbrs
        cdef SPHFunctionParticle func
        cdef size_t nsrc, i, j, k, s_idx
        cdef size_t np
        cdef double nr[3], dnr[3]
        cdef double radius_scale
        cdef str msg
        cdef Point xi

        cdef ParticleArray dst = self.dest
        cdef numpy.ndarray xd = dst.x
        cdef numpy.ndarray yd = dst.y
        cdef numpy.ndarray zd = dst.z
        cdef numpy.ndarray hd = dst.h
        
        nsrc = len(self.srcs)
        np = self.dest.get_number_of_particles()

        # make sure the 'output' array is of same size
        if (output1 is None or output1.length != np or output2 is None or
            output2.length != np):
            msg = 'length of output array not equal to number of particles'
            raise ValueError, msg

        for i from 0 <= i < np:
            radius_scale = self.kernel.radius()*hd[i]
            xi = Point(xd[i], yd[i], zd[i])
            dnr[0] = dnr[1] = dnr[2] = 0.0
            nr[0] = nr[1] = nr[2] = 0.0

            exclude_index = -1
            if exclude_self:
                exclude_index = i

            for k from 0 <= k < nsrc:
                nps = self.nbr_locators[k]
                func = self.sph_funcs[k]
                
                indx, dsts = nps.get_nearest_particles(xi, radius_scale, 
                                                       exclude_index)
                nindx = len(indx)

                for j from 0 <= j < nindx:
                    s_idx = indx[j]
                    func.eval(s_idx, i, self.kernel, &nr[0], &dnr[0])

            if dnr[0] != 0.0:
                output1.data[i] = nr[0]/dnr[0]
            else:
                output1.data[i] = nr[0]
            if dnr[1] != 0.0:
                output2.data[i] = nr[1]/dnr[1]
            else:
                output2.data[i] = nr[1]
            if dnr[2] != 0.0:
                output3.data[i] = nr[2]/dnr[2]
            else:
                output3.data[i] = nr[2]

###########################################################################
