"""
General purpose code for SPH computations.

This module provdies the SPHBase class, which does the actual SPH summation.


**TODO**

 - Particle filtering functions : Instead of checking the 'tag' property by
   default, for every particle in the destination, abstract it out to a
   function. This function in the SPHBase implementation will check for the
   'tag' property before considering them for SPH interpolation. Derived
   classes if needed may check for any other property as desired.

   Similarly, for each neighbor of a given particle (that has passed the prior
   check), check if you want that neighbor to be included in the SPH
   interpolation. 

   Thus we have two functions - dest_particle_check and source_particle_check,
   which are called before considering each dest and each source particle. This
   could provide a cleaner method for implementing other SPH formulations, where
   some particles may need to be discarded.

   This will however introduce those extra calls for every particle, which
   will have a performance hit.

   The other option to this is to implement the whole sph summation again in the
   derived class(instead of just the filter functions). It will definitely be
   faster, but will involve more code(much of which will be similar) and more
   maintainence.
     
"""

#include "stdlib.pxd"
from stdlib cimport *

cimport numpy

# logging import
import logging
logger=logging.getLogger()

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.point cimport Point
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator
from pysph.base.nnps cimport NbrParticleLocatorBase
from pysph.base.particle_tags cimport *

from pysph.sph.sph_func cimport SPHFunctionParticle

###############################################################################
# `SPHBase` class.
###############################################################################
cdef class SPHBase:
    """
    A general purpose class for SPH calculations.

    **Members**

     - sources - a list of source particle arrays involved in this computation.
     - dest - the particle array containing points where interpolation needs to
       be done. 
     - sph_funcs - a list of SPHFunctionParticle classes, one for each (source,
     - dest) pair. Each element of this list should be of the same class.
     - h - name of the array containing the particles interaction radius
       property.
     - nbr_locators - a list of neighbor locator for each (source, dest)
       pair. 
     - kernel - the kernel to be used in this calculation.
     - nnps_manager - the NNPSManager to be used to get neighbor locators.

    """    
    def __cinit__(self, list sources=[], ParticleArray dest=None, KernelBase
                 kernel=None, list sph_funcs=[], NNPSManager nnps_manager=None,
                 str h='h', bint setup_internals=True, *args, **kwargs): 
        """
        Constructor.
        """

        self.sources = []
        self.sources[:] = sources
        self.nbr_locators = []
        self.sph_funcs = []
        self.sph_funcs[:] = sph_funcs
        
        self.dest = dest
        self.kernel = kernel

        self.nnps_manager=nnps_manager
        self.h = h

        self.valid_call = 0

        if setup_internals:
            self.setup_internals()

    cpdef setup_internals(self):
        """
        Setup internals.
        """
        # check if the data is sane.
        if (len(self.sources) == 0 or self.nnps_manager is None
            or self.dest is None or self.kernel is None or len(self.sph_funcs)
            == 0):
            logger.warn('invalid input to setup_internals')
            logger.info('sources : %s'%(self.sources))
            logger.info('nnps_manager : %s'%(self.nnps_manager))
            logger.info('dest : %s'%(self.dest))
            logger.info('kernel : %s'%(self.kernel))
            logger.info('sph_funcs : %s'%(self.sph_funcs))                        
            
            return

        # we need one sph_func for each source.
        if len(self.sph_funcs) != len(self.sources):
            msg = 'One sph function is needed per source'
            raise ValueError, msg

        # now make sure all the sph_funcs are of the same class.
        # and the sources and dest are matched.
        sph_funcs = self.sph_funcs

        for i in range(len(self.sph_funcs)-1):
            if type(sph_funcs[i]) != type(sph_funcs[i+1]):
                msg = 'All sph_funcs should be of same type'
                raise ValueError, msg
            
        for i in range(len(self.sph_funcs)):
            if sph_funcs[i].source != self.sources[i]:
                msg = 'SPHFunctionParticle.source not same as'
                msg += ' SPHBase.sources[%d]'%(i)
                raise ValueError, msg

            if sph_funcs[i].dest != self.dest:
                msg = 'SPHFunctionParticle.dest not same as'
                msg += ' SPHBase.dest'
                raise ValueError, msg

        # now decide, based on the output field requirements of the supplied
        # sphfuncs, which sph* call is valid for this object.
        f0 = self.sph_funcs[0]
        self.valid_call = f0.output_fields()

        if self.valid_call > 3:
            self.valid_call = -1
        
        # create the neighbor locators
        cdef NbrParticleLocatorBase loc
        self.nbr_locators[:] = []
        for src in self.sources:
            loc = self.nnps_manager.get_neighbor_particle_locator(
                src, self.dest, self.kernel.radius())
            logger.info('Using locator : %s, %s, %s'%(src.name, self.dest.name, loc))
            self.nbr_locators.append(loc)
    
    cpdef sph1(self, str output_array, bint exclude_self=False):
        """
        Performs SPH summation using sphfunc to compute contribution of neighbor
        particles. The property computed is either a scalar or a component of a
        vector. The computed property is stored in the array with name given in 
        the parameter output_array.

        """
        
        cdef DoubleArray output = self.dest.get_carray(output_array)
        self.sph1_array(output, exclude_self)

    cpdef sph1_array(self, DoubleArray output, bint exclude_self=False):
        """
        Performs SPH summation using sphfunc to compute contribution of neighbor
        particles. The property computed is either a scalar or a component of a
        vector. The computed property is stored in the array 'output'.

        **Parameters**
        
         - output - the array to store the output values into
         - exclude_self - indicates if each particle itself should be left out
           of the computations.
         - sphfunc - the function to be used for the particle-particle
           interaction. 

        """
        cdef long dest_pid, source_pid
        cdef FixedDestNbrParticleLocator nbr_loc
        cdef LongArray nbrs
        cdef SPHFunctionParticle func
        cdef LongArray tag_array
        cdef long *tag        
        cdef size_t nsrc, i, j, k, s_idx
        cdef size_t np
        cdef double nr, dnr
        cdef double radius_scale
        cdef str msg
        
        # check if this call is valid.
        if self.valid_call != 1:
            msg = 'sph1 cannot be called for the SPHFunctionParticle used'
            raise ValueError, msg

        nsrc = len(self.sources)
        np = self.dest.get_number_of_particles()
        tag_array = self.dest.get_carray('tag')
        tag = tag_array.get_data_ptr()
        nbrs = LongArray()

        # make sure the 'output' array is of same
        # size as the number of particles.
        if output is None or output.length != np:
            msg = 'length of output array not equal to number of particles'
            raise ValueError, msg

        radius_scale = self.kernel.radius()

        for i in range(nsrc):
            nbr_loc = self.nbr_locators[i]
            func = self.sph_funcs[i]
            for j from 0 <= j < np:

                if tag[j] != LocalReal:
                    continue

                dnr = 0.0
                nr = 0.0
                
                nbrs.reset()
                nbr_loc.get_nearest_particles(j, nbrs, exclude_self)
                
                for k from 0 <= k < nbrs.length:
                    s_idx = nbrs.get(k)
                    func.eval(s_idx, j, self.kernel, &nr, &dnr)

                if dnr != 0.0:
                    output.data[j] = nr/dnr
                else:
                    output.data[j] = nr
    
    cpdef sph2(self, str output_array1, str output_array2, bint exclude_self=False):
        """
        Similar to the sph1 function, except that this can handle
        SPHFunctionParticle that compute 2 output fields. For example, a
        SPHFunctionParticle that computes the acceleration due to pressure
        gradient will compute the x and y components of the acceleration and
        store them.

        **Parameters**
        
         - output_array1 - name of the array to hold the first component.
         - output_array2 - name of the array to hold the second component.
         - exclude_self - indicates if each particle itself should be left out
         - of the computation.

        """
        cdef DoubleArray output1 = self.dest.get_carray(output_array1)
        cdef DoubleArray output2 = self.dest.get_carray(output_array2)
        
        self.sph2_array(output1, output2, exclude_self)

    cpdef sph2_array(self, DoubleArray output1, DoubleArray output2, bint
                     exclude_self=False):
        """
        Similar to the sph1 function, except that this can handle
        SPHFunctionParticle that compute 2 output fields. For example, a
        SPHFunctionParticle that computes the acceleration due to pressure
        gradient will compute the x and y components of the acceleration and
        store them.

        **Parameters**
        
         - output1 - the array to store the first output component.
         - output2 - the array to store the second output component.
         - exclude_self - indicates if each particle itself should be left out
           of the computations.

        """
        cdef long dest_pid, source_pid
        cdef FixedDestNbrParticleLocator nbr_loc
        cdef LongArray nbrs
        cdef SPHFunctionParticle func
        cdef LongArray tag_array
        cdef long *tag        
        cdef size_t nsrc, i, j, k, s_idx
        cdef size_t np
        cdef double nr[2], dnr[2]
        cdef double radius_scale
        cdef str msg
        
        # check if this call is valid.
        if self.valid_call != 2:
            msg = 'sph2 cannot be called for the SPHFunctionParticle used'
            raise ValueError, msg

        nsrc = len(self.sources)
        np = self.dest.get_number_of_particles()
        tag_array = self.dest.get_carray('tag')
        tag = tag_array.get_data_ptr()
        nbrs = LongArray()

        # make sure the 'output' array is of same
        # size as the number of particles.
        if (output1 is None or output1.length != np or output2 is None or
            output2.length != np):
            msg = 'length of output array not equal to number of particles'
            raise ValueError, msg

        radius_scale = self.kernel.radius()

        for i in range(nsrc):
            nbr_loc = self.nbr_locators[i]
            func = self.sph_funcs[i]

            for j from 0 <= j < np:

                if tag[j] != LocalReal:
                    continue

                dnr[0] = dnr[1] = 0.0
                nr[0] = nr[1] = 0.0
                
                nbrs.reset()
                nbr_loc.get_nearest_particles(j, nbrs, exclude_self)
                
                for k from 0 <= k < nbrs.length:
                    s_idx = nbrs.get(k)
                    func.eval(s_idx, j, self.kernel, &nr[0], &dnr[0])

                if dnr[0] != 0.0:
                    output1.data[j] = nr[0]/dnr[0]
                else:
                    output1.data[j] = nr[0]

                if dnr[1] != 0.0:
                    output2.data[j] = nr[1]/dnr[1]
                else:
                    output2.data[j] = nr[1]

    cpdef sph3(self, str output_array1, str output_array2, str output_array3,
               bint exclude_self=False): 
        """
        """
        cdef DoubleArray output1 = self.dest.get_carray(output_array1)
        cdef DoubleArray output2 = self.dest.get_carray(output_array2)
        cdef DoubleArray output3 = self.dest.get_carray(output_array3)
        
        self.sph3_array(output1, output2, output3, exclude_self)
        

    cpdef sph3_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
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
        cdef long dest_pid, source_pid
        cdef FixedDestNbrParticleLocator nbr_loc
        cdef LongArray nbrs
        cdef SPHFunctionParticle func
        cdef LongArray tag_array
        cdef long *tag        
        cdef size_t nsrc, i, j, k, s_idx
        cdef size_t np
        cdef double nr[3], dnr[3]
        cdef double radius_scale
        cdef str msg
        
        # check if this call is valid.
        if self.valid_call != 3:
            msg = 'sph3 cannot be called for the SPHFunctionParticle used'
            raise ValueError, msg

        nsrc = len(self.sources)
        np = self.dest.get_number_of_particles()
        tag_array = self.dest.get_carray('tag')
        tag = tag_array.get_data_ptr()
        nbrs = LongArray()

        # make sure the 'output' array is of same
        # size as the number of particles.
        if (output1 is None or output1.length != np or output2 is None or
            output2.length != np):
            msg = 'length of output array not equal to number of particles'
            raise ValueError, msg

        radius_scale = self.kernel.radius()

        for i in range(nsrc):
            nbr_loc = self.nbr_locators[i]
            func = self.sph_funcs[i]

            for j from 0 <= j < np:

                if tag[j] != LocalReal:
                    continue

                dnr[0] = dnr[1] = dnr[2] = 0.0
                nr[0] = nr[1] = nr[2] = 0.0
                
                nbrs.reset()
                nbr_loc.get_nearest_particles(j, nbrs, exclude_self)
                
                for k from 0 <= k < nbrs.length:
                    s_idx = nbrs.get(k)
                    func.eval(s_idx, j, self.kernel, &nr[0], &dnr[0])

                if dnr[0] != 0.0:
                    output1.data[j] = nr[0]/dnr[0]
                else:
                    output1.data[j] = nr[0]
                if dnr[1] != 0.0:
                    output2.data[j] = nr[1]/dnr[1]
                else:
                    output2.data[j] = nr[1]
                if dnr[2] != 0.0:
                    output3.data[j] = nr[2]/dnr[2]
                else:
                    output3.data[j] = nr[2]

    cpdef sphn(self, list op_names, bint exclude_self=False):
        """
        """
        cdef list arr_lst = []
        cdef int num_arrays, i
        cdef str name
        cdef DoubleArray arr
        num_arrays = len(op_names)

        for i in range(num_arrays):
            name = op_names[i]
            arr = self.dest.get_carray(name)
            arr_lst.append(arr)

        self.sphn_array(arr_lst, exclude_self)

    cpdef sphn_array(self, list op_arrays, bint exclude_self=False):
        """
        """
        """
        Similar to the sph1 function, except that this can handle
        SPHFunctionParticle that compute any number of output fields greater
        than 3. When output fields are less than or equal to 3 , use the
        functions defined above. This function will be slower than the others.

        **Parameters**
         
         - op_arrays - list of DoubleArray where the output needs to be stored.
         - exclude_self - indicates if each particle itself should be left out
           of the computations.

        """
        cdef long dest_pid, source_pid
        cdef FixedDestNbrParticleLocator nbr_loc
        cdef LongArray nbrs
        cdef SPHFunctionParticle func
        cdef LongArray tag_array
        cdef long *tag        
        cdef size_t nsrc, i, j, k, s_idx, num_fields
        cdef size_t np
        cdef double *nr, *dnr, *data
        cdef double radius_scale
        cdef str msg
        cdef DoubleArray output_array
        
        # check if this call is valid.
        if self.valid_call != -1:
            msg = 'sphn cannot be called for the SPHFunctionParticle used'
            raise ValueError, msg

        nsrc = len(self.sources)
        np = self.dest.get_number_of_particles()
        tag_array = self.dest.get_carray('tag')
        tag = tag_array.get_data_ptr()
        nbrs = LongArray()

        func = self.sph_funcs[0]
        num_fields = func.output_fields()

        # make sure that we have the required number of output arrays.
        if len(op_arrays) != num_fields:
            msg = 'Number of output arrays should be the same as the number of'
            msg += ' output fields.'
            raise ValueError, msg
        
        # make sure length of each array is equal to number of particles in
        # dest.
        num_arrays = len(op_arrays)
        for i in range(num_arrays):
            output_array = op_arrays[i]
            if output_array.length != np:
                msg = 'Length of output array not equal to number of particles' 
                raise ValueError, msg

        # allocate data for the numerator and denominator.
        nr = <double*>malloc(sizeof(double)*num_fields)
        dnr = <double*>malloc(sizeof(double)*num_fields)
        radius_scale = self.kernel.radius()

        for i in range(nsrc):
            nbr_loc = self.nbr_locators[i]
            func = self.sph_funcs[i]

            for j from 0 <= j < np:

                if tag[j] != LocalReal:
                    continue

                for k from 0 <= k < num_fields:
                    dnr[k] = 0.0
                    nr[k] = 0.0
                
                nbrs.reset()
                nbr_loc.get_nearest_particles(j, nbrs, exclude_self)
                
                for k from 0 <= k < nbrs.length:
                    s_idx = nbrs.get(k)
                    func.eval(s_idx, j, self.kernel, &nr[0], &dnr[0])

                # now accumulate the results into the arrays.
                for k from 0 <= k < num_fields:
                    output_array = op_arrays[k]
                    if dnr[k] != 0.0:
                        output_array.data[j] = nr[k]/dnr[k]
                    else:
                        output_array.data[j] = nr[k]
        
        # free data
        free(<void*>nr)
        free(<void*>dnr)
