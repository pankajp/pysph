"""
Definition file for sph_calc
"""

# numpy import
cimport numpy

# standard imports
from pysph.base.nnps cimport NNPSManager, NbrParticleLocatorBase
from pysph.base.kernels cimport KernelBase
from pysph.base.carray cimport DoubleArray, LongArray, IntArray
from pysph.base.particle_array cimport ParticleArray
from pysph.sph.sph_func cimport SPHFunctionParticle

from pysph.sph.kernel_correction cimport KernelCorrectionManager

cdef class SPHCalc:
    """ A general purpose class for SPH calculations. """
    cdef public ParticleArray dest

    cdef public int kernel_correction
    cdef public bint bonnet_and_lok_correction
    cdef public bint rkpm_first_order_correction

    cdef public bint nbr_info

    cdef public list funcs
    cdef public list nbr_locators
    cdef public list sources

    #kernel correction
    cdef public KernelCorrectionManager correction_manager

    cdef public KernelBase kernel    
    cdef public LongArray nbrs 
    cdef public object particles
    cdef public bint integrates
    cdef public list updates, update_arrays
    
    cdef public list from_types, on_types
    cdef public int nupdates
    cdef public int nsrcs
    cdef public str id
    cdef public str tag

    cdef public int dim

    #identifier for the calc's source and destination arrays
    cdef public int dnum
    cdef public str snum

    cdef public NNPSManager nnps_manager

    cpdef sph(self, str output_array1=*, str output_array2=*, 
              str output_array3=*, bint exclude_self=*) 
    
    cpdef sph_array(self, DoubleArray output1, DoubleArray output2,
                    DoubleArray output3, bint exclude_self=*)

    cdef setup_internals(self)
    cpdef check_internals(self)
