"""
Definition file for sph_calc
"""

# numpy import
cimport numpy

# standard imports
from pysph.base.nnps cimport NNPSManager, NbrParticleLocatorBase
from pysph.base.kernels cimport MultidimensionalKernel
from pysph.base.carray cimport DoubleArray, LongArray, IntArray
from pysph.base.particle_array cimport ParticleArray
from pysph.sph.sph_func cimport SPHFunctionParticle

cdef class SPHBase:
    """ A general purpose class for SPH calculations. """
    cdef public ParticleArray dest

    cdef public int dnum
    cdef public bint nbr_info

    cdef public list funcs
    cdef public list nbr_locators
    cdef public list sources

    cdef public MultidimensionalKernel kernel    
    cdef public LongArray nbrs 
    cdef public object particles
    cdef public bint integrates
    cdef public list updates, update_arrays
    
    cdef public list from_types, on_types
    cdef public int nupdates
    cdef public int nsrcs
    cdef public str id

    cdef NNPSManager nnps_manager

    cpdef sph(self, str output_array1=*, str output_array2=*, 
              str output_array3=*, bint exclude_self=*) 
    
    cpdef sph_array(self, DoubleArray output1, DoubleArray output2,
                    DoubleArray output3, bint exclude_self=*)

    cdef eval(self, size_t i, double* nr, double* dnr, bint exclude_self)
    cdef setup_internals(self)
    cpdef check_internals(self)
    
cdef class SPHCalc(SPHBase):
    """ """
    pass

cdef class SPHEquation(SPHBase):
    """ """
    pass
