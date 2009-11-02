"""
Definition file for sph_calc
"""

# numpy import
cimport numpy

# standard imports
from pysph.base.nnps cimport NNPSManager
from pysph.base.kernelbase cimport KernelBase
from pysph.base.carray cimport DoubleArray
from pysph.base.particle_array cimport ParticleArray

cdef class SPHBase:
    """
    A general purpose class for SPH calculations.
    """
    cdef public list sources
    cdef public ParticleArray dest
    cdef public list nbr_locators
    cdef public list sph_funcs
    cdef public str h
    cdef public int valid_call

    cdef public KernelBase kernel

    cdef public NNPSManager nnps_manager

    cpdef sph1(self, str output_array, bint exclude_self=*)
    cpdef sph1_array(self, DoubleArray output, bint exclude_self=*)

    cpdef sph2(self, str output_array1, str output_array2, bint
               exclude_self=*)
    cpdef sph2_array(self, DoubleArray output1, DoubleArray output2, bint
                     exclude_self=*)

    cpdef sph3(self, str output_array1, str output_array2, 
               str output_array3, bint exclude_self=*) 
    cpdef sph3_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                     output3, bint exclude_self=*)

    cpdef sphn(self, list output_arrays, bint exclude_self=*)
    cpdef sphn_array(self, list output_arrays, bint exclude_self=*)

    cpdef setup_internals(self)
    
    
