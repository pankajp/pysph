""" Dfinitions for the SPH summation module """

# numpy import
cimport numpy

# standard imports
from pysph.base.kernels cimport MultidimensionalKernel
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPS
from pysph.sph.sph_func cimport SPHFunctionParticle
from pysph.base.point cimport Point

#############################################################################
#`SPHCalc` class
#############################################################################
cdef class SPHCalc:
    """ A general purpose class for SPH calculations. """
    
    #The destination particle array for the computation.
    cdef public ParticleArray dest

    #The list of source particle managers for the destination.
    cdef public list srcs

    #The list of NNPS's for each source/dest pair.
    cdef public list nbr_locators
    
    #The list of sph_funcs for each source/dest pair.
    cdef public list sph_funcs

    #The kernel used for the interpolation.
    cdef public MultidimensionalKernel kernel

    #String literal for the named variable `smoothing length`
    cdef public str h

    #The dimension
    cdef public int dim

    #########################################################################
    #Member functons.
    #########################################################################
    cpdef setup_nnps(self)

    cpdef sph(self, list outputs, bint exclude_self=*)
    cdef sph_array(self, DoubleArray output1, DoubleArray output2, DoubleArray
                    output3, bint exclude_self=*)

#############################################################################
