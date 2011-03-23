#cython: cdivision=True
#base imports 
from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.kernels cimport KernelBase
#from pysph.sph.sph_funcs import SPHFunction

###############################################################################
# `PositionStepping' class.
###############################################################################
cdef class PositionStepping(SPHFunction):
    """ Basic position stepping """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True):

        SPHFunction.__init__(self, source, dest, setup_arrays)
        
        self.id = 'positionstepper'
        self.tag = "position"

    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()
        
        for i in range(np):
            if tag_arr.data[i] == LocalReal:
                output1[i] = self.d_u.data[i]
                output2[i] = self.d_v.data[i]
                output3[i] = self.d_w.data[i]
            else:
                output1[i] = output2[i] = output3[i] = 0
    
##########################################################################
