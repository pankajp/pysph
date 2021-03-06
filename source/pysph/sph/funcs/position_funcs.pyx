#cython: cdivision=True
#base imports 
from pysph.base.particle_array cimport ParticleArray, LocalReal
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.kernels cimport KernelBase

###############################################################################
# `PositionStepping' class.
###############################################################################
cdef class PositionStepping(SPHFunction):
    """ Basic position stepping """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, int dim=3):

        SPHFunction.__init__(self, source, dest, setup_arrays)
        
        self.id = 'positionstepper'
        self.tag = "position"

        self.cl_kernel_src_file = "position_funcs.clt"
        self.cl_kernel_function_name = "PositionStepping"
        self.num_outputs = dim

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.dst_reads.extend( ['u','v','w'][:self.num_outputs] )
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        
        cdef LongArray tag_arr = self.dest.get_carray('tag')

        self.setup_iter_data()
        cdef size_t np = self.dest.get_number_of_particles()

        if self.num_outputs == 3:
            for i in range(np):
                if tag_arr.data[i] == LocalReal:
                    output1.data[i] += self.d_u.data[i]
                    output2.data[i] += self.d_v.data[i]
                    output3.data[i] += self.d_w.data[i]
                else:
                    output1.data[i] = output2.data[i] = output3.data[i] = 0
        elif self.num_outputs == 2:
            for i in range(np):
                if tag_arr.data[i] == LocalReal:
                    output1.data[i] += self.d_u.data[i]
                    output2.data[i] += self.d_v.data[i]
                else:
                    output1.data[i] = output2.data[i] = 0
        elif self.num_outputs == 1:
            for i in range(np):
                if tag_arr.data[i] == LocalReal:
                    output1.data[i] += self.d_u.data[i]
                else:
                    output1.data[i] = 0


    def _set_extra_cl_args(self):
        pass
    
    def cl_eval(self, object queue, object context):

        tmpx = self.dest.get_cl_buffer('_tmpx')
        tmpy = self.dest.get_cl_buffer('_tmpy')
        tmpz = self.dest.get_cl_buffer('_tmpz')

        tag = self.dest.get_cl_buffer('tag')
        d_u = self.dest.get_cl_buffer('u')
        d_v = self.dest.get_cl_buffer('v')
        d_w = self.dest.get_cl_buffer('w')

        self.cl_program.PositionStepping(
            queue, self.global_sizes, self.local_sizes, d_u, d_v,
            d_w, tag, tmpx, tmpy, tmpz)
    
##########################################################################
