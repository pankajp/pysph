#cython: cdivision=True
from pysph.base.point cimport cPoint_sub, cPoint, cPoint_dot
from pysph.base.carray cimport DoubleArray

from pysph.solver.cl_utils import HAS_CL
if HAS_CL:
    import pyopencl as cl

import numpy


###############################################################################
# `SPHRho` class.
###############################################################################
cdef class SPHRho(CSPHFunctionParticle):
    """ SPH Summation Density """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     **kwargs)

        self.id = 'sphrho'
        self.tag = "density"

        self.cl_kernel_src_file = "density_funcs.clt"
        self.cl_kernel_function_name = "SPHRho"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m'] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):
        """ Compute the contribution from source_pid on dest_pid. """

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double w, wa, wb

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            wa = kernel.function(self._dst, self._src, ha)
            wb = kernel.function(self._dst, self._src, hb)

            w = 0.5 * (wa + wb)

        else:
            w = kernel.function(self._dst, self._src, hab)        

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/rhob

        nr[0] += w*self.s_m.data[source_pid]

    def _set_extra_cl_args(self):
        pass

    def cl_eval(self, object queue, object context):

        self.set_cl_kernel_args()

        self.cl_program.SPHRho(
            queue, self.global_sizes, self.local_sizes, *self.cl_args).wait()

################################################################################
# `SPHDensityRate` class.
################################################################################
cdef class SPHDensityRate(SPHFunctionParticle):

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     **kwargs)

        self.id = 'densityrate'
        self.tag = "density"

        self.cl_kernel_src_file = "density_funcs.cl"
        self.cl_kernel_function_name = "SPHDensityRate"
        self.num_outputs = 1

    def set_src_dst_reads(self):
        self.src_reads = []
        self.dst_reads = []

        self.src_reads.extend( ['x','y','z','h','m'] )
        self.dst_reads.extend( ['x','y','z','h','tag'] )

        self.src_reads.extend( ['u','v','w'] )
        self.dst_reads.extend( ['u','v','w'] )

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """ Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """

        cdef cPoint vel, grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        cdef DoubleArray xgc, ygc, zgc

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        vel.x = self.d_u.data[dest_pid] - self.s_u.data[source_pid]
        vel.y = self.d_v.data[dest_pid] - self.s_v.data[source_pid]
        vel.z = self.d_w.data[dest_pid] - self.s_w.data[source_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.set((grada.x + gradb.x)*0.5,
                     (grada.y + gradb.y)*0.5,
                     (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)

        nr[0] += cPoint_dot(vel, grad)*self.s_m.data[source_pid]

#############################################################################
