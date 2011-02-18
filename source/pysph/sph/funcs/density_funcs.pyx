from pysph.base.point cimport Point_new, Point_sub
from pysph.base.carray cimport DoubleArray 

###############################################################################
# `SPHRho` class.
###############################################################################
cdef class SPHRho(SPHFunctionParticle):
    """ SPH Summation Density """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     **kwargs)

        self.id = 'sphrho'
        self.tag = "density"

    cdef void eval(self, int k, int source_pid, int dest_pid, 
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
###############################################################################

################################################################################
# `SPHDensityRate` class.
################################################################################
cdef class SPHDensityRate(SPHFunctionParticle):
    """
    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     **kwargs)

        self.id = 'densityrate'
        self.tag = "density"

    cdef void eval(self, int k, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """

        cdef Point vel, grad, grada, gradb

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
            
        vel = Point_new(0,0,0)

        grad = Point_new(0,0,0)
        grada = Point_new(0,0,0)
        gradb = Point_new(0,0,0)

        vel.x = self.d_u.data[dest_pid] - self.s_u.data[source_pid]
        vel.y = self.d_v.data[dest_pid] - self.s_v.data[source_pid]
        vel.z = self.d_w.data[dest_pid] - self.s_w.data[source_pid]

        if self.hks:
            kernel.gradient(self._dst, self._src, ha, grada)
            kernel.gradient(self._dst, self._src, hb, gradb)

            grad = (grada + gradb) * 0.5

        else:            
            kernel.gradient(self._dst, self._src, hab, grad)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        nr[0] += vel.dot(grad)*self.s_m.data[source_pid]

#############################################################################
