""" File to hold the functions required for the ADKE procedure of Sigalotti """

from pysph.base.point cimport cPoint, cPoint_new, cPoint_sub, cPoint_dot

###############################################################################
# `PilotRho` class.
###############################################################################
cdef class PilotRho(SPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True, double h0=1.0, **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)
        self.h0 = h0

        self.id = "pilotrho"
        self.tag = "adke"
    
    cdef void eval(self, int k, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """ Compute the contribution from source_pid on dest_pid.

        The expression used is:

        ..math::

        <\rho_p> = \sum_{b=1}^{b=N} m_b\, W(x_a-x_b, h0)

        h0 is a constant that is set upon instance creation.

        """
        cdef double h = self.h0

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double w

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        w = kernel.function(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/rhob

        nr[0] += w*self.s_m.data[source_pid]        

###############################################################################
# `SPHDivergence` class.
###############################################################################
cdef class SPHVelocityDivergence(SPHFunctionParticle):
    """ Compute the pilot estimate of density for the ADKE algorithm """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 setup_arrays=True, **kwargs):
        
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays,
                                     **kwargs)

        self.id = "vdivergence"
        self.tag = "vdivergence"

    cdef void eval(self, int k, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """ Compute the contribution from source_pid on dest_pid.

        The expression used is:

        ..math::

        <\nabla\,\cdot v>_a = \frac{1}{\rho_a}\sum_{b=1}^{b=N} m_b\,
        (v_b-v_a)\cdot\,\nabla_a\,W_{ab}

        h0 is a constant that is set upon instance creation.

        """

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double h = 0.5 * (ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]

        cdef cPoint grad, grada, gradb, vba

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        vba = cPoint(self.s_u.data[source_pid] - self.d_u.data[dest_pid],
                    self.s_v.data[source_pid] - self.d_v.data[dest_pid],
                    self.s_w.data[source_pid] - self.d_w.data[dest_pid])

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.set((grada.x + gradb.x)*0.5,
                     (grada.y + gradb.y)*0.5,
                     (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, h)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        nr[0] += (1.0/rhoa) * mb * cPoint_dot(grad, vba)
