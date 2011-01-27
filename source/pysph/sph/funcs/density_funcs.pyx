from pysph.base.point cimport Point_new, Point_sub

###############################################################################
# `SPHRho` class.
###############################################################################
cdef class SPHRho(SPHFunctionParticle):
    """ SPH Summation Density """
    cdef void eval(self, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """ Compute the contribution from source_pid on dest_pid. """

        cdef double h = 0.5*(self.s_h.data[source_pid] + \
                                 self.d_h.data[dest_pid])

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double w

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
            
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        #w = kernel.function(self._dst, self._src, h)
        w = self.kernel_function_evaluation[dest_pid][source_pid]

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
    cdef void eval(self, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """

        cdef Point vel, grad        
        cdef double h = 0.5*(self.s_h.data[source_pid] + \
                                 self.d_h.data[dest_pid])

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        vel = Point_new(0,0,0)
        grad = Point_new(0,0,0)

        vel.x = self.d_u.data[dest_pid] - self.s_u.data[source_pid]
        vel.y = self.d_v.data[dest_pid] - self.s_v.data[source_pid]
        vel.z = self.d_w.data[dest_pid] - self.s_w.data[source_pid]


        grad = self.kernel_gradient_evaluation[dest_pid][source_pid]
        #kernel.gradient(self._dst, self._src, h, grad)
        
        #assert grad.x == other_grad.x
        #assert grad.y == other_grad.y
        #assert grad.z == other_grad.z

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        nr[0] += vel.dot(grad)*self.s_m.data[source_pid]

#############################################################################
