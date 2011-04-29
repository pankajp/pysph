#cython: cdivision=True
cdef extern from "math.h":
    double sqrt(double)

from pysph.base.point cimport cPoint_sub, cPoint_new, cPoint, cPoint_dot, \
        cPoint_norm
################################################################################
# `MonaghanArtificialVsicosity` class.
################################################################################
cdef class MonaghanArtificialVsicosity(SPHFunctionParticle):
    """
        INSERTFORMULA

    """
    #Defined in the .pxd file
    #cdef public double c
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, alpha=1, beta=1, 
                 gamma=1.4, eta=0.1, **kwargs):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.id = 'momavisc'
        self.tag = "velocity"

        self.cl_kernel_src_file = "viscosity_funcs.cl"
        self.cl_kernel_function_name = "MonaghanArtificialVsicosity"

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w','cs']
        self.dst_reads = ['x','y','z','h','p',
                          'u','v','w','cs','rho','tag']

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        cdef cPoint va, vb, vab, rab
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        va = cPoint(self.d_u.data[dest_pid], self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])

        vb = cPoint(self.s_u.data[source_pid], self.s_v.data[source_pid],
                   self.s_w.data[source_pid])

        ca = self.d_cs.data[dest_pid]
        cb = self.s_cs.data[source_pid]
        
        rab = cPoint_sub(self._dst, self._src)
        vab = cPoint_sub(va, vb)
        dot = cPoint_dot(vab, rab)
    
        rhoa = self.d_rho.data[dest_pid]

        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]

        piab = 0
        if dot < 0:
            alpha = self.alpha
            beta = self.beta
            eta = self.eta
            gamma = self.gamma

            cab = 0.5 * (ca + cb)

            rhoab = 0.5 * (rhoa + rhob)

            mu = hab*dot
            mu /= (cPoint_norm(rab) + eta*eta*hab*hab)
            
            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab
    
        tmp = piab
        tmp *= -mb

        grad = cPoint(0,0,0)
        grada = cPoint(0,0,0)
        gradb = cPoint(0,0,0)

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

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z

################################################################################
# `SPHViscosityMomentum` class.
################################################################################
cdef class MorrisViscosity(SPHFunctionParticle):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str mu='mu', *args, **kwargs):
        """
        Constructor.
        """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays=True,
                                     **kwargs)

        self.mu = mu
        self.id = "morrisvisc"
        self.tag = "velocity"

        self.src_reads.extend( ['u','v','w'] )
        self.dst_reads.extend( ['u','v','w','rho'] )

        self.cl_kernel_src_file = "viscosity_funcs.cl"
        self.cl_kernel_function_name = "MorrisViscosity"

    def set_src_dst_reads(self):
        self.src_reads = ['x','y','z','h','m','rho','u','v','w',self.mu]
        self.dst_reads = ['x','y','z','h','rho','u','v','w','tag',self.mu]

    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)

        self.d_mu = self.dest.get_carray(self.mu)
        self.s_mu = self.source.get_carray(self.mu)

        self.src_reads.append(self.mu)
        self.dst_reads.append(self.mu)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        cdef cPoint grad, grada, gradb
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double mua = self.d_mu.data[dest_pid]
        cdef double mub = self.s_mu.data[source_pid]

        cdef double temp = 0.0
        cdef cPoint rab, va, vb, vab
        cdef double dot

        va = cPoint(self.d_u.data[dest_pid], 
                   self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])
        
        vb = cPoint(self.s_u.data[source_pid],
                   self.s_v.data[source_pid],
                   self.s_w.data[source_pid])
        
        vab = cPoint_sub(va,vb)
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab = cPoint_sub(self._dst,self._src)

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

        dot = cPoint_dot(grad, rab)
            
        temp = mb*(mua + mub)*dot/(rhoa*rhob)
        temp /= (cPoint_norm(rab) + 0.01*hab*hab)

        nr[0] += temp*vab.x
        nr[1] += temp*vab.y
        nr[2] += temp*vab.z
#############################################################################
