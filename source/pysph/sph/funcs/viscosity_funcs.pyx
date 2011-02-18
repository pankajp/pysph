cdef extern from "math.h":
    double sqrt(double)

from pysph.base.point cimport Point_sub, Point_new
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

    cdef void eval(self, int k, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        """
        cdef Point va, vb, vab, rab
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef Point grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        va = Point_new(self.d_u.data[dest_pid], self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])

        vb = Point_new(self.s_u.data[source_pid], self.s_v.data[source_pid],
                   self.s_w.data[source_pid])

        ca = self.d_cs.data[dest_pid]
        cb = self.s_cs.data[source_pid]
        
        rab = Point_sub(self._dst, self._src)
        vab = Point_sub(va, vb)
        dot = vab.dot(rab)
    
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
            mu /= (rab.norm() + eta*eta*hab*hab)
            
            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab
    
        tmp = piab
        tmp *= -mb

        grad = Point_new(0,0,0)
        grada = Point_new(0,0,0)
        gradb = Point_new(0,0,0)

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


    cpdef setup_arrays(self):
        """
        """
        SPHFunctionParticle.setup_arrays(self)

        self.d_mu = self.dest.get_carray(self.mu)
        self.s_mu = self.source.get_carray(self.mu)

    cdef void eval(self, int k, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """
        """
        cdef Point grad, grada, gradb
        
        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]
        
        cdef double hab = 0.5*(ha + hb)

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]

        cdef double mua = self.d_mu.data[dest_pid]
        cdef double mub = self.s_mu.data[source_pid]

        cdef double temp = 0.0
        cdef Point rab, va, vb, vab
        cdef double dot

        va = Point_new(self.d_u.data[dest_pid], 
                       self.d_v.data[dest_pid],
                       self.d_w.data[dest_pid])
        
        vb = Point_new(self.s_u.data[source_pid],
                       self.s_v.data[source_pid],
                       self.s_w.data[source_pid])
        
        vab = Point_sub(va,vb)
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        rab = Point_sub(self._dst,self._src)

        grad = Point_new(0,0,0)
        grada = Point_new(0,0,0)
        gradb = Point_new(0,0,0)
        
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

        dot = rab.dot(grad)
            
        temp = mb*(mua + mub)*dot/(rhoa*rhob)
        temp /= (rab.norm() + 0.01*hab*hab)

        nr[0] += temp*vab.x
        nr[1] += temp*vab.y
        nr[2] += temp*vab.z
#############################################################################
