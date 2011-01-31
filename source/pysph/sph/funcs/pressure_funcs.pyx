from pysph.base.point cimport Point_new, Point_sub

cdef extern from "math.h":
    double sqrt(double)

################################################################################
# `SPHPressureGradient` class.
################################################################################
cdef class SPHPressureGradient(SPHFunctionParticle):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.id = 'pgrad'

    cdef void eval(self, int source_pid, int dest_pid, 
                   KernelBase kernel, double *nr, double *dnr):
        """
        """
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double temp = 0.0
        cdef Point grad = Point_new(0,0,0)

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        temp = (pa/(rhoa*rhoa) + pb/(rhob*rhob))
        temp *= -mb
        
        kernel.gradient(self._dst, self._src, h, grad)
        #grad = self.kernel_gradient_evaluation[dest_pid][source_pid]

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z
            
#############################################################################


################################################################################
# `MomentumEquation` class.
################################################################################
cdef class MomentumEquation(SPHFunctionParticle):
    """
        INSERTFORMULA

    """
    #Defined in the .pxd file
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma
    #cdef public double eta

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, alpha=1, beta=1, gamma=1.4, 
                 eta=0.1):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.id = 'momentumequation'
        
    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
    
    
        cdef Point va, vb, vab, rab, grad
        cdef double Pa, Pb, rhoa, rhob, rhoab, mb
        cdef double dot, tmp
        cdef double ca, cb, mu, piab, alpha, beta, eta

        cdef double hab = 0.5*(self.s_h.data[source_pid] + \
                                   self.d_h.data[dest_pid])

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
    
        Pa = self.d_p.data[dest_pid]
        rhoa = self.d_rho.data[dest_pid]        

        Pb = self.s_p.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        mb = self.s_m.data[source_pid]

        tmp = Pa/(rhoa*rhoa) + Pb/(rhob*rhob)
        
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
    
        tmp += piab
        tmp *= -mb

        grad = Point_new(0,0,0)

        #grad = self.kernel_gradient_evaluation[dest_pid][source_pid]

        kernel.gradient(self._dst, self._src, hab, grad)        

        #assert grad.x == other_grad.x
        #assert grad.y == other_grad.y
        #assert grad.z == other_grad.z

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z
        
###############################################################################
