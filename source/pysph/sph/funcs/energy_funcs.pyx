cdef extern from "math.h":
    double sqrt(double)

##############################################################################
cdef class EnergyEquationNoVisc(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True):
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.id = 'energyeqn'
    
    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

        """
        cdef double dot, tmp, h
        cdef Point grad = Point()
        cdef Point va, vb, vab

        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double mb = self.s_m.data[source_pid]

        h = 0.5 * (self.d_h.data[dest_pid] + \
                       self.s_h.data[source_pid])

        va = Point(self.d_u.data[dest_pid],
                   self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])
        
        vb = Point(self.s_u.data[source_pid],
                   self.s_v.data[source_pid],
                   self.s_w.data[source_pid])

        vab = va - vb
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        kernel.gradient(self._dst, self._src, h, grad)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        dot = grad.dot(vab)
        tmp = 0.5*mb*(pa/(rhoa*rhoa) + pb/(rhob*rhob))

        nr[0] += tmp*dot
##############################################################################

##############################################################################
cdef class EnergyEquationAVisc(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    #Defined in the .pxd file
    #cdef double beta, alpha, cs, gamma, eta

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=True,  beta=1.0, alpha=1.0, 
                 gamma=1.4, eta=0.1):

        SPHFunctionParticle.__init__(source, dest, setup_arrays)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = eta
        self.id = 'energyavisc'
    
    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

        """

        cdef double test, gamma, alpha, beta, cs
        cdef double pa, rhoa, pb, rhob, cab, h, mu, prod, rhoab
        cdef Point rab, grad, va, vb, vab

        va = Point(self.d_u.data[dest_pid],
                   self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])
        
        vb = Point(self.s_u.data[source_pid],
                   self.s_v.data[source_pid],
                   self.s_w.data[source_pid])
        
        vab = va - vb
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = self._dst - self._src
        test = vab.dot(rab)

        grad = Point()
        
        if test < 0.0:
            gamma = self.gamma 
            alpha = self.alpha
            beta = self.beta
            cs = self.cs
            eta = self.eta

            pa = self.d_p.data[dest_pid]
            pb = self.s_p.data[source_pid]
            rhoa = self.d_rho.data[dest_pid]
            rhob = self.s_rho.data[source_pid]
            mb = self.s_m.data[source_pid]
            h = 0.5 * (self.d_h.data[dest_pid] + \
                           self.s_h.data[source_pid])

            cab = 0.5*(self.d_cs.data[dest_pid] + self.s_cs.data[source_pid])

            rhoab = 0.5 * (rhoa + rhob)

            mu = (h * test) / (rab.norm() + eta*eta*h*h)
            kernel.gradient(self._dst, self._src, h, grad)

            if self.rkpm_first_order_correction:
                pass

            if self.bonnet_and_lok_correction:
                self.bonnet_and_lok_gradient_correction(dest_pid, grad)

            prod  = (-alpha*cab*mu + beta*mu*mu)/(rhoab)
            nr[0] += 0.5 * mb * prod * vab.dot(grad)
        else:
                pass
##############################################################################


################################################################################
# `EnergyEquation` class.
################################################################################
cdef class EnergyEquation(SPHFunctionParticle):
    """
        INSERTFORMULA

    """
    #Defined in the .pxd file
    #cdef public double alpha
    #cdef public double beta
    #cdef public double gamma
    #cdef public double eta

    def __init__(self, ParticleArray source, dest,  bint setup_arrays=True,
                 alpha=1.0, beta=1.0, gamma=1.4, eta=0.1):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.id = 'energyequation'
        
    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
    
    
        cdef Point va, vb, vab, rab, grad, tmp1
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
        
        va = Point(self.d_u.data[dest_pid], self.d_v.data[dest_pid],
                   self.d_w.data[dest_pid])

        vb = Point(self.s_u.data[source_pid], self.s_v.data[source_pid],
                   self.s_w.data[source_pid])
        
        rab = self._dst - self._src
        vab = va - vb
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

            cab = 0.5 * (self.d_cs.data[dest_pid] + self.s_cs.data[source_pid])

            rhoab = 0.5 * (rhoa + rhob)

            mu = hab*dot
            mu /= (rab.norm() + eta*eta*hab*hab)
            
            piab = -alpha*cab*mu + beta*mu*mu
            piab /= rhoab

        grad = Point()
            
        kernel.gradient(self._dst, self._src, hab, grad)

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, grad)

        tmp += piab
        
        tmp1 = vab * tmp
        
        tmp = tmp1.dot(grad)

        nr[0] += 0.5*mb*tmp
        
###############################################################################
