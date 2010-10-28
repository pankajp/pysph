cdef extern from "math.h":
    double sqrt(double)

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
                 bint setup_arrays=True, c=-1, alpha=1, beta=1, 
                 gamma=1.4, eta=0.1):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.c = c
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.id = 'momavisc'

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        """
        cdef double piab, muab

        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double pa = self.d_p.data[dest_pid]
        cdef double pb = self.s_p.data[source_pid]

        cdef double rhoab = 0.5*(rhoa + rhob)
        cdef double cab

        cdef double temp = 0.0
        cdef Point grad = Point()
        cdef Point rab, va, vb, vab
        cdef double dot

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

        if vab.dot(rab) < 0.0:
            alpha = self.alpha
            beta = self.beta        
            
            kernel.gradient(self._dst, self._src, h, grad)

            if self.rkpm_first_order_correction:
                grad *= (1 + self.first_order_kernel_correction_term(dest_pid))

            muab = h*vab.dot(rab)/(rab.norm() + 0.01*h*h) 

            if self.c < 1e-14:
                cab = 0.5 * (sqrt(1.4*pa/rhoa) + sqrt(1.4*pb/rhob))
            else:
                cab = self.c

            piab = -muab*(alpha*cab - beta*muab)/rhoab
            piab *= mb
            grad *= piab
                
            nr[0] -= grad.x
            nr[1] -= grad.y
            nr[2] -= grad.z
#############################################################################


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
        self.mu = mu
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays=True)

    cpdef setup_arrays(self):
        """
        """
        self.d_mu = self.dest.get_carray(self.mu)
        self.d_mu = self.source.get_carray(self.mu)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhoa = self.d_rho.data[dest_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double mua = self.d_mu.data[dest_pid]
        cdef double mub = self.s_mu.data[source_pid]

        cdef double temp = 0.0
        cdef Point grad = Point()
        cdef Point rab, va, vb, vab
        cdef double dot

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
        kernel.gradient(self._dst, self._src, h, grad)

        if self.rkpm_first_order_correction:
            grad *= (1 + self.first_order_kernel_correction_term(dest_pid))

        dot = rab.dot(grad)
            
        temp = mb*(mua + mub)*dot/(rhoa*rhob)
        temp /= (rab.norm() + 0.01*h*h)

        nr[0] += temp*vab.x
        nr[1] += temp*vab.y
        nr[2] += temp*vab.z
#############################################################################
