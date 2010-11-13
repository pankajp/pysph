from pysph.base.point cimport Point_new, Point_sub

###############################################################################
# `XSPHCorrection' class.
###############################################################################
cdef class XSPHCorrection(SPHFunctionParticle):
    """ Basic XSPH """

    #Defined in the .pxd file

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 bint setup_arrays=True, double eps = 0.5):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.eps = eps
        self.id = 'xsph'

    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        The expression used is:

        """

        cdef double temp, w

        cdef double h=0.5*(self.s_h.data[source_pid] + \
                               self.d_h.data[dest_pid])

        cdef double rhoab = 0.5*(self.s_rho.data[source_pid] + \
                                     self.d_rho.data[dest_pid])

        cdef Point Va = Point_new(self.d_u.data[dest_pid],
                              self.d_v.data[dest_pid],
                              self.d_w.data[dest_pid])

        cdef Point Vb = Point_new(self.s_u.data[source_pid],
                              self.s_v.data[source_pid],
                              self.s_w.data[source_pid])

        cdef Point Vba = Point_sub(Vb, Va)

        cdef double mb = self.s_m.data[source_pid]

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
            dnr[0] += w*mb/self.s_rho.data[source_pid]

        temp = mb * w/rhoab

        nr[0] += temp*Vba.x*self.eps
        nr[1] += temp*Vba.y*self.eps
        nr[2] += temp*Vba.z*self.eps
        
##########################################################################


###############################################################################
# `XSPHDensityRate' class.
###############################################################################
cdef class XSPHDensityRate(SPHFunctionParticle):
    """ Basic XSPHDensityRate """

    #Defined in the .pxd file
    #cdef DoubleArray s_ubar, s_vbar, s_wbar

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None or self.dest is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        #Setup the XSPH correction terms
        self.s_ubar = self.array.get_carray('ubar')
        self.s_vbar = self.array.get_carray('vbar')
        self.s_wbar = self.array.get_carray('wbar')

    cdef void eval(self, int source_pid, int dest_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """
        Perform an SPH intterpolation of the property `prop_name`

        The expression used is:

        """
        cdef Point grad = Point()
        cdef double h=0.5*(self.s_h.data[source_pid] + \
                               self.d_h.data[dest_pid])

        cdef Point Va = Point(self.d_u.data[dest_pid]+ \
                                  self.d_ubar.data[dest_pid],

                              self.d_v.data[dest_pid]+ \
                                  self.d_vbar.data[dest_pid],

                              self.d_w.data[dest_pid]+ \
                                  self.d_wbar.data[dest_pid])

        cdef Point Vb = Point(self.s_u.data[source_pid]+ \
                                  self.s_ubar.data[source_pid],

                              self.s_v.data[source_pid]+ \
                                  self.s_vbar.data[source_pid],
                              
                              self.s_w.data[source_pid]+ \
                                  self.s_wbar.data[source_pid])

        cdef Point Vab = Va - Vb
        cdef double mb = self.s_m.data[source_pid]
        cdef double temp

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

        temp = Vab.dot(grad)
        
        nr[0] += temp*mb
###############################################################################
