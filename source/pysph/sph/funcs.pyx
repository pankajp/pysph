""" Implementations for the various sph functions. """


################################################################################
# `SPH` class.
################################################################################
cdef class SPH(SPHFunctionParticle3D):
    """ Basic SPH Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str prop_name='',  *args, **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle3D.__init__(self, source, dest)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None or self.dest is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle3D.setup_arrays(self)

        self.d_prop = self.source.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}
            
        """

        cdef double h=0.5*(self.s_h.data[source_pid] + 
                           self.d_h.data[dest_pid])
        
        cdef Point src_position = self._src
        cdef Point dst_position = self._dst

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
    
        cdef double w = kernel.function(dst_position, src_position, h)
        cdef double temp = w*self.s_mass.data[source_pid]/(
            self.s_rho.data[source_pid])
        
        nr[0] += self.s_prop[source_pid]*temp
###########################################################################


###############################################################################
# `SPHRho3D` class.
###############################################################################
cdef class SPHRho(SPHFunctionParticle3D):
    """
    SPH function to compute density for 3d particles.

    All 3 coordinate arrays should be available.
    """
    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] + \
                                 self.d_h.data[dest_pid])

        cdef Point src_position = self._src
        cdef Point dst_position = self._dst

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        cdef double w = kernel.function(dst_position, src_position, h)
        
        nr[0] += w*self.s_mass.data[source_pid]

################################################################################
# `SPHDensityRate` class.
################################################################################
cdef class SPHDensityRate(SPHFunctionParticle3D):
    """
    """
    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid.
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] + \
                                 self.d_h.data[dest_pid])
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef Point vel = Point()
        cdef Point grad = Point()

        vel.x = self.d_velx.data[dest_pid] - self.s_velx.data[source_pid]
        vel.y = self.d_vely.data[dest_pid] - self.s_vely.data[source_pid]
        vel.z = self.d_velz.data[dest_pid] - self.s_velz.data[source_pid]

        kernel.gradient(self._dst, self._src, h, grad)
        
        nr[0] += vel.dot(grad)*self.s_mass.data[source_pid]

#############################################################################

################################################################################
# `SPHPressureGradient` class.
################################################################################
cdef class SPHPressureGradient(SPHFunctionParticle3D):
    """
    Computes pressure gradient using the formula 

        INSERTFORMULA

    """
    def __init__(self, ParticleArray source, ParticleArray dest,
                 str pressure='p', *args, **kwargs):
        """
        Constructor.
        """
        self.pressure = pressure
        SPHFunctionParticle3D.__init__(self, source, dest)

    cpdef setup_arrays(self):
        """
        """
        if self.source is None or self.dest is None:
            return

        SPHFunctionParticle3D.setup_arrays(self)
        
        self.d_pressure = self.dest.get_carray(self.pressure)
        self.s_pressure = self.source.get_carray(self.pressure)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])

        cdef double temp = 0.0
        cdef Point grad = Point()

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        temp = self.s_pressure.data[source_pid]/(
            self.s_rho.data[source_pid]*self.s_rho.data[source_pid])
        
        temp += self.d_pressure[dest_pid]/(
            self.d_rho.data[dest_pid]*self.d_rho.data[dest_pid])

        temp *= self.s_mass.data[source_pid]
        
        kernel.gradient(self._dst, self._src, h, grad)

        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z        
#############################################################################


