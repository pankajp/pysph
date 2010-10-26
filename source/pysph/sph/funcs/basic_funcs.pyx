#cython: profile=True

""" Implementations for the basic SPH functions """

cdef extern from "math.h":
    double sqrt(double)

################################################################################
# `SPH` class.
################################################################################
cdef class SPH(SPHFunctionParticle):
    """ Basic SPH Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name=''):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.

        Notes:
        ------
        By default, the arrays are not setup. This lets us initialize 
        the function and then explicitly set the prop_name before
        invoking setup_arrays.
        
        """
        assert prop_name != '', 'You must provide a property name '

        self.id = 'sph'
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}
            
        """
        
        cdef double h, w, temp
        cdef double rhob, mb, fb

        rhob = self.s_rho.data[source_pid]
        fb = self.s_prop.data[source_pid]
        mb = self.s_m.data[source_pid]

        h = 0.5*(self.s_h.data[source_pid] + 
                 self.d_h.data[dest_pid])
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        w = kernel.function(self._dst, self._src, h)
        
        nr[0] += w*mb*fb/rhob

###########################################################################


###############################################################################
# `SPHSimpleDerivative` class.
###############################################################################
cdef class SPHSimpleDerivative(SPHFunctionParticle):
    """ Basic SPH Derivative Interpolation.  """

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

        Note:
        -----
        By default, the arrays are not setup. This lets us set the prop
        name after intialization and then setup the arrays.
        
        """
        assert prop_name != '', 'You must provide a property name '

        self.id = 'sphd'
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        """
        cdef double h, temp
        cdef Point grad

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
        
        grad = Point()
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
        
        kernel.gradient(self._dst, self._src, h, grad)
        
        temp = self.s_prop[source_pid]
        temp *= self.s_m.data[source_pid]/self.s_rho.data[source_pid]
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

###########################################################################

################################################################################
# `SPHGrad` class.
################################################################################
cdef class SPHGrad(SPHFunctionParticle):
    """ Basic SPH Gradient Interpolation.  """

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

        Note:
        -----
        By default, the arrays are not setup. This lets us set the prop
        name after intialization and then setup the arrays.
        
        """
        assert prop_name != '', 'You must provide a property name '

        self.id = 'sphgrad'
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.dest_prop = self.dest.get_carray(self.prop_name)
        self.source_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        """
        cdef double h, temp
        cdef Point grad

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
        
        grad = Point()
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        kernel.gradient(self._dst, self._src, h, grad)
            
        temp = (self.s_prop[source_pid] -  self.d_prop[dest_pid])
        
        temp *= self.s_m.data[source_pid]/self.s_rho.data[source_pid]
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

###########################################################################

###############################################################################
# `SPHLaplacian` class.
###############################################################################
cdef class SPHLaplacian(SPHFunctionParticle):
    """ Estimation of the laplacian of a function.  The expression is
     taken from the papeer: 

     "Accuracy of SPH viscous flow models",
     David I. Graham and Jason P. Huges, IJNME, 2008, 56, pp 1261-1269.

     """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name='',  *args, **kwargs):
        """ Constructor

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.

        Notes:
        ------
        By default, the arrays are not setup. This lets us initialize 
        the function and then explicitly set the prop_name before
        invoking setup_arrays.
        
        """
        assert prop_name != '', 'You must provide a property name '

        self.id = 'sphlaplacian'
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        
    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.s_prop = self.source.get_carray(self.prop_name)
        self.d_prop = self.dest.get_carray(self.prop_name)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \\nabla_aW_{ab}
            
        """
        cdef double h, mb, rhob, fb, fa, tmp, dot
        cdef Point rab, grad

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])
        
        mb = self.s_m.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        fb = self.s_prop[source_pid]
        fa = self.d_prop[dest_pid]
        tmp, dot
            
        grad = Point()
        rab = Point()
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        rab = self._dst - self._src
            
        kernel.gradient(self._dst, self._src, h, grad)
        
        dot = rab.dot(grad)            

        tmp = 2*mb*(fa-fb)/(rhob*rab.length())
        
        nr[0] += tmp*dot

###########################################################################

################################################################################
# `CountNeighbors` class.
################################################################################
cdef class CountNeighbors(SPHFunctionParticle):
    """ Count Neighbors.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        self.id = 'nbrs'
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        nr[0] += 1
###########################################################################

################################################################################
# `KernelGradientCorrectionTerms` class.
################################################################################
cdef class KernelGradientCorrectionTerms(SPHFunctionParticle):
    """ Count Neighbors.  """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        self.id = 'kgc'
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        
        cdef Point grad = Point()
        cdef Point rab

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = self._dst - self._src

        kernel.gradient(self._dst, self._src, h, grad)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        grad *= tmp
        
        nr[0] -= grad.x * rab.x * rab.x
        nr[1] -= grad.y * rab.x * rab.y
        nr[2] -= grad.z * rab.y * rab.y

##########################################################################




