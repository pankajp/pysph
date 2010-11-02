# cython: profile=True

""" Implementations for the various sph functions. """


cdef extern from "math.h":
    double sqrt(double)

cdef class SPHEval(SPHFunctionPoint):
    """ Basic SPH Interpolation at a point. """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, str prop_name='', 
                 *args, **kwargs):
        """ Constructor for SPHEval

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.

        """
        assert prop_name != '', 'Supply a valid prop name'
        
        self.prop_name = prop_name
        SPHFunctionPoint.__init__(self, source, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionPoint.setup_arrays(self)

        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, Point pnt, int source_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}
            
        """

        cdef double h=self.s_h.data[source_pid]
        cdef double mb = self.s_mass.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double fb = self.s_prop[source_pid]
        
        cdef Point src_position = self._src

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        cdef double w = kernel.function(pnt, src_position, h)

        nr[0] += w*mb*fb/rhob        
###########################################################################


###############################################################################
# `SPHSimpleDerivativeEval` class.
###############################################################################
cdef class SPHSimpleDerivativeEval(SPHFunctionPoint):
    """ Basic SPH Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, str prop_name='', 
                 *args, **kwargs):
        """ Constructor for SPHEval

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
        assert prop_name != '', 'Supply a valid prop name'

        self.prop_name = prop_name
        SPHFunctionPoint.__init__(self, source, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionPoint.setup_arrays(self)

        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, Point pnt, int source_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """ 
        Perform an SPH intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \\nabla_aW_{ab}
            
        """

        cdef double h=self.s_h.data[source_pid]
        cdef double mb = self.s_mass.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double fb = self.s_prop[source_pid]
        cdef double tmp 
        
        cdef Point grad = Point()

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        kernel.gradient(pnt, self._src, h, grad)
        tmp = mb*fb/rhob

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z

###########################################################################


###############################################################################
# `CSPMEval` class.
###############################################################################
cdef class CSPMEval(SPHFunctionPoint):
    """ CSPM Interpolation of a function"""

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, str prop_name='', 
                 *args, **kwargs):
        """ Constructor for SPHEval

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
        assert prop_name != '', 'Supply a valid prop name'
        self.prop_name = prop_name
        SPHFunctionPoint.__init__(self, source, setup_arrays = True)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionPoint.setup_arrays(self)

        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, Point pnt, int source_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """ 
        Perform an CSPM intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = frac{\sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}}{\sum_{b=1}^{N}m_bW_{ab}}
            
        """

        cdef double h=self.s_h.data[source_pid]
        cdef double mb = self.s_mass.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double fb = self.s_prop[source_pid]
        
        cdef Point src_position = self._src

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        cdef double tmp = kernel.function(pnt, src_position, h)*mb/rhob

        nr[0] += tmp*fb
        dnr[0] += tmp

###############################################################################

###############################################################################
# `CSPMDerivativeEval` class.
###############################################################################
cdef class CSPMDerivativeEval(SPHFunctionPoint):
    """ CSPM Interpolation of a function"""

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, str prop_name='', 
                 *args, **kwargs):
        """ Constructor for SPHEval

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

        assert prop_name != '', 'Supply a valid prop name'
        self.prop_name = prop_name
        SPHFunctionPoint.__init__(self, source, setup_arrays = False)

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        if self.source is None:
            return

        #Setup the basic properties like m, x rho etc.
        SPHFunctionPoint.setup_arrays(self)

        self.s_prop = self.source.get_carray(self.prop_name)

    cdef void eval(self, Point pnt, int source_pid,
                   KernelBase kernel, double *nr, double *dnr):
        """ 
        Perform an CSPM intterpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = frac{\sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}}{\sum_{b=1}^{N}m_bW_{ab}}
            
        """

        cdef double h=self.s_h.data[source_pid]
        cdef double mb = self.s_mass.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double fb = self.s_prop[source_pid]
        
        cdef Point grad = Point()

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        kernel.gradient(pnt, self._src, h, grad)
        cdef double tmp = grad.x*mb/rhob

        nr[0] += tmp*fb
        dnr[0] += tmp

###############################################################################




