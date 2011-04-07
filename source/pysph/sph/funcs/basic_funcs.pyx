#cython: cdivision=True
""" Implementations for the basic SPH functions """

from pysph.base.point cimport cPoint_new, cPoint_sub

cdef extern from "math.h":
    double sqrt(double)

################################################################################
# `SPH` class.
################################################################################
cdef class SPH(CSPHFunctionParticle):
    """ Basic SPH Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name='rho', **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.

        """
        self.prop_name = prop_name
        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'sph'

        self.cl_kernel_src_file = "/home/kunalp/pysph/source/pysph/sph/funcs/basic_funcs.cl"

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

        self.src_reads.append(self.prop_name)
        self.dst_reads.append(self.prop_name)

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        """ 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \W_{ab}
            
        """
        
        cdef double w, wa, wb,  temp
        cdef double rhob, mb, fb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

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

        if self.hks:
            wa = kernel.function(self._dst, self._src, ha)
            wb = kernel.function(self._dst, self._src, hb)

            w = 0.5 * (wa + wb)

        else:
            w = kernel.function(self._dst, self._src, hab)
        
        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            dnr[0] += w*mb/rhob
        
        nr[0] += w*mb*fb/rhob        

###########################################################################


###############################################################################
# `SPHSimpleGradient` class.
###############################################################################
cdef class SPHSimpleGradient(SPHFunctionParticle):
    """ Basic SPH Derivative Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str prop_name='rho',  *args, **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphd'

        self.cl_kernel_src_file = "basic_funcs.cl"

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

        self.src_reads.append(self.prop_name)
        self.dst_reads.append(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid,
                       KernelBase kernel, double *nr):
        """ 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        """
        cdef double temp
        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
            
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.set((grada.x + gradb.x)*0.5,
                     (grada.y + gradb.y)*0.5,
                     (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)
        
        temp = self.s_prop[source_pid]
        temp *= self.s_m.data[source_pid]/self.s_rho.data[source_pid]

        if self.rkpm_first_order_correction:
            pass

        if self.bonnet_and_lok_correction:
            self.bonnet_and_lok_gradient_correction(dest_pid, &grad)
            
        nr[0] += temp*grad.x
        nr[1] += temp*grad.y
        nr[2] += temp*grad.z

###########################################################################

################################################################################
# `SPHGrad` class.
################################################################################
cdef class SPHGradient(SPHFunctionParticle):
    """ Basic SPH Gradient Interpolation.  """

    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop, d_prop

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str prop_name='rho',  *args, **kwargs):
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
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphgrad'

        self.cl_kernel_src_file = "basic_funcs.cl"

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.d_prop = self.dest.get_carray(self.prop_name)
        self.s_prop = self.source.get_carray(self.prop_name)

        self.src_reads.append(self.prop_name)
        self.dst_reads.append(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """ 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        """
        cdef double temp
        cdef cPoint grad, grada, gradb

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h=0.5*(self.s_h.data[source_pid] + 
               self.d_h.data[dest_pid])
    
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

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
     taken from the paper: 

     "Accuracy of SPH viscous flow models",
     David I. Graham and Jason P. Huges, IJNME, 2008, 56, pp 1261-1269.

     """
    #Defined in the .pxd file
    #cdef str prop_name
    #cdef DoubleArray s_prop

    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str prop_name='rho',  *args, **kwargs):
        """ Constructor

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        
        """
        self.prop_name = prop_name
        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True,
                                     *args, **kwargs)
        self.id = 'sphlaplacian'

        self.cl_kernel_src_file = "basic_funcs.cl"

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)
        self.s_prop = self.source.get_carray(self.prop_name)
        self.d_prop = self.dest.get_carray(self.prop_name)

        self.src_reads.append(self.prop_name)
        self.dst_reads.append(self.prop_name)

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):
        """ 
        Perform an SPH interpolation of the property `prop_name` 

        The expression used is:
        
        ..math :: <f(\vec{r}>_a = \sum_{b = 1}^{N}f_b\frac{m_b}{\rho_b}\, 
        \\nabla_aW_{ab}
            
        """
        cdef double mb, rhob, fb, fa, tmp, dot
        cdef cPoint grad, grada, gradb, rab

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])
        
        mb = self.s_m.data[source_pid]
        rhob = self.s_rho.data[source_pid]
        fb = self.s_prop[source_pid]
        fa = self.d_prop[dest_pid]
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        rab.x = self._dst.x-self._src.x
        rab.y = self._dst.y-self._src.y
        rab.z = self._dst.z-self._src.z
        
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
        
        dot = cPoint_dot(rab, grad)
        tmp = 2*mb*(fa-fb)/(rhob*cPoint_length(rab))
        
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

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'nbrs'

    cdef void eval_single(self, size_t dest_pid,
                          KernelBase kernel, double *result):
        result[0] += self.nbr_locator.get_nearest_particles(dest_pid).length

###########################################################################

################################################################################
# `KernelGradientCorrectionTerms` class.
################################################################################
cdef class BonnetAndLokKernelGradientCorrectionTerms(CSPHFunctionParticle):
    """ Evaluate the matrix terms eq(45) in "Variational and
    momentum preservation aspects of Smooth Particle Hydrodynamic
    formulations", Computer Methods in Applied Mechanical Engineering,
    180, (1997), 97-115

    Note:
    -----
    The matrix would need to be inverted to calculate the correction terms!

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'kgc'

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid, 
                            KernelBase kernel, double *nr, double *dnr):
        cdef cPoint grada, gradb, grad
        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double Vb = mb/rhob

        cdef double ha = self.d_h.data[dest_pid]
        cdef double hb = self.s_h.data[source_pid]

        cdef double hab = 0.5 * (ha + hb)    
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint rba = cPoint_sub(self._src, self._dst)

        if self.hks:
            grada = kernel.gradient(self._dst, self._src, ha)
            gradb = kernel.gradient(self._dst, self._src, hb)

            grad.set((grada.x + gradb.x)*0.5,
                     (grada.y + gradb.y)*0.5,
                     (grada.z + gradb.z)*0.5)

        else:            
            grad = kernel.gradient(self._dst, self._src, hab)

        #m11
        nr[0] += Vb * grad.x * rba.x

        #m12 = m21
        nr[1] += Vb * grad.x * rba.y

        #m13 = m31
        nr[2] += Vb * grad.x * rba.z

        #m22
        dnr[0] += Vb * grad.y * rba.y
        
        #m23 = m32
        dnr[1] += Vb * grad.y * rba.z
        
        #m33
        dnr[2] += Vb * grad.z * rba.z

##########################################################################


################################################################################
# `FirstOrderCorrectionMatrix` class.
################################################################################
cdef class FirstOrderCorrectionMatrix(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 14) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        CSPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'liu-correction'

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid, 
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w
        
        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        cdef cPoint rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        tmp *= w

        nr[0] += tmp * rab.x * rab.x 
        nr[1] += tmp * rab.x * rab.y 
        nr[2] += tmp * rab.y * rab.y 

        dnr[0] -= tmp * rab.x
        dnr[1] -= tmp * rab.y
        dnr[2] -= tmp * rab.z

##########################################################################

################################################################################
# `FirstOrderKernelCorrectionTermsForAlpha` class.
################################################################################
cdef class FirstOrderCorrectionTermAlpha(SPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    #Defined in the .pxd file
    def __init__(self, ParticleArray source, ParticleArray dest,
                 *args, **kwargs):
        """ Constructor """

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays = True)
        self.id = 'alpha-correction'

    cpdef setup_arrays(self):
        """ Setup the arrays required to read data from source and dest. """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)

        self.rkpm_d_beta1 = self.dest.get_carray("rkpm_beta1")
        self.rkpm_d_beta2 = self.dest.get_carray("rkpm_beta2")
        self.rkpm_d_alpha = self.dest.get_carray("rkpm_alpha")
      
        self.rkpm_d_dbeta1dx = self.dest.get_carray("rkpm_dbeta1dx")
        self.rkpm_d_dbeta1dy = self.dest.get_carray("rkpm_dbeta1dy")

        self.rkpm_d_dbeta2dx = self.dest.get_carray("rkpm_dbeta2dx")
        self.rkpm_d_dbeta2dy = self.dest.get_carray("rkpm_dbeta2dy")

    cdef void eval_nbr(self, size_t source_pid, size_t dest_pid, 
                       KernelBase kernel, double *nr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, beta, tmp1, tmp2, tmp3, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        beta1 = self.rkpm_d_beta1.data[dest_pid]
        beta2 = self.rkpm_d_beta2.data[dest_pid]

        dbeta1dx = self.rkpm_d_dbeta1dx[dest_pid]
        dbeta1dy = self.rkpm_d_dbeta1dy[dest_pid]

        dbeta2dx = self.rkpm_d_dbeta2dx[dest_pid]
        dbeta2dy = self.rkpm_d_dbeta2dy[dest_pid]

        alpha = self.rkpm_d_alpha[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob
        
        grad = kernel.gradient(self._dst, self._src, h)

        tmp3 = Vb*(1.0 + (beta1*rab.x + beta2*rab.y))
        
        tmp1 = Vb*w* (dbeta1dx*rab.x + beta1 + rab.y*dbeta2dx) + tmp3*grad.x

        tmp2 = Vb*w* (dbeta1dy*rab.x + beta2 + rab.y*dbeta2dy) + tmp3*grad.y
        
        #alpha
        nr[0] += Vb*w * (1 + (beta1*rab.x + beta2*rab.y))

        #dalphadx
        nr[1] += -tmp1

        #dalphady
        nr[2] += -tmp2

################################################################################
# `FirstOrderCorrectionMatrixGradient` class.
################################################################################
cdef class FirstOrderCorrectionMatrixGradient(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, beta, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob

        grad = kernel.gradient(self._dst, self._src, h)
        
        nr[0] += 2*Vb*w*rab.x + Vb*rab.x*rab.x*grad.x

        nr[1] += Vb*rab.x*rab.x*grad.y

        nr[2] += Vb*w*rab.y + Vb*rab.y*rab.x*grad.x

        dnr[0] += Vb*rab.x*(rab.y*grad.y + w)

        dnr[1] += Vb*rab.y*rab.y*grad.x

        dnr[2] += 2*Vb*rab.y*w + Vb*rab.y*rab.y*grad.y

##########################################################################

################################################################################
# `FirstOrderCorrectionVectorGradient` class.
################################################################################
cdef class FirstOrderCorrectionVectorGradient(CSPHFunctionParticle):
    """ Kernel correction terms (Eq 15) in "Correction and
    Stabilization of smooth particle hydrodynamics methods with
    applications in metal forming simulations" by Javier Bonnet and
    S. Kulasegaram

    """

    #Defined in the .pxd file
    cdef void eval_nbr_csph(self, size_t source_pid, size_t dest_pid,
                            KernelBase kernel, double *nr, double *dnr):

        cdef double mb = self.s_m.data[source_pid]
        cdef double rhob = self.s_rho.data[source_pid]
        cdef double tmp = mb/rhob
        cdef double w, Vb
        
        cdef cPoint rab, grad

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = cPoint_sub(self._dst, self._src)

        h = 0.5*(self.s_h.data[source_pid] +
                 self.d_h.data[dest_pid])

        w = kernel.function(self._dst, self._src, h)
        Vb = mb/rhob        
        
        grad = kernel.gradient(self._dst, self._src, h)
        
        nr[0] += -Vb*rab.x*grad.x - Vb*w

        nr[1] += -Vb*rab.x*grad.y

        nr[2] += -Vb*rab.y*grad.x

        dnr[0] += -Vb*rab.y*grad.y - Vb*w

##########################################################################
