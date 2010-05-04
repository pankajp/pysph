""" Implementations for the summations defined in the paper: "SPH and
Riemann Sovers" by J.J. Monaghan, JCP, 136, 298-307. """

# Author: Kunal Puri <kunalp@aero.iitb.ac.in>
# Copyright (c) 2010, Kunal Puri.

cdef extern from "math.h":
    double sqrt(double)

cdef inline double sqr(double x):
    return x*x

cdef inline double Vsig(double ca, double cb, Point vab, Point j,
                        beta = 1.0):
    """ 
    Return the signal velocity between a pair of particles.
    The expression used is:
    
    ..math :: v_{sig}(a,b)=(c_a^2+\beta(\vec{ab}\cdot \vec{j})^2)^\frac{1}{2} +
                          (c_b^2 + \beta(\vec{ab}\cdot \vec{j})^2)^\frac{1}{2} -
                          \vec{ab}\cdot \vec{j}
                              
    """
    cdef double tmp = vab.dot(j)    
    cdef double tmp1 = ca*ca + beta * tmp * tmp
    cdef double tmp2 = cb*cb + beta * tmp * tmp
    return sqrt(tmp1) + sqrt(tmp2) - tmp

############################################################################
#`PressureGradient` class
############################################################################
cdef class PressureGradient(SPHFunctionParticle3D):
    """ 
    The gradient of pressure in the momentum equation.

    .. math :: \frac{Dv_a}{Dt} = -\sum_b m_b \left(\frac{P_b}
    {\rho_{a}^{2-\sigma}\rho_b^{\sigma}} + \frac{P_a}{\rho_a^{\sigma}
    {\rho_b}^{2 - \sigma}}\right)\nabla_a W_{ab}
    
    """
    #Defined in the .pxd file
    #cdef public str pressure
    #cdef public DoubleArray s_p, d_p
    #cdef public double sigma

    def __init__(self, ParticleArray source, ParticleArray dest,
                 str p = 'p', sig = 2, 
                 bint setup_arrays=True, *args, **kwargs):
        
        self.p = p
        SPHFunctionParticle3D.__init__(self, source, dest)

    cpdef setup_arrays(self):
        if self.source is None or self.dest is None:
            return

        SPHFunctionParticle3D.setup_arrays(self)
        
        self.d_p = self.dest.get_carray(self.p)
        self.s_p = self.source.get_carray(self.p)

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):

        cdef double h = 0.5*(self.s_h.data[source_pid]+self.d_h.data[dest_pid])

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

############################################################################
#`MomentumEquationAvisc` class
############################################################################
cdef class MomentumEquationAvisc(SPHFunctionParticle3D):
    """ 
    The artificial viscosity contribution to the momentum equation is 
    defined as: 

    .. math :: \frac{Dv_a}{Dt} = \sum_b m_b \left(\frac{K
    v_{sig}}{\rho_{ab}} (v_{ab}\cdot \vec{j})\, \nabla_a W_{ab}
    
    """
    #Defined in the .pxd file
    #cdef public str pressure
    #cdef public DoubleArray s_pressure, d_pressure
    #cdef public Point j
    #cdef public Point vsig 
    #cdef public double k
    #cdef public double gamma
    #cdef public double beta    

    def __init__(self, ParticleArray source, ParticleArray dest,
                 pressure = 'p', gamma = 1.4, beta = 1.0, k = 0.5,
                 bint setup_arrays=True, *args, **kwargs):
                 
        """ 
        Constructor Notes:
        -----------------
        We assume default string literals for positions, velocities, mass, 
        density and smoothing. 
        That is, x is refered to as `x`, velocity components as `u`, `v`, `w`, 
        mass as `m` and so on.

        """
        SPHFunctionParticle3D.__init__(self, source, dest, *args, **kwargs)

        self.pressure = pressure
        self.gamma = gamma
        self.beta = beta
        self.k = k
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """ Setup the source and destination pressure arrays """

        msg = 'Either source or destinatio is None!'
        assert not self.source is None or self.dest is None, msg
        self.d_pressure = self.dest.get_carray(self.pressure)
        self.s_pressure = self.source.get_carray(self.pressure)

    cpdef int output_fields(self) except -1:
        """ Return the number of components this computes """
        return 3
    
    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel,  double *nr, double *dnr):
        """ The main algorithm for artificial viscosity computation
        
        Parameters:
        -----------
        source_pid -- Index of the source particle 
        dest_pid -- Index for the destination particle
        kernel -- The kernel used in the simulation. This may be 1D, 2D or 3D
        nr -- Pointer to an array 
        dnr -- Pointer to an array

        """
        cdef int pi = source_pid
        cdef int pj = dest_pid

        cdef double h = 0.5 * (self.s_h.data[pi] + self.d_h.data[pj])
        cdef double rho = 0.5 * (self.s_rho.data[pi] + self.d_h.data[pj])
        cdef Point grad = Point(0,0,0)
        cdef double tmp, ca, cb, gamma
        
        cdef Point vab = Point(self.s_velx.data[pi] - self.d_velx[pj],
                               self.s_vely.data[pi] - self.d_vely[pj],
                               self.s_velz.data[pi] - self.d_velz[pj],
                               )

        self.j = Point(self.s_x.data[pi] - self.d_x[pj],
                  self.s_y.data[pi] - self.d_y[pj],
                  self.s_z.data[pi] - self.d_z[pj],
                  )
        self.j /= self.j.length()
        gamma = self.gamma

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        ca = sqrt(gamma*self.d_pressure.data[pj]/self.d_rho.data[pj])
        ca = sqrt(gamma*self.s_pressure.data[pi]/self.s_rho.data[pi])
        self.vsig = Vsig(ca, cb, vab, self.j, self.beta)

        kernel.gradient(self._dst, self._src, h, grad)
        tmp = self.k * self.d_m.data[pj] * self.vsig * vab.dot(self.j)/rho

        nr[0] += tmp*grad.x
        nr[1] += tmp*grad.y
        nr[2] += tmp*grad.z

############################################################################

############################################################################
#`EnergyEquation` class
############################################################################
cdef class EnergyEquation(SPHFunctionParticle3D):
    """ 
    The artificial viscosity contribution to the momentum equation is 
    defined as: 

    .. math :: \frac{P_a}{rho_a^2}\sum_b m_b\vec{ab}\cdot
    \nabla_aW_{ab} + \sum_b
    m_b\frac{Kv_{sig}}{\rho_{ab}}\left(\vec{j}f(e_a - e_b)\right)
    \cdot\,\nabla_aW_{ab}
    
    """
    #Defined in the .pxd file
    #cdef public str pressure
    #cdef public DoubleArray s_pressure, d_pressure
    #cdef public double k
    #cdef public double f    

    def __init__(self, ParticleArray source, ParticleArray dest,
                 energy = 'e', pressure = 'p', 
                 gamma = 1.4, k = 0.5, beta = 1.0, f = 0.5,
                 bint setup_arrays = True, *args, **kwargs):
                 
        """ Constructor for the class  """
        SPHFunctionParticle3D.__init__(self, source, dest, *args, **kwargs)
        self.pressure = pressure
        self.energy = energy
        self.k = k
        self.f = f
        self.beta = beta
        self.gamma = gamma
        if setup_arrays:
            self.setup_arrays()

    cpdef setup_arrays(self):
        """ Setup the source and destination pressure arrays """

        msg = 'Either source or destinatio is None!'
        assert not self.source is None or self.dest is None, msg
        self.d_pressure = self.dest.get_carray(self.pressure)
        self.s_pressure = self.source.get_carray(self.pressure)
        self.d_energy = self.dest.get_carray(self.energy)
        self.s_energy = self.source.get_carray(self.energy)
        
    cpdef int output_fields(self) except -1:
        """ Return the number of components this computes """
        return 3
    
    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ The main algorithm for artificial viscosity computation
        
        Parameters:
        -----------
        source_pid -- Index of the source particle 
        dest_pid -- Index for the destination particle
        kernel -- The kernel used in the simulation. This may be 1D, 2D or 3D
        nr -- Pointer to an array 
        dnr -- Pointer to an array

        """
        cdef int pi = source_pid
        cdef int pj = dest_pid
        cdef double Pa, rhoa, ea, eb, ca, cb
        cdef double tmp1, tmp2

        cdef double h = 0.5 * (self.s_h.data[pi] + self.d_h.data[pj])
        cdef double rho = 0.5 * (self.s_rho.data[pi] + self.d_h.data[pj])
        cdef Point grad = Point(0,0,0)
                
        cdef Point vab = Point(self.s_velx.data[pi] - self.d_velx[pj],
                               self.s_vely.data[pi] - self.d_vely[pj],
                               self.s_velz.data[pi] - self.d_velz[pj],
                               )

        self.j = Point(self.s_x.data[pi] - self.d_x[pj],
                  self.s_y.data[pi] - self.d_y[pj],
                  self.s_z.data[pi] - self.d_z[pj],
                  )
        self.j /= self.j.length()
        gamma = self.gamma

        ca = sqrt(gamma*self.d_pressure.data[pj]/self.d_rho.data[pj])
        ca = sqrt(gamma*self.s_pressure.data[pi]/self.s_rho.data[pi])
        self.vsig = Vsig(ca, cb, vab, self.j, self.beta)

        Pa = self.s_pressure.data[pi]
        ea = self.s_energy.data[pi]
        rhoa = self.s_rho.data[pi]
        eb = self.d_energy.data[pj]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        kernel.gradient(self._dst, self._src, h, grad)

        tmp1 = self.k * self.d_m.data[pj] * self.vsig/rho
        tmp2 = self.f*(ea - eb)
        tmp2 *= self.j.dot(grad)
        tmp1 *= tmp2

        nr[0] += tmp1 + Pa/sqr(rhoa)*self.d_m.data[pj]*(vab.dot(grad))

############################################################################

############################################################################
#`EnergyEquationAvisc` class
############################################################################
cdef class EnergyEquationAvisc(SPHFunctionParticle3D):
    """
    Definition for the artificial viscosity contribution to the energy
    equation between a pair of particles. 
    The expression used is:
    
    .. math :: \frac{De_a}{Dt} = \sum_b \frac{m_b K v_{sig}}{\rho_{ab}} \left
    (\vec{j}(0.5( (\vec{a}\cdot\vec{j})^2 - (\vec{b}\cdot\vec{j})^2 ) ) -
    \vec{v_a}(\vec{v_{ab}}) )\cdot \nabla_a W_{ab}
    
   """
    #Defined in the .pxd file
    #cdef public Point j
    #cdef public Point vsig 
    #cdef public double k
    #cdef public double gamma
    #cdef public double beta    

    def __init__(self, ParticleArray source, ParticleArray dest,
                 gamma = 1.4, k = 0.5, beta = 1.0,
                 bint setup_arrays=False, *args, **kwargs):
        """ 
        Constructor  
        """
        SPHFunctionParticle3D.__init__(self, source, dest, *args, **kwargs)

        self.gamma = gamma
        self.beta = beta
        self.k = k
        if setup_arrays:
            self.setup_arrays()

    cpdef int output_fields(self) except -1:
        """ Return the number of components this computes """
        return 3

    cdef void eval(self, int source_pid, int dest_pid, 
                   MultidimensionalKernel kernel,  double *nr, double *dnr):
        """ The main algorithm for artificial viscosity computation
        
        Parameters:
        -----------
        source_pid -- Index of the source particle 
        dest_pid -- Index for the destination particle
        kernel -- The kernel used in the simulation. This may be 1D, 2D or 3D
        nr -- Pointer to an array 
        dnr -- Pointer to an array

        """
        cdef int pi = source_pid
        cdef int pj = dest_pid

        cdef double h = 0.5 * (self.s_h.data[pi] + self.d_h.data[pj])
        cdef double rho = 0.5 * (self.s_rho.data[pi] + self.d_h.data[pj])
        cdef Point grad = Point(0,0,0)
        cdef double ca, cb, gamma
        cdef Point tmp, va, vb, vab
        
        va = Point(self.s_velx.data[pi], self.s_velx.data[pi],
                   self.s_velx.data[pi])
        vb = Point(self.d_velx.data[pj], self.d_velx.data[pj],
                   self.d_velx.data[pj])
        vab = va - vb

        self.j = Point(self.s_x.data[pi] - self.d_x[pj],
                  self.s_y.data[pi] - self.d_y[pj],
                  self.s_z.data[pi] - self.d_z[pj],
                  )
        self.j /= self.j.length()
        gamma = self.gamma

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        ca = sqrt(gamma*self.d_pressure.data[pj]/self.d_rho.data[pj])
        ca = sqrt(gamma*self.s_pressure.data[pi]/self.s_rho.data[pi])
        self.vsig = Vsig(ca, cb, vab, self.j, self.beta)

        kernel.gradient(self._dst, self._src, h, grad)
        tmp = self.j * (0.5 * (sqr( va.dot(self.j) ) - sqr( vb.dot(self.j))))
        tmp -= va * (vab.dot(self.j))
        tmp *= self.k * self.d_m.data[pj] * self.vsig/rho

        nr[0] += tmp.dot(grad)

############################################################################
