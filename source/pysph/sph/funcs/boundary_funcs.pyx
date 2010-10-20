#############################################################################
# `MonaghanBoundaryForce` class.
#############################################################################
cdef class MonaghanBoundaryForce(SPHFunctionParticle):
    """ Class to compute the boundary force for a fluid boundary pair """ 

    #Defined in the .pxd file
    #cdef public double cs
    #cdef public double double delp
    #cdef public DoubleArray arr_tx, arr_ty, arr_tz, arr_nx, arr_ny, arr_nz

    def __init__(self, ParticleArray source, ParticleArray dest,
                 bint setup_arrays=False, double cs = 0, double delp = 0):

        self.id = 'monaghanbforce'
        self.cs = cs
        self.delp = delp
        SPHFunctionParticle.__init__(self, source, dest, 
                                       setup_arrays = False)

    cpdef setup_arrays(self):
        """ Setup the arrays needed for the function """

        #Setup the basic properties like m, x rho etc.
        SPHFunctionParticle.setup_arrays(self)
        
        self.s_tx = self.array.get_carray("tx")
        self.s_ty = self.array.get_carray("ty")
        self.s_tz = self.array.get_carray("tz")
        self.s_nx = self.array.get_carray("nx")
        self.s_ny = self.array.get_carray("ny")
        self.s_nz = self.array.get_carray("nz")

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """ Perform the boundary force computation """

        cdef double x, y, nforce, tforce, force
        cdef double beta, q
        cdef Point tang, norm, rab

        cdef double h = self.d_h.data[dest_pid]
        cdef double ma = self.d_m.data[dest_pid]
        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        norm = Point(self.s_nx.data[source_pid], self.s_ny.data[source_pid],
                     self.s_nz.data[source_pid])
        
        tang = Point(self.s_tx.data[source_pid], self.s_ty.data[source_pid],
                     self.s_tz.data[source_pid])
        
        rab = self._dst - self._src
        x = rab.dot(tang)
        y = rab.dot(norm)

        #Evaluate the tangential force
        #if 0 < abs(x) < self.delp:
        #tforce = 1 - abs(x)/self.delp
        #else:
        #tforce = 0.0

        #Evaluate the normal force
        beta = 0.02 * self.cs * self.cs/y
        q = y/h

        if abs(x) < self.delp:
            if 0 < q < 2./3:
                nforce =  2./3
            elif 2./3 < q < 1.0:
                nforce = (2*q - 1.5*q*q)
            elif 1. < q < 2:
                nforce = 0.5 * (2-q)*(2-q)
            else:
                nforce = 0.0                
                    
            force = (mb/(ma+mb)) * nforce * beta
        else:
            force = 0.0
        
        assert abs(norm.y - 1) < 1e-14
        assert abs(norm.x) < 1e-14
        assert abs(norm.z) < 1e-14
        
        nr[0] += force*norm.x
        nr[1] += force*norm.y
        nr[2] += force*norm.z

##############################################################################


##############################################################################
cdef class BeckerBoundaryForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    #Defined in the .pxd file
    #cdef public double cs

    def __init__(self, ParticleArray source, dest, bint setup_arrays=True,
                 double sound_speed=0.0):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.id = 'beckerbforce'
        self.sound_speed = sound_speed

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

            f_ak = \frac{m_k}{m_a + m_k}\tau{x_a,x_k}\frac{x_a -
            x_k}{\lvert x_a - x_k \rvert} where \tau{x_a, x_k} =
            0.02\frac{c_s^2}{\lvert x_a - x_k|\rvert}\begin{cases}
            2/3 & 0 < q < 2/3\\ (2q - 3/2q^2) & 2/3 < q < 1\\
            1/2(2-q)^2 & 1 < q < 2\\ 0 & \text{otherwise}
            \end{cases}
        
        """
        cdef double norm, nforce, force
        cdef double beta, q
        cdef Point rab, rabn

        cdef double h = self.d_h.data[dest_pid]
        cdef double ma = self.d_m.data[dest_pid]
        cdef double mb = self.s_m.data[source_pid]

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]

        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]
            
        rab = self._dst - self._src
        norm = rab.length()
        rabn = rab/norm

        #Evaluate the normal force
        q = norm/h
        beta = 0.02 * self.sound_speed * self.sound_speed/norm
        
        if q > 0.0 and q <= 2.0/3.0:
            nforce = 2.0/3.0

        elif q > 2.0/3.0 and q <= 1.0:
            nforce = (2.0*q - (3.0/2.0)*q*q)

        elif q > 1.0 and q < 2.0:
            nforce = (0.5)*(2.0 - q)*(2.0 - q)

        force = (mb/(ma+mb)) * nforce * beta
            
        nr[0] += force*rabn.x
        nr[1] += force*rabn.y
        nr[2] += force*rabn.z
##############################################################################

##############################################################################
cdef class LennardJonesForce(SPHFunctionParticle):
    """
    Class to compute the interaction of a boundary particle on a fluid 
    particle.
    """

    #Defined in the .pxd file
    #cdef public double D
    #cdef public double ro
    #cdef public double p1, p2

    def __init__(self, ParticleArray source, dest, bint setup_arrays=True,
                 double D=0, double ro=0, double p1=0, double p2=0):

        SPHFunctionParticle.__init__(self, source, dest, setup_arrays)
        self.id = 'lenardbforce'
        self.D = D
        self.ro = ro
        self.p1 = p1
        self.p2 = p2

    cdef void eval(self, int source_pid, int dest_pid,
                   MultidimensionalKernel kernel, double *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

        """
        cdef double norm, force, tmp, tmp1, tmp2
        cdef Point rab, rabn

        cdef double ro = self.ro
        cdef double D = self.D

        self._src.x = self.s_x.data[source_pid]
        self._src.y = self.s_y.data[source_pid]
        self._src.z = self.s_z.data[source_pid]
        
        self._dst.x = self.d_x.data[dest_pid]
        self._dst.y = self.d_y.data[dest_pid]
        self._dst.z = self.d_z.data[dest_pid]

        rab = self._dst - self._src
        norm = rab.length()
        rabn = rab/norm

        #Evaluate the normal force
        if norm <= ro:
            tmp = ro/norm
            tmp1 = tmp**self.p1
            tmp2 = tmp**self.p2
            
            force = D*(tmp1 - tmp2)/(norm*norm)
        else:
            force = 0.0
            
        nr[0] += force*rabn.x
        nr[1] += force*rabn.y
        nr[2] += force*rabn.z

##############################################################################
