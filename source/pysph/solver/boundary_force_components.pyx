"""
Compoents to compute forces at boundaries.
"""

# c-function imports
cdef extern from "math.h":
    double sqrt(double)

# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.point cimport Point
from pysph.base.kernelbase cimport *
from pysph.sph.sph_func cimport *
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.sph_component cimport *
from pysph.solver.solver_base cimport *


################################################################################
# `RepulsiveBoundaryKernel` class.
################################################################################
cdef class RepulsiveBoundaryKernel(Kernel3D):
    """
    Kernel class to be used for boundary SPH summations.

    This is a kernel used for boundary force
    calculation in the Becker07.

    * NOTE *
    - The function and gradient computations are very 
      similar. We __do not__ expect the function to be 
      used anywhere. Only the gradient should be used.
      The function is implemented for debugging purposes,
      the gradient is kept for performance.
    - __DO NOT__ put this kernel in the kernel_list in the 
      test_kernels3d module. It will fail, as it is not
      meant to be a generic SPH kernel.
    - __DO NOT__ use this kernel for anything else, you
      may not get expected results.

    ** References **
    1. [becker07] Weakly Compressible SPH for free surface flows.

    """
    cdef double function(self, Point p1, Point p2, double h):
        """
        """
        cdef Point dir = p1-p2
        cdef double dist = sqrt(dir.norm())
        cdef double q = dist/h
        cdef double temp = 0.0

        dir.x = dir.x/dist
        dir.y = dir.y/dist
        dir.z = dir.z/dist

        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        return temp

    cdef void gradient(self, Point p1, Point p2, double h, Point grad):
        """
        """
        cdef Point dir = p1-p2
        cdef double dist = sqrt(dir.norm())
        cdef double q = dist/h
        cdef double temp = 0.0

        dir.x = dir.x/dist
        dir.y = dir.y/dist
        dir.z = dir.z/dist
        
        if q > 0.0 and q <= 2.0/3.0:
            temp = 2.0/3.0
        elif q > 2.0/3.0 and q <= 1.0:
            temp = (2.0*q - (3.0/2.0)*q*q)
        elif q > 1.0 and q < 2.0:
            temp = (0.5)*(2.0 - q)*(2.0 - q)

        temp *= 0.02
        temp /= dist

        grad.x = (temp*dir.x)
        grad.y = (temp*dir.y)
        grad.z = (temp*dir.z)

    cdef double laplacian(self, Point p1, Point p2, double h):
        """
        """
        raise NotImplementedError('laplacian for the BoundaryKernel not implemented')
    
    cpdef double radius(self):
        return 2.0

################################################################################
# `SPHRepulsiveBoundaryFunction` class.
################################################################################
cdef class SPHRepulsiveBoundaryFunction(SPHFunctionParticle3D):
    """
    Class to compute iteration of boundary and non-boundary particles.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str h='h', str mass='m', str rho='rho',
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='w', bint
                 setup_arrays=True,
                 SpeedOfSound speed_of_sound = None,
                 *args, **kwargs):
        """
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, rho,
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays, *args,
                                       **kwargs)

        
        self.speed_of_sound = speed_of_sound

    cpdef int output_fields(self) except -1:
        """
        """
        return 3

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 

        evaluate boundary forces as described in becker07
        
        ::math::

            f_ak = \frac{m_k}{m_a + m_k}\tau{x_a,x_k}\frac{x_a - x_k}{\lvert x_a -
            x_k \rvert}
            where 
            \tau{x_a, x_k} = 0.02\frac{c_s^2}{\lvert x_a - x_k|\rvert}\begin{cases}
            2/3 & 0 < q < 2/3\\
            (2q - 3/2q^2) & 2/3 < q < 1\\
            1/2(2-q)^2 & 1 < q < 2\\
            0 & \text{otherwise}
            \end{cases}
        
        """
        cdef double h = 0.5*(self.s_h.data[source_pid] +
                             self.d_h.data[dest_pid])
        cdef double mtemp, cs_2
        cdef Point dir = Point()

        make_coords_3d(self.s_x, self.s_y, self.s_z, self._pnt1, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, self._pnt2, dest_pid)
        
        mtemp = self.s_mass.data[source_pid]/(
            self.s_mass.data[source_pid]+self.d_mass.data[dest_pid])
        
        cs_2 = self.speed_of_sound.value
        cs_2 *= cs_2

        kernel.gradient(self._pnt2, self._pnt1, h, dir)

        nr[0] += cs_2*dir.x
        nr[1] += cs_2*dir.y
        nr[2] += cs_2*dir.z
################################################################################
# `SPHRepulsiveBoundaryForceComponent` class.
################################################################################
cdef class SPHRepulsiveBoundaryForceComponent(SPHComponent):
    """
    Component to compute repulsive boundary forces.
    """
    def __cinit__(self, SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_All,
                  SpeedOfSound speed_of_sound=None,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.source_dest_mode = SPHSourceDestMode.Group_All
        self.speed_of_sound = speed_of_sound
        
        # setup the accepted destination types.
        self.dest_types =[EntityTypes.Entity_Fluid]

        # setup the accepted source types.
        self.source_types = [EntityTypes.Entity_Solid]
        
        # allow for solids and liquids in the inputs
        self.add_input_entity_type(EntityTypes.Entity_Fluid)
        self.add_input_entity_type(EntityTypes.Entity_Solid)

        # use the default kernel always
        self.kernel = RepulsiveBoundaryKernel()

        # setup the class to use the SPHRepulsiveBoundaryFunction
        self.sph_func_class = SPHRepulsiveBoundaryFunction

    cpdef int update_property_requirements(self) except -1:
        """
        Setup the property requirements of this component.
        """
        for t in self.source_types:
            self.add_read_prop_requirement(t, ['m', 'h'])

        for t in self.dest_types:
            self.add_read_prop_requirement(t, ['m', 'h'])

            self.add_write_prop_requirement(t, 'ax')
            self.add_write_prop_requirement(t, 'ay')
            self.add_write_prop_requirement(t, 'az')

            self.add_private_prop_requirement(t, 'bacclr_x')
            self.add_private_prop_requirement(t, 'bacclr_y')
            self.add_private_prop_requirement(t, 'bacclr_z')

        return 0

    cpdef int setup_component(self) except -1:
        """
        Setup the component before execution.
        """
        if self.setup_done == True:
            return 0
        
        logger.info('Setting up component %s'%(self.name))

        if self.speed_of_sound is None:
            if self.solver is not None:
                self.speed_of_sound = self.solver.speed_of_sound
        
        if self.speed_of_sound is None:
            logger.warn('Using speed of sound = 100.0')
            self.speed_of_sound = SpeedOfSound(100.0)

        logger.info('Speed of Sound %f'%(self.speed_of_sound.value))

        # call the SPH components setup function.
        SPHComponent.setup_component(self)

        self.setup_done = True

        return 0

    cdef int compute(self) except -1:
        """
        """
        cdef int nd
        cdef int i
        cdef SPHBase calc
        cdef ParticleArray parr
        
        self.setup_component()

        nd = len(self.dest_list)

        for i from 0 <= i < nd:
            e = self.dest_list[i]
            parr = e.get_particle_array()
            
            calc = self.sph_calcs[i]

            calc.sph3('bacclr_x', 'bacclr_y', 'bacclr_z')

            parr.ax += parr.bacclr_x
            parr.ay += parr.bacclr_y
            parr.az += parr.bacclr_z        

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
                             SPHFunctionParticle sph_func):
        """
        Sets up component specific properties of the sph function used.
        """
        cdef SPHRepulsiveBoundaryFunction func = \
            <SPHRepulsiveBoundaryFunction>sph_func
        
        func.speed_of_sound = self.speed_of_sound
