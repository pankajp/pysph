"""
Module containing components to compute viscosity.
"""

# logging import
import logging
logger = logging.getLogger()


# local imports
from pysph.base.nnps cimport NNPSManager
from pysph.base.kernels cimport KernelBase
from pysph.sph.sph_func cimport *
from pysph.solver.sph_component cimport *
from pysph.solver.solver_base cimport *
from pysph.solver.speed_of_sound cimport SpeedOfSound
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid

###############################################################################
# `SPHMonaghanArtVisc3D` class.
###############################################################################
cdef class SPHMonaghanArtVisc3D(SPHFunctionParticle3D):
    """
    SPH function to compute artificial viscosity.
    """
    def __init__(self, ParticleArray source, ParticleArray dest, 
                 str h='h', str mass='m', str rho='rho',
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='w', bint
                 setup_arrays=True,
                 double alpha=0.03,
                 double epsilon=1.0,
                 SpeedOfSound speed_of_sound=None,
                 *args, **kwargs):
        """
        Constructor.
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, rho,
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays, *args,
                                       **kwargs)

        self.alpha = alpha
        self.epsilon = epsilon
        self.speed_of_sound = speed_of_sound

    cpdef int output_fields(self) except -1:
        """
        Returns the number of output fields required by this function.
        """
        return 3

    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        Compute the contribution of particle at source_pid on particle at
        dest_pid. 
        """
        cdef Point vel_ds = Point_new()
        cdef Point pos_ds = Point_new()
        cdef Point grad = Point_new()
        cdef double pi_ab = 0.0
        cdef double h = 0.5*(self.s_h.data[source_pid] + self.d_h.data[dest_pid])
        cdef double temp = 0.0
        cdef double nu = 0.0
        cdef double vel_dot_pos = 0.0

        make_coords_3d(self.s_x, self.s_y, self.s_z, self._pnt1, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, self._pnt2, dest_pid)

        pos_ds.x = self._pnt2.x - self._pnt1.x
        pos_ds.y = self._pnt2.y - self._pnt1.y
        pos_ds.z = self._pnt2.z - self._pnt1.z

        vel_ds.x = self.d_velx.data[dest_pid] - self.s_velx.data[source_pid]
        vel_ds.y = self.d_vely.data[dest_pid] - self.s_vely.data[source_pid]
        vel_ds.z = self.d_velz.data[dest_pid] - self.s_velz.data[source_pid]

        vel_dot_pos = vel_ds.dot(pos_ds)

        if vel_dot_pos < 0.0:
            nu = 2.0*self.alpha*h*self.speed_of_sound.value
            nu /= (self.s_rho.data[source_pid] + self.d_rho.data[dest_pid])
            temp = vel_dot_pos/(pos_ds.norm() + self.epsilon*h*h)
            pi_ab = (-1.0)*nu*temp
            kernel.gradient(self._pnt2, self._pnt1, h, grad)

        nr[0] += pi_ab*grad.x
        nr[1] += pi_ab*grad.y
        nr[2] += pi_ab*grad.z
        
###############################################################################
# `MonaghanArtViscComponent` class.
###############################################################################
cdef class MonaghanArtViscComponent(SPHComponent):
    """
    Component to compute artificial viscosity.
    """
    
    def __cinit__(self, SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_None,
                  double alpha=0.03,
                  double epsilon=0.01,
                  double beta=1.0,
                  SpeedOfSound speed_of_sound=None,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        
        if solver is not None:
            self.speed_of_sound = solver.speed_of_sound
        else:
            self.speed_of_sound = speed_of_sound

        # artificial viscosity to be only computed on fluids.
        self.dest_types=[Fluid]
        
        # default we accept only fluids.
        self.add_input_entity_type(Fluid)

        # the sph function class to use.
        self.sph_func_class = SPHMonaghanArtVisc3D

    cpdef int update_property_requirements(self) except -1:
        """
        Setup the property requirements of this component.
        """
        # all sources need velocity arrays.
        for t in self.source_types:
            self.add_read_prop_requirement(t, ['u', 'v', 'w', 'rho'])

        # all detination types need the following.
        for t in self.dest_types:
            self.add_read_prop_requirement(t, ['u', 'v', 'w', 'rho'])

            self.add_write_prop_requirement(t, 'ax')
            self.add_write_prop_requirement(t, 'ay')
            self.add_write_prop_requirement(t, 'az')

            self.add_private_prop_requirement(t, 'avisc_x')
            self.add_private_prop_requirement(t, 'avisc_y')
            self.add_private_prop_requirement(t, 'avisc_z')

        return 0

    cdef int compute(self) except -1:
        """
        The compute function.
        """
        cdef int num_dest
        cdef int i
        
        cdef EntityBase e
        cdef SPHBase calc
        cdef ParticleArray parr

        self.setup_component()

        num_dest = len(self.dest_list)
        
        for i in range(num_dest):
            e = self.dest_list[i]
            parr = e.get_particle_array()
            
            calc = self.sph_calcs[i]

            calc.sph3('avisc_x', 'avisc_y', 'avisc_z', True)

            parr.ax -= parr.avisc_x
            parr.ay -= parr.avisc_y
            parr.az -= parr.avisc_z

        return 0

    cpdef int setup_component(self) except -1:
        """
        Setup component before execution.
        """
        if self.setup_done == True:
            return 1
        
        logger.info('Setting up component %s'%(self.name))

        if self.speed_of_sound is None:
            if self.solve is not None:
                self.speed_of_sound = self.solver.speed_of_sound
        
        logger.info('viscosity_components.pyx : speed of sound %f'%(
                self.speed_of_sound.value))

        # now call the SPHComponent's setup_component
        SPHComponent.setup_component(self)
    
        self.setup_done = True

        return 0

    cpdef setup_sph_function(self, EntityBase source, EntityBase dest,
                             SPHFunctionParticle sph_func):
        """
        Sets up component specific properties of the sph function used.
        """
        cdef SPHMonaghanArtVisc3D func = <SPHMonaghanArtVisc3D>sph_func

        func.alpha = self.alpha
        func.epsilon = self.epsilon
        func.speed_of_sound = self.speed_of_sound
        
        logger.info('Setting func : %s %s %s'%(source.name, dest.name, sph_func))
        logger.info('Setting alpha to : %f'%(self.alpha))
        logger.info('Setting epsilon to : %f'%(self.epsilon))
        logger.info('Setting sound speed to : %f'%(self.speed_of_sound.value))
