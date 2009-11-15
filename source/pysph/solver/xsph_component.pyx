"""
Component to implement XSPH Velocity correction.
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.sph.sph_func cimport *
from pysph.solver.sph_component cimport *
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.integrator_base cimport *

################################################################################
# `XSPHFunction3D` class.
################################################################################
cdef class XSPHFunction3D(SPHFunctionParticle3D):
    """
    """
    def __init__(self, ParticleArray source, ParticleArray dest, str h='h', str
                 mass='m', str rho='rho',
                 str coord_x='x', str coord_y='y', str coord_z='z',
                 str velx='u', str vely='v', str velz='z', bint
                 setup_arrays=True, *args, **kwargs):
        """
        Constructor.
        """
        SPHFunctionParticle3D.__init__(self, source, dest, h, mass, rho,
                                       coord_x, coord_y, coord_z,
                                       velx, vely, velz, setup_arrays, *args,
                                       **kwargs)
        
    cdef void eval(self, int source_pid, int dest_pid, KernelBase kernel, double
                   *nr, double *dnr):
        """
        """
        cdef double w, rho_bar, fac, h
        cdef Point vel = Point()
        cdef Point s_pos = self._pnt1
        cdef Point d_pos = self._pnt2
        
        make_coords_3d(self.s_x, self.s_y, self.s_z, s_pos, source_pid)
        make_coords_3d(self.d_x, self.d_y, self.d_z, d_pos, dest_pid)

        h = 0.5*(self.s_h.data[source_pid] + self.d_h.data[dest_pid])
        w = kernel.function(d_pos, s_pos, h)
        
        vel.x = self.s_u.data[source_pid] - self.d_u.data[dest_pid]
        vel.y = self.s_v.data[source_pid] - self.d_v.data[dest_pid]
        vel.z = self.s_w.data[source_pid] - self.d_w.data[dest_pid]
        rho_bar = (self.d_rho[dest_pid] + self.s_rho[source_pid])*0.5
        rho_bar = 1./rho_bar

        fac = rho_bar*self.s_mass.data[source_pid]*w
        nr[0] += vel.x*fac
        nr[1] += vel.y*fac
        nr[2] += vel.z*fac

    cpdef int output_fields(self) except -1:
        """
        Returns 3 - this computes three values - correction of each velocity
        component.
        """
        return 3

################################################################################
# `XSPHVelocityComponent` class.
################################################################################
cdef class XSPHVelocityComponent(SPHComponent):
    """
    Component to compute velocity correction using the XSPH method.
    """
    category = 'misc_component'
    identifier = 'xsph_correction'

    def __cinit__(self, name='', solver=None,
                  component_manager=None,
                  entity_list=[],
                  nnps_manager=None,
                  kernel=None,
                  source_list=[],
                  dest_list=[],
                  source_dest_setup_auto=True,
                  source_dest_mode=SPHSourceDestMode.Group_None,
                  epsilon=0.5,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.source_types = [EntityTypes.Entity_Fluid]
        self.dest_types = [EntityTypes.Entity_Fluid]

        self.sph_func_class = XSPHFunction3D
        self.add_input_entity_type(EntityTypes.Entity_Fluid)

        self.epsilon = epsilon

    cpdef int update_property_requirements(self) except -1:
        """
        Update the property requirements of the component.

        For each particle
            - read - u, v, w, m, rho, x, y, z
            - private - del_u, del_v, del_w

        """
        cdef dict rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        cdef dict pp = self.information.get_dict(self.PARTICLE_PROPERTIES_PRIVATE)

        fluid_props_read = rp[EntityTypes.Entity_Fluid]
        fluid_props_read.extend(['u', 'v', 'w', 'm', 'rho'])

        fluid_props_pr = pp[EntityTypes.Entity_Fluid]
        fluid_props_pr = [{'name':'del_u', 'default':1.0},
                          {'name':'del_v', 'default':1.0},
                          {'name':'del_w', 'default':1.0}]
        
    cpdef int py_compute(self) except -1:
        """
        """
        for i in range(len(self.dest_list)):
            e = self.dest_list[i]
            calc = self.sph_calcs[i]
            parr = e.get_particle_array()

            calc.sph3('del_u', 'del_v', 'del_w', exclude_self=True)

            parr.del_u *= self.epsilon
            parr.del_v *= self.epsilon
            parr.del_w *= self.epsilon

    def get_velocity_corrections(self, particle_array):
        """
        Convenience function to get the velocity corrections for a particular
        particle object. To get the correct values of the velocity corrections
        , call the compute method of the component and then use this function.

        """
        return particle_array.del_u, particle_array.del_v, particle_array.del_w


################################################################################
# `XPSHPositionStepper` class.
################################################################################
cdef class EulerXSPHPositionStepper(ODEStepper):
    """
    Position stepper with XSPH correction.
    """
    def __cinit__(self, str name='', SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  str prop_name='',
                  list integrands=[], list integrals=[],
                  TimeStep time_step=None,
                  double epsilon=0.5,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.epsilon = epsilon

    cpdef int setup_component(self) except -1:
        """
        Sets up the component before executing.
        """
        if self.setup_done == True:
            return 0

        if self.prop_name != 'position':
            logger.warn(
                'XSPHPositionStepper used for stepping %s',self.prop_name)

        # array setup will be the same as done by the ODEStepper
        ODEStepper.setup_component(self)

        # create the xsph component and add arrays to the entities as required.
        self.xsph_component = XSPHVelocityComponent(name=self.name+'comp',
                                                    solver=self.solver,
                                                    component_manager=self.cm,
                                                    entity_list=self.entity_list,
                                                    source_dest_setup_auto=True,
                                                    epsilon=self.epsilon)
        self.setup_done = True

        return 0

    cdef int compute(self) except -1:
        """
        Performs simple euler integration by the time_step, for the said arrays
        for each entity. The XSPH velocity correction is added to the rate
        before stepping.

        Each entity 'MUST' have a particle array representation. Otherwise
        stepping won't be done currently.

        """
        cdef EntityBase e
        cdef int i, num_entities
        cdef int num_props, p, j
        cpdef ParticleArray parr
        cdef numpy.ndarray an, bn, an1, correction
        cdef DoubleArray _an, _bn, _an1
        
        # make sure the component has been setup
        self.setup_component()

        # compute the velocity corrections.
        self.xsph_component.compute()

        num_entities = len(self.entity_list)
        num_props = len(self.integrand_names)

        for i from 0 <= i < num_entities:
            e = self.entity_list[i]
            
            parr = e.get_particle_array()
            dels = self.xsph_component.get_velocity_corrections(parr)

            if parr is None:
                logger.warn('no particle array for %s'%(e.name))
                continue
            
            for j from 0 <= j < num_props:
                an = parr._get_real_particle_prop(self.integral_names[j])
                bn = parr._get_real_particle_prop(self.integrand_names[j])
                an1 = parr._get_real_particle_prop(self.next_step_names[j])

                correction = dels[i]

                an1[:] = an + (bn + correction)*self.time_step.value        
