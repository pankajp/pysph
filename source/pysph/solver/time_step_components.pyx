"""
Components to compute time step.
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.carray cimport DoubleArray, LongArray
from pysph.base.particle_array cimport ParticleArray
from pysph.base.nnps cimport NNPSManager, \
    FixedDestinationNbrParticleLocator
from pysph.base.point cimport Point

from pysph.solver.entity_base cimport EntityBase
from pysph.solver.speed_of_sound cimport SpeedOfSound
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.solver_base cimport *
from pysph.solver.time_step cimport TimeStep

cdef extern from "math.h":
    double fabs(double)
    float INFINITY

cimport numpy    
import numpy

################################################################################
# `TimeStepComponent` class.
################################################################################
cdef class TimeStepComponent(SolverComponent):
    """
    Base class for all components computing time step.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  TimeStep time_step=None,
                  double max_time_step = -1.0,
                  double min_time_step = -1.0,
                  *args, **kwargs):
        """
        Constructor.
        """
        if solver is None:
            self.time_step = time_step
        else:
            self.time_step = self.solver.time_step

        self.max_time_step = max_time_step
        self.min_time_step = min_time_step

    cpdef int setup_component(self) except -1:
        """
        """
        pass

    cdef int compute(self) except -1:
        """
        """
        raise NotImplementedError, 'TimeStepComponent::compute'


################################################################################
# `MonaghanKosUpdate` class.
################################################################################
cdef class MonaghanKosTimeStepComponent(TimeStepComponent):
    """
    Component to compute time step based on the paper 'Solitary waves on a
    cretan beach'.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  TimeStep time_step=None,
                  double max_time_step=0.0001,
                  double min_time_step=-1.0,
                  SpeedOfSound speed_of_sound=None,
                  NNPSManager nnps_manager=None,
                  double beta=0.3,
                  double default_visc_kernel_radius = 2.0,
                  list viscosity_category_names=['viscosity',
                                                 'viscosity_components'], 
                  *args, **kwargs):
        """
        Constructor.
        """
        self._sigma_arrays = []
        self.beta = beta
        self.nbr_locators = []
        self.viscosity_category_names = []
        self.viscosity_category_names[:] = viscosity_category_names
        self.default_visc_kernel_radius = default_visc_kernel_radius
        self.viscosity_kernel_radius = default_visc_kernel_radius

        self._indices = LongArray(0)
        
        if solver is not None:
            self.speed_of_sound = solver.speed_of_sound
            self.nnps_manager = solver.nnps_manager
        else:
            self.speed_of_sound = speed_of_sound
            self.nnps_manager = nnps_manager

        # accept all entities in input.
        self.add_input_entity_type(EntityTypes.Entity_Base)
            
    cpdef int update_property_requirements(self) except -1:
        """
        Update the property requirements of this component.
        """
        cdef dict input_types = self.information.get_dict(self.INPUT_TYPES)
        
        for t in input_types.keys():
            self.add_write_prop_requirement(t, '_ts_sigma', 0.0)
            self.add_read_prop_requirement(t, ['u', 'v', 'w', 'h'])        
        
    cpdef int setup_component(self) except -1:
        """
        Setup the component.
        """
        if self.setup_done == True:
            return 0
        
        to_remove = []
        for e in self.entity_list:
            parr = e.get_particle_array()

            if parr is None:
                to_remove.append(e)

        # remove entities not providing particle arrays.
        for e in to_remove:
            logger.warn('Removing entity : %s'%(e.name))
            self.entity_list.pop(e)

        # now create the neighbor locators for the component.
        if self.nnps_manager is None:
            if self.solver is not None:
                self.nnps_manager = self.solver.nnps_manager
        
        if self.nnps_manager is None:
            msg = 'NNPS Manager not present - cannot proceed'
            logger.error(msg)
            raise SystemError, msg

        radius_scale = self._find_viscosity_kernel_radius()

        self.nbr_locators[:] = []
        
        for e1 in self.entity_list:
            d_parr = e1.get_particle_array()
            e1_locators = []
            for e2 in self.entity_list:
                s_parr = e2.get_particle_array()
                nbr_loc = self.nnps_manager.get_neighbor_particle_locator(
                    s_parr, d_parr, radius_scale)
                e1_locators.append(nbr_loc)
            self.nbr_locators.append(e1_locators)
    
        self.setup_done = True

    cdef int compute(self) except -1:
        """
        Compute.
        """
        cdef int num_entites, i, np, j
        cdef EntityBase e
        cdef ParticleArray parr
        cdef DoubleArray _sigma, _h
        cdef double max_denom, val, min_ts
        
        max_denom = -INFINITY
        
        # make sure component is setup properly.
        self.setup_component()

        num_entites = len(self.entity_list)

        if num_entites == 0:
            self.time_step.value = self.max_time_step
            logger.warn('No input entities, using %f'%(self.max_time_step))
            return 0

        for i from 0 <= i < num_entites:

            self._compute_sigmas(i)

            e = self.entity_list[i]
            parr = e.get_particle_array()
            np = parr.get_number_of_particles()
            _sigma = parr.get_carray('_ts_sigma')
            _h = parr.get_carray('h')

            for j from 0 <= j < np:
                val = self.speed_of_sound.value + _sigma.data[j]
                if max_denom < val:
                    max_denom = val
            
            min_ts = _h.data[0]*self.beta/max_denom
        
        self.time_step.value = self._ts_range_check(min_ts)

        logger.info('New time step : %1.10f'%(self.time_step.value))

        return 0

    cdef double _ts_range_check(self, double new_ts_value):
        """
        """
        if self.min_time_step < 0:
            if self.max_time_step < 0:
                return new_ts_value
            else:
                if new_ts_value > self.max_time_step:
                    return self.max_time_step
                else:
                    return new_ts_value
        else:
            if self.max_time_step > 0:
                if new_ts_value < self.min_time_step:
                    return self.min_time_step
                elif new_ts_value > self.max_time_step:
                    return self.max_time_step
                else:
                    return new_ts_value
            else:
                if new_ts_value < self.min_time_step:
                    return self.min_time_step
                else:
                    return new_ts_value

        return -1.0

    cdef int _compute_sigmas(self, int entity_index) except -1:
        """
        Compute sigma for every particle.
        """
        cdef EntityBase e, source_e
        cdef ParticleArray s_parr, d_parr
        cdef int np, i, j, idx, k
        cdef int num_entities
        cdef list e_locators
        cdef FixedDestinationNbrParticleLocator s_loc
        cdef Point s_pnt, d_pnt, v_ab, r_ab
        cdef LongArray indices = LongArray(0)
        cdef double search_radius = 0.0
        cdef DoubleArray _h, s_x, s_y, s_z, d_x, d_y, d_z, s_u, s_v, s_w, d_u,\
            d_v, d_w, d_sigma
        cdef double dist

        d_pnt = Point()
        s_pnt = Point()
        r_ab = Point()
        v_ab = Point()

        e = self.entity_list[entity_index]
        d_parr = e.get_particle_array()
        np = d_parr.get_number_of_particles()
        num_entities = len(self.entity_list)
        e_locators = self.nbr_locators[entity_index]

        _h = d_parr.get_carray('h')
        d_x = d_parr.get_carray('x')
        d_y = d_parr.get_carray('y')
        d_z = d_parr.get_carray('z')
        d_u = d_parr.get_carray('u')
        d_v = d_parr.get_carray('v')
        d_w = d_parr.get_carray('w')

        d_sigma = d_parr.get_carray('_ts_sigma')
        # set all values to -INFINITY
        
        for i from 0 <= i < np:
            d_sigma.data[i] = -INFINITY
        
        for i from 0 <= i < num_entities:
            source_e = self.entity_list[i]

            s_parr = source_e.get_particle_array()
            s_loc = e_locators[i]
            s_x = s_parr.get_carray('x')
            s_y = s_parr.get_carray('y')
            s_z = s_parr.get_carray('z')
            s_u = s_parr.get_carray('u')
            s_v = s_parr.get_carray('v')
            s_w = s_parr.get_carray('w')
            
            for j from 0 <= j < np:
                
                indices.reset()

                s_loc.get_nearest_particles(j, indices, 1.0, True)

                d_pnt.x = d_x.data[j]
                d_pnt.y = d_y.data[j]
                d_pnt.z = d_z.data[j]
                
                for k from 0 <= k < indices.length:
                    idx = indices.data[k]

                    s_pnt.x = s_x.data[idx]
                    s_pnt.y = s_y.data[idx]
                    s_pnt.z = s_z.data[idx]

                    r_ab.x = d_pnt.x - s_pnt.x
                    r_ab.y = d_pnt.y - s_pnt.y
                    r_ab.z = d_pnt.z - s_pnt.z

                    v_ab.x = d_u.data[j] - s_u.data[idx]
                    v_ab.y = d_v.data[j] - s_v.data[idx]
                    v_ab.z = d_w.data[j] - s_w.data[idx]

                    dist = s_pnt.distance(d_pnt)

                    if dist < 1e-12:
                        continue

                    val = fabs(r_ab.dot(v_ab)*_h.data[j]/(dist))

                    if val > d_sigma.data[j]:
                        d_sigma.data[j] = val
            
    cpdef double _find_viscosity_kernel_radius(self):
        """
        Find the max radius involved in viscosity computations.
        
        Find components in the solver that "may" represent the viscosity
        components. Get their kernels and use the maximum kernel value found
        here.
        """
        
        cdef double max_radius = -INFINITY

        cc = self.solver.component_categories

        for category in self.viscosity_category_names:
            cat = cc.get(category)

            if cat is None:
                continue

            for comp in cat:
                if hasattr(comp, 'kernel'):
                    if comp.kernel.radius() > max_radius:
                        max_radius = comp.kernel.radius()

        if max_radius < 0:
            self.viscosity_kernel_radius = self.default_visc_kernel_radius
        else:
            self.viscosity_kernel_radius = max_radius
            
        return self.viscosity_kernel_radius


cdef class MonaghanKosForceBasedTimeStepComponent(MonaghanKosTimeStepComponent):
    """
    In addition to above time step considerations, also uses the forces on the
    bodies to compute the time step.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  TimeStep time_step=None,
                  double max_time_step=0.0001,
                  double min_time_step=-1.0,
                  SpeedOfSound speed_of_sound=None,
                  NNPSManager nnps_manager=None,
                  double beta=0.3,
                  double default_visc_kernel_radius = 2.0,
                  list viscosity_category_names=['viscosity',
                                                 'viscosity_components'], 
                  double scale=1.0,
                  *args, **kwargs):

        """
        Constructor.
        """
        self.scale = scale
    
    cdef int compute(self) except -1:
        """
        """

        cdef EntityBase e
        cdef ParticleArray parray
        cdef int i, num_entities
        cdef numpy.ndarray f, a_mag, h, m, dt_2
        cdef double min_ts
        cdef double min_dt_2
        self.setup_component()

        # call the parent classes compute to get the time step.
        MonaghanKosTimeStepComponent.compute(self)

        min_ts = self.time_step.value

        num_entities = len(self.entity_list)

        for i from 0 <= i < num_entities:
            
            e = self.entity_list[i]

            parray = e.get_particle_array()
            
            
            if (parray.has_array('ax') is False or
                parray.has_array('ay') is False or
                parray.has_array('az') is False):
                continue
            
            
            a_mag = (numpy.power(parray.ax, 2) + 
                     numpy.power(parray.ay, 2) +
                     numpy.power(parray.az, 2))
            a_mag = numpy.sqrt(a_mag)

            if numpy.allclose(a_mag, 1e-09):
                continue

            m = parray.get('m')
            h = parray.get('h')
            
            f = a_mag*m

            dt_2  = numpy.divide(h, f)
            min_dt_2 = numpy.min(dt_2)

            if self.scale*min_dt_2 < min_ts:
                min_ts = self.scale*min_dt_2

        self.time_step.value = self._ts_range_check(min_ts)
        logger.info('New time step : %1.10f'%(self.time_step.value))

        return 0
