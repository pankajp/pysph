"""
Module contains various basic RK integrators.
"""

# standard imports
import logging
logger = logging.getLogger()

cimport numpy

# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.integrator_base cimport *
from pysph.solver.solver_base cimport *
from pysph.solver.component_factory import ComponentFactory as cfac

################################################################################
# `RK2Integrator` class.
################################################################################
cdef class RK2Integrator(Integrator):
    """
    Runge-Kutta-2 integrator class.
    """
    def __cinit__(self, str name='', SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  int dimension=3,
                  *args, **kwargs):
        """
        Constructor.
        """
        self._stepper_info = {}

        self._step1_default_steppers = {}
        self._step2_default_steppers = {}        
        
        self.information.set_dict(self.DEFAULT_STEPPERS,
                                  self._step1_default_steppers)
        self.step_being_setup = 1

    def setup_defualt_steppers(self):
        """
        Sets up the default integrator to be used by this integrator, when no
        other stepping information is provided along with the property step
        info.       
        
        """
        self._step1_default_steppers['default'] = 'euler'
        self._step2_default_steppers['default'] = 'rk2_second_step'

    cpdef int setup_component(self) except -1:
        """
        Setup the runge kutta 2 integrator.
        """

        # setup pre-integration components
        self._setup_pre_integration_components()

        # init internal data for setting up the first step.
        self._init_for_step_setup(1)

        # setup the first step.
        self._setup_step()

        # init internal data for setting up the second step.
        self._init_for_step_setup(2)

        # setup the second step.
        self._setup_step()

        # setup post-integration components.
        self._setup_post_integration_components()

    def _init_for_step_setup(self, step_num):
        """
        """
        self.step_being_setup = step_num
        
        # set the default steppers to _step1_default_steppers
        if step_num == 1:
            self.information.set_dict(self.DEFAULT_STEPPERS,
                                      self._step1_default_steppers)
        else:
            self.information.set_dict(self.DEFAULT_STEPPERS,
                                      self._step2_default_steppers)

        # set the property steppers appropriately.
        curr_stepper = self._stepper_info
        io = self.information.get_list(self.INTEGRATION_ORDER)
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)

        for prop_name in io:
            prop_stepper_info = self._stepper_info.get(prop_name)
            prop_info_1 = ip.get(prop_name)
            stepper_info = prop_info_1.get('steppers')
            stepper_info.clear()
            
            if prop_stepper_info is None:
                continue

            # from the stepper info of this integrator, get information about
            # stepper for this step-number - 1/2.
            step_info = prop_stepper_info.get(step_num)
            
            if step_info is None:
                continue

            stepper_info.update(step_info)
        
    def _setup_property_copiers(self, prop_stepper_dict):
        """
        Setup the copiers for the current step being setup.
        """
        if self.step_being_setup == 1:
            self._setup_property_copiers_for_first_step(prop_stepper_dict)
        else:
            self._setup_property_copiers_for_second_step(prop_stepper_dict)

    def _setup_property_copiers_for_first_step(self, prop_stepper_dict):
        """
        Setup property copiers for the 1st step.
        """
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        io = self.information.get_list(self.INTEGRATION_ORDER)

        prev_steppers = {}
        
        for prop_name in io:
            stepper_list = prop_stepper_dict[prop_name]
            prop_info = ip.get(prop_name)

            for stepper in stepper_list:
                self._add_prev_copiers(stepper, stepper.integral_names, prop_name,
                                       '_lhs')
                self._add_prev_copiers(stepper, stepper.integrand_names,
                                       prop_name, '_rhs')

            # now copy the next values of integrals (lhs) into current values.
            for stepper in stepper_list:
                cop_name = 'next_copier_'+prop_name
                copier = cfac.get_component('copiers', 'copier',
                                            name=cop_name,
                                            solver=self.solver,
                                            component_manager=self.cm,
                                            entity_list=stepper.entity_list,
                                            from_arrays=stepper.next_step_names,
                                            to_arrays=stepper.integral_names)
                if copier is None:
                    msg = 'Could not create copier for %s'%(prop_name)
                    logger.warn('Could not create copier for %s'%(prop_name))
                    raise SystemError, msg

                self.execute_list.append(copier)

    def _add_prev_copiers(self, stepper_obj, array_names, prop_name,
                          copier_tag):
        """
        Adds a copier to copy current values to prev arrays.
        """        
        cop_name='prev_copier_'+prop_name+copier_tag
        prev_names = []
        for a_name in array_names:
            prev_name = a_name + '_prev'
            prev_names.append(prev_name)
            
            copier = cfac.get_component('copiers', 'copier',
                                        name=cop_name,
                                        solver=self.solver,
                                        component_manager=self.cm,
                                        entity_list=stepper_obj.entity_list,
                                        from_arrays=array_names,
                                        to_arrays=prev_names)
            if copier is None:
                msg = 'Could not create copier for %s'%(prop_name)
                logger.warn('Could not create copier for %s'%(prop_name))
                raise SystemError, msg
            
            self.execute_list.append(copier)                    

    def _setup_property_copiers_for_second_step(self, prop_stepper_dict):
        """
        Setup property copiers for the 2nd step.
        """
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        io = self.information.get_list(self.INTEGRATION_ORDER)

        for prop_name in io:
            stepper_list = prop_stepper_dict[prop_name]
            # create a copier for each stepper in the stepper_list.
            for stepper in stepper_list:
                cop_name = 'copier_2_'+prop_name
                copier = cfac.get_component('copiers', 'copier', 
                                            name=cop_name, 
                                            solver=self.solver,
                                            entity_list=stepper.entity_list,
                                            from_arrays=stepper.next_step_names,
                                            to_arrays=stepper.integral_names)
                if copier is None:
                    msg = 'Could not create copier for %s'%(prop_name)
                    logger.warn('Could not create copier for %s'%(prop_name))
                    raise SystemError, msg

                self.execute_list.append(copier)    

    cpdef int update_property_requirements(self) except -1:
        """
        Update property requirements.
        """
        cdef str prop_name
        cdef dict ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
        cdef dict wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
                
        for prop_name in ip.keys():
            prop_info = ip.get(prop_name)
            intgnd = prop_info.get('integrand')
            intgl = prop_info.get('integral')
            
            e_types = prop_info.get('entity_types')

            if e_types is None or len(e_types) == 0:
                e_props = wp.get(EntityTypes.Entity_Base)
                if e_props is None:
                    e_props = []
                    wp[EntityTypes.Entity_Base] = e_props
                # meaning this property is to be applied to all entity types. 
                # add the integrand and integral arrays
                for i in range(len(intgnd)):
                    p = intgnd[i]
                    e_props.append({'name':p, 'default':None})
                    p = p+'_prev'
                    e_props.append({'name':p, 'default':None})
                    p = intgl[i]
                    e_props.append({'name':p, 'default':None})
                    p = p+'_prev'
                    e_props.append({'name':p, 'default':None})
                    
            else:
                for e_type in e_types:
                    e_props = wp.get(e_type)
                    if e_props is None:
                        e_props = []
                        wp[e_type] = e_props
                    for i in range(len(intgnd)):
                        p = intgnd[i]
                        e_props.append({'name':p, 'default':None})
                        p = p+'_prev'
                        e_props.append({'name':p, 'default':None})
                        p = intgl[i]
                        e_props.append({'name':p, 'default':None})
                        p = p+'_prev'
                        e_props.append({'name':p, 'default':None})
        return 0

################################################################################
# `RK2SecondStep` class.
################################################################################
cdef class RK2SecondStep(ODEStepper):
    """
    Class to perform the 2nd step of runge kutta 2 integrator.
    """
    def __cinit__(self, str name='', SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  str prop_name='',
                  list integrands=[], list integrals=[],
                  TimeStep time_step=None, *args, **kwargs):
        """
        Constructor.
        """
        pass

    cpdef int compute(self) except -1:
        """
        Performs the second step of a RK2 integrator.
        """
        cdef EntityBase e
        cdef int i, num_entities
        cdef int num_props, p, j
        cpdef ParticleArray parr
        cdef numpy.ndarray an, bn, an_1, bn_1_2
       
        # make sure the component has been setup
        self.setup_component()
        
        num_entities = len(self.entity_list)
        num_props = len(self.integrand_names)

        for i from 0 <= i < num_entities:
            e = self.entity_list[i]
            
            parr = e.get_particle_array()
    
            if parr is None:
                logger.warn('no particle array for %s'%(e.name))
                continue
            
            for j from 0 <= j < num_props:
                
                # get values at prev step (now stored in _prev arrays)
                an = parr._get_real_particle_prop(
                    self.integral_names[j]+'_prev')

                bn = parr._get_real_particle_prop(
                    self.integrand_names[j]+'_prev')

                bn_1_2 = parr._get_real_particle_prop(
                    self.integrand_names[j])
                
                an_1 = parr._get_real_particle_prop(
                    self.next_step_names[j])
                
                an_1[:] = an + (bn+bn_1_2)*self.time_step.value*0.5