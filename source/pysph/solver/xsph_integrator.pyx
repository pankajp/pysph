"""
Set of xsph integrators.
"""

# logger import
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.solver_base cimport *
from pysph.solver.integrator_base cimport *
from pysph.solver.runge_kutta_integrator cimport *
###############################################################################
# `EulerXSPHIntegrator` class.
###############################################################################
cdef class EulerXSPHIntegrator(Integrator):
    """
    Euler integrator to perform xsph corrected stepping of velocity.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  int dimension=3,
                  *args, **kwargs):
        """
        Constructor.
        """
        pass

    def __init__(self, name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 dimension=3,
                 *args, **kwargs):

        Integrator.__init__(self, name=name, solver=solver,
                            component_manager=component_manager,
                            entity_list=entity_list,
                            dimension=dimension, *args, **kwargs)
        
        self.setup_position_stepper()

    cpdef int update_property_requirements(self) except -1:
        """
        Update the property requirements of the integrator.
        """
        Integrator.update_property_requirements(self)

        # add the property required for storing the velocity corrections.
        cdef dict wp = self.particle_props_write
        
        fluid_props = wp.get(Fluid)

        if fluid_props is None:
            fluid_props = []
            wp[Fluid] = fluid_props

        fluid_props.extend([{'name':'del_u', 'default':0.0},
                            {'name':'del_v', 'default':0.0},
                            {'name':'del_w', 'default':0.0}])
        return 0

    def setup_position_stepper(self):
        """
        Sets the position stepper for fluids to use XSPHPositionStepper
        """
        ip = self.information[self.INTEGRATION_PROPERTIES]
        prop_info = ip.get('position')
        
        if prop_info is None:
            logger.warn('Position stepping not enabled')
            return

        steppers = prop_info.get('steppers')
        integrand = prop_info.get('integrand')
        integral = prop_info.get('integral')

        if steppers is None:
            steppers = {}
            prop_info['steppers'] = steppers
        
        # set fluids to use xsph steppers
        steppers[Fluid] = 'euler_xsph_position_stepper'

###############################################################################
# `RK2XSPHIntegrator` class.
###############################################################################
cdef class RK2XSPHIntegrator(RK2Integrator):
    """
    RK2 integrator to perform xsph corrected stepping of velocity.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None,
                  ComponentManager component_manager=None,
                  list entity_list=[],
                  int dimension=3,
                  *args, **kwargs):
        """
        Constructor.
        """
        pass

    def __init__(self, name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 dimension=3,
                 *args, **kwargs):
        RK2Integrator.__init__(self, name=name, solver=solver,
                               component_manager=component_manager,
                               entity_list=entity_list,
                               dimension=dimension, *args, **kwargs)

        self.setup_position_stepper()

    cpdef int update_property_requirements(self) except -1:
        """
        Update the property requirements of the integrator.
        """
        RK2Integrator.update_property_requirements(self)

        # add the property required for storing the velocity corrections.
        cdef dict wp = self.particle_props_write
        
        fluid_props = wp.get(Fluid)

        if fluid_props is None:
            fluid_props =  []
            wp[Fluid] = fluid_props

        fluid_props.extend([{'name':'del_u', 'default':0.0},
                            {'name':'del_v', 'default':0.0},
                            {'name':'del_w', 'default':0.0},
                            {'name':'u_correct_prev', 'default':0.0},
                            {'name':'v_correct_prev', 'default':0.0},
                            {'name':'w_correct_prev', 'default':0.0}
                            ])

        return 0

    def setup_position_stepper(self):
        """
        Sets the position stepper for fluids to use XSPHPositionStepper
        """
        ip = self.information[self.INTEGRATION_PROPERTIES]
        prop_info = ip.get('position')
        
        if prop_info is None:
            logger.warn('Position stepping not enabled')
            return

        # setup the _stepper_info dict appropriately.
        prop_info = self._stepper_info.get('position')
        
        if prop_info is None:
            prop_info = {}
            self._stepper_info['position'] = prop_info
        
        # for the first step 
        first_step = prop_info.get(1)
        if first_step is None:
            first_step = {}
            prop_info[1] = first_step

        first_step[Fluid] = 'rk2_xsph_step1_position_stepper'
        
        second_step = prop_info.get(2)

        if second_step is None:
            second_step = {}
            prop_info[2] = second_step

        second_step[Fluid] =(
            'rk2_xsph_step2_position_stepper')
