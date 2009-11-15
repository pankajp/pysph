"""
Set of xsph integrators.
"""

# logger import
import logging
logger = logging.getLogger()

# local imports
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.solver_base cimport *
from pysph.solver.integrator_base cimport *

################################################################################
# `EulerXSPHIntegrator` class.
################################################################################
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

        self.setup_position_stepper()

    cpdef int update_property_requirements(self) except -1:
        """
        Update the property requirements of the integrator.
        """
        Integrator.update_property_requirements(self)

        # add the property required for storing the velocity corrections.
        cdef dict wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        
        fluid_props = wp[EntityTypes.Entity_Fluid]
        fluid_props.extend([{'name':'del_u', 'default':0.0},
                            {'name':'del_v', 'default':0.0},
                            {'name':'del_w', 'default':0.0}])

    def setup_position_stepper(self):
        """
        Sets the position stepper for fluids to use XSPHPositionStepper
        """
        ip = self.information.get_dict(self.INTEGRATION_PROPERTIES)
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
        steppers[EntityTypes.Entity_Fluid] = 'euler_xsph_position_stepper'
