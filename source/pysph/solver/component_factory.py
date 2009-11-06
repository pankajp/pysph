"""
Factory to create any component.
"""

class ComponentFactory:
    """
    Factory class to create any component.
    """
    def __init__(self):
        """
        """
        raise SystemError, 'Do not Instantiate the ComponentFactory class'

    @staticmethod
    def get_component(comp_category, comp_type):
        """
        Creates and returns the requested component type.

        **Parameters**
            
            - comp_category - category to which the component belogns.
            - comp_name - string identifying the type of component needed.

        """
        if comp_category == 'ode_stepper':
            return ComponentFactory.get_ode_stepper(comp_type)
        elif comp_category == 'integrator':
            return ComponentFactory.get_integrator(comp_type)
        else:
            return None

    @staticmethod
    def get_ode_stepper(comp_type):
        """
        Creates and returns the requested ode stepper.
        """
        import pysph.solver.integrator_base

        if comp_type == 'base' or comp_type == 'euler':
            return pysph.solver.integrator_base.ODESteper()
        else:
            return None
        
    @staticmethod
    def get_integrator(comp_type):
        """
        Creates and returns the requested integrator.
        """
        import pysph.solver.integrator_base

        if comp_type == 'base' or comp_type =='euler':
            return pysph.solver.integrator_base.Integrator()
        else:
            return None
