"""
Module containing various density components.
"""

# standard imports


# local imports
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D
from pysph.solver.sph_component import SPHComponent
from pysph.solver.entity_types import EntityTypes

################################################################################
# `SPHDensityComponent` class.
################################################################################
class SPHDensityComponent(SPHComponent):
    """
    Basic SPH density computing component.

    """
    category = 'density_component'
    identifier = 'sph_density'


    def __init__(self, name='', solver=None, 
                 component_manager=None,
                 entity_list=[], 
                 nnps_manager=None,
                 kernel=None,
                 source_list=[],
                 dest_list=[],
                 source_types=[EntityTypes.Entity_Fluid],
                 dest_types=[EntityTypes.Entity_Fluid],
                 source_dest_setup_auto=True,
                 source_dest_mode=SPHSourceDestMode.Group_None,
                 sph_class=SPHBase,
                 sph_func_class=SPHRho3D,
                 *args, **kwargs):
        """
        Constructor.
        """

        self.add_input_entity_type(EntityTypes.Entity_Fluid)

    def update_property_requirements(self):
        """
        Update the property requirements of the component.
        
        For every entity type that this component will accept as input, add the
        following properties:
            - read - m and h
            - write - rho, _tmp1
        """
        inp_types = self.information.get_dict(self.INPUT_TYPES)
        wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        
        rp.clear()
        wp.clear()

        for key in input_types.keys():
            rp[key] = [{'name':'m', 'default':1.0},
                       {'name':'h', 'default':1.0}]
            wp[key] = [{'name':'rho', 'default':1000.0},
                       {'name':'_tmp1', 'default':0.0}]

    def py_compute(self):
        """
        """
        for i in range(len(self.dest_list)):
            e = self.dest_list[i]
            calc = self.sph_calcs[i]
            parr = e.get_particle_array()
            
            # compute the new densities in the _tmp1 array.
            calc.sph1('_tmp1')
            
            # copy values from _tmp1 to rho
            parr.rho[:] = parr._tmp1
