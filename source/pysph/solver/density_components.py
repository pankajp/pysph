"""
Module containing various density components.
"""

# standard imports


# local imports
from pysph.sph.sph_calc import SPHBase
from pysph.sph.density_funcs import SPHRho3D, SPHDensityRate3D
from pysph.solver.sph_component import *
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid

###############################################################################
# `SPHDensityComponent` class.
###############################################################################
class SPHDensityComponent(PYSPHComponent):
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
                 source_dest_setup_auto=True,
                 source_dest_mode=SPHSourceDestMode.Group_None,
                 *args, **kwargs):
        """
        Constructor.
        """
        self.source_types=[Fluid]
        self.dest_types=[Fluid]

        self.sph_func_class = SPHRho3D

        self.add_input_entity_type(Fluid)

    def update_property_requirements(self):
        """
        Update the property requirements of the component.
        
        For every entity type that this component will accept as input, add the
        following properties:
            - read - m and h
            - write - rho, _tmp1
        """
        for t in self.source_types:
            self.add_read_prop_requirement(t, ['m', 'rho', 'h'])

        for t in self.dest_types:
            self.add_write_prop_requirement(e_type=t, prop_name='rho',
                                            default_value=0.)
            self.add_write_prop_requirement(e_type=t, prop_name='_tmp1',
                                            default_value=0.0)

        return 0

    def py_compute(self):
        """
        """
        # make sure component is setup.
        self.setup_component()

        for i in range(len(self.dest_list)):
            e = self.dest_list[i]
            calc = self.sph_calcs[i]
            parr = e.get_particle_array()
            
            # compute the new densities in the _tmp1 array.
            calc.sph1('_tmp1')
            
            # copy values from _tmp1 to rho
            parr.rho[:] = parr._tmp1
            
        return 0

###############################################################################
# `SPHDensityRateComponent` class.
###############################################################################
class SPHDensityRateComponent(PYSPHComponent):
    """
    Component to compute the rate of change of density.
    """
    category = 'density_component'
    identifier = 'sph_density_rate'

    def __init__(self, name='', solver=None, 
                 component_manager=None,
                 entity_list=[], 
                 nnps_manager=None,
                 kernel=None,
                 source_list=[],
                 dest_list=[],
                 source_dest_setup_auto=True,
                 source_dest_mode=SPHSourceDestMode.Group_None,
                 *args, **kwargs):
        """
        Constructor.
        """
        self.source_types=[Fluid]
        self.dest_types=[Fluid]

        self.sph_func_class = SPHDensityRate3D

        self.add_input_entity_type(Fluid)

    def update_property_requirements(self):
        """
        Update the property requirements of the component.
        
        For every entity type that this component will accept as input, add the
        following properties:
            - read - m and h
            - write - rho, _tmp1
        """
        inp_types = self.input_types

        for t in self.source_types:
            self.add_read_prop_requirement(t, ['m', 'h', 'u', 'v', 'w'])

        for t in self.dest_types:
            self.add_read_prop_requirement(t, ['m', 'h', 'u', 'v', 'w'])
            self.add_write_prop_requirement(t, prop_name='rho_rate',
                                            default_value=0.0)
            self.add_write_prop_requirement(t, prop_name='_tmp1',
                                            default_value=0.0)
        return 0

    def py_compute(self):
        """
        """
        # make sure component is setup.
        self.setup_component()

        for i in range(len(self.dest_list)):
            e = self.dest_list[i]
            calc = self.sph_calcs[i]
            parr = e.get_particle_array()
            
            # compute the rate of change in the _tmp1 array.
            calc.sph1('_tmp1')
            
            # add the rate compute my this component to the density rates.
            parr.rho_rate[:] = parr._tmp1
            
        return 0
    
