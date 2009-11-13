"""
Contains various components to compute pressure gradients.
"""

# local imports
from pysph.base.particle_array cimport ParticleArray

from pysph.sph.pressure_funcs cimport SPHSymmetricPressureGradient3D
from pysph.solver.sph_component cimport *
from pysph.solver.entity_types cimport EntityTypes
from pysph.solver.entity_base cimport Fluid



################################################################################
# `SPHSymmetricPressureGradientComponent` class.
################################################################################
cdef class SPHSymmetricPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    def __cinit__(self, SolverBase solver=None, 
                  ComponentManager component_manager=None, 
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  list source_types=[EntityTypes.Entity_Fluid],
                  list dest_types=[EntityTypes.Entity_Fluid],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_By_Type,
                  type sph_class=SPHBase,
                  type sph_func_class=SPHSymmetricPressureGradient3D,
                  *args, **kwargs):
        """
        Constructor.
        """        
        # only accept fluids.
        self.add_input_entity_type(EntityTypes.Entity_Fluid)
        
    cpdef int update_property_requirements(self) except -1:
        """
        Update the components property requirements.
        """
        cdef dict rp = self.information.get_dict(self.PARTICLE_PROPERTIES_READ)
        cdef dict pp = self.information.get_dict(
            self.PARTICLE_PROPERTIES_PRIVATE)
        cpdef dict wp = self.information.get_dict(self.PARTICLE_PROPERTIES_WRITE)
        
        rp[EntityTypes.Entity_Fluid] = ['p', 'h', 'm']

        wp[EntityTypes.Entity_Fluid] = [{'name':'ax', 'default':0.0},
                                        {'name':'ay', 'default':0.0},
                                        {'name':'az', 'default':0.0}]

        pp[EntityTypes.Entity_Fluid] = [{'name':'pacclr_x', 'default':0.},
                                        {'name':'pacclr_y', 'default':0.},
                                        {'name':'pacclr_z', 'default':0.}]

    cdef int compute(self) except -1:
        """
        """
        cdef int num_dest = len(self.dest_list)
        cdef int i
        
        cdef Fluid fluid
        cdef SPHBase calc
        cdef ParticleArray parr
        
        for i from 0 <= i < num_dest:
            fluid = self.dest_list[i]
            parr = fluid.get_particle_array()
            calc = self.sph_calcs[i]

            calc.sph3('pacclr_x', 'pacclr_y', 'pacclr_z', True)
            
            parr.ax += parr.pacclr_x
            parr.ay += parr.pacclr_y
            parr.az += parr.pacclr_z
