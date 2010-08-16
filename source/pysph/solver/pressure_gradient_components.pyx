"""
Contains various components to compute pressure gradients.
"""

# local imports
from pysph.base.particle_array cimport ParticleArray

from pysph.sph.pressure_funcs cimport SPHSymmetricPressureGradient3D
from pysph.solver.sph_component cimport *
from pysph.solver.solid import Solid
from pysph.solver.fluid import Fluid
from pysph.solver.fluid cimport Fluid

################################################################################
# `SPHSymmetricPressureGradientComponent` class.
################################################################################
cdef class SPHSymmetricPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    def __cinit__(self, str name='',
                  SolverBase solver=None, 
                  ComponentManager component_manager=None, 
                  list entity_list=[],
                  NNPSManager nnps_manager=None,
                  KernelBase kernel=None,
                  list source_list=[],
                  list dest_list=[],
                  bint source_dest_setup_auto=True,
                  int source_dest_mode=SPHSourceDestMode.Group_By_Type,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.source_dest_mode = SPHSourceDestMode.Group_By_Type
        self.source_types = [Fluid]
        self.dest_types = [Fluid]
        
        self.sph_func_class = SPHSymmetricPressureGradient3D

        # only accept fluids.
        self.add_input_entity_type(Fluid)
        
    cpdef int update_property_requirements(self) except -1:
        """
        Update the components property requirements.
        """
        for t in self.source_types:
            self.add_read_prop_requirement(t, ['m', 'p', 'h', 'rho'])

        for t in self.dest_types:
            self.add_read_prop_requirement(t, ['p', 'h', 'rho'])

            self.add_write_prop_requirement(t, 'ax')
            self.add_write_prop_requirement(t, 'ay')
            self.add_write_prop_requirement(t, 'az')

            self.add_private_prop_requirement(t, 'pacclr_x')
            self.add_private_prop_requirement(t, 'pacclr_y')
            self.add_private_prop_requirement(t, 'pacclr_z')
            
        return 0

    cdef int compute(self) except -1:
        """
        """
        cdef int num_dest
        cdef int i
        
        cdef Fluid fluid
        cdef SPHBase calc
        cdef ParticleArray parr
        
        # call setup component.
        self.setup_component()

        num_dest = len(self.dest_list)
        
        for i from 0 <= i < num_dest:
            fluid = self.dest_list[i]
            parr = fluid.get_particle_array()
            calc = self.sph_calcs[i]

            calc.sph3('pacclr_x', 'pacclr_y', 'pacclr_z', True)
            
            parr.pacclr_x *= -1.0
            parr.pacclr_y *= -1.0
            parr.pacclr_z *= -1.0

            parr.ax += parr.pacclr_x
            parr.ay += parr.pacclr_y
            parr.az += parr.pacclr_z

        return 0
