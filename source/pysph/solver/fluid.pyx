"""
Module to contain class to represent fluids.
"""

# local imports
from pysph.base.cell cimport CellManager
from pysph.solver.fluid import Fluid
from pysph.solver.entity_base cimport *

cdef class Fluid(EntityBase):
    """
    Base class to represent fluids.
    """
    def __cinit__(self, str name='', dict properties={}, 
                  ParticleArray particles=None, 
                  *args, **kwargs):
        """
        Constructor.
        """
        self.type = Fluid
        self.particle_array = particles

        # create an empty particle array if nothing give from input.
        if self.particle_array is None:
            self.particle_array = ParticleArray(name=self.name)
        
        # name of particle array same as name of entity.
        self.particle_array.name = self.name

        # add any default properties that are required of fluids in all kinds
        # of simulations.
        self.add_entity_property('rho', 1000.)
        self.add_entity_property('max_density_variation', 1.0)
        self.add_entity_property('actual_density_variation', 1.0)

    cpdef ParticleArray get_particle_array(self):
        """
        Returns the ParticleArray representing this entity.
        """
        return self.particle_array

    cpdef add_arrays_to_cell_manager(self, CellManager cell_manager):
        """
        Add all arrays that need to be binned for this entity to the cell
        manager.
        
        """
        cell_manager.add_array_to_bin(self.particle_array)
