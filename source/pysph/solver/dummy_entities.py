"""
Set of dummy entities used for tests.
"""

# local imports
from pysph.base.particle_array import ParticleArray
from pysph.solver.entity_base import EntityBase
from pysph.solver.entity_types import EntityTypes

class DummyEntity(EntityBase):
    """
    Dummy entity class for test purposes.
    """
    def __init__(self, name='', properties={}, particle_props={}, *args, **kwargs):
        """
        """

        self.type = EntityTypes.Entity_Dummy
        self.parr = ParticleArray(name=self.name, **particle_props)

    def get_particle_array(self):
        return self.parr
