"""
Module containing the ParticleManager class.

**Issues**
- Define functionality clearly.
- 
"""
class ParticleManager:
    """
    Class to manage all particle related information. 

    **Member Variables**
     - particle_arrays - list of particle arrays.
     - polygon_arrays - list of polygon arrays.
     - cell_manager - cell manager for spatial subdivision.

    """
    ######################################################################
    # `object` interface
    ######################################################################
    def __init__(self, name='', *args, **kwargs):
        """
        """
        self.name = name

        self.particle_arrays = {}
        self.polygon_arrays = {}
        self.cell_manager = CellManager(particle_manager=self)

    ######################################################################
    # `Public` interface
    ######################################################################
    def add_polygon_array(self, polygon_array, track=False):
        """
        Add a polygon array.

	**Parameters**
         
         - polygon_array - the new polygon array to be added.
         - track - whether the array should be tracked by the cell manager.

    	**Notes**

	**Helper Functions**

    	**Issues**

    	"""
        pass

    def add_particle_array(self, particle_array, track=False):
        """
        Add a particle array.

	**Parameters**
         - particle_array - the particle array to be added.
         - track - whether the array should be tracked by the cell manager.

    	**Notes**

	**Helper Functions**

    	**Issues**

    	"""
        pass

    def update(self):
        """
        Update internal state.
        """
        pass


    ######################################################################
    # Non-public interface
    ######################################################################
    
