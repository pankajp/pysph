"""
Module contains classes to represent polygon arrays.
"""

# local imports
from pysph.base.particle_array import ParticleArray

cdef class PolygonArray:
    """
    Class to represent array of polygons.

    **Member Variables**

     - vertex_array - a ParticleArray containing vertex information.
     - cell_array - Information about how each polygon in the polygon array are
       formed.
     - particle_rep - The particle representation of all polygons in this
       polygon array are stored in this particle array. 
     - cell_array - a list of LongArrays. Each contains indices into the vertex
       array, indicating the points that make this polygon.
     - cell_information - Properties at each cell.

    **Notes**
    
     - Just encapsulates polygonal data.
    
    **Issues**

     - We may have to give ParticleArray like accessors to this class, so that
       properties etc. can be got in a similar manner.
     - Will also have to provide functions to add vertices, cells etc.
    
    """
    def __init__(self, name='', vertex_array=None, cell_array=[],
                 cell_information={}, particle_rep=None):
        """
        Constructor.
        """
        self.vertex_array = vertex_array
        self.cell_array = cell_array
        self.cell_information = cell_information
        self.name = name
        self.particle_rep = particle_rep

    def get_number_of_polygons(self):
        """
        Returns the number of polygons.
        """
        return len(self.cell_array)

    def get_number_of_vertices(self):
        """
        Returns the number of vertices making up this polygon array.
        """
        return self.vertex_array.get_number_of_particles()
