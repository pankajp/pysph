# local imports
from pysph.base.particle_array cimport ParticleArray
from pysph.base.carray cimport LongArray

cdef class PolygonArray:
    """
    A class to hold the following kinds of polygons:
     
     - points - 0d polygons.
     - lines - 1d polygons.
     - all 2d polygons.

    This class is modeled similar to the vtkPolyData class.

    """
    cdef public ParticleArray vertex_array

    cdef public LongArray points
    cdef public LongArray edges
    cdef public LongArray polygons

    cdef public dict cell_information 
    cdef public str name

    cdef public ParticleArray particle_rep
