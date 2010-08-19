"""
Contains base classes for representing any kind of geometric objects.
"""

# local imports
from pysph.base.particle_array cimport ParticleArray



###############################################################################
# `GeometryBase` class.
###############################################################################
cdef class GeometryBase:
    """
    Base class for all geometries associtaed with entities.
    """    
    
    def __cinit__(self, object entity=None, *args, **kwargs):
        """
        """
        self.entity = entity

    cpdef get_tangent_plane(self, Point pnt, Plane p):
        """
        """
        raise NotImplementedError, 'GeometryBase::get_tangent_plane'
    
    cpdef get_normal(self, Point pnt, Point normal):
        """
        """
        raise NotImplementedError, 'GeometryBase::get_normal'

    cpdef double get_shortest_distance(self, Point pnt):
        """
        """
        raise NotImplementedError, 'GeometryBase::get_shortest_distance'

    cpdef ParticleArray get_particle_array(self):
        """
        """
        return None

###############################################################################
# `AnalyticalGeometry` class.
###############################################################################
cdef class AnalyticalGeometry(GeometryBase):
    """
    Base class to represent geometries that can be represented by some
    analytical equations - like planes cylinders, sphere etc.

    """
    def __cinit__(self, object entity=None, reference_points=None, *args, **kwargs):
        """
        """
        if reference_points is None:
            self.reference_points = reference_points
        else:
            self.reference_points = ParticleArray()      
            rp = self.reference_points
            rp.add_property({'name':'x', 'data':[0]},
                            {'name':'y', 'data':[0]},
                            {'name':'z', 'data':[0]},
                            {'name':'u', 'data':[0]},
                            {'name':'v', 'data':[0]},
                            {'name':'w', 'data':[0]})

    cpdef ParticleArray get_particle_array(self):
        """
        """
        return self.reference_points


###############################################################################
# `PolygonalGeometry` class.
###############################################################################
cdef class PolygonalGeometry(GeometryBase):
    """
    Class to represent geometries that are a collection of polygons.
    """
    def __cinit__(self, object entity=None, *args, **kwargs):
        """
        """
        pass

    cpdef ParticleArray get_particle_array(self):
        """
        """
        return None
