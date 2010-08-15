"""
Contains base classes for representing any kind of geometric objects.
"""

# local imports
from pysph.base.point cimport Point
from pysph.base.plane cimport Plane
from pysph.base.particle_array cimport ParticleArray


################################################################################
# `GeometryBase` class.
################################################################################
cdef class GeometryBase:
    """
    Base class for all geometries associtaed with entities.
    """

    # reference to the entity which this geometry represents.
    cdef object entity
    
    cpdef get_tangent_plane(self, Point pnt, Plane p)
    cpdef get_normal(self, Point pnt, Point normal)
    cpdef double get_shortest_distance(self, Point pnt)

    cpdef ParticleArray get_particle_array(self)

################################################################################
# `AnalyticalGeometry` class.
################################################################################ 
cdef class AnalyticalGeometry(GeometryBase):
    """
    Base class to represent geometries that can be represented by some
    analytical equations - like planes cylinders, sphere etc.

    """
    
    # set of points about which this object is defined. This could as well be
    # just a single point - an origin in body space coords, about which this
    # object is defined.
    cdef ParticleArray reference_points

################################################################################
# `PolygonalGeometry` class.
################################################################################ 
cdef class PolygonalGeometry(GeometryBase):
    """
    Class to represent geometries that are a collection of polygons.
    """
    #cdef PolygonArray polygon_array
    pass
