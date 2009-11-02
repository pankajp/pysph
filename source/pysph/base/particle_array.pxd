cimport numpy as np
from pysph.base.particle_tags cimport ParticleTag
from pysph.base.carray cimport LongArray

cdef class ParticleArray:
    cdef public dict properties
    cdef public list property_arrays
    cdef public dict temporary_arrays
    cdef public object particle_manager
    cdef public str name
    cdef public bint is_dirty
    cdef public dict standard_name_map
    cdef public long default_particle_tag

    cdef object _create_c_array_from_npy_array(self, np.ndarray arr)
    cdef _check_property(self, str)

    cpdef set_dirty(self, bint val)

    cpdef get_carray(self, str prop)

    cpdef int get_number_of_particles(self)
    cpdef remove_particles(self, LongArray index_list)
    cpdef remove_tagged_particles(self, long tag)

    # new functionality to be added.

    # function to add any property
    #cpdef add_property(self, str prop_name, str data_type, list default)

    # function to remove particles with particular value of a flag property.
    #cpdef remove_flagged_particles(self, str flag_name, int flag_value)

