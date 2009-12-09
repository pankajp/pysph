cimport numpy as np
from pysph.base.particle_tags cimport ParticleTag
from pysph.base.carray cimport LongArray

cdef class ParticleArray:
    """
    Maintains various properties for particles.
    """
    # dictionary to hold the properties held per particle.
    cdef public dict properties
    cdef public list property_arrays

    # default value associated with each property
    cdef public dict default_values

    # dictionary to hold temporary arrays - we can do away with this.
    cdef public dict temporary_arrays

    # the particle manager of which this is part of.
    cdef public object particle_manager

    # name associated with this particle array
    cdef public str name

    # indicates if coordinates of particles has changed.
    cdef public bint is_dirty
    
    # indicate if the particle configuration has changed.
    cdef public bint indices_invalid

    # the number of real particles.
    cdef public long num_real_particles

    cdef object _create_c_array_from_npy_array(self, np.ndarray arr)
    cdef _check_property(self, str)

    cdef np.ndarray _get_real_particle_prop(self, str prop)

    cpdef set_dirty(self, bint val)
    cpdef set_indices_invalid(self, bint val)

    cpdef get_carray(self, str prop)

    cpdef int get_number_of_particles(self)
    cpdef remove_particles(self, LongArray index_list)
    cpdef remove_tagged_particles(self, long tag)
    
    # function to add any property
    cpdef add_property(self, dict prop_info)
    
    # increase the number of particles by num_particles
    cpdef extend(self, int num_particles)

    cpdef has_array(self, str arr_name)

    #function to remove particles with particular value of a flag property.
    cpdef remove_flagged_particles(self, str flag_name, int flag_value)

    # function to get indices of particles that have a particle integer property
    # set to the specified value.
    #cpdef int get_flagged_particles(self, str flag_name, int flag_value,
    # LongArray flag_value)
    
    # get requested properties of selected particles
    # cpdef get_props(self, LongArray indices, *args)

    # set the properties of selected particles int the parray.
    #cpdef int set_particle_props(self, LongArray indices, **props)

    # aligns all the real particles in contiguous positions starting from 0
    cpdef int align_particles(self) except -1

    # add particles from the parray to self.
    cpdef int append_parray(self, ParticleArray parray) except -1

    # create a new particle array with the given particles indices and the
    # properties. 
    cpdef ParticleArray extract_particles(self, LongArray index_array, list
                                          props=*)

    # sets the value of the property flag_name to flag_value for the particles
    # in indices.
    cpdef set_flag(self, str flag_name, int flag_value, LongArray indices)
    cpdef set_tag(self, long tag_value, LongArray indices)
