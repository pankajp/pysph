"""
Represents a collection of particles.

**Classes**
 
 - Particle - a particle proxy class.
 - ParticleArray - class to represent an array of particles.

**Default Property Names**

 - X coordinate - 'x'
 - Y coordinate - 'y'
 - Z coordinate - 'z'
 - Particle X Velocity - 'u'
 - Particle Y Velocity - 'v'
 - Particle Z Velocity - 'w'
 - Particle mass - 'm'
 - Particle density - 'rho'
 - Particle Interaction Radius - 'h'

The default property names is not enforced here. It is enforced by having
default values to property names in various constructors.

**Issues**

 - Do we need to have a particle class separately, and make this an array of 
   those particles ?
 - Do we have a Particle class that will be a proxy to the data in the 
   ParticleArray ? It should provide get and set functions to access the 
   properties of the particle. May make some access more natural.

"""

# numpy imports
cimport numpy
import numpy

# Local imports
from pysph.base.carray cimport *
from pysph.base.particle_tags cimport *

class Particle(object):
    """
    A simple proxy object that acts as an accessor to the ParticleArray.

    """
    def __init__(self, index=0, particle_array=None):
        """
        """
        self.index = 0
        self.particle_array = particle_array

    def get(self, prop):
        """
        Get the value of the property "prop" for the particle at self.index.

        """
        pass

    def set(self, prop, value):
        """
        """
        pass

cdef class ParticleArray:
    """
    Class to represent a collection of particles.

    **Member variables**
     - particle_manager - the particle_manager to which this array belongs.
     - name - name of this particle array.
     - properties - for every property, gives the index into the list where the
       property array is found.
     - property_arrays - list of arrays, one for each property.
     - temporary_arrays - dict of temporary arrays. temporary arrays are just
       for internal calculations.
     - is_dirty - indicates if the particle positions have changed.
     - standard_name_map - a map from a few standard property names to the
       actual names used in this particle array.

    **Notes**
     - the property 'tag' is always the 0th property.
    
    **Issues**
     - Decision on standard_name_map to be made.

    """
    ######################################################################
    # `object` interface
    ######################################################################
    def __cinit__(self, object particle_manager=None, str name='',
                  default_particle_tag=LocalReal, **props):
        """
        Constructor.

	**Parameters**

         - particle_manager - particle manager managing this array.
         - name - name of this particle array.
         - props - dictionary of properties for every particle in this array.

    	"""
        self.properties = {'tag':0}
        self.property_arrays = [LongArray(0)]
        self.temporary_arrays = {}

        self.particle_manager = particle_manager
        self.name = name

        self.is_dirty = True

        self.standard_name_map = {}
        self.default_particle_tag = default_particle_tag
        
        if props:
            self.initialize(**props)

    def __getattr__(self, name):
        """
        Convenience, to access particle property arrays as an attribute.
        
        A numpy array is returned. Look at the get() functions documentation for
        care on using this numpy array.

        """
        keys = self.properties.keys() + self.temporary_arrays.keys()
        if name in keys:
            return self.get(name)
        else:
            raise AttributeError, 'property %s not found'%(name)

    def __setattr__(self, name, value):
        """
        Convenience, to set particle property arrays as an attribute.
        """
        keys = self.properties.keys() + self.temporary_arrays.keys()
        if name in keys:
            self.set(**{name:value})
        else:
            raise AttributeError, 'property %s not found'%(name)

    ######################################################################
    # `Public` interface
    ######################################################################
    cpdef set_dirty(self, bint value):
        """
        Set the is_dirty variable to given value
        """
        self.is_dirty = value

    def clear(self):
        """
        Clear all data held by this array.
        """
        self.properties = {'tag':0}
        self.property_arrays[:] = [LongArray(0)]
        self.temporary_arrays.clear()
        self.is_dirty = True

    def initialize(self, **props):
        """
        Initialize the particle array with the given props.

        **Parameters**

         - props - dictionary containing various property arrays. All these
           arrays are expected to be numpy arrays or objects that can be
           converted to numpy arrays.

        **Notes**

         - This will clear any existing data.
         - As a rule internal arrays will always be either long or double
           arrays. 

        **Helper Functions**

         - _create_c_array_from_npy_array

        """
        cdef int nprop, nparticles
        cdef bint tag_present = False
        cdef numpy.ndarray a, arr, npyarr
        cdef LongArray tagarr
        cdef str prop

        self.clear()

        nprop = len(props)

        if nprop == 0:
            return 

        # check if the 'tag' array has been given as part of the properties.
        if props.has_key('tag'):
            a = numpy.asarray(props['tag'], dtype=numpy.long)
            self.property_arrays[0].resize(a.size)
            self.property_arrays[0].set_data(a)
            props.pop('tag')
            tag_present = True
        
        # add the property names to the properties dict and the arrays to the
        # property array.
        for prop in props:
            if self.properties.has_key(prop):
                raise ValueError, 'property %s already exists'%(prop)

            arr = numpy.asarray(props[prop])
            carr = self._create_c_array_from_npy_array(arr)
            self.property_arrays.append(carr)
            self.properties[prop] = len(self.property_arrays)-1
            
        # if tag was not present in the input set of properties, add tag with
        # default value.
        if tag_present == False:
            nparticles = len(props.values()[0])
            tagarr = self.property_arrays[0]
            tagarr.resize(nparticles)
            # set them to the default value
            npyarr = tagarr.get_npy_array()
            npyarr[:] = self.default_particle_tag

    cpdef int get_number_of_particles(self):
        """
        Return the number of particles.
        """
        if len(self.property_arrays) > 1:
            prop0 = self.property_arrays[0]
            return prop0.length
        else:
            return 0

    def add_temporary_array(self, arr_name):
        """
        Add temporary double with name arr_name.

        **Parameters**

         - arr_name - name of the temporary array needed. It should be different
           from any property name.
           
        """
        if self.properties.has_key(arr_name):
            raise ValueError, 'property (%s) exists'%(arr_name)
        
        np = self.get_number_of_particles()
        
        if not self.temporary_arrays.has_key(arr_name):
            carr = DoubleArray(np)
            self.temporary_arrays[arr_name] = carr            
        
    cpdef remove_particles(self, LongArray index_list):
        """
        Remove particles whose indices are given in index_list.

        We repeatedy interchange the values of the last element and values from
        the index_list and reduce the size of the array by one. This is done for
        every property and temporary arrays that is being maintained.
    
	**Parameters**
        
         - index_list - an array of indices, this array should be a LongArray.

        **Algorithm**::
        
         if index_list.length > number of particles
             raise ValueError

         sorted_indices <- index_list sorted in ascending order.
        
         for every every array in property_array
             array.remove(sorted_indices)

         for every array in temporary_arrays:
             array.remove(sorted_indices)

    	"""
        cdef str msg
        cdef numpy.ndarray sorted_indices
        cdef BaseArray prop_array
        cdef int num_arrays, i
        cdef list temp_arrays

        if index_list.length > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError, msg
        
        sorted_indices = numpy.sort(index_list.get_npy_array())
        num_arrays = len(self.property_arrays)
        
        for i from 0 <= i < num_arrays:
            prop_array = self.property_arrays[i]
            prop_array.remove(sorted_indices, 1)

        temp_arrays = self.temporary_arrays.values()
        num_arrays = len(temp_arrays)
        for i from 0 <= i < num_arrays:
            prop_array = temp_arrays[i]
            prop_array.remove(sorted_indices, 1)

        self.is_dirty = True
    
    cpdef remove_tagged_particles(self, long tag):
        """
        Remove particles that have the given tag.

        **Parameters**

         - tag - the type of particles that need to be removed.

        """
        cdef LongArray indices = LongArray()
        cdef LongArray tag_array = self.property_arrays[0]
        cdef long *tagarrptr = tag_array.get_data_ptr()
        cdef int i
        
        # find the indices of the particles to be removed.
        for i from 0 <= i < tag_array.length:
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles.
        self.remove_particles(indices)

        if indices.length > 0:
            self.is_dirty = True

    def add_particles(self, **particle_props):
        """
        Add particles in particle_array to self.
    
	**Parameters**

         - particle_props - a dictionary containing numpy arrays for various
           particle properties.
         
    	**Notes**
         
         - all properties should have same length arrays.
         - all properties should already be present in this particles array.
           if new properties are seen, an exception will be raised.
         - temporary arrays are not to be specified here, only particle
           properties.

    	**Issues**

         - should the input parameter type be changed ?

    	"""
        if len(particle_props) == 0:
            return

        # check if the input properties are valid.
        for prop in particle_props.keys():
            self._check_property(prop)

        num_extra_particles = len(particle_props.values()[0])
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self.properties:
            prop_id = self.properties[prop]
            arr = self.property_arrays[prop_id]

            if prop in particle_props.keys():
                arr.extend(particle_props[prop])
            else:
                arr.resize(new_num_particles)
                if prop == 'tag':
                    # set the default_particle_tag for newly added
                    # particles to the default_particle_tag.
                    nparr = arr.get_npy_array()
                    nparr[old_num_particles:] = self.default_particle_tag
        
        # now extend the temporary arrays.
        for arr in self.temporary_arrays.values():
            arr.resize(new_num_particles)

        if num_extra_particles > 0:
            self.is_dirty = True

    def get_property_index(self, prop_name):
        """
        Get the index into the property array where the prop_name property is
        located.

        """
        return self.properties.get(prop_name)

    def get(self, *args):
        """
        Return the numpy array for the 'prop_name' property.
        
        **Parameters**

         - args - a list of property names.

        **Notes**

         - The returned numpy array does **NOT** own its data. Other operations
           may be performed.

        """
        nargs = len(args)
        result = []
        if nargs == 0:
            return 
        
        # make sure all prop names are valid names
        for arg in args:
            self._check_property(arg)
        
        for arg in args:
            if arg in self.properties:
                arg_id = self.properties[arg]
                result.append(self.property_arrays[arg_id].get_npy_array())
            elif self.temporary_arrays.has_key(arg):
                result.append(self.temporary_arrays[arg].get_npy_array())

        if nargs == 1:
            return result[0]
        else:
            return tuple(result)        

    def set(self, **props):
        """
        Set properties from numpy arrays or objects that can be converted into
        numpy arrays.

        **Parameters**

         - props - a dictionary of properties containing the arrays to be set.

        **Notes**

         - the properties being set must already be present in the properties
           dict. 
         - the size of the data should match the array already present.

        **Issues**

         - Should the is_dirty flag be set here ? This would involve some checks
           like if the 'x', 'y' or 'z' properties were set. I do not think this
           is the correct place for setting the is_dirty flag. Let the module
           setting the coordinates handle that.

        """
        cdef str prop
        cdef int nprops = len(props)
        cdef list prop_names = props.keys()
        cdef int i

        for i in range(nprops):
            prop = prop_names[i]
            self._check_property(prop)
            
        for prop in props.keys():
            proparr = numpy.asarray(props[prop])
            if self.properties.has_key(prop):
                prop_id = self.properties[prop]
                self.property_arrays[prop_id].set_data(proparr)
            elif self.temporary_arrays.has_key(prop):
                self.temporary_arrays[prop].set_data(proparr)
    
    cpdef get_carray(self, str prop):
        """
        Return the c-array corresponding to the property or temporary array.
        """
        cdef int prop_id

        if self.properties.has_key(prop):
            prop_id = self.properties.get(prop)
            return self.property_arrays[prop_id]
        elif self.temporary_arrays.has_key(prop):
            return self.temporary_arrays[prop]

    ######################################################################
    # Non-public interface
    ######################################################################
    cdef _check_property(self, str prop):
        """
        Check if a property is present or not.
        """
        if self.temporary_arrays.has_key(prop) or self.properties.has_key(prop):
            return
        else:
            raise AttributeError, 'property %s not present'%(prop)
        
    cdef object _create_c_array_from_npy_array(self, numpy.ndarray np_array):
        """
        Create and return  a carray array from the given numpy array.

        **Notes**
         - this function is used only when a C array needs to be
           created (in the initialize function).

        """
        cdef int np = np_array.size
        cdef object a 
        if np_array.dtype is numpy.int32 or np_array.dtype is numpy.int64:
            a = LongArray(np)
            a.set_data(np_array)
        elif np_array.dtype == numpy.float32:
            a = FloatArray(np)
            a.set_data(np_array)
        elif np_array.dtype == numpy.double:
            a = DoubleArray(np)
            a.set_data(np_array)
        else:
            raise TypeError, 'unknown numpy data type passed'

        return a
