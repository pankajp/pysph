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

 - Setting of property values of real particles only.
   We need a facility to retrieve properties of real particles only. In many
   cases, such a requirement is seen. For example, consider a parallel algorithm
   where some particles are fetched from neighbor processors and are only used
   as sources - their values are only read - any computation of their properties
   is done by the remote processor. All particles of an entity will however be
   stored as part of a single particle array. Now we may need to integrate
   properties of only the real paritcles and not the copies of the remote
   ones. How should this be done elegantly ?

   One way is to store all the real particles in continguos positions starting
   from 0. This way we can easily get slices of the numpy array containing the
   real particles easily.

"""

# logging imports
import logging
logger = logging.getLogger()

# numpy imports
cimport numpy
import numpy

# Local imports
from pysph.base.carray cimport *
from pysph.base.particle_tags cimport *

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
                  default_particle_tag=LocalReal, *args, **props):
        """
        Constructor.

	**Parameters**

         - particle_manager - particle manager managing this array.
         - name - name of this particle array.
         - props - dictionary of properties for every particle in this array.

    	"""
        self.properties = {'tag':LongArray(0), 'group':LongArray(0)}
        self.default_values = {'tag':default_particle_tag, 'group':0}
        
        self.temporary_arrays = {}

        self.particle_manager = particle_manager

        self.name = name
        self.is_dirty = True

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
            return self._get_real_particle_prop(name)
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

    cpdef has_array(self, str arr_name):
        """
        Returns true if the array arr_name is present.
        """
        return self.properties.has_key(arr_name)

    def clear(self):
        """
        Clear all data held by this array.
        """
        self.properties = {'tag':LongArray(0), 'group':LongArray(0)}
        tag_def_values = self.default_values['tag']
        self.default_values.clear()
        self.default_values = {'tag':tag_def_values, 'group':0}
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
        # if props.has_key('tag'):
        #     tag_prop = props['tag']
        #     data = tag_prop.get('data')
        #     a = numpy.asarray(tag_prop['data'], dtype=numpy.long)
        #     self.properties['tag'].resize(a.size)
        #     self.properties['tag'].set_data(a)
        #     props.pop('tag')
        
        # add the property names to the properties dict and the arrays to the
        # property array.
        for prop in props.keys():
            # if self.properties.has_key(prop):
            #     raise ValueError, 'property %s already exists'%(prop)
            prop_info = props[prop]
            prop_info['name'] = prop
            self.add_property(prop_info)
            
        # if tag was not present in the input set of properties, add tag with
        # default value.
        # if (tag_present == False and
        #     self.get_number_of_particles() > 0):
        #     nparticles = self.get_number_of_particles()
        #     tagarr = self.properties['tag']
        #     tagarr.resize(nparticles)
        #     # set them to the default value
        #     npyarr = tagarr.get_npy_array()
        #     npyarr[:] = self.default_values['tag']

        self.align_particles()

    cpdef int get_number_of_particles(self):
        """
        Return the number of particles.
        """
        if len(self.properties.values()) > 0:
            prop0 = self.properties.values()[0]
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
        cdef list property_arrays

        if index_list.length > self.get_number_of_particles():
            msg = 'Number of particles to be removed is greater than'
            msg += 'number of particles in array'
            raise ValueError, msg
        
        sorted_indices = numpy.sort(index_list.get_npy_array())
        num_arrays = len(self.properties.keys())

        property_arrays = self.properties.values()
        
        for i from 0 <= i < num_arrays:
            prop_array = property_arrays[i]
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
        cdef LongArray tag_array = self.properties['tag']
        cdef long *tagarrptr = tag_array.get_data_ptr()
        cdef int i
        
        # find the indices of the particles to be removed.
        for i from 0 <= i < tag_array.length:
            if tagarrptr[i] == tag:
                indices.append(i)

        # remove the particles.
        self.remove_particles(indices)

        if indices.length > 0:
            # realign particles
            self.align_particles()
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
            return 0

        # check if the input properties are valid.
        for prop in particle_props.keys():
            self._check_property(prop)

        num_extra_particles = len(particle_props.values()[0])
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        for prop in self.properties:
            arr = self.properties[prop]

            if prop in particle_props.keys():
                d_type = arr.get_npy_array().dtype
                s_arr = numpy.asarray(particle_props[prop], dtype=d_type)
                arr.extend(s_arr)
            else:
                arr.resize(new_num_particles)
                # set the properties of the new particles to the default ones.
                nparr = arr.get_npy_array()
                nparr[old_num_particles:] = self.default_values[prop]
        
        # now extend the temporary arrays.
        for arr in self.temporary_arrays.values():
            arr.resize(new_num_particles)

        if num_extra_particles > 0:
            # make sure particles are aligned properly.
            self.align_particles()
            self.is_dirty = True

        return 0

    cpdef int append_parray(self, ParticleArray parray):
        """
        Similar to add_particles above, except that particles come from a
        particle array.

        New properties - that are not there in self, will be added.
        """
        if parray.get_number_of_particles() == 0:
            return 0
        
        num_extra_particles = parray.get_number_of_particles()
        old_num_particles = self.get_number_of_particles()
        new_num_particles = num_extra_particles + old_num_particles

        # extend current arrays by the required number of particles
        self.extend(num_extra_particles)
        
        for prop_name in parray.properties.keys():
            arr = self.properties.get(prop_name)
            if arr is not None:
                nparr_source = parray.get_carray(prop_name).get_npy_array()
                nparr_dest = arr.get_npy_array()
                nparr_dest[old_num_particles:] = nparr_source
            else:
                # meaning this property is not there in self.
                self.add_property({'name':prop_name,
                                   'default':parray.default_values[prop_name]})
                # now add the values to the end of the created array
                nparr_dest = self.properties[prop_name].get_npy_array()
                nparr_source = parray.get_carray(prop_name).get_npy_array()
                nparr_dest[old_num_particles:] = nparr_source

        if num_extra_particles > 0:
            self.align_particles()
            self.is_dirty = True

        return 0

    cpdef extend(self, int num_particles):
        """
        Increase the total number of particles by the requested amount.
        """
        if num_particles <= 0:
            return
        
        cdef int old_size = self.get_number_of_particles()
        cdef int new_size = old_size + num_particles
        cdef BaseArray arr
        cdef numpy.ndarray nparr

        for key in self.properties.keys():
            arr = self.properties[key]
            arr.resize(new_size)
            nparr = arr.get_npy_array()
            nparr[old_size:] = self.default_values[key]

        for arr in self.temporary_arrays.values():
            arr.resize(new_size)

        self.align_particles()

    def get_property_index(self, prop_name):
        """
        Get the index into the property array where the prop_name property is
        located.

        """
        return self.properties.get(prop_name)

    cdef numpy.ndarray _get_real_particle_prop(self, str prop_name):
        """
        cdef'ed function to get the npy array corresponding to only real
        particles of a given property.

        No checks are performed. Only call this after making sure that the
        property required already exists.
        """
        cdef BaseArray prop_array
        prop_array = self.properties.get(prop_name)
        if prop_array is not None:
            return prop_array.get_npy_array()[:self.num_real_particles]
        else:
            prop_array = self.temporary_arrays.get(prop_name)
            if prop_array is not None:
                return prop_array.get_npy_array()[:self.num_real_particles]
            else:
                return None

    def get(self, *args, only_real_particles=True):
        """
        Return the numpy array for the  property names in *args.
        
        **Parameters**

         - only_real_particles - indicates if properties of only real particles
           need to be returned or all particles to be returned. By default only
           real particles will be returned.
         - args - a list of property names.

        **Notes**

         - The returned numpy array does **NOT** own its data. Other operations
           may be performed.

        """
        cdef int nargs = len(args)
        cdef list result = []
        cdef str arg
        cdef int i
        cdef BaseArray arg_array

        if nargs == 0:
            return
        
        if only_real_particles == True:
            for i from 0 <= i < nargs:
                arg = args[i]
                self._check_property(arg)
            
                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(
                        arg_array.get_npy_array()[:self.num_real_particles])
                elif self.temporary_arrays.has_key(arg):
                    result.append(
                        self.temporary_arrays[arg].get_npy_array()[
                            :self.num_real_particles])
        else:
            for i from 0 <= i < nargs:
                arg = args[i]
                self._check_property(arg)
            
                if arg in self.properties:
                    arg_array = self.properties[arg]
                    result.append(arg_array.get_npy_array())
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
        cdef BaseArray prop_array
        cdef int nprops = len(props)
        cdef list prop_names = props.keys()
        cdef int i

        for i in range(nprops):
            prop = prop_names[i]
            self._check_property(prop)
            
        for prop in props.keys():
            proparr = numpy.asarray(props[prop])
            if self.properties.has_key(prop):
                prop_array = self.properties[prop]
                prop_array.set_data(proparr)
            elif self.temporary_arrays.has_key(prop):
                self.temporary_arrays[prop].set_data(proparr)
            # if the tag property is being set, the alignment will have to be
            # changed.                 
            if prop == 'tag':
                self.align_particles()
            if prop == 'x' or prop == 'y' or prop == 'z':
                self.set_dirty(True)
    
    cpdef get_carray(self, str prop):
        """
        Return the c-array corresponding to the property or temporary array.
        """
        cdef int prop_id

        if self.properties.has_key(prop):
            return self.properties[prop]
        elif self.temporary_arrays.has_key(prop):
            return self.temporary_arrays[prop]

    cpdef add_property(self, dict prop_info):
        """
        Add a new property based on information in prop_info.

        **Params**
        
            - prop_info - a dict with the following keys: 

                - 'name' - compulsory key
                - 'type' - specifying the data type of this property.
                - 'default' - specifying the default value of this property.
                - 'data' - specifying the data associated with each particle.
           
                type, default and data are optional keys. They will take the
                following default values:
                type - 'double' by default
                default - 0 by default
                data - if not present, an array with all values set to default will
                be used for this property.                

        **Notes**
            
            If there are no particles currently in the particle array, and a
            new property with some particles is added, all the remaining
            properties will be resized to the size of the newly added array.

            If there are some particles in the particle array, and a new
            property is added without any particles, then this new property will
            be resized according to the current size.

            If there are some particles in the particle array and a new property
            is added with a different number of particles, then an error will be
            raised.

        **Issue**

            - it is best not to add properties with dat when you already have
              particles in the particle array. Reason is that, the particles in
              the particle array will be stored so that the 'Real' particles are
              in the top of the list and later the dummy ones. The data in your
              property array should be matched to the particles appropriately.
              This may not always be possible when there are particles of
              different type in the particle array. 
            - Properties without any values can be added anytime.
            - While initializing particle arrays only using the add_property
              function, you will have to call align_particles manually to make
              sure particles are aligned properly.

        """
        cdef str prop_name, data_type
        cdef object data, default
        cdef bint array_size_proper = False

        prop_name = prop_info.get('name')
        data_type = prop_info.get('type')
        data = prop_info.get('data')
        default = prop_info.get('default')

        if prop_name is None:
            logger.error('Cannot add property with no name')
            raise ValueError

        # check if the property is already present, if so display warning
        # message and exit
        # if self.properties.has_key(prop_name):
        #     logger.warn(
        #         'Property %s already present, cannot add again'%(prop_name))
        #     logger.warn('Use set() to set values for a property')
        #     return

        # make sure the size of the supplied array is consistent.
        if (data is None or self.get_number_of_particles() == 0 or   
            len(data) == 0):
            array_size_proper = True
        else:
            if self.get_number_of_particles() == len(data):
                array_size_proper = True

        if array_size_proper == False:
            logger.error('Array sizes incompatible')
            raise ValueError, 'Array sizes incompatible'
        
        # setup the default values
        if default is None:
            if not self.properties.has_key(prop_name):
                default = 0
            else:
                default = self.default_values[prop_name]
                
        self.default_values[prop_name] = default

        # array sizes are compatible, now resize the required arrays
        # appropriately and add.
        if self.get_number_of_particles() == 0:
            if data is None or len(data) == 0:
                # if a property with that name already exists, do not do
                # anything.
                if not self.properties.has_key(prop_name):
                    # just add the property with a zero array.
                    self.properties[prop_name] = self._create_carray(
                        data_type, 0, default)
            else:
                # new property has been added with some particles, while no
                # particles are currently present. First resize the current
                # properties to this new length, and then add this new
                # property.
                for prop in self.properties.keys():
                    arr = self.properties[prop]
                    arr.resize(len(data))
                    arr.get_npy_array()[:] = self.default_values[prop]
                
                if self.properties.has_key(prop_name):
                    # just add the particles to the already existing array.
                    d_type = self.properties[prop_name].get_npy_array().dtype
                    arr = numpy.asarray(data, dtype=d_type)
                    self.properties[prop_name].set_data(arr)
                else:
                    # now add the new property array
                    # if a type was specifed create that type of array
                    if data_type is None:
                        # get an array for this data
                        arr = numpy.asarray(data, dtype=numpy.double)
                        self.properties[prop_name] = (
                            self._create_c_array_from_npy_array(arr))
                    else:
                        arr = self._create_carray(data_type, len(data), default)
                        np_arr = arr.get_npy_array()
                        np_arr[:] = numpy.asarray(data)
                        self.properties[prop_name] = arr
        else:
            if data is None or len(data) == 0:
                # new property is added without any initial data, resize it to
                # current particle count.
                if not self.properties.has_key(prop_name):
                    arr = self._create_carray(data_type,
                                              self.get_number_of_particles(),
                                              default)
                    self.properties[prop_name] = arr
            else:
                if self.properties.has_key(prop_name):
                    d_type = self.properties[prop_name].get_npy_array().dtype
                    arr = numpy.asarray(data, dtype=d_type)
                    self.properties[prop_name].set_data(arr)
                    # realign the particles if the tag variable is being set.
                else:
                    if data_type is None:
                        # just add the property array
                        arr = numpy.asarray(data, dtype=numpy.double)
                        self.properties[prop_name] = (
                            self._create_c_array_from_npy_array(arr))
                    else:
                        arr = self._create_carray(data_type, len(data), default)
                        np_arr = arr.get_npy_array()
                        arr.get_npy_array()[:] = numpy.asarray(data)
                        self.properties[prop_name] = arr
                        
    ######################################################################
    # Non-public interface
    ######################################################################
    def _create_carray(self, data_type, size, default=0):
        """
        Create a carray of the requested type, and of requested size.

        **Parameters**

            - data_type - string representing the 'c' data type - eg. 'int' for
            integers. 
            - size - the size of the requested array
            - default - the default value to initialize the array with.

        """
        if data_type == None or data_type == 'double':
            arr = DoubleArray(size)
        if data_type == 'long':
            arr = LongArray(size)
        if data_type == 'float':
            arr = FloatArray(size)
        if data_type == 'int':
            arr = IntArray(size)

        if size > 0:
            arr.get_npy_array()[:] = default

        return arr
            
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
            raise TypeError, 'unknown numpy data type passed %s'%(np_array.dtype)

        return a

    cpdef int align_particles(self) except -1:
        """
        Moves all 'LocalReal' particles to the begining of the array. 

        This makes retrieving numpy slices of properties of 'LocalReal'
        particles possible. This facility will be required frequently.

        **Algorimth**::
        
            index_arr = LongArray(n)
            
            next_insert = 0
            for i from 0 to n
                p <- ith particle
                if p is LocalReal
                    if i != next_insert
                        tmp = index_arr[next_insert]
                        index_arr[next_insert] = i
                        index_arr[i] = tmp
                        next_insert += 1
                    else
                        index_arr[i] = i
                        next_insert += 1
                else
                    index_arr[i] = i

             # we now have the new index assignment.
             # swap the required values as needed.
             for every property array:
                 for i from 0 to n:
                     if index_arr[i] != i:
                         tmp = prop[i]
                         prop[i] = prop[index_arr[i]]
                         prop[index_arr[i]] = tmp
        """
        cdef size_t i, num_particles
        cdef size_t next_insert
        cdef long tmp
        cdef LongArray index_array, tag_arr
        cdef BaseArray arr
        cdef int num_arrays
        cdef list arrays
        cdef long num_real_particles = 0
        cdef long num_moves = 0

        next_insert = 0
        num_particles = self.get_number_of_particles()

        tag_arr = self.get_carray('tag')

        # malloc the new index array
        index_array = LongArray(num_particles)
        
        for i from 0 <= i < num_particles:
            if tag_arr.data[i] == LocalReal:
                num_real_particles += 1
                if i != next_insert:
                    tmp = index_array.data[next_insert]
                    index_array.data[next_insert] = i
                    index_array.data[i] = tmp
                    next_insert += 1
                    num_moves += 1
                else:
                    index_array.data[i] = i
                    next_insert += 1
            else:
                index_array.data[i] = i

        self.num_real_particles = num_real_particles
        # we now have the aligned indices. Rearrage the particles particles
        # accordingly.
        arrays = self.properties.values()
        num_arrays = len(arrays)
        
        for i from 0 <= i < num_arrays:
            arr = arrays[i]
            arr._align_array(index_array)

        # now the temporary arrays
        arrays = self.temporary_arrays.values()
        num_arrays = len(arrays)
        for i from 0 <= i < num_arrays:
            arr = arrays[i]
            arr._align_array(index_array)

        if num_moves > 0:
            self.is_dirty = True
