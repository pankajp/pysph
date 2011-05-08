from pysph.sph.sph_func import SPHFunction


cdef class PropertyGet(SPHFunction):
    """ function to get property arrays (upto 3 arrays) """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 prop_names=['rho'], **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array
        dest -- The destination particle array
        prop_names -- The properties to get (upto 3 particle arrays)
        """
        self.prop_names = prop_names
        SPHFunction.__init__(self, source, dest, setup_arrays=True)
        self.num_outputs = len(prop_names)
        self.id = 'property_get'
    
    cpdef setup_arrays(self):
        #Setup the basic properties like m, x rho etc.
        SPHFunction.setup_arrays(self)
        self.d_props = [self.dest.get_carray(i) for i in self.prop_names]
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef double** output = [output1.data, output2.data, output3.data]
        cdef long i, np=self.dest.get_number_of_particles()
        cdef int n
        cdef DoubleArray arr
        for n in range(self.num_outputs):
            arr = self.d_props[n]
            for i in range(np):
                output[n][i] = arr.data[i]

cdef class PropertyAdd(SPHFunction):
    """ function to add property arrays (any number of arrays) """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 prop_names=['rho'], constant=0.0, **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array
        dest -- The destination particle array
        prop_names -- The properties to get the sum of
        constant -- a constant value to add to the result
        """
        self.prop_names = prop_names
        self.constant = constant
        SPHFunction.__init__(self, source, dest, setup_arrays=True)
        self.num_outputs = 1
        self.id = 'property_add'
    
    cpdef setup_arrays(self):
        #Setup the basic properties like m, x rho etc.
        SPHFunction.setup_arrays(self)
        self.d_props = [self.dest.get_carray(i) for i in self.prop_names]
        self.num_props = len(self.d_props)
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef long i, np=self.dest.get_number_of_particles()
        cdef int n
        cdef DoubleArray arr=self.d_props[0]
        for i in range(np):
            output1.data[i] = arr.data[i]
        for n in range(1, self.num_props):
            arr = self.d_props[n]
            for i in range(np):
                output1.data[i] += arr.data[i] + self.constant

cdef class PropertyNeg(SPHFunction):
    """ function to return the negative of upto 3 particle arrays """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 prop_names=['rho'], **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        prop_names -- The properties to get the inverse of (upto 3)
        """
        self.prop_names = prop_names
        SPHFunction.__init__(self, source, dest, setup_arrays=True)
        self.num_outputs = len(prop_names)
        self.id = 'property_neg'
    
    cpdef setup_arrays(self):
        #Setup the basic properties like m, x rho etc.
        SPHFunction.setup_arrays(self)
        self.d_props = [self.dest.get_carray(i) for i in self.prop_names]
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef double** output = [output1.data, output2.data, output3.data]
        cdef long i, np=self.dest.get_number_of_particles()
        cdef int n
        cdef DoubleArray arr
        for n in range(self.num_outputs):
            arr = self.d_props[n]
            for i in range(np):
                output[n][i] = -arr.data[i]

cdef class PropertyMul(SPHFunction):
    """ function to get product of property arrays (any number of arrays) """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 prop_names=['rho'], constant=0.0, **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        prop_names -- The properties to get product of
        constant -- a constant value to multiply to the result
        """
        self.prop_names = prop_names
        self.constant = constant
        SPHFunction.__init__(self, source, dest, setup_arrays=True)
        self.num_outputs = 1
        self.id = 'property_mul'
    
    cpdef setup_arrays(self):
        #Setup the basic properties like m, x rho etc.
        SPHFunction.setup_arrays(self)
        self.d_props = [self.dest.get_carray(i) for i in self.prop_names]
        self.num_props = len(self.d_props)
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef long i, np=self.dest.get_number_of_particles()
        cdef int n
        cdef DoubleArray arr=self.d_props[0]
        for i in range(np):
            output1.data[i] = arr.data[i]
        for n in range(1, self.num_props):
            arr = self.d_props[n]
            for i in range(np):
                output1.data[i] *= arr.data[i] * self.constant

cdef class PropertyInv(SPHFunction):
    """ function to return the inverse of upto 3 particle arrays """
    def __init__(self, ParticleArray source, ParticleArray dest=None,
                 prop_names=['rho'], **kwargs):
        """ Constructor for SPH

        Parameters:
        -----------
        source -- The source particle array.
        dest -- The destination particle array.
        prop_names -- The properties to get inverse of (upto 3)
        """
        self.prop_names = prop_names
        SPHFunction.__init__(self, source, dest, setup_arrays=True)
        self.num_outputs = len(prop_names)
        self.id = 'property_inv'
    
    cpdef setup_arrays(self):
        #Setup the basic properties like m, x rho etc.
        SPHFunction.setup_arrays(self)
        self.d_props = [self.dest.get_carray(i) for i in self.prop_names]
    
    cpdef eval(self, KernelBase kernel, DoubleArray output1,
               DoubleArray output2, DoubleArray output3):
        cdef double** output = [output1.data, output2.data, output3.data]
        cdef long i, np=self.dest.get_number_of_particles()
        cdef int n
        cdef DoubleArray arr
        for n in range(self.num_outputs):
            arr = self.d_props[n]
            for i in range(np):
                output[n][i] = 1/arr.data[i]
