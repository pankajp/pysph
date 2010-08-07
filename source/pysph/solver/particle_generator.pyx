"""
Contains base class for classes generating particle configurations.
"""

# standard imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.kernels cimport KernelBase
from pysph.base.particle_array cimport ParticleArray
from pysph.solver.base cimport Base

cdef class MassComputationMode:
    """
    Enum class to decide the method to compute the mass of the generated
    particles.
    """
    Ignore = 0
    Set_Constant = 1
    Compute_From_Density = 2
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        raise SystemError, "Do not instantiate the %s class"%(
            MassComputationMode)
            
cdef class DensityComputationMode:
    """
    Enum class to decide the method to compute the density of the generated
    particles.
    """
    Ignore = 0
    Set_Constant = 1
    Compute_From_Mass = 2
    def __init__(self, *args, **kwargs):
        """
        Constructor.
        """
        raise SystemError, "Do not instantiate the %s class"%(
            DensityComputationMode)

cdef class ParticleGenerator(Base):
    """
    Base class for classes generating particle configurations.
    """
    def __cinit__(self, 
                  list output_particle_arrays=[],
                  double particle_mass=-1.0,
                  int mass_computation_mode=MassComputationMode.Compute_From_Density,
                  double particle_density=1000.0,
                  int density_computation_mode=DensityComputationMode.Set_Constant, 
                  double particle_h=0.1,
                  KernelBase kernel=None,
                  *args, **kwargs):
        """
        Constructor.
        """
        self.output_particle_arrays = []
        self.output_particle_arrays[:] = output_particle_arrays
        
        self.particle_mass = particle_mass
        self.mass_computation_mode = mass_computation_mode
        self.particle_density=particle_density
        self.density_computation_mode = density_computation_mode
        self.particle_h=particle_h
        self.kernel = kernel

    def __init__(self,
                 output_particle_arrays=[],
                 particle_mass=-1.0,
                 mass_computation_mode=MassComputationMode.Compute_From_Density,
                 particle_density=1000.0,
                 density_computation_mode=DensityComputationMode.Set_Constant,
                 particle_h=0.1,
                 KernelBase kernel=None,
                 *args, **kwargs):
        """
        Python constructor.
        """
        pass
        
    cpdef get_particles(self):
        """
        Returns a ParticleArray or a list of ParticleArrays.
        """
        if self.validate_setup() == False:
            return None

        self.generate_func()

        if self.num_output_arrays() == 1:
            return self.output_particle_arrays[0]
        else:
            return tuple(self.output_particle_arrays)

    cpdef generate_func(self):
        """
        Function that does the generator specific actions. Implement this in the
        derived classes.
        """
        raise NotImplementedError, 'ParticleGenerator::generate_func'

    cpdef bint validate_setup(self):
        """
        Performs basic validation of the input parameters. Derived classes may
        have to reimplement this to add specific validations.
        """
        if (self.mass_computation_mode ==
            MassComputationMode.Compute_From_Density and
            self.density_computation_mode == 
            DensityComputationMode.Compute_From_Mass):
            msg = 'Cannot compute mass and density from each other'
            msg += '\nOne of them needs to be specified.'
            print msg
            logger.warn('Cannot compute mass and density from each other')
            logger.warn('One of them needs to be specified.')
            return False
        
        if ((self.mass_computation_mode == 
             MassComputationMode.Compute_From_Density or 
             self.density_computation_mode == 
             DensityComputationMode.Compute_From_Mass) 
            and (self.kernel is None)):
            msg = 'Cannot compute mass/density without a kernel'
            print msg
            logger.warn(msg)
            return False

        return True

    cpdef num_output_arrays(self):
        """
        Returns the number of output arrays this generator is going to
        produce. Implement this in all concrete generators.
        """
        raise NotImplementedError, 'ParticleGenerator::num_output_arrays'

    cpdef get_coords(self):
        """
        Function to just generate the coordinates of the required points are
        return three numpy arrays.
        """
        raise NotImplementedError, 'ParticleGenerator::get_coords'

    cpdef _setup_outputs(self):
        """
        Setup the output arrays as needed.
        """
        num_outputs = self.num_output_arrays()
        if len(self.output_particle_arrays) != num_outputs:
            # create one particle array each for each output.
            num_outputs = self.num_output_arrays()
            self.output_particle_arrays[:] = [None]*num_outputs
            for i in range(num_outputs):
                self.output_particle_arrays[i] = ParticleArray()                

        # make sure the setup arrays have the required properties.
        # first add the coordinate arrays
        for i in range(num_outputs):
            pa = self.output_particle_arrays[i]
            
            # add x-coords 
            pa.add_property({'name':'x'})
            pa.add_property({'name':'y'})
            pa.add_property({'name':'z'})
            
            # if 'h' setting is needed add that too.
            if self.particle_h != -1.:
                pa.add_property({'name':'h'})
            
            if (self.mass_computation_mode != MassComputationMode.Ignore or
                self.density_computation_mode != DensityComputationMode.Ignore):
                pa.add_property({'name':'m'})
                pa.add_property({'name':'rho'})
            
