"""
Base class for classes generating particle configurations.
"""

# local imports
from pysph.base.kernels cimport KernelBase
from pysph.base.particle_array cimport ParticleArray

cdef class MassComputationMode:
    """
    Enum class to decide the method to compute the mass of the generated
    particles.
    """
    pass

cdef class DensityComputationMode:
    """
    Enum class to decide the method to compute the density of the generated
    particles.
    """
    pass

cdef class ParticleGenerator:
    """
    Base class for classes generating particle configurations.
    """

    # list of particle arrays containing the newly generated paritlces. These
    # may be suppied by the user or be created fresh.
    cdef public list output_particle_arrays
    
    # if mass should be computed for the particles and the method to compute the mass.
    cdef public int mass_computation_mode
    # a constant mass to use for all particles.
    cdef public double particle_mass

    # if density should be computed or not and the method to compute the densities.
    cdef public int density_computation_mode
    # a constact density to set for all particles.
    cdef public double particle_density

    # the interaction radius for all particles.
    # if this is -1.0, the interaction radius is assumed to be already set for
    # the particles and that value is used for the above computation.
    cdef public double particle_h

    # the kernel to be used in computing the above quantities.
    cdef public KernelBase kernel
    
    # function to return the number of output arrays generated by this generator.
    cpdef num_output_arrays(self)
    
    # generate and return the particle arrays.
    cpdef get_particles(self)
    
    # just get the coords of the generated points.
    cpdef get_coords(self)

    # setup the output arrays
    cpdef _setup_outputs(self)

    # makes sure input configuration is valid.
    cpdef bint validate_setup(self) 
    
    # the actual particle generation logic
    cpdef generate_func(self)
