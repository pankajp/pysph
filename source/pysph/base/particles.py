from cell import CellManager
from nnps import NNPSManager
from particle_array import ParticleArray
from particle_types import ParticleType

Fluid = ParticleType.Fluid
Solid = ParticleType.Solid

# MPI conditional imports
HAS_MPI = True
try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI = False
else:
    from pysph.parallel.parallel_cell import ParallelCellManager

import numpy

class Particles(object):
    """ A particle array for solid and fluid particles.
    
    Works as a wrapper around ParticleArray, CellManager and 
    NNPSManager. Use this to get neighbor locators for an SPH simulation
    involving with the one particle array only paradigm.

    Using the ParticleType data structure, we can retreive tagged particles.
        
    Example:
    --------
    In [1]: import pysph.base.api as base

    In [2]: x = linspace(-pi,pi,101)
    
    In [3]: pa = base.get_particle_array(x=x)
    
    In [4]: particles = base.Particles(pa,kfac=2)

    In [5]: particles.update()
    
    In [6]: neighbors = base.LongArray()
    
    In [7]: particles.get_neighbors(0, from_types=[ParticleType.Fluid],
                                    exclude_self=False)
    
    to get all neighbors with tag `0` (ParticleType.Fluid).

    Notes:
    ------
    The `kfac` argument at construction determines the radius of search for 
    the neighbor locator. The actual search radius is then calculated as 
    `rad = h*kfac` for each particle.

    """
    
    def __init__(self, arrays=[], in_parallel=False,
                 load_balancing=True):
        """ Construct a representation of a particle array 

        Parameters:
        -----------
        pa -- Particle Array generated via `get_particle_array`

        Notes:
        ------
        Default properties like those defined in `get_particle_array` are
        assumed to exist.
        
        """

        self.arrays=arrays
        self.in_parallel = in_parallel
        self.load_balancing = load_balancing

        if not in_parallel:
            self.cell_manager = CellManager(arrays_to_bin=arrays)
        else:
            self.cell_manager = ParallelCellManager(
                arrays_to_bin=arrays, load_balancing=load_balancing)
            self.pid = self.cell_manager.pid

        self.nnps_manager = NNPSManager(cell_manager=self.cell_manager,
                                        variable_h=False)

        self.correction_manager = None

        #call an update on the particles
        self.update()

    def update(self):
        """ Update the status of the neighbor locators and cell manager.
        
        Call this function if any particle has moved to properly get the 
        required neighbors.
        
        Notes:
        ------
        This calls the nnps_manager's update function, which in turn 
        sets the `is_dirty` bit for the locators and cell manager based on
        the particle array's `is_dirty` information.
        
        """
        err = self.nnps_manager.py_update()
        assert err != -1, 'NNPSManager update failed! '

        if self.correction_manager:
            self.correction_manager.update()

        self.needs_update = False

    def get_named_particle_array(self, name):
        """ Return the named particle array if it exists """
        has_array = False

        for array in self.arrays:
            if array.name == name:
                arr = array
                has_array = True                
                
        if has_array:
            return arr
        else:
            print 'Array %s does not exist!' %(name)

    def update_remote_particle_properties(self, props=None):
        """ Perform a remote particle property update. 
        
        This function needs to be called when the remote particles
        on one processor need to be updated on account of computations 
        on another physical processor.

        """
        if self.in_parallel:
            self.cell_manager.update_remote_particle_properties(props=[props])

    def barrier(self):
        if self.in_parallel:
            self.cell_manager.barrier()

###############################################################################

def get_particle_array(**props):
    """ Create and return a particle array with default properties 
    
    Parameters:
    -----------
    props -- A dictionary of properties requested

    Example Usage:
    --------------
    In [1]: import particle

    In [2]: x = linspace(0,1,10)

    In [3]: pa = particle.get_particle_array(x=x)

    In [4]: pa
    Out[4]: <pysph.base.particle_array.ParticleArray object at 0x9ec302c>
 
    """ 
        
    nprops = len(props)
    
    prop_dict = {}
    name = ""
    particle_type = Fluid

    default_props = ['x','y','z','u','v','w','m','h','p','e','rho','cs',
                     'tmpx','tmpy','tmpz',]
    
    for prop in props.keys():
        if prop in ['name','type']:
            pass
        else:
            assert type(props[prop]) == numpy.ndarray, 'Numpy array required!'
            prop_dict[prop] = {'data':props[prop]}

    for prop in default_props:
        if prop not in props.keys():
            prop_dict[prop] = {'name':prop}

    if props.has_key('name'):
        name = props['name']

    if props.has_key("type"):
        particle_type = props["type"]
        assert particle_type in [Fluid, Solid], 'Type not understood!'

    pa = ParticleArray(name=name, particle_type=particle_type, **prop_dict)

    return pa
