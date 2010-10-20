#base imports 
import pysph.base.api as base

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
    
    def __init__(self, pa, kfac = 2, in_parallel=False,
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

        self.pa = pa

        if not in_parallel:
            self.cell_manager = base.CellManager(arrays_to_bin=[pa,])
        else:
            self.cell_manager = base.ParallelCellManager(
                arrays_to_bin=[pa,], load_balancing=load_balancing)

        self.nnps_manager = base.NNPSManager(cell_manager=self.cell_manager,
                                             variable_h=True)
        self.kfac = kfac
        self.nbr_loc = self.nnps_manager.get_neighbor_particle_locator(
            pa, pa, kfac)

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

        self.needs_update = False

    def get_neighbors(self, pid, neighbors, from_types = [0,1], 
                      exclude_self=False):

        """ Get tagged neighbors for a given particle.

        Parameters:
        -----------
        pid -- the destination particle id whose neighbors are sought.
        from_types -- a list of accepted types (ParticleType)

        Notes:
        ------
        Raises an error if the manager's update has not been called
        
        The particle array should define a `type` property which holds 
        integer values for the kind of particle:
        
        0:base.ParticleType.Fluid, 1:base.ParticleType.Solid

        """

        assert self.needs_update == False, 'Only get neighbors if binned!'

        pa = self.pa
        type = pa.get('type', only_real_particles=False)

        neighbors.reset()
        self.nbr_loc.py_get_nearest_particles(pid, neighbors, exclude_self)

        to_remove = [i for i, j in enumerate(neighbors)\
                         if type[j] not in from_types]

        neighbors.remove(numpy.array(to_remove))
        
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
    if nprops == 0:
        return 
    
    prop_dict = {}

    default_props = ['x','y','z','u','v','w','m','h','p','e','rho',
                     'tmpx','tmpy','tmpz', 'type', '_p']
    
    for prop in props.keys():
        assert type(props[prop]) == numpy.ndarray, 'Numpy array required!'
        if prop == 'bflag':
            prop_dict[prop] = {'data':props[prop], 'type':'int'}
        else:
            prop_dict[prop] = {'data':props[prop]}

    for prop in default_props:
        if prop not in props.keys():
            if prop == 'type':
                prop_dict[prop] = {'name':prop, 'type':'int'}
            else:
                prop_dict[prop] = {'name':prop}
            
    return base.ParticleArray(**prop_dict)
