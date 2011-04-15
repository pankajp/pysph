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
    """ A collection of particles and related data structures that
    hat define an SPH simulation.

    In pysph, particle properties are stored in a ParticleArray. The
    array may represent a particular type of particle (solid, fluid
    etc). Valid types are defined in base.particle_types.

    Indexing of the particles is performed by a CellManager and
    nearest neighbors are obtained via an instance of NNPSManager.

    Particles is a collection of these data structures to provide a
    single point access to

    (a) Hold all particle information
    (b) Update the indexing scheme when particles have moved.
    (d) Update remote particle properties in parallel runs.
    (e) Barrier synchronizations across processors

    Data Attributes:
    ----------------

    arrays -- a list of particle arrays in the simulation.

    cell_manager -- the CellManager for spatial indexing.

    nnps_manager -- the NNPSManager for neighbor queries.

    correction_manager -- a kernel KernelCorrectionManager if kernel
    correction is used. Defaults to None

    misc_prop_update_functions -- A list of functions to evaluate
    properties at the beginning of a sub step.

    variable_h -- boolean indicating if variable smoothing lengths are
    considered. Defaults to False

    in_parallel -- boolean indicating if running in parallel. Defaults to False

    load_balancing -- boolean indicating if load balancing is required.
    Defaults to False.

    pid -- processor id if running in parallel

    Example:
    ---------

    In [1]: import pysph.base.api as base

    In [2]: x = linspace(-pi,pi,101)
    
    In [3]: pa = base.get_particle_array(x=x)
    
    In [4]: particles = base.Particles(arrays=[pa], in_parallel=True,
                                       load_balancing=False, variable_h=True)


    Notes:
    ------

    An appropriate cell manager (CellManager/ParallelCellManager) is
    created with reference to the 'in_parallel' attribute.

    Similarly an appropriate NNPSManager is created with reference to
    the 'variable_h' attribute.

    """
    
    def __init__(self, arrays=[], in_parallel=False, variable_h=False,
                 load_balancing=True, update_particles=True,
                 periodic_domain=None, min_cell_size=-1):
        
        """ Constructor

        Parameters:
        -----------

        arrays -- list of particle arrays in the simulation

        in_parallel -- flag for parallel runs

        variable_h -- flag for variable smoothing lengths

        load_balancing -- flag for dynamic load balancing.

        periodic_domain -- the periodic domain for periodicity

        """

        # set the flags

        self.variable_h = variable_h
        self.in_parallel = in_parallel
        self.load_balancing = load_balancing

        self.arrays=arrays        

        self.kernel = None

        # create the cell manager

        if not in_parallel:
            self.cell_manager = CellManager(arrays_to_bin=arrays,
                                            min_cell_size=min_cell_size,
                                            periodic_domain=periodic_domain)
        else:
            self.cell_manager = ParallelCellManager(
                arrays_to_bin=arrays, load_balancing=load_balancing)

            self.pid = self.cell_manager.pid

        # create the nnps manager

        self.nnps_manager = NNPSManager(cell_manager=self.cell_manager,
                                        variable_h=variable_h)

        # set defaults
        
        self.correction_manager = None        
        self.misc_prop_update_functions = []

        # call an update on the particles (i.e index)
        
        if update_particles:
            self.update()

    def update(self, cache_neighbors=False):
        """ Update the status of the Particles.

        Parameters:
        -----------

        cache_neighbors -- flag for caching kernel interactions 
        
        Notes:
        -------
        
        This function must be called whenever particles have moved and
        the indexing structure invalid. After a call to this function,
        particle neighbors will be accurately returned. 

        Since particles move at the end of an integration
        step/sub-step, we may perform any other operation that would
        be required for the subsequent step/sub-step. Examples of
        these are summation density, equation of state, smoothing
        length updates, evaluation of velocity divergence/vorticity
        etc. 

        All other properties may be updated by appending functions to
        the list 'misc_prop_update_functions'. These functions must
        implement an 'eval' method which takes no arguments. An example
        is the UpdateDivergence function in 'sph.update_misc_props.py'
        
        """

        # update the cell structure

        err = self.nnps_manager.py_update()
        assert err != -1, 'NNPSManager update failed! '

        # update any other properties (rho, p, cs, div etc.)
            
        self.evaluate_misc_properties()

        # evaluate kernel correction terms

        if self.correction_manager:
            self.correction_manager.update()

    def evaluate_misc_properties(self):
        """ Evaluate properties from the list of functions. """
        
        for func in self.misc_prop_update_functions:
            func.eval()

    def add_misc_function(self, func, operation, kernel):
        """ Add a function to be performed when particles are updated

        Parameters:
        -----------
        func -- The function to perform. Defined in sph.update_functions
        operation -- the calc operation that is required for the function
        kernel -- the kernel used to setup the calcs.

        Example:
        --------

        The conduction coefficient required for the artificial heat
        requires the velocity divergence at a particle. This must be
        available at the start of every substep of an integration step.

        """

        calcs = operation.get_calcs(self, kernel)
        self.misc_prop_update_functions.append(func(calcs))

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
            self.cell_manager.update_remote_particle_properties(props=props)

    def barrier(self):
        """ Synchronize all processes """
        if self.in_parallel:
            self.cell_manager.barrier()

    def get_neighbor_particle_locator(self, src, dst, radius_scale=2.0):
        """ Return a neighbor locator from the NNPSManager """
        return self.nnps_manager.get_neighbor_particle_locator(
            src, dst, radius_scale)

###############################################################################

def get_particle_array(floating_point_default="double", **props):
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
                     'tmpx','tmpy','tmpz']
    
    #Add the properties requested
    
    np = 0

    for prop in props.keys():
        if prop in ['name','type']:
            pass
        else:
            assert type(props[prop]) == numpy.ndarray, 'Numpy array required!'
            np = len(props[prop])
            if prop == 'idx':
                prop_dict[prop] = {'data':props[prop], 'type':'int'}
            else:
                data = props[prop]
                prop_dict[prop] = {'data':data, 'type':floating_point_default}
            
    # Add the default props

    for prop in default_props:
        if prop not in props.keys():
            prop_dict[prop] = {'name':prop, 'type':floating_point_default}

    # Add the property idx

    if not prop_dict.has_key('idx') and np != 0:
        prop_dict['idx'] = {'name':'idx', 'data':numpy.arange(np),
                            'type':'int'}
            
    #handle the name and particle_type information separately

    if props.has_key('name'):
        name = props['name']

    if props.has_key("type"):
        particle_type = props["type"]
        assert particle_type in [Fluid, Solid], 'Type not understood!'

    pa = ParticleArray(name=name, particle_type=particle_type, **prop_dict)

    return pa
