# Standard imports.
import logging
from optparse import OptionParser
# PySPH imports.
from pysph.base.particles import Particles, get_particle_array

# MPI conditional imports
HAS_MPI = True
try:
    from mpi4py import MPI
except ImportError:
    HAS_MPI = False
else:
    from pysph.parallel.load_balancer import LoadBalancer


##############################################################################
# `Application` class.
############################################################################## 
class Application(object):
    """Class used by any SPH application.
    """

    def __init__(self, solver=None, load_balance=True):
        """
        Constructor.

        **Parameters**

         - solver - An instance of a `Solver`.  This can also be set
                    later using the `set_solver` method.

         - load_balance - A boolean which determines if automatic load
                          balancing is to be performed or not.

        """
        self._solver = solver
        self.load_balance = load_balance

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if HAS_MPI:
            self.comm = comm = MPI.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()

        self._setup_optparse()
        
    def _setup_optparse(self):
        usage = """
        %prog [options] 

        Note that you may run this program via MPI and the run will be
        automatically parallelized.  To do this run::

         $ mpirun -n 4 /path/to/your/python %prog [options]
   
        Replace '4' above with the number of processors you have.
        Below are the options you may pass.
        """
        parser = OptionParser(usage)
        self.opt_parse = parser   

    def setup_logging(self, filename='application.log', 
                      loglevel=logging.WARNING,
                      stream=True):
        """Setup logging for the application.
        
        **Parameters**

         - filename - The filename to log messages to.  If this is None
                      or an empty string, no file is used.

         - loglevel - The logging level.

         - stream - Boolean indicating if logging is also printed on
                    stderr.
        """
        # logging setup
        logger = logging.getLogger()
        logger.setLevel(loglevel)
        # Setup the log file.
        if filename is not None and len(filename) > 0:
            lfn = filename
            if self.num_procs > 1:
                lfn = filename + '.%d'%self.rank
            logging.basicConfig(level=loglevel, filename=lfn,
                                filemode='w')
        if stream:
            logger.addHandler(logging.StreamHandler())

    def process_command_line(self):
        """Parse any command line arguments.  Add any new options before
        this is called."""
        (options, args) = self.opt_parse.parse_args()
        self.options = options
        self.args = args

    def create_particles(self, callable, *args, **kw):
        """ Create particles given a callable and any arguments to it.
        This will also automatically distribute the particles among
        processors if this is a parallel run.  Returns the `Particles`
        instance that is created.
        """

        num_procs = self.num_procs
        rank = self.rank
        data = None
        if rank == 0:
            # Only master creates the particles.
            pa = callable(*args, **kw)

            if num_procs > 1:
                # Use the offline load-balancer to distribute the data
                # initially. Negative cell size forces automatic computation. 
                data = LoadBalancer.distribute_particles(pa, 
                                                         num_procs=num_procs, 
                                                         cell_size=-1)
        if num_procs > 1:
            # Now scatter the distributed data.
            pa = self.comm.scatter(data, root=0)

        self.particle_array = pa
        in_parallel = num_procs > 1
        self.particles = Particles(arrays=[pa], in_parallel=in_parallel,
                                   load_balancing=self.load_balance)
        return self.particles

    def set_solver(self, solver):
        """Set the application's solver.  This will call the solver's
        `setup_integrator` method."""
        self._solver = solver
        solver.setup_integrator(self.particles)

    def run(self):
        """Run the application."""
        self._solver.solve()

