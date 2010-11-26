# Standard imports.
import logging, os
from optparse import OptionParser
from os.path import basename, splitext
import sys

from utils import mkdir

# PySPH imports.
from pysph.base.particles import Particles, get_particle_array, ParticleArray

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

    def __init__(self, load_balance=True, fname=None):
        """
        Constructor.

        **Parameters**

         - load_balance - A boolean which determines if automatic load
                          balancing is to be performed or not.

        """
        self._solver = None 
        self.load_balance = load_balance

        if fname == None:
            fname = sys.argv[0].split('.')[0]

        self.fname = fname

        # MPI related vars.
        self.comm = None
        self.num_procs = 1
        self.rank = 0
        if HAS_MPI:
            self.comm = comm = MPI.COMM_WORLD
            self.num_procs = comm.Get_size()
            self.rank = comm.Get_rank()
        
        self._log_levels = {'debug': logging.DEBUG,
                           'info': logging.INFO,
                           'warning': logging.WARNING,
                           'error': logging.ERROR,
                           'critical': logging.CRITICAL,
                           'none': None}

        self._setup_optparse()

        self.path = None
    
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

        # Add some default options.
        parser.add_option("-b", "--no-load-balance", action="store_true",
                         dest="no_load_balance", default=False,
                         help="Do not perform automatic load balancing "\
                              "for parallel runs.")
        # -v
        valid_vals = "Valid values: %s"%self._log_levels.keys()
        parser.add_option("-v", "--loglevel", action="store",
                          type="string",
                          dest="loglevel",
                          default='warning',
                          help="Log-level to use for log messages. " +
                               valid_vals)
        # --logfile
        parser.add_option("--logfile", action="store",
                          type="string",
                          dest="logfile",
                          default=None,
                          help="Log file to use for logging, set to "+
                               "empty ('') for no file logging.")
        # -l 
        parser.add_option("-l", "--print-log", action="store_true",
                          dest="print_log", default=False,
                          help="Print log messages to stderr.")
        # --final-time
        parser.add_option("--final-time", action="store",
                          type="float",
                          dest="final_time",
                          default=None,
                          help="Total time for the simulation.")
        # --time-step
        parser.add_option("--time_step", action="store",
                          type="float",
                          dest="time_step",
                          default=None,
                          help="Time-step to use for the simulation.")
        # -q/--quiet.
        parser.add_option("-q", "--quiet", action="store_true",
                         dest="quiet", default=False,
                         help="Do not print any progress information.")

        # -o/ --output
        parser.add_option("-o", "--output", action="store",
                          dest="output", default=self.fname,
                          help="File name to use for output")

        # --output-freq.
        parser.add_option("--freq", action="store",
                          dest="freq", default=20, type="int",
                          help="Printing frequency for the output")
        
        # -d/ --detailed-output.
        parser.add_option("-d", "--detailed-output", action="store_true",
                         dest="detailed_output", default=False,
                         help="Dump detailed output.")

        # --directory
        parser.add_option("--directory", action="store",
                         dest="output_dir", default=".",
                         help="Dump output in the specified directory.")

        # -k/--kernel-correction
        parser.add_option("-k", "--kernel-correction", action="store",
                          dest="kernel_correction", type="int",
                          default=-1,
                          help="""Use Kernel correction.
                                  0 - Bonnet and Lok correction
                                  1 - RKPM first order correction""")

        # --xsph
        parser.add_option("--xsph", action="store", dest="eps", type="float",
                          default=None, 
                          help="Use XSPH correction with epsilon value")

    def _setup_logging(self, filename=None, 
                      loglevel=logging.WARNING,
                      stream=True):
        """Setup logging for the application.
        
        **Parameters**

         - filename - The filename to log messages to.  If this is None
                      a filename is automatically chosen and if it is an
                      empty string, no file is used.

         - loglevel - The logging level.

         - stream - Boolean indicating if logging is also printed on
                    stderr.
        """
        # logging setup
        logger = logging.getLogger()
        logger.setLevel(loglevel)
        # Setup the log file.
        if filename is None:
            filename = splitext(basename(sys.argv[0]))[0] + '.log'
        if len(filename) > 0:
            lfn = os.path.join(self.path,filename)
            if self.num_procs > 1:
                logging.basicConfig(level=loglevel, filename=lfn,
                                    filemode='w')
        if stream:
            logger.addHandler(logging.StreamHandler())

    ######################################################################
    # Public interface.
    ###################################################################### 
    def process_command_line(self):
        """Parse any command line arguments.  Add any new options before
        this is called.  This also sets up the logging automatically.
        """
        (options, args) = self.opt_parse.parse_args()
        self.options = options
        self.args = args
        
        # Setup logging based on command line options.
        level = self._log_levels[options.loglevel]

        #save the path where we want to dump output
        self.path = os.path.abspath(options.output_dir)
        mkdir(self.path)

        if level is not None:
            self._setup_logging(options.logfile, level,
                                options.print_log)

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
        if isinstance(pa, (ParticleArray,)):
            pa = [pa]
        self.particles = Particles(arrays=pa, in_parallel=in_parallel,
                                   load_balancing=self.load_balance)
        return self.particles

    def set_solver(self, solver):
        """Set the application's solver.  This will call the solver's
        `setup_integrator` method."""
        self._solver = solver
        dt = self.options.time_step
        if dt is not None:
            solver.set_time_step(dt)
        tf = self.options.final_time
        if tf is not None:
            solver.set_final_time(tf)

        #setup the solver output file name
        fname = self.options.output

        if HAS_MPI:
            comm = self.comm 
            rank = self.rank
            
            if not self.num_procs == 0:
                fname += '_' + str(rank)

        solver.set_output_fname(fname)
        solver.set_print_freq(self.options.freq)
        solver.set_output_printing_level(self.options.detailed_output)
        solver.set_output_directory(self.options.output_dir)
        solver.set_kernel_correction(self.options.kernel_correction)

        if self.options.eps:
            solver.set_xsph(self.options.eps)
        
        solver.setup_integrator(self.particles)

    def run(self):
        """Run the application."""
        self._solver.solve(not self.options.quiet)

