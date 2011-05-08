""" An implementation of a general solver base class """

import os
from utils import PBar, savez_compressed, savez
from cl_utils import get_cl_devices, HAS_CL

import pysph.base.api as base

from pysph.sph.kernel_correction import KernelCorrectionManager
from pysph.sph.sph_calc import SPHCalc, CLCalc
import pysph.sph.api as sph

from sph_equation import SPHOperation, SPHIntegration

from integrator import EulerIntegrator
from cl_integrator import CLEulerIntegrator

if HAS_CL:
    import pyopencl as cl

import logging
logger = logging.getLogger()

Fluids = base.ParticleType.Fluid

class Solver(object):
    """ Base class for all PySPH Solvers

    **Attributes**
    
    - particles -- the particle arrays to operate on

    - integrator_type -- the class of the integrator. This may be one of any 
      defined in solver/integrator.py

    - kernel -- the kernel to be used throughout the calculations. This may 
      need to be modified to handle several kernels.

    - operation_dict -- an internal structure indexing the operation id and 
      the corresponding operation as a dictionary

    - order -- a list of strings specifying the order of an SPH simulation.
    
    - t -- the internal time step counter

    - pre_step_functions -- a list of functions to be performed before stepping

    - post_step_functions -- a list of functions to execute after stepping

    - pfreq -- the output print frequency

    - dim -- the dimension of the problem

    - kernel_correction -- flag to indicate type of kernel correction.
      Defaults to -1 for no correction

    - pid -- the processor id if running in parallel

    - eps -- the epsilon value to use for XSPH stepping. 
      Defaults to -1 for no XSPH

    - position_stepping_operations -- the dictionary of position stepping 
      operations.
    
    """
    
    def __init__(self, dim, integrator_type):

        self.particles = None
        self.integrator_type = integrator_type
        self.dim = dim

        self.default_kernel = base.CubicSplineKernel(dim=dim)

        self.initialize()
        self.setup_solver()

        self.with_cl = False
        self.cl_integrator_types = {EulerIntegrator:CLEulerIntegrator}

    def initialize(self):
        """ Perform basic initializations """

        self.particles = None
        self.operation_dict = {}
        self.order = []
        self.t = 0
        self.count = 0
        self.execute_commands = None

        self.pre_step_functions = []
        self.post_step_functions = []
        self.pfreq = 100

        self.kernel_correction = -1

        self.pid = None
        self.eps = -1

        self.position_stepping_operations = {}

        self.print_properties = ['x','u','m','h','p','rho',]

        if self.dim > 1:
            self.print_properties.extend(['y','v'])

        if self.dim > 2:
            self.print_properties.extend(['z','w'])

    def switch_integrator(self, integrator_type):
        """ Change the integrator for the solver """

        if self.particles == None:
            raise RuntimeError, "There are no particles!"

        if self.with_cl:
            self.integrator_type = self.cl_integrator_types[integrator_type]

        else:
            self.integrator_type = integrator_type

        self.setup_integrator(self.particles)

    def add_operation_step(self, types, xsph=False, eps=0.5):
        
        """ Specify an acceptable list of types to step

        Parameters
        ----------
        types : a list of acceptable types eg Fluid, Solid

        Notes
        -----
        The types are defined in base/particle_types.py

        """
        updates = ['x','y','z'][:self.dim]

        kernel = base.DummyKernel(dim=self.dim)
        
        id = 'step'
        
        self.add_operation(SPHIntegration(
                sph.PositionStepping.withargs(dim=self.dim), on_types=types,
                updates=updates, id=id, kernel=kernel)
                           )

    def add_operation_xsph(self, eps, hks=False):
        """ Set the XSPH operation if requested

        Parameters
        ----------
        eps : the epsilon value to use for XSPH stepping
        
        Notes
        -----
        The position stepping operation must be defined. This is because
        the XSPH operation is setup for those arrays that need stepping.

        The smoothing kernel used for this operation is the CubicSpline

        """       
        
        assert eps > 0, 'Invalid value for XSPH epsilon: %f' %(eps)
        self.eps = eps

        # create the xsph stepping operation
                
        id = 'xsph'
        err = "position stepping function does not exist!"
        assert self.operation_dict.has_key('step'), err

        types = self.operation_dict['step'].on_types
        updates = self.operation_dict['step'].updates

                           
        self.add_operation(SPHIntegration(

            sph.XSPHCorrection.withargs(eps=eps, hks=hks, dim=self.dim), from_types=types,
            on_types=types, updates=updates, id=id, kernel=self.default_kernel)

                           )        

    def add_operation(self, operation, before=False, id=None):
        """ Add an SPH operation to the solver.

        Parameters
        ----------
        operation : the operation (:class:`SPHOperation`) to add
        before : flag to indicate insertion before an id. Defaults to False
        id : The id where to insert the operation. Defaults to None

        Notes
        -----
        An SPH operation typically represents a single equation written
        in SPH form. SPHOperation is defined in solver/sph_equation.py

        The id for the operation must be unique. An error is raised if an
        operation with the same id exists.

        Similarly, an error is raised if an invalid 'id' is provided 
        as an argument.

        Examples
        --------
        (1)        

        >>> solver.add_operation(operation)
        
        This appends an operation to the existing list. 

        (2)
        
        >>> solver.add_operation(operation, before=False, id=someid)
        
        Add an operation after an existing operation with id 'someid'

        (3)
        
        >>> solver.add_operation(operation, before=True, id=soleid)
        
        Add an operation before the operation with id 'someid'
        

        """
        err = 'Operation %s exists!'%(operation.id)
        assert operation.id not in self.order, err
        assert operation.id not in self.operation_dict.keys()

        self.operation_dict[operation.id] = operation
            
        if id:
            msg = 'The specified operation doesnt exist'
            assert self.operation_dict.has_key(id), msg  + ' in the calcs dict!'
            assert id in self.order, msg + ' in the order list!'

            if before:                
                self.order.insert(self.order.index(id), operation.id)

            else:
                self.order.insert(self.order.index(id)+1, operation.id)
            
        else:
            self.order.append(operation.id)

    def replace_operation(self, id, operation):
        """ Replace an operation.

        Parameters
        ----------
        id : the operation with id to replace
        operation : The replacement operation

        Notes
        -----
        The id to replace is taken from the provided operation. 
        
        An error is raised if the provided operation does not exist.

        """

        msg = 'The specified operation doesnt exist'
        assert self.operation_dict.has_key(id), msg  + ' in the op dict!'
        assert id in self.order, msg + ' in the order list!'

        self.operation_dict.pop(id)

        self.operation_dict[operation.id] = operation

        idx = self.order.index(id)
        self.order.insert(idx, operation.id)
        self.order.remove(id)

    def remove_operation(self, id_or_operation):
        """ Remove an operation with id

        Parameters
        ----------
        id_or_operation : the operation to remove

        Notes
        -----
        Remove an operation with either the operation object or the 
        operation id.

        An error is raised if the operation is invalid.
        
        """
        if type(id_or_operation) == str:
            id = id_or_operation
        else:
            id = id_or_operation.id

        assert id in self.operation_dict.keys(), 'id doesnt exist!'
        assert id in self.order, 'id doesnt exist!'

        self.order.remove(id)
        self.operation_dict.pop(id)

    def set_order(self, order):
        """ Install a new order 

        The order determines the manner in which the operations are
        executed by the integrator.

        The new order and existing order should match else, an error is raised

        """
        for equation_id in order:
            msg = '%s in order list does not exist!'%(equation_id)
            assert equation_id in self.order, msg
            assert equation_id in self.operation_dict.keys(), msg

        self.order = order

    def setup_position_step(self):
        """ Setup the position stepping for the solver """
        pass

    def setup_integrator(self, particles=None):
        """ Setup the integrator for the solver

        The solver's processor id is set if the in_parallel flag is set 
        to true.

        The order of the integrating calcs is determined by the solver's 
        order attribute.

        This is usually called at the start of a PySPH simulation.

        By default, the kernel correction manager is set for all the calcs.
        
        """
        
        if particles:
            self.particles = particles

            self.particles.kernel = self.default_kernel

            if particles.in_parallel:
                self.pid = particles.cell_manager.pid

            self.integrator = self.integrator_type(particles, calcs=[])

            # set the calcs for the integrator

            for equation_id in self.order:
                operation = self.operation_dict[equation_id]

                if operation.kernel is None:
                    operation.kernel = self.default_kernel
                
                calcs = operation.get_calcs(particles, operation.kernel)

                self.integrator.calcs.extend(calcs)

            if self.with_cl:
                self.integrator.setup_integrator(self.cl_context)
            else:
                self.integrator.setup_integrator()

            # Setup the kernel correction manager for each calc

            calcs = self.integrator.calcs
            particles.correction_manager = KernelCorrectionManager(
                calcs, self.kernel_correction)

    def append_particle_arrrays(self, arrays):
        """ Append the particle arrays to the existing particle arrays """

        if not self.particles:
            print 'Warning!, particles not defined'
            return
        
        for array in self.particles.arrays:
            array_name = array.name
            for arr in arrays:
                if array_name == arr.name:
                    array.append_parray(arr)

        self.setup_integrator(self.particles)

    def set_final_time(self, tf):
        """ Set the final time for the simulation """
        self.tf = tf

    def set_time_step(self, dt):
        """ Set the time step to use """
        self.dt = dt

    def set_print_freq(self, n):
        """ Set the output print frequency """
        self.pfreq = n

    def set_output_fname(self, fname):
        """ Set the output file name """
        self.fname = fname    

    def set_output_printing_level(self, detailed_output):
        """ Set the output printing level """
        self.detailed_output = detailed_output

    def add_print_properties(self, props):
        """ Add a list of properties to print """
        for prop in props:
            if not prop in self.print_properties:
                self.print_properties.append(prop)

    def set_output_directory(self, path):
        """ Set the output directory """
        self.output_directory = path

    def set_kernel_correction(self, kernel_correction):
        """ Set the kernel correction manager for each calc """
        self.kernel_correction = kernel_correction
        
        for id in self.operation_dict:
            self.operation_dict[id].kernel_correction=kernel_correction

    def set_cl(self, with_cl=False):
        """ Set the flag to use OpenCL

        This option must be set after all operations are created so that
        we may switch the default SPHCalcs to CLCalcs.

        The solver must also setup an appropriate context which is used
        to setup the ParticleArrays on the device.
        
        """
        self.with_cl = with_cl

        if with_cl:
            if not HAS_CL:
                raise RuntimeWarning, "PyOpenCL not found!"

            for equation_id in self.order:
                operation = self.operation_dict[equation_id]

                # set the type of calc to use for the operation

                operation.calc_type = CLCalc

            self.integrator_type = self.cl_integrator_types[
                self.integrator_type]

            # Setup the OpenCL context
            self.setup_cl()

    def set_command_handler(self, callable, command_interval=1):
        """ set the `callable` to be called at every `command_interval` iteration
        
        the `callable` is called with the solver instance as an argument
        """
        self.execute_commands = callable
        self.command_interval = command_interval

    def solve(self, show_progress=False):
        """ Solve the system

        Notes
        -----
        Pre-stepping functions are those that need to be called before
        the integrator is called. 

        Similarly, post step functions are those that are called after
        the stepping within the integrator.

        """
        self.count = 0
        maxval = int((self.tf - self.t)/self.dt +1)
        bar = PBar(maxval, show=show_progress)

        while self.t < self.tf:
            self.t += self.dt
            self.count += 1
            
            #update the particles explicitly

            self.particles.update()

            # perform any pre step functions
            
            for func in self.pre_step_functions:
                func.eval(self, self.count)

            # perform the integration 

            logger.info("Time %f, time step %f "%(self.t, self.dt))

            self.integrator.integrate(self.dt)

            # perform any post step functions
            
            for func in self.post_step_functions:
                func.eval(self, self.count)

            # dump output

            if self.count % self.pfreq == 0:
                self.dump_output(*self.print_properties)

            bar.update()
        
            if self.execute_commands is not None:
                if self.count % self.command_interval == 0:
                    self.execute_commands(self)

        bar.finish()

    def dump_output(self, *print_properties):
        """ Print output based on level of detail required
        
        The default detail level (low) is the integrator's calc's update 
        property for each named particle array.
        
        The higher detail level dumps all particle array properties.
        
        """

        fname = self.fname + '_' 
        props = {}

        cell_size = self.particles.cell_manager.cell_size

        for pa in self.particles.arrays:
            name = pa.name
            _fname = os.path.join(self.output_directory,
                                  fname + name + '_' + str(self.t) + '.npz')
            
            if self.detailed_output:
                savez(_fname, dt=self.dt, **pa.properties)

            else:
                for prop in print_properties:
                    props[prop] = pa.get(prop)

                savez(_fname, dt=self.dt, cell_size=cell_size, 
                      np = pa.num_real_particles, **props)

    def setup_solver(self):
        """ Implement the basic solvers here 

        All subclasses of Solver may implement this function to add the 
        necessary operations for the problem at hand.

        Look at solver/fluid_solver.py for an example.

        """
        pass                

    def setup_cl(self):
        """ Setup the OpenCL context and other initializations """

        if HAS_CL:
            devices = get_cl_devices()

            gpu_devices = devices['GPU']
            cpu_devices = devices['CPU']

            # choose the first available device by default
            if len(gpu_devices) > 0:
                devices = gpu_devices
                device = gpu_devices[0]

            elif len(cpu_devices) > 0:
                devices = cpu_devices
                device = cpu_devices[0]

            else:
                raise RuntimeError, "Did not find any OpenCL devices!"

            self.cl_context = cl.Context(devices=devices)

############################################################################
