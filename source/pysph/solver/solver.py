""" An implementation of a general solver base class """

import logging
import os
from utils import PBar, savez_compressed, savez

from pysph.sph.kernel_correction import KernelCorrectionManager
from pysph.sph.function import XSPHCorrection, PositionStepping

from sph_equation import SPHSummationODE, SPHSimpleODE

logger = logging.getLogger()

class Solver(object):
    """ Base class for all PySPH Solvers

    Attributes:
    ------------
    particles -- the particle arrays to operate on

    integrator_type -- the class of the integrator. This may be one of any 
                       defined in solver/integrator.py

    kernel -- the kernel to be used throughout the calculations. This may 
              need to be modified to handle several kernels.

    operation_dict -- an internal structure indexing the operation id and 
                      the corresponding operation as a dictionary

    order -- a list of strings specifying the order of an SPH simulation.
    
    t -- the internal time step counter

    pre_step_functions -- a list of functions to be performed before stepping

    post_step_functions -- a list of functions to execute after stepping

    pfreq -- the output print frequency

    dim -- the dimension of the problem

    kernel_correction -- flag to indicate type of kernel correction.
                         Defaults to -1 for no correction

    pid -- the procesor id if running in parallel

    eps -- the epsilon value to use for XSPH stepping. 
           Defaults to -1 for no XSPH

    position_stepping_operations -- the dictionary of position stepping 
                                    operations.
    
    """
    
    def __init__(self, kernel, integrator_type):
        self.particles = None
        self.integrator_type = integrator_type
        self.kernel = kernel
        self.operation_dict = {}
        self.order = []
        self.t = 0

        self.pre_step_functions = []
        self.post_step_functions = []
        self.pfreq = 100

        self.dim = kernel.dim
        self.kernel_correction = -1

        self.pid = None
        self.eps = -1

        self.position_stepping_operations = {}

        self.setup_solver()

    def to_step(self, types):
        """ Specify an acceptable list of types to step

        Parameters:
        -----------
        
        types -- a list of acceptable types eg Fluid, Solid

        Notes:
        ------
        The types are defined in base/particle_types.py

        """
        updates = ['x']
        if self.dim > 1:
            updates.append('y')
            if self.dim > 2:
                updates.append('z')
                
        id = 'step'
        
        self.position_stepping_operations[id] = SPHSimpleODE(
            PositionStepping(), on_types=types, updates=updates, id=id)
        
    def add_operation(self, operation, before=False, id=None):
        """ Add an SPH operation to the solver.

        Parameters:
        -----------
        operation -- the operation to add
        before -- flag to indicate insertion before an id. Defaults to False
        id -- The id where to insert the operation. Defaults to None

        Notes:
        ------
        An SPH operation typically represents a single equation written
        in SPH form. The different types of operations are:
        
        (i) SPHSimpleODE
        (ii) SPHSummationODE
        (iii) SPHSummation
        (iv) SPHAssignment
        
        and are defined in solver/sph_equation.py

        The id for the operation must be unique. An error is raised if an
        operation with the same id exists.

        Simillarly, an error is raised if an invalid 'id' is provided 
        as an argument.

        Usage Examples:
        ---------------
        (1)        

        >>> solver.add_operation(operation)
        
        This appends an operation to the existing list. 

        (2)
        
        >>> solver.add_operation(operation, before=False, id=someid)
        
        Add an operation after an existing operation with id 'someid'

        (3)
        
        >>> solver.add_operation(operation, before=True, id=soleid)
        
        Add an operation befor the operation with id 'someid'
        

        """
        err = 'Operation %s exists!'%(operation.id)
        assert operation.id not in self.order, err
        assert operation.id not in self.operation_dict.keys()

        self.operation_dict[operation.id] = operation
            
        if id:
            msg = 'The specified operation dosent exist'
            assert self.operation_dict.has_key(id), msg  + ' in the calcs dict!'
            assert id in self.order, msg + ' in the order list!'

            if before:                
                self.order.insert(self.order.index(id), operation.id)

            else:
                self.order.insert(self.order.index(id)+1, operation.id)
            
        else:
            self.order.append(operation.id)

    def replace_operation(self, operation):
        """ Replace an operation.

        Parameters:
        -----------
        operation -- The replacement operation

        Notes:
        ------
        The id to replace is taken from the provided operation. 
        
        An error is raised if the provided operation does not exist.

        """
        id = operation.id

        msg = 'The specified operation dosent exist'
        assert self.operation_dict.has_key(id), msg  + ' in the op dict!'
        assert id in self.order, msg + ' in the order list!'

        self.operation_dict.pop(id)
        self.operation_dict[operation.id] = operation

    def remove_operation(self, id_or_operation):
        """ Remove an operation with id

        Parameters:
        -----------
        id_or_operation -- the operation to remove

        Notes:
        ------
        Remove an operation with either the operation object or the 
        operation id.

        An error is raised if the operation is invalid.
        
        """
        if type(id_or_operation) == str:
           id = id_or_operation
        else:
            id = id_or_operation.id

        assert id in self.calc_dict.keys(), 'id dosent exist!'
        assert id in self.order, 'id dosent exist!'

        self.order.remove(id)
        self.operation_dict.pop(id)

    def set_order(self, order):
        """ Install a new order 

        Notes:
        ------
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

        Notes:
        ------
        The solver's processor id is set if the in_parallel flag is set 
        to true.

        The order of the integrating calcs is determined by the solver's 
        order attribute.

        This is usually called at the start of a PySPH simulation.

        By default, the kernel correction manager is set for all the calcs.
        
        """
        
        if particles:
            self.particles = particles

            if particles.in_parallel:
                self.pid = particles.cell_manager.pid

            self.integrator = self.integrator_type(particles, calcs=[])

            for equation_id in self.order:
                operation = self.operation_dict[equation_id]
                calcs = operation.get_calcs(particles,self.kernel)
                self.integrator.calcs.extend(calcs)

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

    def set_xsph(self, eps):
        """ Set the XSPH operation if requested

        Parameters:
        -----------
        eps -- the epsilon value to use for XSPH stepping
        
        Notes:
        ------
        The position stepping operation must be defined. This is because
        the XSPH operation is setup for those arrays that need stepping.

        """       
        
        assert eps > 0, 'Invalid value for XSPH epsilon: %f' %(eps)
        self.eps = eps

        # create the xsph stepping operation
                
        id = 'xsph'
        err = "position stepping function does not exist!"
        assert self.position_stepping_operations.has_key('step'), err

        types = self.position_stepping_operations['step'].on_types
        updates = self.position_stepping_operations['step'].updates

        self.position_stepping_operations[id] = SPHSummationODE(

            XSPHCorrection(eps=eps), from_types=types,
            on_types=types,  updates=updates, id=id )        

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

    def set_output_directory(self, path):
        """ Set the out put directory """
        self.path = path

    def set_kernel_correction(self, kernel_correction):
        """ Set the kernel correction manager for each calc """
        self.kernel_correction = kernel_correction
        
        for id in self.operation_dict:
            self.operation_dict[id].kernel_correction=kernel_correction
        
    def solve(self, show_progress=False):
        """ Solve the system

        Notes:
        ------
        Pre-stepping functions are those that need to be called before
        the integrator is called. 

        Similarlly, post step functions are those that are called after
        the stepping within the integrator.

        """
        tf = self.tf
        dt = self.dt

        count = 0
        maxval = int((tf - self.t)/dt +1)
        bar = PBar(maxval, show=show_progress)

        while self.t < tf:
            self.t += dt
            count += 1
            
            #update the particles explicitly

            self.particles.update()

            #perform any pre step functions
            
            for func in self.pre_step_functions:
                func.eval(self.particles, count)

            #perform the integration 

            logger.info("TIME %f"%(self.t))

            self.integrator.integrate(dt)

            #perform any post step functions
            
            for func in self.post_step_functions:
                func.eval(self.particles, count)

            #dump output
            if count % self.pfreq == 0:
                self.dump_output(self.t)

            logger.info("Time %f, time step %f "%(self.t, dt))
            bar.update()

        bar.finish()

    def dump_output(self, t):
        """ Print output based on level of detail required
        
        The default detail level (low) is the integrator's calc's update 
        property for each named particle array.
        
        The higher detail level dumps all particle array properties.
        
        """

        fname = self.fname + '_' 
        props = {}

        for pa in self.particles.arrays:
            name = pa.name
            _fname=os.path.join(self.path,fname + name + '_' + str(t) + '.npz')
            
            if self.detailed_output:
                savez(_fname, dt=self.dt, **pa.properties)

            else:
                #set the default properties
                props['x'] = pa.get('x')
                props['u'] = pa.get('u')
                props['h'] = pa.get('h')
                props['m'] = pa.get('m')
                props['e'] = pa.get('e')
                props['p'] = pa.get('p')
                props['rho'] = pa.get('rho')                

                if self.dim > 1:
                    props['y'] = pa.get('y')
                    props['v'] = pa.get('v')                

                    if self.dim > 2:
                        props['z'] = pa.get('z')
                        props['w'] = pa.get('w')

                savez(_fname, dt=self.dt, **props)                        

    def setup_solver(self):
        """ Implement the basic solvers here 

        Notes:
        ------
        All subclasses of Solver may implement this function to add the 
        necessary operations for the problem at hand.

        Look at solver/fluid_solver.py for an example.

        """
        pass


############################################################################
