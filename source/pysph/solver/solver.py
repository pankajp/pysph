""" An implementation of a general solver base class """

import logging
import os
from utils import PBar, savez_compressed, savez

from pysph.sph.kernel_correction import KernelCorrectionManager

logger = logging.getLogger()

class Solver(object):
    """ Intended as the base class for all solvers """
    
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
        self.setup_solver()

    def add_operation(self, operation, before=False, id=None):
        """ Add an SPH operation to the solver.

        Parameters:
        -----------
        operation -- the operation to add
        before -- flag to indicate insertion before an id. Defaults to False
        id -- The id where to insert the operation. Defaults to None

        Notes:
        ------
        By default, a call to add_operation appends the operation.

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

        Algorithm:
        ----------
        get the id of the operation to be replaced
        assert the replacement is valid
        set the particles and kernel for the operation
        pop the existing operation from the calc_dict
        add the new operation to the calc_dict        

        """
        id = operation.id

        msg = 'The specified operation dosent exist'
        assert self.operation_dict.has_key(id), msg  + ' in the op dict!'
        assert id in self.order, msg + ' in the order list!'

        self.operation_dict.pop(id)
        self.operation_dict[operation.id] = operation

    def remove_operation(self, id_or_operation):
        """ Remove an operation with id
        
        Algorithm:
        ----------
        assert the id is valid
        remove the id from the order list
        pop the operation from the calc_dict
        
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
        The new order and existing order should match else an error is raised

        """
        for equation_id in order:
            msg = '%s in order list does not exist!'%(equation_id)
            assert equation_id in self.order, msg
            assert equation_id in self.operation_dict.keys(), msg

        self.order = order

    def setup_integrator(self, particles=None):
        """ Setup the integrator for the solver

        Notes:
        ------
        The order for the calcs is provided by the order list and the calc
        is maintained in the calc_dict.

        Using this, the integrator's calcs can be setup in order
        
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

            #setup the kernel correction manager for each calc
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
        self.tf = tf

    def set_time_step(self, dt):
        self.dt = dt

    def set_print_freq(self, n):
        self.pfreq = n

    def set_output_fname(self, fname):
        self.fname = fname    

    def set_output_printing_level(self, detailed_output):
        self.detailed_output = detailed_output

    def set_output_directory(self, path):
        self.path = path

    def set_kernel_correction(self, kernel_correction):
        self.kernel_correction = kernel_correction
        
        for id in self.operation_dict:
            self.operation_dict[id].kernel_correction=kernel_correction
        
    def solve(self, show_progress=False):
        """ Solve the system by repeatedly calling the integrator """
        tf = self.tf
        dt = self.dt

        t = 0
        count = 0
        maxval = int((tf - t)/dt +1)
        bar = PBar(maxval, show=show_progress)

        while t < tf:
            t += dt
            count += 1
            
            #update the particles explicitly

            self.particles.update()

            #perform any pre step functions
            
            for func in self.pre_step_functions:
                func.eval(self.particles, count)

            #perform the integration 

            self.integrator.integrate(dt)

            #perform any post step functions
            
            for func in self.post_step_functions:
                func.eval(self.particles, count)

            #dump output
            if count % self.pfreq == 0:
                self.dump_output(t)

            logger.info("Time %f, time step %f "%(t, dt))
            bar.update()

        self.t += t
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
        """ Implement the basic solvers here """
        pass


############################################################################
