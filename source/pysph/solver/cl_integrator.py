from integrator import Integrator
from cl_utils import HAS_CL, get_pysph_root, get_cl_include,\
     get_scalar_buffer, cl_read, get_real

if HAS_CL:
    import pyopencl as cl

from os import path
import numpy

class CLIntegrator(Integrator):

    def setup_integrator(self, context):
        """ Setup the additional particle arrays for integration.

        Parameters:
        -----------

        context -- the OpenCL context

        setup_cl on the calcs must be called when all particle
        properties on the particle array are created. This is
        important as all device buffers will created.

        """
        Integrator.setup_integrator(self)

        self.setup_cl(context)

        self.cl_precision = self.particles.get_cl_precision()

    def setup_cl(self, context):
        """ OpenCL setup """

        self.context = context
                
        for calc in self.calcs:
            calc.setup_cl(context)

        # setup the OpenCL Program
        root = get_pysph_root()
        src = cl_read(path.join(root, 'solver/integrator.cl'), 
                      self.particles.get_cl_precision())

        self.program = cl.Program(context, src).build(get_cl_include())

    def set_initial_buffers(self):
        """ Set the initial arrays for each calc

        The initial array is the update property of a calc appended with _0
        Note that multiple calcs can update the same property and this 
        will not replicate the creation of the initial arrays.

        In OpenCL, we call the EnqueueCopyBuffer with source as the
        current update property and destination as the initial
        property array.
        
        """        
        calcs = self.calcs

        ncalcs = len(calcs)
        for i in range(ncalcs):
            calc = calcs[i]
            queue = calc.queue

            if calc.integrates:
                updates = calc.updates
                nupdates = len(updates)

                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                    initial_prop = self.initial_props[calc.id][j]

                    update_prop_buffer = pa.get_cl_buffer(update_prop)
                    initial_prop_buffer = pa.get_cl_buffer(initial_prop)

                    cl.enqueue_copy_buffer(queue=queue, src=update_prop_buffer,
                                           dst=initial_prop_buffer).wait()

                    pa.read_from_buffer()

    def reset_current_buffers(self, calcs):
        """ Reset the current arrays """
        
        ncalcs = len(calcs)
        for i in range(ncalcs):
            calc = calcs[i]
            queue = calc.queue

            if calc.integrates:

                updates = calc.updates
                nupdates = len(updates)

                pa = self.arrays[calc.dnum]
            
                for j in range(nupdates):
                    update_prop = updates[j]
                    initial_prop = self.initial_props[calc.id][j]

                    # get the device buffers
                    update_prop_buffer = pa.get_cl_buffer(update_prop)
                    initial_prop_buffer = pa.get_cl_buffer(initial_prop)

                    # reset the current property to the initial array

                    cl.enqueue_copy_buffer(queue=queue,src=initial_prop_buffer,
                                           dst=update_prop_buffer)
                    

    def eval(self, calcs):
        """ Evaluate each calc and store in the k list if necessary """

        ncalcs = len(calcs)
        particles = self.particles
        
        k_num = 'k' + str(self.cstep)
        for i in range(ncalcs):
            calc = calcs[i]
            queue = calc.queue

            updates = calc.updates
            nupdates = calc.nupdates

            # get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]
            
            # Evaluate the calc. The result is stored in cl_tmpx, cl_tmpy, ...

            calc.sph()

            pa.read_from_buffer()

            for j in range(nupdates):
                update_prop = updates[j]
                step_prop = self.step_props[j]

                #step_array = pa.get(step_prop)
                step_prop_buffer = pa.get_cl_buffer(step_prop) 

                if not calc.integrates:

                    update_prop_buffer = pa.get_cl_buffer(update_prop)

                    cl.enqueue_copy_buffer(queue, src=step_prop_buffer,
                                           dst=update_prop_buffer).wait()
                                       
                    # ensure that all processes have reached this point

                    particles.barrier()

                    # update neighbor information if 'h' has been updated

                    if calc.tag == "h":
                        particles.update()

                    # update the remote particle properties

                    self.rupdate_list[calc.dnum] = [update_prop]

                    particles.update_remote_particle_properties(
                        self.rupdate_list)
                    
                else:
                    k_prop = self.k_props[calc.id][k_num][j]

                    k_prop_buffer = pa.get_cl_buffer(k_prop)

                    cl.enqueue_copy_buffer(queue, src=step_prop_buffer,
                                           dst=k_prop_buffer).wait()

                pass

        #ensure that the eval phase is completed for all processes

        particles.barrier()

    def step(self, calcs, dt):
        """ Perform stepping for the integrating calcs """

        ncalcs = len(calcs)

        cl_dt = get_real(dt, self.cl_precision)

        k_num = 'k' + str(self.cstep)
        for i in range(ncalcs):
            calc = calcs[i]
            queue = calc.queue

            if calc.integrates:
                
                updates = calc.updates
                nupdates = calc.nupdates

                # get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]
                np = pa.get_number_of_particles()

                for j in range(nupdates):
                    update_prop = updates[j]
                    k_prop = self.k_props[calc.id][k_num][j]

                    current_buffer = pa.get_cl_buffer(update_prop)
                    step_buffer = pa.get_cl_buffer(k_prop)
                    tmp_buffer = pa.get_cl_buffer('tmpx')
                
                    self.program.step(queue, (np,1,1), (1,1,1),
                                      current_buffer, step_buffer,
                                      tmp_buffer, cl_dt)

                    cl.enqueue_copy_buffer(queue, src=tmp_buffer,
                                           dst=current_buffer)
                    
                pass
            pass

        # Increment the step by 1

        self.cstep += 1

##############################################################################
#`CLEulerIntegrator` class 
##############################################################################
class CLEulerIntegrator(CLIntegrator):
    """ Euler integration of the system X' = F(X) with the formula:
    
    X(t + h) = X + h*F(X)
    
    """    
    def __init__(self, particles, calcs):
        CLIntegrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def final_step(self, calc, dt):
        """ Perform the final step for the integrating calc """
        updates = calc.updates
        nupdates = calc.nupdates
        queue = calc.queue

        pa = self.arrays[calc.dnum]
        np = pa.get_number_of_particles()

        cl_dt = get_real(dt, self.cl_precision)

        for i in range(nupdates):
            initial_prop = self.initial_props[calc.id][i]
            k_prop = self.k_props[calc.id]['k1'][i]
            update_prop = updates[i]

            initial_buffer = pa.get_cl_buffer(initial_prop)
            update_buffer = pa.get_cl_buffer(update_prop)
            k1_buffer = pa.get_cl_buffer(k_prop)
            tmp_buffer = pa.get_cl_buffer('tmpx')
           
            self.program.step(queue, (np,1,1), (1,1,1),
                              initial_buffer, k1_buffer,
                              tmp_buffer, cl_dt).wait()

            cl.enqueue_copy_buffer(queue, src=tmp_buffer,
                                   dst=initial_buffer).wait()

            cl.enqueue_copy_buffer(queue, src=tmp_buffer,
                                   dst=update_buffer).wait()

    def integrate(self, dt):

        # set the initial buffers

        self.set_initial_buffers()

        # evaluate the calcs

        self.eval(self.calcs)

        # step the update properties for each integrating calc

        for calc in self.calcs:
            if calc.integrates:
                self.final_step(calc, dt)
                
        # update the partilces
        
        self.particles.update()

#############################################################################
