import logging
logger = logging.getLogger()

#############################################################################
#`Integrator` class
#############################################################################
class Integrator(object):
    """ The base class for all integrators. Currently, the following 
    integrators are supported:
    
    (a) Forward Euler Integrator
    (b) RK2 Integrator
    (c) RK4 Integrator
    (d) Predictor Corrector Integrator
    (e) Leap Frog Integrator

    The integrator operates on a list of SPHBase objects which define the 
    interaction between a single destination particle array and a list of 
    source particle arrays.

    An instance of SPHBase is called a `calc` and thus, the integrator
    operates on a list of calcs.

    A calc can be integrating or non integrating depending on the
    operation it represents. For example, the summation density and
    density rate operations result in calcs that are non integrating
    and integrating respectively. Note that both of them operate on
    the same LHS variable, namely the density.

    In addition, the integrator differentiates between a calc that
    steps position and calcs that step other physical variables. This
    is because in an SPH operation, the position of the particles is
    updated after the primitive variables (rho, p, v) are stepped.

    Each of the integrators thus step the primitive variables prior to
    stepping of the position.

    Data Attributes:
    ================

    particles:
    ----------
    The manager for the particle arrays used in the simulation.

    calcs:
    ------
    The list of SPHBase operations (calcs) that is used for stepping.

    icalcs:
    -------
    An internal list of integrating and non position calcs. 

    pcalcs:
    -------
    An internal list of position calcs. Used for stepping particle positions.

    nsteps:
    -------
    The number of steps for the integrator. Eg: RK2 has nsteps=2

    cstep:
    ------
    Current step. Used for storing intermediate step values.

    
    The following example is applicable to the description of
    'initial_props', 'step_props' and 'k_props' to follow:

    Consider a dam break simulation using an RK2 integratoe. The
    operations are

    (a) Tait equation (updates=['p','cs'])
    (b) Density Rate (updates=['rho'])
    (c) Momentum equation with avisc  (updates = ['u','v'])
    (d) Gravity force (updates = ['u','v'])
    (e) Position Stepping (updates=['x','y'])
    (f) XSPH Correction (updates=['x','y'])    

    initial_props:
    --------------
    The initial property names for the LHS in the systemm of equations.

    The structure is a dictionary indexed by the calc id (which must
    be unique) with the value being a list of strings, one for each
    update property of the calc. 

    For the example simulation the intitial_props would look like:

    {'eos':['p_0','cs_0'], 'density_rate':['rho_0'], 

    'mom':['u_0','v_0'], 'gravity':['u_0','v_0'], 

    'step':['x_0','y_0'], 'xsph':['x_0','y_0']}

    For each calc, these subscripted variables are added to the
    particle arrays.

    step_props:
    -----------
    The property names for the result of an RHS evaluation.
   
    The order of the calcs define the integration sequence. Any calc that
    does not integrate is evaluated and the value set immediately. 
    The stepping for the properties, that is, the integrating phase 
    is done after all the calcs are evaluated.

    For each calc and each update, the integrator creates a unique property 
    name for the particle array. This prop name is used when the calc's sph
    function is called. In this way, no overwriting of data occurs as each 
    separate call to sph stores the result with a unique property name.
    
    An example is the shock tube problem which uses 4 calcs, summation 
    density, equation of state, momentum equation and energy equation which
    update 'rho', 'p', ('u' ,'v') and 'e' respectively. The unique property 
    arrays created, and stored for reference in `calling_sequence` are:

    [['rho_00'], ['p_10'], ['u_20', 'v_21'], ['e_30']]

    We can see that the first subscript corresponds to the calc number while
    the second subscript to the calc's update number.

    The list is stored in a data attribute `calling_sequence`. This makes it 
    easy to call the calc's sph function with the list corresponding to this
    calc. For example, the momentum equation (calc 2) call to sph 
    would be:

    momentum_equation.sph(*calling_sequence[2]),

    to store the result in `u_20` and `v_20`.

    initial_arrays:
    ===============
    The initial arrays are required in multi step integrators to get to the
    final step. These are stored as follows:

    momentum_equation updates 'u' and 'v' and so, the initial values are
    u_0 and v_0 which are stored in the particle array.
    
    During the final integration step, care must be taken to ensure that the
    initial arrays are also stepped. This would be necessary since multiple
    calcs could be updating the same property and we want the cumulative 
    result of these.

    The `k` list
    ===================
    Consider the integration of a system X' = F(X) using an RK4 scheme.
    The scheme reads:
    
    X(t + h) = h/6 * (K1 + 2K2 + 2K3 + K4)

    where, `h` is the time step and K's are defined as 
    
    K1 = F(X)
    K2 = F(X + 0.5*K1)
    K3 = F(X + 0.5*K2)
    K4 = F(X + K3)

    the `k` data attribute stores the K vectors for each calc and each 
    update property. The list is indexed by the step number with
    the value a dictionary indexed on the calc number and the value of that
    being a dictionary indexed on the update property. The result of this is
    the calc's eval that is set.

    The k dictionary is used in the final step to update the property for the
    particles.

    The do_step
    =====================
    Each calc must define the property it is concerned with and also if this
    property is required to be integrated. The need for differentiating 
    these cases arises from the nature of the SPH process.

    Consider the evolution of density by summation and by the continuity 
    equation. In the summation density approach, the result is simply 
    assigned to the particles density. Moreover, the density is 
    updated (along with pressure) before stepping of any other property.
    In the continuity equation approach, the result is used to integrate the
    density along with all other properties.

    The do step iterates over each calc and calls it's eval function with
    the appropriate calling sequence which should be set by a call to 
    `setup_integrator`. If the calc is non integrating, the evaluated value
    is assigned to the calc's update property. The result for each integrating
    calc is stored in the `k` dictionary.

    After all the evals have been called, we are ready for integrating step.
    The result of all the integrating calcs have been stored in the `k` 
    dictionary. The current array is retrieved via the calc's update property
    and the step array is retrieved through the k dictionary. Stepping is
    as simple as `updated_array = current_array + step_array*dt`

    The integrate step
    ===================
    This is the function to be called while integrating an SPH system.

    """

    def __init__(self, particles=None, calcs=[], pcalcs = []):
        self.particles = particles

        self.calcs = calcs
        self.icalcs = []
        self.pcalcs = []

        self.nsteps = 1
        self.cstep = 1

        self.initial_props = {}
        self.step_props = {}
        self.k_props = {}

        self.setup_done = False

        self.rupdate_list = []

    def set_rupdate_list(self):
        for i in range(len(self.particles.arrays)):
            self.rupdate_list.append([])

    def setup_integrator(self):
        """ Setup the information required for the stepping
        
        Notes:
        ------
        A call to this function must be done before any integration begins.
        This function sets up the properties required for storing the results
        of the calc operation as well as setting the calling_sequence variable.

        Algorithm:
        ----------
        initialize calling_sequence to []
        initialize k to []
        for each step in the integrator
           append a dictionary to k
        for each calc in the calcs
            append an empty list to calling_sequence
            append a dictionary to k corresponding to the calc number
            for each prop in update of the calc
                assign a unique prop name based on calc and update number
                add the property to the particle array
                append to the newly appended list this prop in calling_sequence
                set the value for k[<nstep>][<ncalc>][update_prop] to None

                
        Example:
        --------
        For the shock tube problem with summation density, equation of state, 
        momentum equation and energy equation updating 'rho', 'p', 'u', and 'e'
        respectively, the calling sequence looks like:
        
        [['rho_00'], ['p_10'], ['u_20'], ['e_30']]
        
        """
        
        logger.info("Setup Integrator called")

        self.arrays = self.particles.arrays

        self.initial_props = {}
        self.step_props = {}
        self.k_props = {}

        calcs = self.calcs
        
        self.pcalcs = [calc for calc in calcs if calc.tag == 'position']
        self.ncalcs = [calc for calc in calcs if not calc.tag == 'position']
        self.icalcs = [calc for calc in calcs if not calc.tag == 'position'
                       and calc.integrates==True]

        ncalcs = len(calcs)

        for i in range(ncalcs):
            
            calc = calcs[i]

            #get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]

            updates = calc.updates
            nupdates = len(updates)

            # make an entry in the step and k array for this calc

            self.initial_props[calc.id] = []
            self.step_props[calc.id] = []
            self.k_props[calc.id] = {}

            for j in range(nupdates):
                prop = updates[j]

                # define and add the initial and step properties

                prop_step = prop+'_'+str(i)+str(j)
                prop_initial = prop+'_0'

                pa.add_property({'name':prop_step})
                pa.add_property({'name':prop_initial})

                # set the step array and initial array

                self.initial_props[calc.id].append(prop_initial)
                self.step_props[calc.id].append(prop_step)

                for l in range(self.nsteps):
                    k_num = 'k'+str(l+1)

                    if calc.integrates:

                        if not self.k_props[calc.id].has_key(k_num):
                            self.k_props[calc.id][k_num] = []
                        
                        # add the k array name 
                        
                        k_name = k_num + '_' + prop + str(i) + str(j)
                        pa.add_property({'name':k_name})

                        self.k_props[calc.id][k_num].append(k_name)

        #indicate that the setup is complete

        self.set_rupdate_list()
        self.setup_done = True

    def set_initial_arrays(self):
        """ Set the initial arrays for each calc

        The initial array is the update property of a calc appended with _0
        Note that multiple calcs can update the same property and this 
        will not replicate the creation of the initial arrays. 
        
        """        
        if logger.level < 30:
            logger.info("Integrator: Set initial arrays called")
            
        calcs = self.calcs

        ncalcs = len(calcs)
        for i in range(ncalcs):
            calc = calcs[i]
            updates = calc.updates
            nupdates = len(updates)

            pa = self.arrays[calc.dnum]

            for j in range(nupdates):
                prop = updates[j]

                prop_initial = self.initial_props[calc.id][j]

                pa.set(**{prop_initial:pa.get(prop)})

    def reset_current_arrays(self, calcs):
        """ Reset the current arrays """
        
        if logger.level < 30:
            logger.info("Integrator: Setting current arrays")

        ncalcs = len(calcs)
        for i in range(ncalcs):
            calc = calcs[i]

            updates = calc.updates
            nupdates = len(updates)

            pa = self.arrays[calc.dnum]
            
            for j in range(nupdates):
                prop = updates[j]

                #reset the current property to the initial array

                initial_prop = self.initial_props[calc.id][j]
                pa.set(**{prop:pa.get(initial_prop)})

    def do_step(self, calcs, dt):
        """ Perform one step for the integration
        
        This is an intermediate step in a multi step integrator wherein 
        the step arrays are set in the `k` list. First, each eval is 
        called and the step arrays are stored in the `k` list and then
        for each integrating calc, the current state of the particles is
        advanced with respect to the initial position and the `k` value 
        from a previous step.

        """
        particles = self.particles

        if logger.level < 30:
            logger.info("Integrator:do_step")

        ncalcs = len(calcs)
        
        # Evaluate the RHS

        self.eval(calcs)

        # ensure that the eval phase is completed for all processes

        particles.barrier()

        # Reset the current arrays (Required for multi step integrators)

        self.reset_current_arrays(calcs)
    
        # Step the LHS

        self.step(calcs, dt)

        # ensure that all processors have stepped the local particles

        particles.barrier()

    def eval(self, calcs):
        """ Evaluate each calc and store in the k list if necessary """

        if logger.level < 30:
            logger.info("Integrator:eval")

        ncalcs = len(calcs)
        particles = self.particles
        
        k_num = 'k' + str(self.cstep)
        for i in range(ncalcs):
            calc = calcs[i]

            updates = calc.updates
            nupdates = calc.nupdates

            #get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]
            
            if logger.level < 30:
                logger.info("Integrator:eval: operating on calc %d, %s"%(
                        i, calc.id))

            # Evaluatte the calc

            step_props = self.step_props[calc.id]
            calc.sph(*step_props)

            for j in range(nupdates):
                update_prop = updates[j]
                step_prop = step_props[j]

                step_array = pa.get(step_prop)

                if not calc.integrates:

                    #set the evaluated property
                    if logger.level < 30:
                        logger.info("""Integrator:eval: setting the prop 
                                    %s for calc %d, %s"""
                                    %(update_prop,  i, calc.id))

                    pa.set(**{update_prop:step_array.copy()})

                    # ensure that all processes have reached this point

                    particles.barrier()

                    # update the remote particle properties

                    self.rupdate_list[calc.dnum] = [update_prop]

                    if logger.level < 30:
                        logger.info("""Integrator:eval: updating remote particle
                                     properties %s"""%(self.rupdate_list))
 
                    particles.update_remote_particle_properties(
                        self.rupdate_list)
                    
                else:
                    k_prop = self.k_props[calc.id][k_num][j]

                    pa.set(**{k_prop:step_array.copy()})

                pass

        #ensure that the eval phase is completed for all processes

        particles.barrier()

    def step(self, calcs, dt):
        """ Perform stepping for the integrating calcs """

        ncalcs = len(calcs)

        k_num = 'k' + str(self.cstep)
        for i in range(ncalcs):
            calc = calcs[i]

            if calc.integrates:
                
                updates = calc.updates
                nupdates = calc.nupdates

                # get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                    k_prop = self.k_props[calc.id][k_num][j]

                    current_arr = pa.get(update_prop)
                    step_array = pa.get(k_prop)

                    if logger.level < 30:
                        k_name = k_num + '_' + update_prop + str(i) + str(j)
                        logger.info("""Integrator:do_step: Updating the k array
                                    %s """%(k_name))

                    updated_array = current_arr + step_array*dt
                  
                    pa.set(**{update_prop:updated_array})

                pass
            pass

        # Increment the step by 1

        self.cstep += 1

    def integrate(self, dt, count):
        raise NotImplementedError

##############################################################################


##############################################################################
#`EulerIntegrator` class 
##############################################################################
class EulerIntegrator(Integrator):
    """ Euler integration of the system X' = F(X) with the formula:
    
    X(t + h) = X + h*F(X)
    
    """    
    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 1

    def final_step(self, calc, dt):
        """ Perform the final step for the integrating calc """
        updates = calc.updates
        nupdates = calc.nupdates

        pa = self.arrays[calc.dnum]

        for i in range(nupdates):
            update_prop = updates[i]
            
            initial_prop = self.initial_props[calc.id][i]
            k_prop = self.k_props[calc.id]['k1'][i]

            initial_array = pa.get(initial_prop)
            k1_arr = pa.get(k_prop)
            
            updated_array = initial_array + k1_arr * dt
            
            pa.set(**{update_prop:updated_array})
            pa.set(**{initial_prop:updated_array})

    def integrate(self, dt):

        # set the intitial arrays for all calcs
        
        self.set_initial_arrays()
        
        # evaluate the k1 arrays for non position calcs

        self.eval(self.ncalcs)
        
        # perform the final step for each integrating non position calc
        
        for calc in self.icalcs:
            self.final_step(calc, dt)

        # Now step the position calcs

        self.eval(self.pcalcs)

        for calc in self.pcalcs:
            self.final_step(calc, dt)

        #update the particles to get the new neighbors

        self.particles.update()

#############################################################################


##############################################################################
#`RK2Integrator` class 
##############################################################################
class RK2Integrator(Integrator):
    """ RK2 Integration for the system X' = F(X) with the formula:

    X(t + h) = h/2 * (K1 + K2)

    where,
    
    K1 = F(X)
    K2 = F(X + 0.5h*K1)

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 2

    def final_step(self, calc, dt):
        """ Perform the final step for the RK2 integration """
        updates = calc.updates
        pa = self.arrays[calc.dnum]

        for j in range(len(updates)):
            update_prop = updates[j]
            
            initial_prop = self.initial_props[calc.id][j]            
            k1_props = self.k_props[calc.id]['k1']
            k2_props = self.k_props[calc.id]['k2']
            
            initial_arr = pa.get(initial_prop)
            k1_arr = pa.get(k1_props[j])
            k2_arr = pa.get(k2_props[j])

            updated_array = initial_arr + 0.5 * dt * (k1_arr + k2_arr)

            pa.set(**{update_prop:updated_array})
            pa.set(**{initial_prop:updated_array})

    def integrate(self, dt):
        
        # set the initial arrays

        self.set_initial_arrays()

        # evaluate the k arrays

        while self.cstep != self.nsteps:

            # eval and step the non position calcs
            self.do_step(self.ncalcs, dt)

            self.cstep = 1

            # eavl and step the position calcs
            self.do_step(self.pcalcs, dt)
            
            # update the particle positions
            self.particles.update()

        # eval the k2 arrays for the non position calcs
        self.eval(self.ncalcs)
        
        for calc in self.icalcs:
            self.final_step(calc, dt)

        # now eval and step the position calcs
        self.eval(self.pcalcs)
        
        for calc in self.pcalcs:
            self.final_step(calc, dt)

        # reset the step counter and update the particles

        self.cstep = 1
        self.particles.update()


##############################################################################
#`RK4Integrator` class 
##############################################################################
class RK4Integrator(Integrator):
    """ RK4 Integration of a system X' = F(X) using the scheme
    
    X(t + h) = h/6 * (K1 + 2K2 + 2K3 + K4)

    where, `h` is the time step and K's are defined as 
    
    K1 = F(X)
    K2 = F(X + 0.5*K1)
    K3 = F(X + 0.5*K2)
    K4 = F(X + K3)

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 4

    def final_step(self, calc, dt):
        """ Perform the final step for RK4 integration """
        pa = self.arrays[calc.dnum]
        updates = calc.updates
        
        for j in range(len(updates)):
            update_prop = updates[j]

            initial_prop = self.initial_props[calc.id][j]
            k1_prop = self.k_props[calc.id]['k1'][j]
            k2_prop = self.k_props[calc.id]['k2'][j]
            k3_prop = self.k_props[calc.id]['k3'][j]
            k4_prop = self.k_props[calc.id]['k4'][j]

            initial_array = pa.get(initial_prop)
            k1_array = pa.get(k1_prop)
            k2_array = pa.get(k2_prop)
            k3_array = pa.get(k3_prop)
            k4_array = pa.get(k4_prop)

            updated_array = initial_array + (dt/6.0) *\
                (k1_array + 2*k2_array + 2*k3_array + k4_array)
                    
            pa.set(**{update_prop:updated_array})
            pa.set(**{initial_prop:updated_array})

    def integrate(self, dt):

        # set the initial arrays

        self.set_initial_arrays()

        # evaluate the k arrays

        while self.cstep != self.nsteps:
        
            ################ K1 #################################w
            
            # Eval and step the k1 arrays for non position calcs

            self.do_step(self.ncalcs, 0.5*dt)

            self.cstep = 1
            
            # Eval and step the position calcs

            self.do_step(self.pcalcs, 0.5*dt)

            # update the particles

            self.particles.update()

            ################ K2 #################################
            
            self.do_step(self.ncalcs, 0.5*dt)
            
            self.cstep = 2

            self.do_step(self.pcalcs, 0.5*dt)

            self.particles.update()

            ################ K3 #################################
            
            self.do_step(self.ncalcs, dt)
            
            self.cstep = 3

            self.do_step(self.pcalcs, dt)

            self.particles.update()

        # eval the k4 arrays for the non position calcs

        self.eval(self.ncalcs)
        
        for calc in self.icalcs:
            self.final_step(calc, dt)

        # now eval and step the position calcs

        self.eval(self.pcalcs)
        
        for calc in self.pcalcs:
            self.final_step(calc, dt)

        # reset the step counter and update the particles

        self.cstep = 1
        self.particles.update()            

##############################################################################
#`RK4Integrator` class 
##############################################################################
class PredictorCorrectorIntegrator(Integrator):
    """ Predictor Corrector Integration of a system X' = F(X) using the scheme
    
    the prediction step:
    
    X(t + h/2) = X + h/2 * F(X)

    the correction step
    
    X(t + h/2) = X + h/2 * F(X(t + h/2))

    the final step:
    
    X(t + h) = 2*X(t + h/2) - X

    where, `h` is the time step 

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)

    def final_step(self, calc, dt):
        """ Perform the final step in the PC integration method """
        pa = self.arrays[calc.dnum]
        updates = calc.updates
        
        for j in range(calc.nupdates):
            update_prop = updates[j]
            initial_prop = self.initial_props[calc.id][j]
            
            current_array = pa.get(update_prop)
            initial_array = pa.get(initial_prop)
            
            updated_array = 2*current_array - initial_array

            pa.set(**{update_prop:updated_array})
            pa.set(**{initial_prop:updated_array})

    def integrate(self, dt):
        
        # set the initial arrays

        self.set_initial_arrays()

        ################### Predict #################################
        
        self.do_step(self.ncalcs,0.5*dt)

        self.cstep = 1

        self.do_step(self.pcalcs, 0.5*dt)

        self.particles.update()

        self.cstep = 1

        ################### Correct #################################
        
        self.do_step(self.ncalcs, 0.5*dt)

        self.cstep = 1

        self.do_step(self.pcalcs, 0.5*dt)

        self.particles.update()

        ################### Step #################################        
        
        # Step the non position calcs

        for calc in self.icalcs:
            self.final_step(calc, dt)

        # Step the position calcs

        for calc in self.pcalcs:
            self.final_step(calc, dt)

        # Reset the step counter and update the particles

        self.cstep = 1
        self.particles.update()


##############################################################################
#`LeapFrogIntegrator` class 
##############################################################################
class LeapFrogIntegrator(Integrator):
    """ Leap frog integration of a system :
    
    \frac{Dv}{Dt} = F
    \frac{Dr}{Dt} = v
    \frac{D\rho}{Dt} = D
    
    the prediction step:
    
    vbar = v_0 + h * F_0
    r = r_0 + h*v_0 + 0.5 * h * h * F_0
    rhobar = rho_0 + h * D_0

    correction step:
    v = vbar + 0.5*h*(F - F_0)
    
    rho = rhobar + 0.5*h*(D - D_0)

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 2

    def add_correction_for_position(self, dt):
        ncalcs = len(self.icalcs)

        pos_calc = self.pcalcs[0]

        pos_calc_pa = self.arrays[pos_calc.dnum]
        pos_calc_updates = pos_calc.updates
        
        for i in range(ncalcs):
            calc = self.icalcs[i]

            if calc.tag == "velocity":
                
                pa = self.particles.arrays[calc.dnum]
                
                updates = calc.updates
                for j in range(calc.nupdates):
                    
                    update_prop = pos_calc_updates[j]
                    k1_prop = self.k1_props['k1'][calc.id][j]

                    #the current position

                    current_arr = pos_calc_pa.get(update_prop)

                    step_array = pa.get(k1_prop)

                    updated_array = current_arr + 0.5*dt*dt*step_array
                    pos_calc_pa.set(**{update_prop:updated_array})            

    def final_step(self, calc, dt):
        pa = self.arrays[calc.dnum]
        updates = calc.updates

        for j in range(len(updates)):
            update_prop = updates[j]

            k1_prop = self.k_props[calc.id]['k1'][j]
            k2_prop = self.k_props[calc.id]['k2'][j]

            k1_array = pa.get(k1_prop)
            k2_array = pa.get(k2_prop)

            current_array = pa.get(update_prop)

            updated_array = current_array + 0.5*dt*(k2_array - k1_array)
            
            pa.set(**{update_prop:updated_array})

    def integrate(self, dt):
        
        # set the initial arrays
        
        self.set_initial_arrays()

        # eval and step the non position calcs at the current state
        
        self.do_step(self.ncalcs, dt)

        self.cstep = 1

        # eval and step the position calcs
        
        self.do_step(self.pcalcs, dt)

        # add correction for the positions

        self.add_correction_for_position(dt)

        # ensure all processors have reached this point, then update

        self.particles.barrier()
        self.particles.update()

        # eval and step the non position calcs

        self.eval(self.ncalcs)

        for calc in self.icalcs:
            self.final_step(calc, dt)

        self.cstep = 1
        
############################################################################## 
