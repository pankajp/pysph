import logging

logger = logging.getLogger()

#############################################################################
#`Integrator` class
#############################################################################
class Integrator(object):
    """ Encapsulate the Euler integration step with the calcs 

    calling_sequence:
    =================
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
    intial arrays are also stepped. This would be necessary since multiple
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
    dictionay. The current array is retreived via the calc's update property
    and the step array is retreived through the k dictionary. Stepping is
    as simple as `updated_array = current_array + step_array*dt`

    The integrate step
    ===================
    This is the function to be called while integrating an SPH system.

    """

    def __init__(self, particles=None, calcs=[], pcalcs = []):
        self.particles = particles
        self.calcs = calcs
        self.calling_sequence = []
        self.nsteps = 1
        self.step = 1
        self.setup_done = False

        self.rupdate_list = []

        self.pcalcs = []
        self.initial_props = {}
        self.step_props = {}
        self.k_props = {}

    def setup_integrator(self):
        """ Setup the information required for the stepping
        
        Notes:
        ------
        A call to this function must be done before any integration begins.
        This function sets up the properties required for storing the results
        of the calc operation as well as setting the calling_sequence variable.

        Algorithm:
        ----------
        intialize calling_sequence to []
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

        self.calling_sequence = []

        ncalcs = len(self.calcs)       

        for i in range(ncalcs):
            
            calc = self.calcs[i]

            #get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]

            updates = calc.updates
            nupdates = len(updates)

            #append an entry for the calling sequence for this calc

            self.calling_sequence.append([])

            for j in range(nupdates):
                #get the calc's update property

                prop = updates[j]

                #define and add the step property name

                prop_step = prop+'_'+str(i)+str(j)
                pa.add_property({'name':prop_step})

                #define and add the initial property name

                prop_initial = prop+'_0'
                pa.add_property({'name':prop_initial})

                #set the calling sequence
                self.calling_sequence[-1].append(prop_step)

                for l in range(self.nsteps):
                    if calc.integrates:

                        #get and add the k array name for the property

                        k_name = 'k'+str(l+1)+'_'+ prop+str(i)+str(j)
                        pa.add_property({'name':k_name})

        #indicate that the setup is complete

        self.set_rupdate_list()
        self.setup_done = True

    def set_rupdate_list(self):
        for i in range(len(self.particles.arrays)):
            self.rupdate_list.append([])

    def set_initial_arrays(self):
        """ Set the initial arrays for the integrator

        The initial array is the update property of a calc appended with _0
        Note that multiple calcs can update the same property and this 
        will not replicate the creation of the initial arrays. 
        
        """        
        if logger.level < 30:
            logger.info("Integrator: Set initial arrays called")
        ncalcs = len(self.calcs)
        for i in range(ncalcs):            
            calc = self.calcs[i]
            updates = calc.updates

            #get the dest particle array for this calc

            pa = self.arrays[calc.dnum]

            nupdates = len(updates)
            for j in range(nupdates):
                #get the calc's update property
                
                prop = updates[j]

                #define the and set the initial property arrays

                prop_initial = prop+'_0'
                pa.set(**{prop_initial:pa.get(prop)})

    def reset_current_arrays(self):
        """ Reset the current arrays """
        
        if logger.level < 30:
            logger.info("Integrator: Setting current arrays")

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            updates = calc.updates

            #get the dest particle array for this calc
            
            pa = self.arrays[calc.dnum]

            nupdates = len(updates)
            for j in range(nupdates):
                #get the calc's update property

                prop = updates[j]

                #reset the current property to the initial array

                prop_initial = prop+'_0'
                pa.set(**{prop:pa.get(prop_initial)})

    def eval(self):
        """ Evaluate each calc and store in the k list if necessary """
        if logger.level < 30:
            logger.info("Integrator:eval")

        calling_sequence = self.calling_sequence
        ncalcs = len(self.calcs)
        particles = self.particles
        
        #call each of the eval functions in order
        for i in range(ncalcs):
            calc = self.calcs[i]
            
            if logger.level < 30:
                logger.info("Integrator:eval: operating on calc %d, %s"%(
                        i, calc.id))

            calc.sph(*calling_sequence[i])

            updates = calc.updates
            nupdates = calc.nupdates

            #get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]

            for j in range(nupdates):
                update_prop = updates[j]
                step_prop = calling_sequence[i][j]
                              
                step_array = pa.get(step_prop)
                if not calc.integrates:

                    #set the evaluated property
                    if logger.level < 30:
                        logger.info("""Integrator:eval: setting the prop 
                                    %s for calc %d, %s"""
                                    %(update_prop,  i, calc.id))

                    pa.set(**{update_prop:step_array.copy()})

                    #ensure that all processes have reached this point

                    particles.barrier()

                    #update the remote particle properties

                    self.rupdate_list[calc.dnum] = [update_prop]

                    if logger.level < 30:
                        logger.info("""Integrator:eval: updating remote particle
                                     properties %s"""%(self.rupdate_list))
 
                    #particles.update_remote_particle_properties([update_prop])
                    particles.update_remote_particle_properties(
                        self.rupdate_list)
                    
                else:
                    k_name = 'k'+str(self.step)+'_'+update_prop+str(i)+str(j)
                    pa.set(**{k_name:step_array.copy()})

                pass

        #ensure that the eval phase is completed for all processes
        particles.barrier()

    def do_step(self, dt):
        """ Perform one step for the integration
        
        This is an intermediate step in a multi step integrator wherein 
        the step arrays are set in the `k` list. First, each eval is 
        called and the step arrays are stored in the `k` list and then
        for each integrating calc, the current state of the particles is
        advanced with respect to the initial position and the `k` value 
        from a previous step.


        """

        if logger.level < 30:
            logger.info("Integrator:do_step")

        calling_sequence = self.calling_sequence
        ncalcs = len(self.calcs)
        particles = self.particles
        
        #call each of the eval functions in order

        for i in range(ncalcs):
            calc = self.calcs[i]
            calc.sph(*calling_sequence[i])

            updates = calc.updates
            nupdates = calc.nupdates

            if logger.level < 30:
                logger.info("Integrator:do_step: operating on calc %d %s"
                            %(i, calc.id))

            #get the destination particle array for this calc
            
            pa = self.arrays[calc.dnum]
            
            for j in range(nupdates):
                update_prop = updates[j]
                step_prop = calling_sequence[i][j]

                step_array = pa.get(step_prop)
                
                if not calc.integrates:
                    #set the evaluated property

                    if logger.level < 30:

                        logger.info("""Integrator:eval: setting the 
                                    prop %s for calc %d, %s"""
                                    %(update_prop,  i, calc.id))

                    pa.set(**{update_prop:step_array})

                    #ensure that all processes have reached this point

                    particles.barrier()

                    #update the remote particle properties
 
                    if logger.level < 30:
                        logger.info("""Integrator:eval: updating remote particle
                                   properties %s"""%(self.rupdate_list))

                    self.rupdate_list[calc.dnum] = [update_prop]
                    particles.update_remote_particle_properties(
                        self.rupdate_list)
                else:
                    k_name = 'k'+str(self.step)+'_'+update_prop+str(i)+str(j)
                    pa.set(**{k_name:step_array.copy()})

                pass
            pass

        #ensure that the eval phase is completed for all processes

        particles.barrier()

        #Reset the current arrays (Required for multi step integrators)

        self.reset_current_arrays()
    
        #step the variables

        for i in range(ncalcs):
            calc = self.calcs[i]
            if calc.integrates:
                
                updates = calc.updates
                nupdates = calc.nupdates
                
                if logger.level < 30:
                    logger.info("""Steping phase for calc %d, %s"""%(i,calc.id))

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]

                    current_arr = pa.get(update_prop)

                    k_name = 'k'+str(self.step)+'_'+update_prop+str(i)+str(j)
                    step_array = pa.get(k_name)

                    if logger.level < 30:
                        logger.info("""Integrator:do_step: Updating the k array
                                    %s """%(k_name))

                    updated_array = current_arr + step_array*dt
                    pa.set(**{update_prop:updated_array})

                pass
            pass

        self.step += 1

        #ensure that all processors have stepped the local particles
        
        particles.barrier()

    def _setup_integrator(self):
        """ Setup the information required for the stepping
        
        Notes:
        ------
        A call to this function must be done before any integration begins.
        This function sets up the properties required for storing the results
        of the calc operation as well as setting the calling_sequence variable.

        Algorithm:
        ----------
        intialize calling_sequence to []
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

        calcs = []
        calcs.extend(self.calcs)
        calcs.extend(self.pcalcs)

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

    def _set_initial_arrays(self):
        """ Set the initial arrays for each calc

        The initial array is the update property of a calc appended with _0
        Note that multiple calcs can update the same property and this 
        will not replicate the creation of the initial arrays. 
        
        """        
        if logger.level < 30:
            logger.info("Integrator: Set initial arrays called")
            
        calcs = []
        calcs.extend(self.calcs)
        calcs.extend(self.pcalcs)

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

    def _reset_current_arrays(self, calcs):
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

    def _do_step(self, calcs, dt):
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
        
        # Evaluate the calcs

        self._eval(calcs)

        # ensure that the eval phase is completed for all processes

        particles.barrier()

        # Reset the current arrays (Required for multi step integrators)

        self._reset_current_arrays(calcs)
    
        # step the variables

        self._step(calcs, dt)

        # ensure that all processors have stepped the local particles

        particles.barrier()

    def _eval(self, calcs):
        """ Evaluate each calc and store in the k list if necessary """

        if logger.level < 30:
            logger.info("Integrator:eval")

        ncalcs = len(calcs)
        particles = self.particles
        
        k_num = 'k' + str(self.step)
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

    def _step(self, calcs, dt):
        """ Perform stepping for the integrating calcs """

        particles = self.particles
        ncalcs = len(calcs)

        k_num = 'k' + str(self.step)
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

                        logger.info("""Integrator:do_step: Updating the k array
                                    %s """%(k_name))

                    updated_array = current_arr + step_array*dt
                  
                    pa.set(**{update_prop:updated_array})

                pass
            pass

        # Increment the step by 1

        self.step += 1

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

        k_num = 'k' + str(self.step)
        for i in range(nupdates):
            update_prop = updates[i]
            
            initial_prop = self.initial_props[calc.id][i]
            k_prop = self.k_props[calc.id][k_num][i]

            initial_array = pa.get(initial_prop)
            k1_arr = pa.get(k_prop)
            
            updated_array = initial_array + k1_arr * dt
            
            pa.set(**{update_prop:updated_array})
            pa.set(**{initial_prop:updated_array})

    def _integrate(self, dt):

        # set the intitial arrays for all calcs
        
        self._set_initial_arrays()
        
        # evaluate the k1 arrays for non position calcs

        self._eval(self.calcs)
        
        # perform the final step for each non position calc
        
        for calc in self.calcs:
            if calc.integrates:
                self.final_step(calc, dt)

        # Now step the position calcs

        self._eval(self.pcalcs)

        for calc in self.pcalcs:
            self.final_step(calc, dt)

        #update the particles to get the new neighbors

        self.particles.update()

    def integrate(self, dt):
        
        #set the intitial arrays
        
        self.set_initial_arrays()
        
        #evaluate the k1 arrays

        self.eval()
        
        #step the update arrays for each integrating calc

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            if calc.integrates:
                
                updates = calc.updates
                nupdates = calc.nupdates

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]

                    initial_prop = update_prop+'_0'
                    initial_arr = pa.get(initial_prop)

                    k_name = 'k1_' + update_prop+str(i)+str(j)
                    step_array = pa.get(k_name)
                    
                    updated_array = initial_arr + step_array*dt
                    pa.set(**{update_prop:updated_array})
                    pa.set(**{initial_prop:updated_array})
                pass
            pass

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

    def integrate(self, dt):
        
        #set the initial arrays
        self.set_initial_arrays()

        while self.step != self.nsteps:
            self.do_step(dt)
            self.particles.update()
            
        self.eval()

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
                
            if calc.integrates:
                updates = calc.updates
                nupdates = calc.nupdates

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                    
                    initial_prop = update_prop+'_0'
                    initial_arr = pa.get(initial_prop)
                        
                    k1 = 'k1_'+update_prop+str(i)+str(j)
                    k2 = 'k2_'+update_prop+str(i)+str(j)
                    
                    k1_arr = pa.get(k1)
                    k2_arr = pa.get(k2)

                    updated_array = initial_arr + 0.5*dt*(k1_arr + k2_arr)
                    
                    pa.set(**{update_prop:updated_array})
                    pa.set(**{initial_prop:updated_array})

        self.step = 1            
        self.particles.update()        
############################################################################## 

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

    def integrate(self, dt):

        #set the initial arrays
        self.set_initial_arrays()

        while self.step != self.nsteps:

            if self.step == 1:
                self.do_step(0.5*dt)
                self.particles.update()
                
            elif self.step == 2:
                self.do_step(0.5*dt)
                self.particles.update()
                
            elif self.step == 3:
                self.do_step(dt)
                self.particles.update()
            
        self.eval()

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            
            if calc.integrates:
                updates = calc.updates
                nupdates = calc.nupdates

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                        
                    initial_prop = update_prop+'_0'
                    initial_arr = pa.get(initial_prop)
                        
                    k1 = 'k1_'+update_prop+str(i)+str(j)
                    k2 = 'k2_'+update_prop+str(i)+str(j)
                    k3 = 'k3_'+update_prop+str(i)+str(j)
                    k4 = 'k4_'+update_prop+str(i)+str(j)
                    
                    k1_arr = pa.get(k1)
                    k2_arr = pa.get(k2)
                    k3_arr = pa.get(k3)
                    k4_arr = pa.get(k4)

                    updated_array = initial_arr + (dt/6.0) *\
                        (k1_arr + 2*k2_arr + 2*k3_arr + k4_arr)
                    
                    pa.set(**{update_prop:updated_array})
                    pa.set(**{initial_prop:updated_array})

        self.step = 1
        self.particles.update()
        
############################################################################## 


##############################################################################
#`RK4Integrator` class 
##############################################################################
class PredictorCorrectorIntegrator(Integrator):
    """ RK4 Integration of a system X' = F(X) using the scheme
    
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

    def integrate(self, dt):

        #set the initial arrays

        self.set_initial_arrays()

        #predict
            
        self.do_step(0.5*dt)

        #correct
        
        self.step = 1
        self.do_step(0.5*dt)

        #step

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            
            if calc.integrates:
                updates = calc.updates
                nupdates = calc.nupdates

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                        
                    initial_prop = update_prop+'_0'
                    initial_arr = pa.get(initial_prop)
                        
                    k1 = 'k1_'+update_prop+str(i)+str(j)
                    
                    k1_arr = pa.get(k1)

                    current_arr = pa.get(update_prop)
                    
                    updated_array = 2*current_arr - initial_arr

                    pa.set(**{update_prop:updated_array})
                    pa.set(**{initial_prop:updated_array})

        self.step = 1
        self.particles.update()
        
############################################################################## 


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
    
    rho = rhobar + 0.5*h*(F - F_0)

    """

    def __init__(self, particles, calcs):
        Integrator.__init__(self, particles, calcs)
        self.nsteps = 2

    def add_correction_for_position(self, dt):
        ncalcs = len(self.calcs)

        vel_updates = {1:['u'], 2:['u', 'v'], 3:['u','v','w']}
        
        for i in range(ncalcs):
            calc = self.calcs[i]

            if calc.tag == "position":
                
                pa = self.particles.arrays[calc.dnum]
                
                updates = calc.updates
                nupdates = len(updates)

                for j in range(nupdates):
                    
                    vupdate_prop = vel_updates[nupdates][j]
                    
                    update_prop = updates[j]

                    #the current position

                    current_arr = pa.get(update_prop)

                    k_name = 'k'+str(self.step)+'_'+vupdate_prop+str(i)+str(j)
                    step_array = pa.get(k_name)

                    updated_array = current_arr + 0.5*dt*dt*step_array
                    pa.set(**{update_prop:updated_array})            

    def integrate(self, dt):

        #set the current arrays
        self.set_initial_arrays()
        self.reset_current_arrays()

        #evaluate the system at the current state

        self.eval()

        #perform the prediction step. The next eval will store in k2 arrays

        self.Step(dt)
        
        #add the prediction step for the position
        self.add_correction_for_position(dt)

        self.particles.barrier()
        self.particles.update()

        self.eval()

        #reset the step to 1

        self.step = 1

        #step but not for the position functions

        ncalcs = len(self.calcs)
        for i in range(ncalcs):
            calc = self.calcs[i]
            
            if calc.integrates and not calc.tag == 'position':
                updates = calc.updates
                nupdates = calc.nupdates

                #get the destination particle array for this calc
            
                pa = self.arrays[calc.dnum]

                for j in range(nupdates):
                    update_prop = updates[j]
                        
                    k1 = 'k1_'+update_prop+str(i)+str(j)                    
                    k1_arr = pa.get(k1)

                    k2 = 'k2_'+update_prop+str(i)+str(j)                    
                    k2_arr = pa.get(k2)

                    current_arr = pa.get(update_prop)
                    
                    updated_array = current_arr + 0.5*dt*(k2_arr - k1_arr)

                    pa.set(**{update_prop:updated_array})

        self.step = 1
        self.particles.update()
        
############################################################################## 
