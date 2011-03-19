""" A wrapper around the sph calc and function to make it simpler """

import pysph.sph.api as sph

import pysph.base.api as base
Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

class SPHOperation(object):
    """ A base class that represents a general SPH operation.

    An operation is defined as any SPH equation appearing in the system
    to be solved. These may be a simple assignment of a certain property
    as in the equation of state, or a summation ODE as for the momentum 
    equation.

    This is intended as a wrapper around the calc and it's various nuances.
    The concrete subclasses for the SPHOperation are

    (i) SPHSimpleODE -- Integrating calc for an equation of the type::
                        \frac{D prop}{Dt} = \vec{F}
    

    (ii) SPHSummationODE -- Integrating calc for an equation of the type::
                           \frac{D prop}{Dt} = \sum_{b=1}^{N} ...


    (iii) SPHSummation -- Nonintegrating calc for an equation of the type::
                          \rho_a = \sum m_b W_{ab}

    (iv) SPHAssignment -- Non integrating calc for an equation of the type::
                          p_a = (gamma-1)*rho*e


    Data Members:
    --------------
    function -- The function (:class:`sph_func.Function`) to use for evaluating
            the RHS in an SPH operation.
    from_types -- The accepted neighbor tags
    on_types -- The accepted self tags
    updates -- The property the operation updates
    id -- A unique id for this operation


    Member Functions:
    ------------------
    get_calc -- Return an appropriate calc for the kind of operation requested.

    """
    
    def __init__(self, function, on_types, updates, id, from_types=[],
                 kernel_correction=-1):

        self.from_types = from_types
        self.on_types = on_types
        self.function = function
        self.updates = updates        
        self.id = id

        self.has_kernel_correction = False
        self.kernel_correction = kernel_correction

        if kernel_correction != -1:
            self.has_kernel_correction = True

    def get_calc_data(self, particles):
        """ Get the data for setting up the calcs """ 
        
        arrays = particles.arrays
        narrays = len(arrays)
        calc_data = {}

        for i in range(narrays):

            #get the destination array
            dst = arrays[i]

            #create an entry in the dict if this is a destination array

            if dst.particle_type in self.on_types:
                calc_data[i] = {'sources':[], 'funcs':[], 'dnum':i, 'id':"",
                                'snum':str(i)}

                #from types can be zero for a no-neighbor calc

                if self.from_types == []:
                    func = self.function.get_func(dst, dst)
                    func.id = self.id

                    calc_data[i]['funcs'] = [func]
                    calc_data[i]['id'] = self.id + '_' + dst.name

                else:
                    #check for the sources for this destination
                    for j in range(narrays):
                        
                    #get the potential source array
                        src = arrays[j]

                    #check if this is a valid source for the destination array
                 
                        if src.particle_type in self.from_types:

                        #get the function for this pair and set the id

                            func = self.function.get_func(source=src, dest=dst)
                            func.id = self.id

                        #make an entry in the dict for this destination

                            calc_data[i]['sources'].append(src)
                            calc_data[i]['funcs'].append(func)
                            calc_data[i]['snum'] = calc_data[i]['snum']+str(j)
                            calc_data[i]['id'] = self.id + '_' + dst.name
                     
        return calc_data

    def get_calcs(self):
        raise NotImplementedError, 'SPHOperation::get_calc'
        
class SPHSimpleODE(SPHOperation):
    """ Return a calc (via get_calc) for an SPH equation of the form
    
    \frac{D prop}{Dt} = \vec{F}
    
    where prop is the property being updated and F is the forcing term which
    may be a scalar or a vector.

    Example:
    --------
    Gravity force in free surface simulations require an evaluation like

    \frac{D\vec{v}}{Dt} = \vec{g}

    Notes:
    ------
    No neighbor information is not needed for evaluation for a particular 
    particle.
    Returns an integrating calc.

    """    
    def get_calcs(self, particles, kernel):
        calcs = []
        calc_data = self.get_calc_data(particles)
        
        arrays = particles.arrays
        narrays = len(arrays)

        for i in range(narrays):
            if calc_data.has_key(i):
                dest = arrays[i]
                srcs = calc_data[i]['sources']
                funcs = calc_data[i]['funcs']
                dnum = calc_data[i]['dnum']
                snum = calc_data[i]['snum']
                id = calc_data[i]['id']
                
                calc = sph.SPHCalc(
                    particles=particles, sources=srcs, dest=dest, 
                    kernel=kernel, funcs=funcs, updates=self.updates,
                    integrates=True, dnum=dnum, nbr_info=False, id=id,
                    kernel_correction=self.kernel_correction,
                    dim=kernel.dim, snum=snum)

                calcs.append(calc)

        return calcs

class SPHSummationODE(SPHOperation):
    """  Return a calc (via get_calc) for an SPH equation of the form

    \frac{D prop}{Dt} = \sum_{b=1}^{N} ...
    
    Examples:
    ---------
    Momentum equation, Artificial viscosity term etc

    Notes:
    ------
    Neighbor information is needed for the evaluation for a particular particle.
    This returns an integrating calc.

    """    
    def get_calcs(self, particles, kernel):

        calcs = []
        calc_data = self.get_calc_data(particles)
        
        arrays = particles.arrays
        narrays = len(arrays)

        for i in range(narrays):
            if calc_data.has_key(i):
                dest = arrays[i]
                srcs = calc_data[i]['sources']
                funcs = calc_data[i]['funcs']
                dnum = calc_data[i]['dnum']
                snum = calc_data[i]['snum']
                id = calc_data[i]['id']
                
                calc = sph.SPHCalc(
                    particles=particles, sources=srcs,
                    dest=dest, kernel=kernel, funcs=funcs,
                    updates=self.updates, integrates=True,
                    dnum=dnum, nbr_info=True, id=id,
                    kernel_correction=self.kernel_correction,
                    dim=kernel.dim, snum=snum)
                
                calcs.append(calc)

        return calcs

class SPHSummation(SPHOperation):
    """  Return a calc (via get_calc) for an SPH equation of the form

    \rho_a = \sum m_b W_{ab}
    
    Examples:
    ---------
    Summation density

    Notes:
    ------
    Neighbor information is needed for the evaluation for a particular particle.
    This is a non integrating calc.

    """    
    def get_calcs(self, particles, kernel):
        calcs = []
        calc_data = self.get_calc_data(particles)

        arrays = particles.arrays
        narrays = len(arrays)

        for i in range(narrays):
            if calc_data.has_key(i):
                dest = arrays[i]
                srcs = calc_data[i]['sources']
                funcs = calc_data[i]['funcs']
                dnum = calc_data[i]['dnum']
                snum = calc_data[i]['snum']
                id = calc_data[i]['id']
                
                calc = sph.SPHCalc(
                    particles=particles, sources=srcs,
                    dest=dest, kernel=kernel, 
                    funcs=funcs, updates=self.updates, 
                    integrates=False,
                    dnum=dnum, nbr_info=True, id=id,
                    kernel_correction=self.kernel_correction,
                    dim=kernel.dim, snum=snum)

                calcs.append(calc)

        return calcs

class SPHAssignment(SPHOperation):
    """  Return a calc (via get_calc) for an SPH equation of the form

    p_a = (gamma-1)*rho*e
    
    Examples:
    ---------
    Equation of state

    Notes:
    ------
    Neighbor information is not needed for evaluation.
    This is a non integrating calc.

    """ 
    def get_calcs(self, particles, kernel):
        calcs = []
        calc_data = self.get_calc_data(particles)
        
        arrays = particles.arrays
        narrays = len(arrays)

        for i in range(narrays):
            if calc_data.has_key(i):
                dest = arrays[i]
                srcs = calc_data[i]['sources']
                funcs = calc_data[i]['funcs']
                dnum = calc_data[i]['dnum']
                snum = calc_data[i]['snum']
                id = calc_data[i]['id']
                
                calc = sph.SPHCalc(
                    particles=particles, sources=srcs,
                    dest=dest, kernel=kernel, 
                    funcs=funcs, updates=self.updates, 
                    integrates=False, dnum=dnum,
                    nbr_info=False, id=id,
                    kernel_correction=self.kernel_correction,
                    dim=kernel.dim, snum=snum)

                calcs.append(calc)

        return calcs

#############################################################################
