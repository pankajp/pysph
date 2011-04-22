""" A wrapper around the sph calc and function to make it simpler """

from pysph.sph.sph_calc import SPHCalc, CLCalc

import pysph.base.api as base
Fluid = base.ParticleType.Fluid
Solid = base.ParticleType.Solid

class SPHOperation(object):
    """ This class that represents a general SPH operation

    An operation is defined as any equation appearing in the system to
    be solved. This may be an assignment of a certain property or the
    evaluation of a certain forcing function.

    Examples of operations are:

    .. math::
             \rho = \sum_{j=1}^{N} m_j\,W_{ij}

             p = (\gamma - 1.0)\rho e

    Each equation necessarily defines a *destination* and posiibly a
    *source* particle on which it operates. This is usually problem
    specific and the SPHCalc objects which compute the necessary
    interactions is constructed from this information given.    

    Data Members:
    --------------

    function -- The function (:class:`sph_func.Function`) to use for evaluating
                the RHS in an SPH operation. One function is created for
                each source-destination pair.

    from_types -- The influencing type of particles. All particle arrays
                  matching these types are sources.

    on_types -- The influenced type of particles. All particle arrays matching
                these types are destinations.

    updates -- A list of strings indicating the destination particle
               properties updated by the resulting SPHCalc object.

    id -- A unique id for this operation
    
    intgrates -- Bool indicating if the RHS evaluated by the Operation
                 is an acceleration.

    Member Functions:
    ------------------
    get_calc -- Return an appropriate calc for the kind of operation requested

    """
    
    def __init__(self, function, on_types, updates, id, kernel=None,
                 from_types=[], kernel_correction=-1, integrates=False):
        """ Constructor

        Parameters:
        -----------

        function -- The SPHFunction for this operation.

        on_types -- valid destination (influenced) particle array types.

        from_types -- valid source (influencing) particle array types

        kernel -- the kernel to use for this operation.

        kernel_correction -- Type of kernel correction to use. (-1) means
                             no correction.

        integrates -- Flag indicating if the function evaluates an
                      acceleration.

        """

        self.from_types = from_types
        self.on_types = on_types
        self.function = function
        self.updates = updates        
        self.id = id
        self.integrates = integrates

        self.kernel = kernel

        self.has_kernel_correction = False
        self.kernel_correction = kernel_correction

        if kernel_correction != -1:
            self.has_kernel_correction = True

        self.calc_type = SPHCalc

    def get_calc_data(self, particles):
        """ Group particle arrays as src-dst pairs with appropriate
        functions as required by SPHCalc

        Return a dictionary of calc properties keyed on particle array
        id in particles which corresponds to the destination particle
        array for that calc.

        The calc properties are:

        sources -- a list of particle arrays considered as sources for this
                   calc

        funcs -- a list of num_sources functions of the same type to operate
                 between destination-source pair.

        dnum -- the destination particle array id in particles for this calc.

        id -- a unique identifier for the calc
        
        """ 
        
        #if issubclass(self.function.get_func_class(), sph.SPHFunctionParticle):
        #    # all nbr requiring funcs are subclasses of SPHFunctionParticle
        #    self.nbr_info = True
        #else:
        #    self.nbr_info = False
        
        arrays = particles.arrays
        narrays = len(arrays)

        calc_data = {}

        for i in range(narrays):

            dst = arrays[i]

            # create an entry in the dict if this is a valid destination array

            if dst.particle_type in self.on_types:
                calc_data[i] = {'sources':[], 'funcs':[], 'dnum':i, 'id':"",
                                'snum':str(i)}

                # if from_types == [], no neighbor info is required!

                if self.from_types == []:
                    func = self.function.get_func(dst, dst)
                    func.id = self.id

                    calc_data[i]['funcs'] = [func]
                    calc_data[i]['id'] = self.id + '_' + dst.name

                else:
                    # check for the sources for this destination
                    for j in range(narrays):
                        
                        src = arrays[j]

                        # check if this is a valid source array
                 
                        if src.particle_type in self.from_types:

                            # get the function with the src dst pair

                            func = self.function.get_func(source=src, dest=dst)
                            func.id = self.id

                            # make an entry in the dict for this destination

                            calc_data[i]['sources'].append(src)
                            calc_data[i]['funcs'].append(func)
                            calc_data[i]['snum'] = calc_data[i]['snum']+str(j)
                            calc_data[i]['id'] = self.id + '_' + dst.name

                    if calc_data[i]['sources'] == []:
                        msg = "No source found for %s operation"%(self.id)
                        raise RuntimeWarning, msg
                     
        return calc_data

    def get_calcs(self, particles, kernel):
        """ Return a list of calcs for the operation.
        
        An SPHCalc is created for each destination particle array for
        the operation. The calc may have a list of sources with one
        function for each  src-dst pair.

        Parameters:
        ------------

        particles -- the collection of particle arrays to consider

        kernel -- the smoothing kernel to use for the operation

        """

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
                
                calc = self.calc_type(
                    
                    particles=particles, sources=srcs, dest=dest,
                    funcs=funcs, kernel=kernel, updates=self.updates,
                    integrates=self.integrates, dnum=dnum, id=id,
                    dim=kernel.dim, snum=snum,
                    kernel_correction=self.kernel_correction, nbr_info=True,

                    )

                calcs.append(calc)

        return calcs
        
class SPHIntegration(SPHOperation):
    """ Return an integrating calc (via get_calc()) for an SPH equation of form
    
        \frac{D prop}{Dt} = \vec{F}
    
    where `prop` is the property being updated and `F` is the forcing
    term which may be a scalar or a vector.

    Example:
    --------

    .. math::
             \frac{D\vec{v}}{Dt} = \vec{g}

             \frac{Dv}{Dt} = \sum_{j=1}^{N} m_j \frac{v_j}{\rho_j}\nablaW_{ij}
    
    Note:
    -----
    This class is just a convenience (and an alias) over creating an integrating
    :class:`SPHOperation` with integrates=True
    
    """    
    def __init__(self, function, on_types, updates, id, kernel=None,
                 from_types=[], kernel_correction=-1):

        SPHOperation.__init__(self, function, on_types, updates, id, kernel,
                              from_types=from_types, integrates=True,
                              kernel_correction=kernel_correction,)
        
