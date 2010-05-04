"""
Contains various components to compute pressure gradients.
"""


################################################################################
# `SPHPressureGradientComponent` class.
################################################################################
cdef class SPHPressureGradientComponent(SPHComponent):
    """
    Computes the pressure gradient using the SPHSymmetricPressureGradient3D
    function.
    """
    def __init__(self, str name='',
                  list entity_list=[],
                  MultidimensionalKernel kernel=None,
                  int _mode=SPHSourceDestMode.byType,
                  *args, **kwargs):
        """
        Constructor.
        """

        self.src_types = [EntityTypes.Entity_Fluid]
        self.dst_types = [EntityTypes.Entity_Fluid]
        
        self.sph_func = SPHPressureGradient

        self.reads = ['p']
        self.updates = ['u','v','w']

        SPHComponent.__init__(self, name, entity_list, kernel, _mode)
        
    cdef int compute(self) except -1:
        """
        """
        cdef int num_dest
        cdef int i
        
        cdef Fluid fluid
        cdef SPHCalc calc
        cdef ParticleArray parr
        
        # call setup component.
        self.setup_component()

        num_dest = len(self.dsts)
        
        reads = self.reads
        writes = self.writes
        updates = self.updates
        
        for i from 0 <= i < num_dest:
            fluid = self.dsts[i]
            parr = fluid.get_particle_array()
            calc = self.calcs[i]

            calc.sph(writes, False)
            
            parr.tmpx *= -1.0
            parr.tmpy *= -1.0
            parr.tmpz *= -1.0

        return 0
##########################################################################

cdef class SPHSummationDensityComponent(SPHComponent):

    def __init__(self, str name='SummationDensity', list entity_list=[],
                 MultidimensionalKernel kernel=None, 
                 int _mode=SPHSourceDestMode.byType, *args, **kwargs):
        
        self.src_types = [EntityTypes.Entity_Fluid]
        self.dst_types = [EntityTypes.Entity_Fluid]

        self.reads =['m']
        self.updates = ['rho']
        
        self.sph_func = SPHRho
        
        SPHComponent.__init__(self, name, entity_list, kernel, _mode)
        
    cdef int compute(self) except -1:
        """
        """
        cdef int ndst
        cdef int i
        
        cdef Fluid fluid
        cdef SPHCalc calc
        cdef ParticleArray parr
        
        # call setup component.
        self.setup_component()

        ndst = len(self.dsts)
        
        writes = self.writes
        rho = self.updates[0]

        for i from 0 <= i < ndst:

            fluid = self.dsts[i]
            parr = fluid.get_particle_array()
            calc = self.calcs[i]

            #Evaluate 
            calc.sph(writes, False)
            
            #Update
            parr.set(rho = parr.tmpx)

        return 0
##########################################################################
