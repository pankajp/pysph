"""
Module containing various classes to represent one step in a solver.
"""

cdef class ODEStepper:
    """
    Class to advance a set of ODE's by one step.
    
    The ODE takes the following form:
    .. math::
        
        \frac{dA}{dt} = B(t)

    To step this ODE one step ahead in time, we integrate B(t) wrt 't' to get
    A(t+1).
    .. math::
    
        A(n+1) = A(n) + \delta t * B(n)

    """
    def __init__(self, list integrands=[], integrals=[], str _step='dt'):
        """
        Constructor.
        """
        pass

    cdef double time_step(self):
        """
        Function that does the stepping.
        """
        pass


cdef class ODEInfo:
    cpdef add_pre_step_component(self, str name, SolverComponent c, int pos=0):
        raise NotImplementedError
    
    cpdef add_post_step_component(self, str name, SolverComponent c, int pos=0):
        raise NotImplementedError
    

    # to copy components from another ODEInfo.
    cpdef copy_components(self, ODEInfo odeinfo):
        raise NotImplementedError
    

cdef class IntegrationStep:
    cdef void step(self):
        raise NotImplementedError
    
    cdef void add_ode(self, ODEInfo ode_info, int pos=0):
        raise NotImplementedError
    
    cdef void add_pre_integration_component(self, SolverComponent c, int pos=0):
        raise NotImplementedError
    
