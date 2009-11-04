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

    cdef void step(self):
        """
        Function that does the stepping.
        """
        pass
