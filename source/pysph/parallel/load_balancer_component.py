"""
Component to perform load balancing in the solver.

This is a wrapper around the LoadBalancer to incorporate it into the solver
framework. 
"""

# logging imports
import logging
logger = logging.getLogger()


# local imports
from pysph.solver.solver_base import UserDefinedComponent
from pysph.parallel.load_balancer import LoadBalancer

class LoadBalancerComponent(UserDefinedComponent):
    """
    """
    def __init__(self, 
                 name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 num_iterations=1, *args, **kwargs):
        """
        """
        UserDefinedComponent.__init__(self, name=name,
                                      solver=solver,
                                      component_manager=component_manager,
                                      entity_list=entity_list)
        self.num_iterations = num_iterations
        self.cell_manager = None
        self.load_balancer = None
        
    def setup_component(self):
        """
        Setup the component here.
        """
        if self.setup_done == True:
            return
    
        if self.solver is None:
            msg = 'Solver object is None'
            logger.error(msg)
            raise SystemError, msg
        
        self.cell_manager = self.solver.cell_manager
        self.load_balancer = self.cell_manager.load_balancer
        self.load_balancer.lb_max_iterations = self.num_iterations

        self.load_balancer.setup()

        rank = self.load_balancer.parallel_controller.rank
        file_name = 'load_balance_' + str(rank)
        f = open(file_name, 'w')
        f.close()
        
        self.setup_done = True

        return 0

    def update_property_requirements(self):
        """
        """
        return 0

    def py_compute(self):
        """
        Perform the load balancing.
        """
        self.setup_component()
                
        rank = self.load_balancer.parallel_controller.rank
        file_name = 'load_balance_' + str(rank)
        import time
        t1 = time.time()

        self.load_balancer.load_balance()
        self.cell_manager.root_cell.exchange_neighbor_particles()

        t2 = time.time()
        
        diff = t2-t1

        f = open(file_name, 'a')
        f.write(str (self.solver.current_iteration) + ' ' + str(diff) + '\n')

        f.close()
        
        return 0
