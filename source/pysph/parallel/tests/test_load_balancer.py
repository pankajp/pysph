""" Tests for the load balancer """

import unittest

from pysph.base.kernels import CubicSplineKernel
from pysph.base.cell import CellManager

from pysph.solver.basic_generators import *

from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.load_balancer_mkmeans import LoadBalancer

def dict_from_kwargs(**kwargs):
    return kwargs

class TestSerialLoadBalancer1D(unittest.TestCase):
    
    def setUp(self):
        lg = LineGenerator(kernel=CubicSplineKernel(1))
        lg.end_point.x = 1.0
        lg.end_point.z = 0.0
        self.pas = [lg.get_particles()]
        self.pas[0].x += 0.1
        self.cell_size = 0.1
        self.dim = 1
        
    def create_solver(self):
        self.cm = cm = ParallelCellManager(self.pas, self.cell_size, self.cell_size)
        #print 'num_cells:', len(cm.cells_dict)
        cm.load_balancing = False # balancing will be done manually
        cm.dimension = self.dim
        
        self.lb = lb = self.cm.load_balancer = LoadBalancer(parallel_solver=None, parallel_cell_manager=self.cm)
        lb.skip_iteration = 1
        lb.threshold_ratio = 10.
        lb.lb_max_iteration = 10
        lb.setup()
    
    def load_balance(self):
        np0 = 0
        nc0 = len(self.cm.cell_dict)
        for cid, cell in self.cm.cell_dict.items():
            np0 += cell.get_number_of_particles()
        lb = self.cm.load_balancer
        for lbargs in self.get_lb_args():
            lb.load_balance(**lbargs)
            self.cm.exchange_neighbor_particles()


def get_lb_args():
    return [
            # This test is only for serial cases
            #dict_from_kwargs(method='normal'),
            dict_from_kwargs(),
            dict_from_kwargs(distr_func='auto'),
            dict_from_kwargs(distr_func='geometric'),
            dict_from_kwargs(distr_func='mkmeans', c=0.3, t=0.2, tr=0.8, u=0.4, e=3, er=6, r=2.0),
            dict_from_kwargs(distr_func='sfc', sfc_func='morton', start_origin=False),
            dict_from_kwargs(distr_func='sfc', sfc_func='morton', start_origin=True),
            #dict_from_kwargs(distr_func='sfc', sfc_func='hilbert', start_origin=False),
            #dict_from_kwargs(distr_func='sfc', sfc_func='hilbert', start_origin=True),
           ]

# function names have 't' instead of 'test' otherwise nose test collector
# assumes them to be test functions
def create_t_func1(lbargs, num_procs):
    """ create and return test functions for load balancing """ 
    def test(self):
        self.create_solver()
        proc_pas = LoadBalancer.distribute_particle_arrays(self.pas, num_procs, self.cell_size, 100, **lbargs)
        nps = [sum([pa.get_number_of_particles() for pa in pas]) for pas in proc_pas]
        self.assertTrue(sum(nps) == sum([pa.get_number_of_particles() for pa in self.pas]))
        for pa in pas:
            self.assertEqual(len(pa.get('x')),pa.get_number_of_particles())
            self.assertEqual(len(pa.get('y')),pa.get_number_of_particles())
        # each proc should have at least one cell since num_cells>num_procs
        for np in nps:
            assert np > 0
    
    test.__name__ = 'test_distribute_particle_arrays_p%d'%(num_procs)
    test.__doc__ = 'distribute_particle_arrays; procs=%d; lbargs=' + str(lbargs)
    
    return test

def gen_ts():
    """ generate test functions and attach them to TestSerialLoadBalancer1D """
    for i, lbargs in enumerate(get_lb_args()):
        for num_procs in [1,5,11]:
            t_method = create_t_func1(lbargs, num_procs)
            t_method.__name__ = t_method.__name__ + '_%d'%(i)
            setattr(TestSerialLoadBalancer1D, t_method.__name__, t_method)

# generate the test functions
gen_ts()


class TestSerialLoadBalancer2D(TestSerialLoadBalancer1D):
    
    def setUp(self):
        lg = RectangleGenerator(kernel=CubicSplineKernel(2))
        self.pas = [lg.get_particles()]
        self.pas[0].x += 0.1
        self.pas[0].y += 0.2
        self.cell_size = 0.1
        self.dim = 2

class TestSerialLoadBalancer3D(TestSerialLoadBalancer1D):
    
    def setUp(self):
        lg = CuboidGenerator(kernel=CubicSplineKernel(3))
        self.pas = [lg.get_particles()]
        # to shift the origin
        self.pas[0].x += 0.1
        self.pas[0].y += 0.2
        self.pas[0].z += 0.3
        self.cell_size = 0.1
        self.dim = 3

    def test_distribute_particles1(self):
        """ test distribute_particles for a single particle_array """
        pa = self.pas[0]
        np = pa.get_number_of_particles()
        pa1, pa2 = LoadBalancer.distribute_particles(pa, 2, -1)
        self.assertEqual(pa1.get_number_of_particles() +
                         pa2.get_number_of_particles(), np)

    def test_distribute_particles2(self):
        """ test distribute_particles with list of particle_array """
        np = self.pas[0].get_number_of_particles()
        pas = LoadBalancer.distribute_particles(self.pas, 2, -1)
        pa1, pa2 = pas[0][0], pas[1][0]
        self.assertEqual(pa1.get_number_of_particles() +
                         pa2.get_number_of_particles(), np)

        pas = LoadBalancer.distribute_particles([pa1, pa2], 2, -1)
        np2 = 0
        for pa in pas[0]+pas[1]:
            np2 += pa.get_number_of_particles()
        self.assertEqual(np2, np)


if __name__ == "__main__":
    unittest.main()
