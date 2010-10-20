""" Tests for the load balancer """

import unittest

from pysph.base.kernels import CubicSplineKernel
from pysph.base.cell import CellManager

from pysph.solver.basic_generators import *

from pysph.parallel.parallel_cell import ParallelCellManager
from pysph.parallel.load_balancer_mkmeans import LoadBalancer
from pysph.parallel.dummy_solver import DummySolver
from pysph.parallel.parallel_component import ParallelComponent

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
        pc = ParallelComponent(name='parallel_component', solver=None)
        self.cm = cm = ParallelCellManager(self.pas, self.cell_size, self.cell_size)
        #print 'num_cells:', len(cm.cells_dict)
        cm.load_balancing = False # balancing will be done manually
        cm.dimension = self.dim
        
        # create a dummy solver - serves no purpose but for a place holder
        pc.solver = cm.solver = self.solver = DummySolver(cell_manager=cm)
        self.lb = lb = self.cm.load_balancer = LoadBalancer(parallel_solver=self.solver, parallel_cell_manager=self.cm)
        lb.skip_iteration = 1
        lb.threshold_ratio = 10.
        lb.lb_max_iteration = 20
        lb.setup()
    
    def get_lb_args(self):
        return [
                # This test is only for serial cases
                #dict_from_kwargs(method='normal'),
                dict_from_kwargs(),
                dict_from_kwargs(distr_func='auto'),
                dict_from_kwargs(distr_func='geometric'),
                dict_from_kwargs(distr_func='mkmeans', c=0.3, t=0.2, tr=0.8, u=0.4, e=3, er=6, r=2.0),
                dict_from_kwargs(distr_func='sfc', sfc_func='morton', start_origin=False),
                dict_from_kwargs(distr_func='sfc', sfc_func='morton', start_origin=True),
                dict_from_kwargs(distr_func='sfc', sfc_func='hilbert', start_origin=False),
                dict_from_kwargs(distr_func='sfc', sfc_func='hilbert', start_origin=True),
               ]
    
    def load_balance(self):
        np0 = 0
        nc0 = len(self.cm.cell_dict)
        for cid, cell in self.cm.cell_dict.items():
            np0 += cell.get_number_of_particles()
        lb = self.cm.load_balancer
        for lbargs in self.get_lb_args():
            lb.load_balance(**lbargs)
            self.cm.exchange_neighbor_particles()
    
    def test_distribute_particle_arrays(self):
        for num_procs in range(1,12,3):
            for lbargs in self.get_lb_args():
                self.create_solver()
                #self.assertEqual(len(self.cm.cells_dict), 11
                proc_pas = LoadBalancer.distribute_particle_arrays(self.pas, num_procs, self.cell_size, 100, **lbargs)
                nps = [sum([pa.get_number_of_particles() for pa in pas]) for pas in proc_pas]
                self.assertTrue(sum(nps) == sum([pa.get_number_of_particles() for pa in self.pas]))
                for pa in pas:
                    self.assertEqual(len(pa.get('x')),pa.get_number_of_particles())
                    self.assertEqual(len(pa.get('y')),pa.get_number_of_particles())
                # each proc should have at least one cell since num_cells>num_procs
                for np in nps:
                    assert np > 0


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
        lg = CuboidGenerator(kernel=CubicSplineKernel(2))
        self.pas = [lg.get_particles()]
        # to shift the origin
        self.pas[0].x += 0.1
        self.pas[0].y += 0.2
        self.pas[0].z += 0.3
        self.cell_size = 0.1
        self.dim = 3



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'TestSerialLoadBalancer3D.test_distribute_particle_arrays']
    unittest.main()
