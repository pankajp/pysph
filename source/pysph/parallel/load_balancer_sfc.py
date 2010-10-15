"""
Contains class to perform load balancing using space filling curves.
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
import numpy

# local imports
from pysph.base.particle_array import ParticleArray
from pysph.base.cell import py_construct_immediate_neighbor_list
from load_balancer import LoadBalancer
import space_filling_curves


###############################################################################
# `LoadBalancerSFC` class.
###############################################################################
class LoadBalancerSFC(LoadBalancer):
    def __init__(self, sfc_func_name='morton', sfc_func_dict=None,
                 start_origin=False, **args):
        LoadBalancer.__init__(self, **args)
        self.method = 'serial_sfc'
        if sfc_func_dict is None:
            sfc_func_dict = space_filling_curves.sfc_func_dict
        self.sfc_func_dict = sfc_func_dict
        self.sfc_func = sfc_func_name
        self.start_origin = start_origin
    
    def load_balance_func_serial_sfc_iter(self, sfc_func_name=None,
                                     start_origin=None, **args):
        """ serial load balance function which uses SFCs
        
        calls the :class:Loadbalancer :meth:load_balance_func_serial
        setting the appropriate sfc function
        """
        if sfc_func_name is None:
            sfc_func_name = self.sfc_func
        if start_origin is None:
            start_origin = self.start_origin
        sfc_func = self.sfc_func_dict[sfc_func_name]
        self.load_balance_func_serial('sfc', sfc_func=sfc_func,
                                      start_origin=start_origin, **args)
        
    def load_redistr_sfc(self, cell_proc, proc_cell_np, sfc_func=None,
                         start_origin=None, **args):
        """ function to redistribute the cells amongst processes using SFCs
        
        This is called by :class:Loadbalancer :meth:load_balance_func_serial
        """
        if sfc_func is None:
            sfc_func = self.sfc_func_dict[self.sfc_func]
        if start_origin is None:
            start_origin = self.start_origin
        num_procs = len(proc_cell_np)
        
        num_cells = len(cell_proc)
        cell_arr = numpy.empty((num_cells, 3))
        for i,cell_id in enumerate(cell_proc):
            cell_arr[i,0] = cell_id.x
            cell_arr[i,1] = cell_id.y
            cell_arr[i,2] = cell_id.z
        dim = 3
        if min(cell_arr[:,2])==max(cell_arr[:,2]):
            dim = 2
            if min(cell_arr[:,1])==max(cell_arr[:,1]):
                dim = 1
        np_per_proc = sum(self.particles_per_proc)/float(self.num_procs)
        cell_ids = cell_proc.keys()
        if start_origin:
            idmin = cell_arr.min(axis=0)
            cell_ids.sort(key=lambda x: sfc_func(x.asarray()-idmin, dim=dim))
        else:
            cell_ids.sort(key=lambda x: sfc_func(x, dim=dim))
        
        ret_cells = [[] for i in range(num_procs)]
        proc_num_particles = [0]*num_procs
        np = 0
        proc = 0
        for cell_id in cell_ids:
            np += self.proc_cell_np[cell_proc[cell_id]][cell_id]
            #print proc, cell_id, np
            ret_cells[proc].append(cell_id)
            if np > np_per_proc:
                proc_num_particles[proc] = np
                np -= np_per_proc
                proc += 1
        
        self.particles_per_proc = [0]*self.num_procs
        
        cell_np = {}
        for cnp in self.proc_cell_np:
            cell_np.update(cnp)
        for proc,cells in enumerate(ret_cells):
            for cid in cells:
                cell_proc[cid] = proc
                self.particles_per_proc[proc] += cell_np[cid]
        self.balancing_done = True
        return cell_proc, self.particles_per_proc

###############################################################################

