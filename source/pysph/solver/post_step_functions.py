""" Post step functions for the solver """

import pickle
import os

import pysph.base.api as base
from pysph.base.cell import py_find_cell_id

class PrintNeighborInformation:

    def __init__(self, rank = 0, path=None, count=10):
        self.rank = rank

        self.count = count
        
        if path:
            self.path = path
        else:
            self.path = "."
    
    def eval(self, particles, count, time):
        
        if not ((count % self.count) == 0):
            return

        nnps = particles.nnps_manager
        locator_cache = nnps.particle_locator_cache
        
        num_locs = len(locator_cache)
        locators = locator_cache.values()

        fname_base = os.path.join(self.path+"/neighbors_"+str(self.rank))

        cell_manager = particles.cell_manager
        cell_size = cell_manager.cell_size
        origin = cell_manager.origin

        for i in range(num_locs):
            loc = locators[i]
            dest = loc.dest
            
            particle_indices = dest.get('idx')

            x, y, z = dest.get("x", "y", "z")

            neighbor_idx = {}
            
            nrp = dest.num_real_particles

            for j in range(nrp):
                neighbors = loc.py_get_nearest_particles(j)
                
                temp = dest.extract_particles(neighbors)
                particle_idx = particle_indices[j]
                
                pnt = base.Point(x[j], y[j], z[j])
                cid = py_find_cell_id(origin, pnt, cell_size)

                idx = temp.get_carray("idx")

                neighbor_idx[particle_idx] = {'neighbors':idx, 'cid':cid}
            
            fname = fname_base + "_" + dest.name + "_" + str(time)
            
            f = open(fname, 'w')
            pickle.dump(neighbor_idx, f)
            f.close()
            
            fname_cells = os.path.join(self.path+"/cells_"+str(self.rank))
            fname_cells += "_" + str(time)
            cell_manager.get_particle_representation(fname_cells)
            
