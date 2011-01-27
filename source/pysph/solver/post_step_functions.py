""" Post step functions for the solver """

import pickle
import os

import pysph.base.api as base

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

        for i in range(num_locs):
            loc = locators[i]
            dest = loc.dest
            
            particle_indices = dest.get('idx')

            neighbor_idx = {}
            
            neighbor_cache = loc.particle_neighbors

            nrp = dest.num_real_particles

            for j in range(nrp):
                neighbors = neighbor_cache[j]
                
                temp = dest.extract_particles(neighbors)
                particle_idx = particle_indices[j]

                idx = temp.get("idx")

                ia = base.IntArray()                
                ia.resize(len(idx))
                ia.set_data(idx)

                neighbor_idx[particle_idx] = ia
            
            fname = fname_base + "_" + dest.name + "_" + str(time)

            f = open(fname, 'w')
            pickle.dump(neighbor_idx, f)
            f.close()
            
