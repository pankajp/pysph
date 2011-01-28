""" SImple plotting for the data """

from utils import get_pickled_data

import visvis as vv

class ParticleInformation(object):

    def __init__(self, nprocs, array_name, time):
        self.nprocs = nprocs
        self.array_name = array_name
        self.time = time

        self.fnames = []

        self.neghbors = {}
        self.cell_ids = {}
        
    def _load_neighbor_data(self):
        
        for i in range(self.nprocs):
            
            fname = "neighbors_" + str(i) + "_" + self.array_name 
            fname += "_" + str(time)
            
            nbr_data = get_pickled_data(fname)

            for idx, data in nbr_data.iteritems():
                self.neighbors[idx] = data["neighbors"]
                self.cell_ids[idx] = data["cid"]
                
    def _load_cell_data(self):
        
        for i in range(self.nprocs):
            
            fname = "cells_" + str(i) + "_" + str(time)
            
            cell_data = get_pickled_data(fname)
            
            
