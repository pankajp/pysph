""" SImple plotting for the data """

from utils import get_pickled_data
import numpy

import pysph.base.api as base

#import visvis as vv

class ParticleInformation(object):

    def __init__(self, fname, nprocs, array_name, time,
                 load_neighbors=True, load_cells=True):
        self.fname = fname
        self.nprocs = nprocs
        self.array_name = array_name
        self.time = time

        self.load_neighbors = load_neighbors
        self.load_cells = load_cells

        self.neighbors = {}
        self.cell_ids = {}

        self.cell_data = {}
        self.particle_data = {}
        self.particle_positions = {}
        self.particle_idx_by_proc = {}
        self.particle_positions_by_proc = {}

        self.np = -1

        # load the data

        if load_neighbors:
            self._load_neighbor_data()

        if load_cells:
            self._load_cell_data()

        self._load_particle_data(fname)
        
    def _load_neighbor_data(self):
        
        for i in range(self.nprocs):
            
            fname = "neighbors_" + str(i) + "_" + self.array_name 
            fname += "_" + str(self.time)
            
            nbr_data = get_pickled_data(fname)

            for idx, data in nbr_data.iteritems():
                self.neighbors[idx] = data["neighbors"]
                self.cell_ids[idx] = data["cid"]
                
    def _load_cell_data(self):
        
        for i in range(self.nprocs):
            
            fname = "cells_" + str(i) + "_" + str(self.time)            
            cell_data = get_pickled_data(fname)

            self.cell_data.update(cell_data)
            
    def _load_particle_data(self, fname):
        
        np = 0

        for i in range(self.nprocs):
            
            self.particle_positions_by_proc[i] = {}

            _fname = fname + "_" + str(i) + "_" + self.array_name + "_" 
            _fname += str(self.time) + ".npz"
            
            data = numpy.load(_fname)
            self.particle_data[i] = data
            
            self.particle_positions_by_proc[i]['x'] = data['x']

            if 'y' in data.files:
                self.particle_positions_by_proc[i]['y'] = data['y']

            if 'z' in data.files:
                self.particle_positions_by_proc[i]['z'] = data['z']

            idx = data['idx']
            self.particle_idx_by_proc[i] = idx

            for j in range(data['np']):
                particle_idx = idx[j]
                self.particle_positions[particle_idx] = {}
                
                self.particle_positions[particle_idx]['x'] = data['x'][j]
                
                if 'y' in data.files:
                    self.particle_positions[particle_idx]['y'] = data['y'][j]

                if 'z' in data.files:
                    self.particle_positions[particle_idx]['z'] = data['z'][j]

            np += data['np']
            
        self.np = np

    def get_particle_position(self, idx):
        
        if idx < 0 or idx > self.np:
            raise RunTimeError, "Invalid Particle Index!"
        
        return base.Point(*self.particle_positions[idx].values())

    def get_cid_for_particle(self, idx):
        if not self.load_cells:
            return

        self.check_particle_id(idx)
        return self.cell_ids[idx]    

    def get_particles_in_cell(self, cid):
        """ Return the particles in the same cell """
        
        if not self.load_cells:
            return 
        
        cell_data = self.cell_data[cid]
        particle_indices = cell_data["positions"]["idx"]
        
        return particle_indices

    def get_particle_info(self, idx):
        if not self.load_neighbors:
            return

        self.check_particle_id(idx)

        cid = self.get_cid_for_particle(idx)
        neighbors = self.neighbors[idx].get_npy_array()
        
        position = self.get_particle_position(idx)
        
        info = {'cid':cid, "position":position, 'neighbors':neighbors}

        for rank, idx_list in self.particle_idx_by_proc.iteritems():
            if idx in idx_list:
                info["rank"] = rank
                break

        return info

    def get_cells_for_proc(self, proc):
        if proc < 0 or proc > self.nprocs:
            raise RunTimeError, "Invalid pid!"

        particle_ids = self.particle_idx_by_proc[proc]        
        return set([self.cell_ids[i] for i in particle_ids])

    def get_coordinates_for_cell(self, cid):
        
        centroid = self.cell_data[cid]['centroid']
        cell_size = self.particle_data[0]['cell_size']

        x1 = centroid[0] - 0.5 * cell_size
        y1 = centroid[1] - 0.5 * cell_size
        
        x2 = x1 + cell_size
        y2 = y1
        
        x3 = x2
        y3 = y2 + cell_size
        
        x4 = x1
        y4 = y3

        x = numpy.array([x1,x2,x3,x4,x1])
        y = numpy.array([y1,y2,y3,y4,y1])
        
        return numpy.rec.fromarrays([x,y], names="x,y")
        
    def check_particle_id(self, idx):
        if idx < 0 or idx > self.np:
            raise RunTimeError, "Invalid Particle Index!"
