
import numpy

from pysph.base.nnps cimport NNPSManager, FixedDestNbrParticleLocator, \
        VarHNbrParticleLocator, NbrParticleLocatorBase
from pysph.base.carray cimport LongArray
from pysph.base.point cimport Point
from pysph.base.nnps import brute_force_nnps
from pysph.base.particle_array import ParticleArray
from pysph.base.carray import LongArray
from pysph.base.cell import CellManager
#from pysph.base.api import brute_force_nnps, ParticleArray, LongArray, \
#        CellManager

from time import time

cdef extern from "time.h":
    ctypedef long clock_t
    clock_t clock()

cdef extern from "stdlib.h":
    int RAND_MAX
    int crand "rand" ()

Nps = {'2d1':(40,40,1), '3d1':(10,10,10), '1d1':(100,1,1),}
       #'2d2':(60,60,1), '3d2':(16,16,16), '1d2':(400,1,1),}


def get_data(variable_h=True, Nps=(40,40,1), cell_size=4.0):
    
    x, y, z = numpy.mgrid[0:Nps[0], 0:Nps[1], 0:Nps[2]]
    x = x.ravel()
    y = y.ravel()
    z = z.ravel()
    h = numpy.ones_like(x)
    
    parr1 = ParticleArray(name='parr1', **{'x':{'data':x}, 'y':{'data':y}, 'z':
                                            {'data':z}, 'h':{'data':h}})
    
    cell_mgr = CellManager(arrays_to_bin=[parr1], min_cell_size=cell_size)
    nnps_mgr = NNPSManager(cell_mgr, variable_h=variable_h)
    nbrl = nnps_mgr.get_neighbor_particle_locator(source=parr1, dest=parr1,
                                                  radius_scale=3.0)
    return nbrl

cpdef dict nnps():
    cdef dict ret = {}
    cdef double t, t1, t2, t3, h
    cdef LongArray output_array
    cdef int np, i
    cdef FixedDestNbrParticleLocator nbrl
    for var_h in [False, True]:
        vh = 'var-h' if var_h else 'fixed-h'
        for nam,nps in Nps.items():
            np = nps[0]*nps[1]*nps[2]
            nbrl = get_data(var_h, nps)
            nbrl.is_dirty = True
            t = time()
            nbrl.update()
            t1 = time() - t
            
            t = time()
            for i in range(np):
                output_array = nbrl.get_nearest_particles(i)
            t2 = time() - t
            
            t = time()
            for i in range(np):
                output_array = nbrl.get_nearest_particles(i, True)
            t3 = time() - t
            
            ret['%s up %s /%d'%(vh,nam,np)] = t1/np
            ret['%s %s /%d'%(vh,nam,np)] = t2/np
            ret['%s es %s /%d'%(vh,nam,np)] = t3/np
    return ret

cpdef nbr_particles_from_cell_list():
    cdef dict ret = {}
    cdef double t, t1, t2, t3, h
    cdef double radius = 1.0
    cdef LongArray output_array = LongArray()
    cdef int np, i
    cdef list cell_list
    cdef Point pnt = Point()
    cdef FixedDestNbrParticleLocator nbrl
    
    for nam,nps in Nps.items():
        np = nps[0]*nps[1]*nps[2]
        cell_list = []
        output_array.reset()
        pnt.set(nps[0]/2.0, nps[1]/2.0, nps[2]/2.0)
        nbrl = get_data(False, nps)
        
        t = time()
        nbrl.cell_manager.get_potential_cells(pnt, radius, cell_list)
        t1 = time() - t
        
        t = time()
        nbrl._get_nearest_particles_from_cell_list(pnt,
                radius, cell_list, output_array)
        t2 = time() - t
        
        ret['cell_mgr get_pot_cells %s %d'%(nam,np)] = t1
        ret['nnps _get_nbr_frm_cell_list %s %d'%(nam,np)] = t2
    return ret

cdef list funcs = [nnps, nbr_particles_from_cell_list]


cpdef bench():
    cdef list timings = []
    for func in funcs:
        timings.append(func())
    return timings
    
