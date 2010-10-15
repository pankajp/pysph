"""
Module to implement parallel decomposition of particles to assign to
different processes during parallel simulations. The method used is an
extension of k-means clustering algorithm
"""

# logging imports
import logging
logger = logging.getLogger()

# standard imports
import numpy

# local imports
from pysph.base.cell import py_construct_immediate_neighbor_list
from load_balancer import LoadBalancer
from load_balancer_sfc import LoadBalancerSFC


class Cluster():
    """Class representing a cluster in k-means clustering"""
    def __init__(self, cells, cell_np, np_req, **kwargs):
        """constructor
        
        kwargs can be used to finetune the algorithm:
        t = ratio of old component of center used in the center calculation
        tr = `t` when the number of particles over/undershoot (reversal)
        u = ratio of nearest cell center in the new center from the remaining
                (1-t) (other component is the centroid) of cells
        e = reciprocal of the exponent of 
                (required particles)/(actual particles) used to
                resize the cluster
        er = `e` on reversal (see `tr`)
        r = clipping of resize factor between (1/r and r)
        """
        self.cells = cells
        self.cell_np = cell_np
        self.dnp = 0
        self.np = 0
        self.dsize = 0.0
        self.size = 1.0
        self.np_req = np_req
        
        # ratio of old component
        self.tr = kwargs.get('tr',0.8)
        # ratio of nearest cell in the new component (other is the centroid)
        self.u = kwargs.get('u',0.4)
        # exponent for resizing
        self.e = kwargs.get('e',3.0)
        self.er = kwargs.get('er',6.0)
        self.r = kwargs.get('r',2.0)
        
        # there's no previous center hence it shouldn't come into calculation
        self.t = 0.0
        
        self.x = self.y = self.z = 0.0
        np = 0
        for cell in self.cells:
            n = self.cell_np[cell]
            np += n
            self.x += (cell.x)#*n
            self.y += (cell.y)#*n
            self.z += (cell.z)#*n
        self.np = np
        np = float(len(self.cells))
        self.x, self.y, self.z = self.x/np,self.y/np,self.z/np
        self.center = numpy.array([self.x, self.y, self.z])
        self.dcenter = self.center*0
        
        # so that initial setting is not way off
        self.move()
        # set the value of t
        self.t = kwargs.get('t',0.2)

    def calc(self):
        """calculate the number of particles and the change in the number of 
        particles (after a reallocation of cells)"""
        np = 0
        for cell in self.cells:
            n = self.cell_np[cell]
            np += n
        self.dnp = np - self.np
        self.np = np
    
    def move(self):
        """move the center depending on the centroid of cells (A),
        the nearest cell to the centroid (B) and the old center(C)
        
        formula: new center = (1-t)(1-u)A + (1-t)uB + tC
        t = tr on reversal (overshoot/undershoot of particles)"""
        x = y = z = 0.0
        for cell in self.cells:
            x += (cell.x)#*n
            y += (cell.y)#*n
            z += (cell.z)#*n
        np = float(len(self.cells))
        med = numpy.array([x/np,y/np,z/np])
        
        dists = []
        for cell in self.cells:
            d = (cell.x-self.x)**2+(cell.y-self.y)**2+(cell.z-self.z)**2
            d = numpy.sqrt(d)
            dists.append(d)
            #md = (cell.x-med[0])**2+(cell.y-med[1])**2+(cell.z-med[2])**2
            #dists[-1] = (dists[-1]+md)/2
        cell = self.cells[numpy.argmin(dists)]
        cc = numpy.array([cell.x, cell.y, cell.z])
        
        t = self.t
        if abs(self.dnp) * ( self.np-self.np_req) > 0:
            t = self.tr
        self.dcenter = (1-t)*(med-self.center + self.u*(cc-med))
        self.x,self.y,self.z = self.center = self.center + self.dcenter
    
    def resize(self):
        """resize the cluster depending on the number of particles and the
        required number of particles
        
        formula: new size = (old_size)*(np_req/np)**(1/e),
        clipped between r and 1/r"""
        e = self.e
        if abs(self.dnp) * ( self.np-self.np_req) > 0:
            e = self.er
        self.dsize = numpy.clip((self.np_req/self.np)**(1./e), 1/self.r, self.r)
        self.size *= self.dsize

class ParDecompose:
    """Partition of cells for parallel solvers"""
    def __init__(self, cell_proc, proc_cell_np, init=True, **kwargs):
        """constructor
        
        kwargs can be used to finetune the algorithm:
        c = (0.3) the ratio of euler distance contribution in calculating the
                distance of particle from cluster center
                (the other component is scaled distance based on cluster size)
        t = (0.2) ratio of old component of center in the center calculation
        tr = (0.8) `t` when the number of particles over/undershoot (reversal)
        u = (0.4) ratio of nearest cell center in the new center from the
                remaining (1-t) (other component is the centroid) of cells
        e = (3) reciprocal of the exponent of 
                (required particles)/(actual particles) used to
                resize the cluster
        er = (6) `e` on reversal (see `tr`)
        r = (2) clipping of resize factor between (1/r and r)
        """
        self.cell_proc = cell_proc
        self.proc_cell_np = proc_cell_np
        self.num_procs = len(proc_cell_np)
        self.c = kwargs.get('c', 0.3)
        if init:
            self.gen_clusters(**kwargs)
    
    def clusters_allocate_cells(self):
        """allocate the cells in the cell manager to clusters based on their
        "weighted distance" from the center of the cluster"""
        for cluster in self.clusters:
            cluster.cells[:] = []
        for cell in self.cell_proc:
            wdists = []
            for cluster in self.clusters:
                s = cluster.size
                d = ( (cell.x-cluster.x)**2 + (cell.y-cluster.y)**2 +
                        (cell.z-cluster.z)**2 )
                d = numpy.sqrt(d)
                c = self.c
                # TODO: choose a better distance function below
                r = d*(c+(1-c)*numpy.exp(-s/d))
                r = numpy.clip(r,0,r)
                wdists.append(r)
            self.clusters[numpy.argmin(wdists)].cells.append(cell)
    
    def get_distribution(self):
        """return the list of cells and the number of particles in each
        cluster to be used for distribution to processes"""
        self.calc()
        proc_cells = self.proc_cells
        proc_num_particles = self.particle_loads
        cell_proc = LoadBalancer.get_cell_proc(proc_cells=proc_cells)
        return cell_proc, proc_num_particles
    
    def cluster_bal_iter(self):
        """perform a single iteration of balancing the clusters
        
        **algorithm**
        
         # move the cluster center based on their cells
         # allocate cells to clusters based on new centers
         # resize the clusters based on the number of particles
         # allocate cells to clusters based on new sizes
        """
        # moving
        for j,cluster in enumerate(self.clusters):
            cluster.move()
        self.clusters_allocate_cells()
        for j,cluster in enumerate(self.clusters):
            cluster.calc()
            #print j, '\t', cluster.center, '\t', cluster.np, '\t', cluster.size
        
        # resizing
        for j,cluster in enumerate(self.clusters):
            cluster.resize()
        self.clusters_allocate_cells()
        for j,cluster in enumerate(self.clusters):
            cluster.calc()
            #print j, '\t', cluster.center, '\t', cluster.np, '\t', cluster.size
        
        self.calc()
    
    def calc(self):
        """calculates the cells in each process, the cell and particle loads
        and the imbalance in the distribution"""
        self.proc_cells = [cluster.cells for cluster in self.clusters]
        self.cell_loads = [sum([len(cell) for cell in self.proc_cells])]
        self.particle_loads = [cluster.np for cluster in self.clusters]
        self.imbalance = LoadBalancer.get_load_imbalance(self.particle_loads)
    
    def gen_clusters(self, proc_cells=None, proc_num_particles=None, **kwargs):
        """generate the clusters to operate on. This is automatically called
        by the constructor if its `init` argument is True (default)"""
        cell_np = {}
        for tmp_cells_np in self.proc_cell_np:
            cell_np.update(tmp_cells_np)
        self.cell_np = cell_np
        if proc_cells is None:
            proc_cells, proc_num_particles = LoadBalancer.distribute_particles_geometric(
                                                self.cell_np, self.num_procs)
        self.np_req = numpy.average(proc_num_particles)
        self.clusters = [Cluster(cells, cell_np, self.np_req, **kwargs) 
                            for cells in proc_cells]
        self.calc()

def distribute_particles(cm, num_procs, max_iter=200, n=5, **kwargs):
    """ distribute particles according to the modified k-means clustering
    algorithm implemented by the `ParDecompose` class
    
    The algorithm runs maximum `max_iter` iterations.
    The solution is assumed converged if the particle distribution is same
    in `n+k` steps out of `n+2k` latest steps
    See :class:`ParDecompose` for the fine-tuning parameters kwargs"""
    pd = ParDecompose(cm, num_procs, **kwargs)
    pd.calc()
    proc_num_particles = pd.particle_loads
    conv = 0
    for t in range(max_iter):
        pd.cluster_bal_iter()
        pd.calc()
        #print t
        
        proc_num_particlesold = proc_num_particles
        proc_num_particles = pd.particle_loads
        imbal = pd.imbalance
        logger.debug('imbalance %g' %imbal)
        if proc_num_particlesold == proc_num_particles:
            conv += 1
            logger.debug('converged in %d iterations' %t)
            if conv > n:
                break
        else:
            conv -= 1
            if conv < 0: conv = 0
    return pd.get_distribution()


###############################################################################
# `LoadBalancerMKMeans` class.
###############################################################################
class LoadBalancerMKMeans(LoadBalancerSFC):
    def __init__(self, **args):
        LoadBalancerSFC.__init__(self, **args)
        self.method = 'serial_mkmeans'
        self.args = args
    
    def load_balance_func_serial_mkmeans(self, **args):
        self.load_balance_func_serial('mkmeans', **args)
    
    def load_redistr_mkmeans(self, cell_proc=None, proc_cell_np=None, max_iter=None, n=3, **args):
        """ distribute particles according to the modified k-means clustering
        algorithm implemented by the `ParDecompose` class
        
        The algorithm runs maximum `max_iter` iterations.
        The solution is assumed converged if the particle distribution is same
        in `n+k` steps out of `n+2k` latest steps
        See :class:`ParDecompose` for the fine-tuning parameters kwargs"""
        args2 = {}
        args2.update(self.args)
        args2.update(args)
        if max_iter is None:
            max_iter = self.lb_max_iterations
        #print args
        pd = ParDecompose(cell_proc, proc_cell_np, **args)
        pd.calc()
        proc_num_particles = pd.particle_loads
        conv = 0
        for t in range(max_iter):
            pd.cluster_bal_iter()
            pd.calc()
            #print t
            
            proc_num_particlesold = proc_num_particles
            proc_num_particles = pd.particle_loads
            imbal = pd.imbalance
            logger.debug('imbalance %g' %imbal)
            if proc_num_particlesold == proc_num_particles:
                conv += 1
                logger.debug('converged in %d iterations' %t)
                if conv > n:
                    logger.debug('mkm converged in %d iterations' %t)
                    break
            else:
                conv -= 1
                if conv < 0: conv = 0
        #self.balancing_done = True
        return pd.get_distribution()

