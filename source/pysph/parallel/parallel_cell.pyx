""" Classes to implement cells and cell manager for a parallel invocation. """
# standard imports
import copy

# numpy imports
import numpy

from pysph.base.cell import INT_INF

# logger imports
import logging
logger = logging.getLogger()

# mpi imports
cimport mpi4py.MPI as MPI

# local imports
from pysph.base.point import Point, IntPoint
from pysph.base.point cimport cPoint, IntPoint, cPoint_new, IntPoint_new, \
            Point_new, IntPoint_from_cIntPoint
from pysph.base import cell
from pysph.base.cell cimport construct_immediate_neighbor_list, find_cell_id
from pysph.base.cell cimport CellManager, Cell
from pysph.base.particle_array cimport ParticleArray, get_dummy_tag, LocalReal
from pysph.base.carray cimport LongArray, DoubleArray

from fast_utils cimport arange_long
from pysph.parallel.parallel_controller cimport ParallelController
from pysph.parallel.load_balancer import LoadBalancer

from python_dict cimport *

TAG_PROC_MAP_UPDATE = 1
TAG_CELL_ID_EXCHANGE = 2
TAG_CROSSING_PARTICLES = 3
TAG_NEW_CELL_PARTICLES = 4
TAG_REMOTE_CELL_REQUEST = 5
TAG_REMOTE_CELL_REPLY = 6
TAG_REMOTE_DATA_REQUEST = 7
TAG_REMOTE_DATA_REPLY = 8
TAG_REMOTE_DATA = 9

cdef extern from 'math.h':
    cdef double ceil(double)
    cdef double floor(double)

###############################################################################
# `share_data` function.
###############################################################################
cpdef dict share_data(int mypid, list send_procs, object data,
                             MPI.Comm comm, int tag=0, bint multi=False,
                             list recv_procs=None):
    """
    Sends the given data to the processors in send_procs list and receives
    data from procs in recv_list. Returns a dictionary containing data from
    each other processor.

    Parameters:
    -----------
    
    mypid - pid where the function is being called.
    send_procs - list of processors to send data to.
    data - the data to be shared.
    comm - the MPI communicator to use.
    tag - a tag for the MPI send/recv calls if needed.
    multi - indicates if 'data' contains data specific to each nbr_proc or not.
            True -> separate data for each proc in send_proc
            False -> Send same data to all procs in  send_proc
    recv_procs - list of processors to receive data from
            None -> same as send_procs

    Notes:
    ------

    `data` is assumed to be a dictionary keyed on processor id to
     which some message must be sent. The value is the message.

     The function returns the dictionary `proc_data` which is keyed on
     processor id. The values are the messages received from the
     respective processor.
     
     The send_procs and recv_procs among all procs must be consistent, i.e.
         proc i has proc j in send_procs => proc j has proc i in recv_procs
         and vice-versa

    """
    cdef int pid=-1, num_procs, i=0
    send_procs = sorted(send_procs)
    if recv_procs is None:
        recv_procs = send_procs[:]
    else:
        recv_procs = sorted(recv_procs)
    num_send_procs = len(send_procs)
    num_recv_procs = len(recv_procs)

    cdef dict proc_data = {}
    
    # recv from procs with lower rank
    while i < num_recv_procs:
        pid = recv_procs[i]
        if pid < mypid:
            i += 1
            proc_data[pid] = comm.recv(source=pid, tag=tag)
        else:
            break
    
    # set self data if needed and remove self from nbrs
    # pid,i set in the above for loop
    if pid == mypid:
        if multi:
            proc_data[mypid] = data[mypid]
        else:
            proc_data[mypid] = data
        del recv_procs[i]
        send_procs.remove(mypid)
    
    # send data to all nbrs
    for pid in send_procs:
        if multi:
            comm.send(data[pid], dest=pid, tag=tag)
        else:
            comm.send(data, dest=pid, tag=tag)
    
    # recv from procs with higher rank
    # i value as set in first loop
    for j in range(i, len(recv_procs)):
        pid = recv_procs[j]
        proc_data[pid] = comm.recv(source=pid, tag=tag)
    
    return proc_data

###############################################################################
# `ProcessorMap` class.
###############################################################################
cdef class ProcessorMap:
    """
    Class to maintain the assignment of processors to geometric regions.

    Notes:
    ------
    
    The processor map bins the cells contained in the cell manager
    based on the bin size provided. This means that a large number of
    cells are aggregated into one box (if the bin size is large).

    The processor map identifies neighboring processors by checking
    the bin ids just created. For example, if a bin on one processor
    has, as a neighbor (as returned by
    construct_immediate_neighbor_list) a bin on another processor, the
    two processors are considered neighbors.
    
    """

    #Defined in the .pxd file
    #cdef public dict local_block_map
    #cdef public dict block_map
    #cdef public dict cell_map
    #cdef public list nbr_procs
    #cdef public int pid
    #cdef public double block_size

    def __init__(self, parallel_controller,
                 double block_size=0.3, *args, **kwargs):
        self.parallel_controller = parallel_controller
        self.local_block_map = {}
        self.block_map = {}
        self.cell_map = {}
        self.nbr_procs = []
        self.pid = parallel_controller.rank
        self.conflicts = {}
        self.load_per_proc = {}
        
        self.block_size = block_size
    
    def __reduce__(self):
        """ Implemented to facilitate pickling of extension types. """
        d = {}
        d['local_block_map'] = self.local_block_map
        d['block_map'] = self.block_map
        d['cell_map'] = self.cell_map
        d['pid'] = self.pid
        d['conflicts'] = self.conflicts
        d['block_size'] = self.block_size
        d['load_per_proc'] = self.load_per_proc

        return (ProcessorMap, (self.parallel_controller,), d)

    def __setstate__(self, d):
        """ Implemented to facilitate pickling of extension types. """
        self.local_block_map = {}
        self.local_block_map.update(d['local_block_map'])
        self.block_map = {}
        self.block_map.update(d['block_map'])
        self.cell_map = {}
        self.cell_map.update(d['cell_map'])
        self.pid = d['pid']
        self.block_size = d['block_size']
        self.conflicts = {}
        self.conflicts.update(d['conflicts'])
        self.load_per_proc = {}
        self.load_per_proc.update(d['load_per_proc'])
    
    def update(self, dict cells_dict, num_particles=-1):
        """ Replace the processor map information with given cells_dict
        
        Notes:
        ------
      
        The block_map data attribute is a dictionary keyed on block
        index with value being the processor owning that block

        Algorithm:
        ----------

        for all cells in cell_dict
            find the index of the cell based on binning with block size
            add this to the block_map attribute
            add the cell to the cell_map attribute
        
        """
        cdef int pid = self.pid
        cdef dict block_map = {}
        cdef Point centroid = Point()
        cdef Cell cell
        cdef IntPoint cid, bid = IntPoint_new(0,0,0)

        self.cell_map.clear()
        
        for cid, cell in cells_dict.iteritems():
            cell.get_centroid(centroid)
            bid.data = find_cell_id(centroid.data, self.block_size)
            block_map[bid.copy()] = pid

            if self.cell_map.has_key(bid):
                self.cell_map[bid].add(cid)
            else:
                self.cell_map[bid.copy()] = set([cid])
        
        self.block_map = block_map
        self.local_block_map = copy.deepcopy(block_map)
        
        if num_particles > 0:
            self.load_per_proc = {self.pid:num_particles}
        else:
            self.load_per_proc = {self.pid:len(block_map)}

    cpdef glb_update_proc_map(self, dict cells_dict):
        """ Update the global processor map.

        Notes:
        ------
        
        After a call to this function, all processors, should have identical
        processor maps.

        Algorithm:
        ----------
        
        - bring local data up to date.
        - receive proc_map from children if any.
        - merge with current p_map.
        - send to parent if not root.
        - receive updated proc_map from root.
        - send updated proc_map to children.

        """
        cdef ParallelController pc = self.parallel_controller
        cdef MPI.Comm comm = pc.comm
        cdef int c_rank
        cdef ProcessorMap c_proc_map, updated_proc_map
        cdef dict block_particles = {}
        
        self.update(cells_dict)
        
        # merge data from all children proc maps.
        for c_rank in pc.children_proc_ranks:
            c_proc_map = comm.recv(source=c_rank,
                                   tag=TAG_PROC_MAP_UPDATE)
            self.merge(c_proc_map)
        
        # we now have partially merged data, send it to parent if not root.
        if pc.parent_rank > -1:
            comm.send(self, dest=pc.parent_rank,
                      tag=TAG_PROC_MAP_UPDATE)

            # receive updated proc map from parent
            updated_proc_map = comm.recv(source=pc.parent_rank,
                                         tag=TAG_PROC_MAP_UPDATE)

            # set our proc data with the updated data.
            PyDict_Clear(self.block_map)
            PyDict_Update(self.block_map, updated_proc_map.block_map)
            
            self.conflicts.clear()
            self.conflicts.update(updated_proc_map.conflicts)
            
            self.load_per_proc.update(updated_proc_map.load_per_proc)

        # send updated data to children.
        for c_rank in pc.children_proc_ranks:
            comm.send(self, dest=c_rank, tag=TAG_PROC_MAP_UPDATE)
    
    cpdef resolve_procmap_conflicts(self, dict trf_particles):
        # calculate num_blocks per proc
        blocks_per_proc = {}
        recv_procs = set([self.pid])
        procs_blocks_particles = {self.pid:{}}
        for proc in range(self.parallel_controller.num_procs):
            blocks_per_proc[proc] = 0
        
        for bid in self.block_map:
            proc = self.block_map[bid]
            if proc > -1:
                blocks_per_proc[proc] += 1
        
        # assign block to proc with least blocks then max rank
        for bid in self.conflicts:
            candidates = list(self.conflicts[bid])
            proc = self.block_map.get(bid, -1)
            if proc < 0:
                # need to resolve conflict, new block
                proc = candidates[0]
                blocks = blocks_per_proc[0]
                
                # find the winning proc for each block
                for i in range(1, len(candidates)):
                    if blocks_per_proc[i] < blocks:
                        proc = candidates[i]
                        blocks = blocks_per_proc[i]
                    elif blocks_per_proc[i] == blocks:
                        if candidates[i] > proc:
                            proc = candidates[i]
                
                self.block_map[bid] = proc
                blocks_per_proc[proc] += 1

            if self.pid in candidates:
                # this is a remote block
                if bid in trf_particles:
                    if proc in procs_blocks_particles:
                        t = trf_particles[bid]
                        procs_blocks_particles[proc][bid] = t
                    else:
                        procs_blocks_particles[proc] = {bid:trf_particles[bid]}
            
            if proc != self.pid:
                # send to winning proc
                if bid in self.local_block_map:
                    del self.local_block_map[bid]
            else:
                # recv from other conflicting procs
                self.local_block_map[bid] = proc
                recv_procs.update(candidates)
        
        logger.info('remote_block_particles: '+str(procs_blocks_particles))
        logger.info('recv_procs: %r'%(recv_procs))
        #if self.pid in recv_procs: recv_procs.remove(self.pid)
        
        send_procs = procs_blocks_particles.keys()
        recv_particles = share_data(self.pid, send_procs, procs_blocks_particles,
                                self.parallel_controller.comm,
                                TAG_CROSSING_PARTICLES, True, list(recv_procs))

        self.find_region_neighbors()
        self.conflicts.clear()
        return recv_particles
    
    cpdef merge(self, ProcessorMap proc_map):
        """ Merge data from other processors proc map into this processor map.

        Parameters:
        -----------

        proc_map -- processor map from another processor.

        Algorithm:
        ----------

        for all bins in the other processor maps bin list
            get the pid of the processor containing that bin
            
            if this proc_map contains the bin index
                assign it to max(pid) from self and other
                add it to conflicts
            else
                add an entry in merged_block_map for this bin index and this pid

        """
        cdef dict merged_block_map = self.block_map
        cdef dict other_block_map = proc_map.block_map
        cdef IntPoint other_bid
        cdef list block_list
        cdef int i, num_blocks, other_proc, local_proc
        
        self.load_per_proc.update(proc_map.load_per_proc)
        
        block_list = other_block_map.keys()
        num_blocks = len(block_list)
        
        # merge conflicts
        for bid in proc_map.conflicts:
            if bid in self.conflicts:
                self.conflicts[bid].update(proc_map.conflicts[bid])
            else:
                self.conflicts[bid] = proc_map.conflicts[bid]
            if bid not in self.block_map:
                self.block_map[bid] = -1
        
        for i in range(num_blocks):
            other_bid = block_list[i]
            other_proc = other_block_map.get(other_bid)
            
            if other_bid in self.block_map:
                # there may be a conflict
                local_proc = self.block_map.get(other_bid)
                
                if local_proc < 0:
                    if other_proc < 0:
                        # new region block
                        self.conflicts.get(other_bid).update(proc_map.conflicts.get(other_bid))
                    else:
                        # remote block into other
                        merged_block_map[other_bid] = other_proc
                        self.conflicts.get(other_bid).update(proc_map.conflicts.get(other_bid, []))
                elif other_proc < 0:
                    # remote block into self
                    self.conflicts[other_bid] = self.conflicts.get(other_bid, set())
                    self.conflicts.get(other_bid).update(proc_map.conflicts.get(other_bid, []))
                elif other_proc == local_proc:
                    # unlikely
                    self.conflicts[other_bid] = self.conflicts.get(other_bid, set())
                    self.conflicts.get(other_bid).update(proc_map.conflicts.get(other_bid, []))
                else:
                    # conflict created at this level
                    merged_block_map[other_bid] = -1
                    self.conflicts[other_bid] = set([local_proc, other_proc])
            else:
                merged_block_map[other_bid] = other_proc
                if other_bid in proc_map.conflicts:
                    self.conflicts[other_bid] = proc_map.conflicts.get(other_bid)
    
    cpdef find_region_neighbors(self):
        """ Find processors that are occupying regions in the
        neighborhood of this processors blocks.

        Notes:
        ------
        
        For each block in the local block map, an immediate neighbor
        list is constructed (27 neighbors). These blocks are checked
        if they belong to any other processor by querying the global
        block map.
        
        """
        cdef int pid = self.pid
        cdef set nb = set([pid])
        cdef dict local_block_map = self.local_block_map
        cdef dict block_map = self.block_map
        cdef list nids = []
        cdef list empty_list = []
        cdef int i, j, num_local_bins, len_nids
        cdef IntPoint nid, bid
        cdef int n_pid

        # for each block in local_pm
        for bid in local_block_map:
            nids[:] = empty_list
            
            # construct list of neighbor block ids in the proc_map.
            construct_immediate_neighbor_list(bid.data, nids, False)
            len_nids = len(nids)
            for i in range(len_nids):
                nid = nids[i]

                # if block exists, collect the occupying processor id
                n_pid = block_map.get(nid, -1)
                if n_pid >= 0:
                    nb.add(n_pid)
        
        self.nbr_procs = sorted(list(nb))

    def __str__(self):
        rep = '\nProcessor Map At proc : %d\n'%(self.pid)
        rep += 'Bin size : %s\n'%(self.block_size)
        rep += 'Region neighbors : %s'%(self.nbr_procs)
        return rep 

###############################################################################
# `ParallelCellManager` class.
###############################################################################
cdef class ParallelCellManager(CellManager):
    """
    Cell manager for parallel invocations.
    """
    def __init__(self, arrays_to_bin=[], min_cell_size=-1.0,
                 max_cell_size=0.5, initialize=True, max_radius_scale=2.0,
                 parallel_controller=None, dimension=3, load_balancing=True,
                 min_block_size=0.0,
                 solver=None,
                 *args, **kwargs):
        """
        Constructor.
        """
        cell.CellManager.__init__(self, arrays_to_bin=arrays_to_bin,
                                  min_cell_size=min_cell_size,
                                  max_cell_size=max_cell_size,
                                  initialize=False,
                                  max_radius_scale=max_radius_scale)

        self.solver=solver
        self.dimension = dimension

        self.glb_bounds_min = [0, 0, 0]
        self.glb_bounds_max = [0, 0, 0]
        self.glb_min_h = 0
        self.glb_max_h = 0

        self.local_bounds_min = [0,0,0]
        self.local_bounds_max = [0,0,0]
        self.local_min_h = 0
        self.local_max_h = 0

        self.max_radius_scale = max_radius_scale
        self.pid = 0

        self.min_block_size = min_block_size

        # set the parallel controller and procesor id

        if parallel_controller is None:
            self.parallel_controller = ParallelController(cell_manager=self)
        else:
            self.parallel_controller = parallel_controller

        self.pid = self.parallel_controller.rank

        # the processor map

        self.proc_map = ProcessorMap(self.parallel_controller)

        # the load balancer

        self.load_balancer = LoadBalancer(parallel_solver=self.solver,
                                          parallel_cell_manager=self)
        self.load_balancing = load_balancing
        
        self.initial_redistribution_done = False

        # start and end indices of particles from each neighbor processor.

        self.remote_particle_indices = {}

        self.trf_particles = {}

        if initialize is True:
            self.initialize()

    def barrier(self):
        self.parallel_controller.comm.barrier()

    cpdef initialize(self):
        """
        Initialization function for the cell manager.

        Algorithm:
        ----------- 

        Find the global/local bounds on data (x, y, z, h)
        setup the processor map
            
        """
        logger.debug('%s initialize called'%(self))
        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        pc = self.parallel_controller
        # exchange bounds and interaction radii (h) information.

        self.update_global_properties()

        # find the block size and cell size to use

        self.compute_block_size(self.min_block_size)

        self.compute_cell_size(self.min_cell_size, self.max_cell_size)

        # setup array indices.

        self.py_rebuild_array_indices()

        # setup the cells_dict
        self.py_setup_cells_dict()
        
        # setup information for the processor map.
        self.setup_processor_map()

        # build a single cell with all the particles
        self._build_cell()

        pc = self.parallel_controller
        logger.info('(%d) cell_size: %g'%(pc.rank,
                                           self.cell_size))

        self.cells_update()
        self.py_reset_jump_tolerance()

        self.initialized = True
    
    def update_global_properties(self):
        """ Exchange bound and smoothing length information among all
        processors.

        Notes:
        ------
        
        At the end of this call, the global min and max values for the 
        coordinates and smoothing lengths are stored in the attributes
        glb_bounds_min/max, glb_min/max_h 

        """
        data_min = {'x':0, 'y':0, 'z':0, 'h':0}
        data_max = {'x':0, 'y':0, 'z':0, 'h':0}
        
        for key in data_min.keys():
            mi, ma = self._find_min_max_of_property(key)
            data_min[key] = mi
            data_max[key] = ma

        self.local_bounds_min[0] = data_min['x']
        self.local_bounds_min[1] = data_min['y']
        self.local_bounds_min[2] = data_min['z']
        self.local_bounds_max[0] = data_max['x']
        self.local_bounds_max[1] = data_max['y']
        self.local_bounds_max[2] = data_max['z']
        self.local_min_h = data_min['h']
        self.local_max_h = data_max['h']

        pc = self.parallel_controller
        
        glb_min, glb_max = pc.get_glb_min_max(data_min, data_max)

        self.glb_bounds_min[0] = glb_min['x']
        self.glb_bounds_min[1] = glb_min['y']
        self.glb_bounds_min[2] = glb_min['z']
        self.glb_bounds_max[0] = glb_max['x']
        self.glb_bounds_max[1] = glb_max['y']
        self.glb_bounds_max[2] = glb_max['z']
        self.glb_min_h = glb_min['h']
        self.glb_max_h = glb_max['h']

        logger.info('(%d) bounds : %s %s'%(pc.rank, self.glb_bounds_min,
                                           self.glb_bounds_max))
        logger.info('(%d) min_h : %f, max_h : %f'%(pc.rank, self.glb_min_h,
                                                   self.glb_max_h))

    cpdef double compute_cell_size(self, double min_size=0, double max_size=0):
        """ Setup the cell size to use from the 'h' values.
        
        Notes:
        ------

        The cell size is set to 2*self.max_radius_scale*self.glb_min_h

        """
        if self.local_max_h > 0:
            self.factor = int(ceil(self.glb_max_h/self.local_max_h))
        else:
            self.factor = 1

        self.cell_size = self.block_size/self.factor
        
        logger.info('(R=%d) cell_size=%g'%(self.parallel_controller.rank,
                                           self.cell_size,))
        return self.cell_size

    cpdef compute_block_size(self, double min_block_size):
        """ Setup the block size to use 
        
        Parameters:
        -----------
        
        min_block_size -- the minimum block size to use

        """
        cdef double block_size = self.max_radius_scale * self.glb_max_h

        if block_size < min_block_size:
            block_size = min_block_size
        
        self.block_size = block_size

    def setup_processor_map(self):
        """ Setup the processor map 

        Parameters:
        -----------
        
        factor -- the aggregate of cells to consider a block.

        Notes:
        ------
        
        Fixme: The block size for the processor map is set to thrice the cell
        size. This could be incorrect as if different processors have
        different cell sizes, the blocks would be different.
        
        """
        proc_map = self.proc_map
        
        proc_map.pid = self.parallel_controller.rank
        
        cell_size = self.cell_size
        proc_map.block_size = cell_size*self.factor

    def _build_cell(self):
        """ Build a cell containing all the particles.
        
        Notes:
        ------
        
        This function is similar to the function in the CellManager,
        except that the cells used are the Parallel variants.

        """
        
        self.jump_tolerance = INT_INF()

        cdef double cell_size = self.cell_size

        cdef double xc = self.local_bounds_min[0] + 0.5 * cell_size
        cdef double yc = self.local_bounds_min[1] + 0.5 * cell_size
        cdef double zc = self.local_bounds_min[2] + 0.5 * cell_size
        cdef IntPoint id = IntPoint_new(0,0,0)
        id.data = find_cell_id(cPoint_new(xc, yc, zc), cell_size)

        cell = Cell(id=id, cell_manager=self, cell_size=cell_size,
                    jump_tolerance=INT_INF())

        # now add all particles of all arrays to this cell.

        num_arrays = len(cell.arrays_to_bin)
        for i in range(num_arrays):
            parray = cell.arrays_to_bin[i]
            num_particles = parray.get_number_of_particles()
            index_arr_source = numpy.arange(num_particles, dtype=numpy.long)
            index_arr = cell.index_lists[i]
            index_arr.resize(num_particles)
            index_arr.set_data(index_arr_source)

        self.cells_dict.clear()
        self.cells_dict[id] = cell

    cpdef int update_status(self) except -1:
        """
        Sets the is_dirty to to true, We cannot decide the dirtyness of this
        cell manager based on just the particle arrays here, as cell managers on
        other processors could have become dirty at the same time.

        We also force an update at this point. Reason being, the delayed updates
        can cause stale neighbor information to be used before the actual update
        happens. 

        """
        self.set_dirty(True)

        self.cells_update()

        return 0

    cpdef remove_remote_particles(self):
        """ Remove all remote particles from the particle arrays.
        
        Notes:
        -------

        Remote particles have the 'local' flag set to 0.
        
        """
        cdef ParticleArray parray
        
        for parray in self.arrays_to_bin:
            parray.remove_flagged_particles('local', 0)
  
    cpdef int cells_update(self) except -1:
        """ Update particle information """

        self.remove_remote_particles()

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        logger.debug('++++++++++++++++ UPDATE BEGIN +++++++++++++++++++++')

        # bin the particles and find the new_cells and remote_cells.

        new_block_cells, remote_block_cells = self.bin_particles()

        
        # create particle copies and mark those particles as remote.

        new_particles_for_neighbors = self.create_new_particle_copies(
                        remote_block_cells, True)
        new_region_particles = self.create_new_particle_copies(
                        new_block_cells, True)

        #logger.debug('remote_blocks: %r'%remote_block_cells)
        #logger.debug('new_blocks: %r'%new_block_cells)
        trf_particles = {}
        trf_particles.update(new_particles_for_neighbors)
        trf_particles.update(new_region_particles)

        self.mark_crossing_particles(remote_block_cells)
        self.assign_new_blocks(new_block_cells)
        
        # update the processor map and resolve the conflicts

        self.remove_remote_particles()
        self.delete_empty_cells()
        self.proc_map.glb_update_proc_map(self.cells_dict)
        recv_particles = self.proc_map.resolve_procmap_conflicts(trf_particles)
        self.add_entering_particles_from_neighbors(recv_particles)
        
        # compute the cell sizes for binning

        self.compute_cell_size()

        # rebin the particles

        self.rebin_particles()

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        # call a load balancer function.

        if self.initialized == True:
            if self.load_balancing == True:
                self.load_balancer.load_balance()

        logger.info('cells_update:'+str([parr.get_number_of_particles()
                                            for parr in self.arrays_to_bin]))

        # exchange neighbor information

        self.exchange_neighbor_particles()

        logger.debug('+++++++++++++++ UPDATE DONE ++++++++++++++++++++')
        return 0

    cpdef rebin_particles(self):
        """ Re-insert particle indices for all arrays.

        Algorithm:
        ----------

        - clear the indices of all the particle arrays that need to be
          binned.  

        - reinsert particle indices of all particle
          arrays that are to be binned.

        """
        cdef int i
        cdef int num_arrays, num_particles
        cdef ParticleArray parr
        cdef LongArray indices
        cdef Cell cell

        num_arrays = len(self.arrays_to_bin)
        
        for i in range(num_arrays):
            for cell in self.cells_dict.values():
                cell.clear_indices(i)
        
        i = 0
        for i in range(num_arrays):
            parr = self.arrays_to_bin[i]
            num_particles = parr.get_number_of_particles()
            indices = arange_long(num_particles, -1)
            self.insert_particles(i, indices)

        # delete any empty cells.

        self.delete_empty_cells()

    cpdef bin_particles(self):
        """ Find the cell configuration caused by the particles moving.

        Notes:
        ------

        This function is called just after the base cell is built
        with all the particles added to it.

        When new cells are created by particles moving into new
        regions, the corresponding block id's from the processor map
        is checked. We distinguish three cases.

        (i)   If the block id exists in the local block map, it implies
              that the cell is created in a region that is assigned to our
              processor. Hence this cell need not be communicated.

        (ii)  If the block id exists in the global processor map and  not in
              the local block map, it implies that the cell belongs to a
              region assigned to some other processor (remote block).
              This cell therefore needs to be communicated to that processor.
            
        (iii) If the block id does not exist in the global processor
              map, it it implies that the cell belongs to an
              unassigned region (new block). We need to determine an 
              appropriate processor to assign this cell to.             

        The function returns two dictionaries corresponding to remote
        block cells and new block cells as discussed above. The
        dictionaries are keyed on block id and have the list of cells
        that have gone into these blocks as value.

        """
        cdef dict new_block_cells = {}
        cdef dict remote_block_cells = {}
        cdef dict collected_data = {}
        cdef IntPoint cid
        cdef Point centroid = Point_new(0,0,0)
        cdef Cell cell
        cdef ProcessorMap proc_map = self.proc_map
        cdef int pid
        cdef IntPoint block_id = IntPoint_new(0,0,0)
        
        #find the new configuration of the cells

        for cid, cell in self.cells_dict.iteritems():
            (<Cell>cell).update(collected_data)

        # if the base cell exists and the initial re-distribution is False,
        # add that to the list of new cells.

        if self.initial_redistribution_done is False:

            # update the global processor map

            c = self.cells_dict.values()[0]

            if (<Cell>c).get_number_of_particles() > 0:
                collected_data[(<Cell>c).id] = c
                
            self.cells_dict.clear()
            self.initial_redistribution_done = True
        
        # we have a list of all new cells created by the cell manager.
        for cid, cell in collected_data.iteritems():

            #find the block id to which the newly created cell belongs
            cell.get_centroid(centroid)
            block_id.data = find_cell_id(centroid.data, proc_map.block_size)
            
            # get the pid corresponding to the block_id

            pid = proc_map.block_map.get(block_id, -1)
            if pid < 0:
                # add to new block particles

                if new_block_cells.has_key(block_id):
                    new_block_cells[block_id].append(cell)
                else:
                    new_block_cells[block_id.copy()] = [cell]
            else:
                # find to which remote processor the block belongs to and add

                if not pid == self.pid:
                    if remote_block_cells.has_key(block_id):
                        remote_block_cells[block_id].append(cell)
                    else:
                        remote_block_cells[block_id.copy()] = [cell]

        return new_block_cells, remote_block_cells

    cpdef create_new_particle_copies(self, dict block_dict_to_copy,
                                     bint mark_src_remote=True,
                                     bint local_only=True):
        """ Make copies of all particles in the given cell dict.
        
        Parameters:
        -----------

        block_dict_to_copy -- the new cell dictionary to copy. This is
                             either the new block cells or the remote
                             block cells as returned form the call to
                             `bin_particles`. The dict is keyed on
                             block id and has a list of cells
                             belonging to that block as value.

        mark_src_remote -- flag to toggle marking the source particles as
                           remote when creating copies.

        Algorithm:
        -----------
        - for each block id in cell_dict_to_copy, get the cell list
            create a list of particle arrays for the copies (num_arrays)
            - for each cell in the cell list
                - get indices of all particles in this cell and add it to
                  the copy arrays created.
                - mark particles as remote and dummy in the copies.

        Notes:
        ------

        The function is called from `cells_update` to make copies for the 
        new and remote block particles.

        For each new or remote block, the function creates a big
        particle array for all particle indices contained in cells
        under the block. Separate such arrays are created for
        different arrays in `arrays_to_bin`.

        The return value is a dictionary keyed on block id with a list
        of copy particle arrays for that block as value. The length if
        this list is of course the number of arrays in
        `arrays_to_bin`.
                
        """
        cdef dict copies = {}
        cdef IntPoint bid
        cdef list cell_list, parray_list
        cdef ParticleArray pa
        cdef int num_arrays = len(self.arrays_to_bin)
        
        for bid, cell_list in block_dict_to_copy.iteritems():

            parray_list = []
            for i in range(num_arrays):
                parray_list.append(ParticleArray())

            for cell in cell_list:
                index_lists = []
                cell.get_particle_ids(index_lists)

                for j in range(num_arrays):

                    s_parr = cell.arrays_to_bin[j]
                    index_array = index_lists[j]

                    d_parr = parray_list[j]
                    pa = s_parr.extract_particles(index_array)
                    
                    if local_only:
                        pa.remove_flagged_particles('local', 0)
                    
                    d_parr.append_parray(pa)
                    d_parr.set_name(s_parr.name)
                    
                    # mark the particles as remote and dummy in src.

                    if mark_src_remote:
                        s_parr.set_flag('local', 0, index_array)
                        s_parr.set_tag(get_dummy_tag(), index_array)

            copies[bid] = parray_list

        return copies
    
    cpdef mark_crossing_particles(self, dict remote_block_dict):
        """ Add crossing blocks to proc_map.conflicts
        
        These will be transferred when glb_update_proc_map is called
        """
        cdef ProcessorMap proc_map = self.proc_map
        cdef IntPoint bid
        
        for bid in remote_block_dict:
            self.proc_map.block_map[bid] = self.proc_map.block_map[bid]
            proc_map.conflicts[bid] = set([self.pid])
    
    cpdef assign_new_blocks(self, dict new_block_dict):
        """
        Assigns cells created in new regions (i.e. regions not assigned to any
        processor) to some processor. Conflicts are resolved using a
        deterministic scheme which returns the same winner in all processors.
        
        Parameters:
        -----------

        new_block_dict -- a dictionary keyed on block id with a list of
                          cells belonging to that block.


        Algorithm:
        -----------

            - share new particles with all neighbor procs in the proc map.
            - resolve conflict.
            - add data assigned to self into the local particle arrays.

        Notes:
        ------
        
        This function is called from `cells_update` after remote particles
        have been exchanged.

        `new_particles` is the dictionary returned from the call to
        `create_new_particle_copies` for new block particles.

        """
        cdef ProcessorMap proc_map = self.proc_map
        cdef IntPoint bid
        
        for bid in new_block_dict:
            proc_map.local_block_map.setdefault(bid, self.pid)
            proc_map.block_map.setdefault(bid, self.pid)
            self.proc_map.conflicts[bid] = set([self.pid])
        proc_map.nbr_procs = [self.pid]

    cpdef transfer_blocks_to_procs(self, dict procs_blocks,
                                   bint mark_remote=True, list recv_procs=None):
        """ Transfer particles in blocks to other procs and receive blocks

        Parameters:
        -----------

        procs_blocks -- dictionary keyed on proc with a list of blocks
                to send to that proc
        mark_remote -- True -> mark local particles in the block as remote
                                (eg. if block belongs to other proc)
                       False -> FIXME: *NOT YET IMPLEMENTED*
                                block belongs to self proc
                                (eg. sending neighbor information to nbr procs)
        recv_procs -- list of procs from which to receive data
                None -> same as the procs_blocks.keys()
        
        Algorithm:
        ----------
        
            - get cells to send for each proc by checking cells in
                self.cells_dict belonging in the blocks to be send to proc
            - create_new_particle_copies() for these cells
            - send this data to procs in procs_blocks.keys() and recv particles
                from procs in recv_procs
            - add_entering_particles_from_neighbors(recv_particles)

        **Data sent to and received from each processor**
            - 'block_id' - block id of received particles
            - 'particles' - particles received located in that block
        
        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef int proc
        cdef dict proc_data = {}
        cdef IntPoint bid
        cdef list parray_list
        cdef ParticleArray parray
        cdef list send_procs = procs_blocks.keys()
        cdef dict remote_block_cells
        
        if not mark_remote:
            raise NotImplementedError, 'transfer_block_to_procs(mark_remote=False)'
        
        for proc in procs_blocks:
            # get particles in the blocks
            remote_block_cells = {}
            for bid in procs_blocks[proc]:
                remote_block_cells[bid] = [self.cells_dict[cid] for cid in
                                                self.get_cells_in_block(bid)]
            proc_data[proc] = self.create_new_particle_copies(remote_block_cells)
        
        send_procs = proc_data.keys()
        logger.info('exchange_particles:'+str(send_procs)+str(recv_procs))
        recv_particles = share_data(self.pid, send_procs, proc_data, comm,
                                    TAG_CROSSING_PARTICLES, True, recv_procs)
        logger.info('recv_particles:'+str(recv_particles))
        rp = []
        for p in recv_particles.values():
            for q in p.values():
                rp.extend(q)
        logger.info('recv_particles_real:'+str([p.num_real_particles for p in rp]))
        logger.info('recv_particles_tot:'+str([p.get_number_of_particles() for p in rp]))
        
        # for each neighbor processor, there is one entry in recv_particles
        # containing all new cells that processor sent to us
        self.add_entering_particles_from_neighbors(recv_particles)

    def transfer_particles_to_procs(self, dict procs_blocks_particles,
                                   list recv_procs=None):
        """ Transfer particles in blocks to other procs and receive blocks

        Parameters:
        -----------

        procs_blocks_particles -- dictionary keyed on proc with a dictionary
            of blocks : particle_arrays to send to that proc
        recv_procs -- list of procs from which to receive data
                None -> same as the procs_blocks.keys()
        
        Algorithm:
        ----------
        
            - send procs_blocks_particles to procs in procs_blocks.keys() and
                recv particles from procs in recv_procs
            - add_entering_particles_from_neighbors(recv_particles)

        **Data sent to and received from each processor**
            - 'block_id' - block id of received particles
            - 'particles' - particles received located in that block
        
        """
        send_procs = procs_blocks_particles.keys()
        recv_particles = share_data(self.pid, send_procs, procs_blocks_particles,
                                self.parallel_controller.comm,
                                TAG_CROSSING_PARTICLES, True, recv_procs)
        
        # for each neighbor processor, there is one entry in recv_particles
        # containing all new cells that processor sent to us
        self.add_entering_particles_from_neighbors(recv_particles)

    cpdef exchange_crossing_particles_with_neighbors(self,dict block_particles):
        """
        Send all particles that crossed into a known neighbors region,
        receive particles that got into our region from a neighbors.

        Parameters:
        -----------

        block_particles -- dictionary keyed on block id with a list of particle 
                           arrays to send to that block

        Notes:
        -------

        Called from cells_update after the remote block and new block
        particles are created.

        `particles` is the dictionary returned from the call to 
        `create_new_particle_copies` for remote particles.

        The processor map should be updated for valid neighbor
        information.

        Algorithm:
        ----------
            
            - invert the remote_cells list, i.e. find the list of cells to be
              sent to each processor.
            - prepare this data for sending.
            - exchange this data with all processors in nbr_procs.
            - we now have a set of particles (in particle arrays) that entered
              our domain.
            - add these particles as real particles into the corresponding
              particle arrays.

        **Data sent to and received from each processor**

            - 'cell_id' - the cell that they have to create.
            - 'particles' - the particles they have to add to the said cells.
        
        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef ProcessorMap proc_map = self.proc_map
        cdef dict proc_data = {}
        cdef int proc_id, num_particles
        cdef IntPoint cid, bid
        cdef dict p_data
        cdef list parray_list
        cdef ParticleArray parray
        cdef list nbr_procs = self.proc_map.nbr_procs

        for proc_id in nbr_procs:
            proc_data[proc_id] = {}
        
        for bid, parray_list in block_particles.iteritems():
            #find the processor to which the block belongs
            pid = proc_map.block_map.get(bid)
            proc_data[pid][bid] = parray_list
        
        logger.info('exchange_particles:'+str(proc_data.keys())+str(proc_map.nbr_procs))
        
        new_particles = share_data(self.pid, nbr_procs, proc_data, comm,
                                   TAG_CROSSING_PARTICLES, True)
        
        # for each neighbor processor, there is one entry in new_particles
        # containing all new cells that processor sent to us.

        self.add_entering_particles_from_neighbors(new_particles)

    cpdef add_entering_particles_from_neighbors(self, dict new_particles):
        """ Add particles that entered into this processors region.

        Parameters:
        -----------
        
        new_particles - the particles received from neighboring processors.

        Notes:
        ------

        called from: `exchange_crossing_particles_with_neighbors`

        new_particles is a dictionary keyed on processor id. Each pid
        indicating the processor from which data is received. The data
        is in the form of a dictionary, keyed on block id belonging to
        this processor and a list of particle arrays for that block.

        Algorithm:
        ----------    
        - for data from each processor
            - add the particle arrays.
        
        """
        for pid, particle_data in new_particles.iteritems():
            self.add_local_particles_to_parray(particle_data)

    cpdef add_local_particles_to_parray(self, dict particle_data):
        """ Add the given particles to the arrays as local particles. """

        cdef IntPoint bid
        cdef list parrays
        cdef ParticleArray s_parr, d_parr
        cdef int num_arrays, i, count
        cdef LongArray indices

        num_arrays = len(self.arrays_to_bin)
        
        for bid, parrays in particle_data.iteritems():
            if bid not in self.proc_map.local_block_map:
                self.proc_map.local_block_map[bid] = self.pid
                self.proc_map.block_map[bid] = self.pid
            
            for i in range(num_arrays):
                s_parr = parrays[i]
                d_parr = self.arrays_to_bin[i]

                np = s_parr.get_number_of_particles()
                ns = d_parr.get_number_of_particles()
                # set the local property to '1'

                s_parr.local[:] = 1
                d_parr.append_parray(s_parr)
                indices = LongArray(np)
                indices.set_data(numpy.arange(ns, np+ns))
                created_cells = self.insert_particles(i, indices)
                self.add_cells_to_cell_map(created_cells)
    
    def add_cells_to_cell_map(self, created_cells):
        for cid in created_cells:
            bid = IntPoint_new(cid.x/self.factor, cid.y/self.factor, cid.z/self.factor)
            if bid not in self.proc_map.local_block_map:
                self.proc_map.local_block_map[bid] = self.pid
                self.proc_map.block_map[bid] = self.pid
            if bid not in self.proc_map.cell_map:
                self.proc_map.cell_map[bid] = set([cid])
            else:
                self.proc_map.cell_map[bid].add(cid)

    cpdef update_remote_particle_properties(self, list props=None):
        """
        Update the properties of the remote particles from the respective
        processors. 

        **Parameters**
            - props - the names of the properties that are to be copied. One
            list of properties for each array that has been binned using the
            cell manager. 

        **Note**
        
             - this function will work correctly only if the particle arrays
             have not been modified since the last parallel update. If the
             particle arrays have been touched, then the start and end indices
             stored for r the particles that are remote copies will become
             invalid and the values will be copied into incorrect locations.

        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef ProcessorMap proc_map = self.proc_map
        cdef list nbr_procs = proc_map.nbr_procs
        cdef IntPoint bid
        cdef int pid, i, si, ei, dest, num_arrays

        cdef dict proc_data, blocks_to_send
        cdef dict local_block_map = proc_map.local_block_map
        cdef dict global_block_map = proc_map.block_map

        cdef list parray_list, pid_index_info, indices
        cdef str prop

        cdef ParticleArray d_parr, s_parr
        
        num_arrays = len(self.arrays_to_bin)
        
        # make sure the props array is correct.

        if props is None:
            props = [None]*len(self.arrays_to_bin)

        # sort the processors in increasing order of ranks.

        if nbr_procs.count(self.pid) == 0:
            nbr_procs.append(self.pid)
            nbr_procs = sorted(nbr_procs)

        # prepare the data for sharing

        proc_data = {}
        blocks_to_send = {}
        for pid in nbr_procs:
            proc_data[pid] = []
            blocks_to_send[pid] = set()

        # get the list of blocks to send to each processor id

        for bid in local_block_map:
            block_neighbors = []
            construct_immediate_neighbor_list(bid.data, block_neighbors, False)
            for nid in block_neighbors:
                
                if global_block_map.has_key(nid):
                    pid = global_block_map.get(nid)
                    if not pid == self.pid:
                        blocks_to_send[pid].add(bid)

        # construct the list of particle arrays to be sent to each processor

        for pid, block_list in blocks_to_send.iteritems():

            cell_list = []
            for bid in block_list:
                
                cell_list.extend(proc_map.cell_map[bid])

            parray_list = self.get_communication_data(num_arrays, cell_list)

            proc_data[pid] = parray_list
        
        # share data with all processors

        proc_data = share_data(self.pid, nbr_procs, proc_data,
                               comm, TAG_REMOTE_DATA, True)

        # now copy the new remote values into the existing local copy.

        for pid, parray_list in proc_data.iteritems():
            pid_index_info = self.remote_particle_indices[pid]

            for i in range(num_arrays):
                indices = pid_index_info[i]
                si = indices[0]
                ei = indices[1]
                s_parr = parray_list[i]

                # check for the required properties in source parray

                if props[i] is not None:
                    for prop in s_parr.properties.keys():
                        if prop not in props[i]:
                            s_parr.remove_property(prop)
                
                d_parr = self.arrays_to_bin[i]
                d_parr.copy_properties(s_parr, si, ei)

    cpdef exchange_neighbor_particles(self):
        """ Exchange neighbor particles.

        Algorithm:
        ----------

        -- use processor map to construct a list of blocks to be sent to
           neighboring processors
           
        -- construct a list of particle arrays (num_arrays) from the
           cells contained in those blocks that need to be
           communicated to each neighboring processor. The particles
           are flagged as remote and dummy. This is the communicated
           data.

        -- after communication, we receive from each neighbor, a list
           of particle arrays (num_arrays) that are remote neighbors
           for us.

        -- setup the remote particle indices that keeps track of the
           indices used to store these particles from each processor.

        -- append these particle arrays to the corresponding particle
           arrays and bin them. The cell manager's `insert_particles`
           can be used with the indices just saved.

        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef ProcessorMap proc_map = self.proc_map
        cdef dict local_block_map = proc_map.local_block_map
        cdef dict global_block_map = proc_map.block_map
        cdef dict proc_data = {}

        cdef int num_arrays = len(self.arrays_to_bin)
        cdef int i, j, pid, dest, num_nbrs
        cdef IntPoint bid, nid

        cdef list parray_list = []
        cdef list index_lists = [] 
        cdef list block_neighbors = []
        cdef list cell_list, pid_index_info
        cdef list nbr_procs = proc_map.nbr_procs

        cdef dict remote_particle_data = {}
        cdef dict blocks_to_send = {}

        cdef ParticleArray parray, s_parr, d_parr
        cdef Cell cell

        cdef list requested_cells, cell_ids, parrays, particle_counts

        cdef LongArray current_counts = LongArray(num_arrays)
        cdef LongArray indices
        self.remove_remote_particles()

        # get the list of blocks to send to each processor id

        for bid in local_block_map:
            block_neighbors = []
            construct_immediate_neighbor_list(bid.data, block_neighbors, False)
            for nid in block_neighbors:
                
                pid = global_block_map.get(nid, -1)
                if pid > -1 and not pid == self.pid:
                    if pid not in blocks_to_send:
                        proc_data[pid] = []
                        blocks_to_send[pid] = set([bid])
                    else:
                        blocks_to_send[pid].add(bid)

        nbr_procs = proc_data.keys()
        # construct the list of particle arrays to be sent to each processor

        for pid, block_list in blocks_to_send.iteritems():

            cell_list = []
            for bid in block_list:
                
                cell_list.extend(proc_map.cell_map[bid])

            parray_list = self.get_communication_data(num_arrays, cell_list)

            proc_data[pid] = parray_list

        # share data with all processors

        proc_data = share_data(self.pid, nbr_procs, proc_data, comm, 
                               TAG_REMOTE_DATA, True)

        # setup the remote particle indices
        
        self.setup_remote_particle_indices()

        for pid, parray_list in proc_data.iteritems():
            pid_index_info = self.remote_particle_indices[pid]

            for i in range(num_arrays):
                s_parr = parray_list[i]
                d_parr = self.arrays_to_bin[i]

                index_info = pid_index_info[i]
                index_info[0] = d_parr.get_number_of_particles()
                
                if s_parr.get_number_of_particles() > 0:
                    index_info[1] = index_info[0] + \
                        s_parr.get_number_of_particles()
                else:
                    index_info[1] = index_info[0]

                d_parr.append_parray(s_parr)

                # now insert the indices into the cells
                
                indices = arange_long(index_info[0], index_info[1])
                new_cells = self.insert_particles(i, indices)

    cpdef list get_cells_in_block(self, IntPoint bid):
        """ return the list of cells in the cells_dict located in block bid """
        cdef list ret = []
        cdef IntPoint p = IntPoint_new(0,0,0)
        for i in range(bid.data.x*self.factor, (bid.data.x+1)*self.factor):
            p.data.x = i
            for j in range(bid.data.y*self.factor, (bid.data.y+1)*self.factor):
                p.data.y = j
                for k in range(bid.data.z*self.factor, (bid.data.z+1)*self.factor):
                    p.data.z = k
                    if p in self.cells_dict:
                        ret.append(p.copy())
        return ret
        
    cpdef list get_particle_indices_in_block(self, IntPoint bid):
        """ Return the list of indices for particles located in block bid """
        cdef int num_arrs = len(self.arrays_to_bin)
        cdef list parray_list = [LongArray() for i in range(num_arrs)]

        for cid in self.proc_map.cell_map:
            cell = self.cells_dict[cid]
            index_lists = []
            cell.get_particle_ids(index_lists)

            for j in range(num_arrs):
                parray_list[j].extend(index_lists[j].get_npy_array())
        
        return parray_list

    cpdef list get_particles_in_block(self, IntPoint bid):
        """ Returns the list of parrays for particles located in block bid. """
        cdef list indices_list = self.get_particle_indices_in_block(bid)
        cdef list parrays = []
        for i in range(len(indices_list)):
            parrays.append(self.arrays_to_bin[i].extract_particles(indices_list[i]))
        
        return parrays

    def setup_remote_particle_indices(self):
        """ Setup the remote particle indices for this processor.

        For every neighboring processor, store two indices for every
        particle array 

        """
        cdef ProcessorMap proc_map = self.proc_map
        cdef int i, pid, num_arrays
        cdef list index_data

        self.remote_particle_indices.clear()
        num_arrays = len(self.arrays_to_bin)
        
        for pid in proc_map.nbr_procs:
            index_data = []
            for i in range(num_arrays):
                index_data.append([-1, -1])
            self.remote_particle_indices[pid] = index_data

    cdef list get_communication_data(self, int num_arrays, list cell_list):
        """ Return a list of particle arrays for the requested cells 

        Parameters:
        -----------

        num_arrays -- the number of arrays in arrays_to_bin
        
        cell_list -- the cells from which the particle data is requested

        """
        cdef list index_lists = []
        cdef list parray_list
        cdef list arrays_to_bin = self.arrays_to_bin

        cdef int i, j
        
        cdef ParticleArray s_parr, d_parr, parray
        cdef LongArray index_array
        cdef Cell cell

        parray_list = []
        for i in range(num_arrays):
            s_parr = arrays_to_bin[i]
            parray_list.append(ParticleArray(name=s_parr.name))
            
        for cid in cell_list:
            cell = self.cells_dict[cid]
            index_lists = []
            cell.get_particle_ids(index_lists)

            for j in range(num_arrays):
                s_parr = arrays_to_bin[j]
                d_parr = parray_list[j]
                
                index_array = index_lists[j]
                parray = s_parr.extract_particles(index_array)

                parray.local[:] = 0
                parray.tag[:] = get_dummy_tag()

                d_parr.append_parray(parray)
                d_parr.set_name(s_parr.name)

        return parray_list
    
    def check_jump_tolerance(self, IntPoint myid, IntPoint newid):
        """ Check if the particle has moved more than the jump tolerance """

        cdef ProcessorMap pmap = self.proc_map
        cdef IntPoint block1, block2, pdiff
        cdef cPoint cent1, cent2

        block1 = IntPoint()
        block2 = IntPoint()
        
        cent1.x = (<double>myid.x + 0.5)*self.cell_size
        cent1.y = (<double>myid.y + 0.5)*self.cell_size
        cent1.z = (<double>myid.z + 0.5)*self.cell_size

        cent2.x = (<double>newid.x + 0.5)*self.cell_size
        cent2.y = (<double>newid.y + 0.5)*self.cell_size
        cent2.z = (<double>newid.z + 0.5)*self.cell_size

        block1 = find_cell_id(cent1, pmap.block_size)
        block2 = find_cell_id(cent2, pmap.block_size)

        pdiff = block1.diff(block2)        
        
        if (abs(pdiff.x) > self.jump_tolerance or abs(pdiff.y) >
            self.jump_tolerance or abs(pdiff.z) >
            self.jump_tolerance):
            
            msg = 'Particle moved  more than one block width\n'

            msg += 'old id : (%d, %d, %d)\n'%(block1.x, block1.y,
                                               block1.z)

            msg += 'new id  : (%d, %d, %d)\n'%(block2.x, block2.y, 
                                               block2.z)

            msg += 'Block Jump Tolerance is : %s, %d\n'%(self, 
                                                         self.jump_tolerance)
            raise RuntimeError, msg

    cpdef update_cell_neighbor_information(self):
        """
        Update each cells neighbor information.
        Requires communication among proc_map neighbors.

        **Algorithm**
            
            - send and receive self information to and from all neighbors in
              proc_map.
            - invert this list to get a large dict of cells and pids containing
              those cells.
            - for each child cell update their neighbor list using this global
              cell dict.

        **Note**

            - The processor map should be up-to-date before this
              function is called.
            
        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef list nbr_procs = self.proc_map.nbr_procs
        cdef IntPoint cid
        cdef int pid
        
        cdef list sorted_nbr_proces = sorted(nbr_procs)
        
        cdef list cell_list = self.cells_dict.keys()
        
        cdef dict nbr_proc_cell_list = share_data(self.pid,
                                                  sorted_nbr_proces,
                                                  cell_list, comm, 
                                                  TAG_CELL_ID_EXCHANGE,
                                                  False)

        
        # add this pids information also here.
        nbr_proc_cell_list[self.pid] = cell_list

        # from this data, construct the nbr_cell_info.
        nbr_cell_info = {}
        for pid, cell_list in nbr_proc_cell_list.iteritems():
            for cid in cell_list:
                n_info = nbr_cell_info.get(cid)
                if n_info is None:
                    nbr_cell_info[cid] = pid
                else:
                    logger.error('Cell %s in more than one processor : %d, %d'%(
                            cid, pid, n_info))
        
        # update the neighbor information of all the children cells.
        for cell in self.cells_dict.values():
            cell.parallel_cell_info.update_neighbor_information(nbr_cell_info)

    def _find_min_max_of_property(self, prop_name):
        """ Find the minimum and maximum of the property among all arrays 
        
        Parameters:
        -----------

        prop_name -- the property name to find the bounds for

        """
        min = 1e20
        max = -1e20

        num_particles = 0
        
        for arr in self.arrays_to_bin:
            
            if arr.get_number_of_particles() == 0:
                continue
            else:
                num_particles += arr.get_number_of_particles()                

                min_prop = numpy.min(arr.get(prop_name))
                max_prop = numpy.max(arr.get(prop_name))

                if min > min_prop:
                    min = min_prop
                if max < max_prop:
                    max = max_prop

        return min, max

    cpdef dict _resolve_conflicts(self, dict data):
        """
        Resolve conflicts when multiple processors are competing for a region
        occupied by the same cell.

        Parameters:
        -----------
            
        - data - a dictionary indexed on block ids. Each entry contains a
            dictionary indexed on process id, containing the number of particles
            that proc adds to that cell.

        Algorithm:
        ----------

        - for each cell
            - if only one pid is occupying that region, that pid is the
              winner
            - sort the competing pids on pid
            - find the maximum number of particles any processor is
              contributing to the region
            - if more than one processor contribute the same number of
              particles, choose the one with the larger pid
        """
        cdef dict winning_procs = {}
        cdef IntPoint cid
        cdef list pids, num_particles, procs
        cdef int max_contribution
                
        for cid, p_data in data.iteritems():
            if len(p_data) == 1:
                winning_procs[cid] = p_data.keys()[0]
                continue
            pids = p_data.keys()
            num_particles = p_data.values()
            pids = sorted(pids)
            max_contribution = max(num_particles)
            procs = []
            for pid in pids:
                if p_data[pid] == max_contribution:
                    procs.append(pid)
            winning_procs[cid] = max(procs)

        for cid, proc in winning_procs.iteritems():
            logger.debug('Cell %s assigned to proc %d'%(cid, proc))

        return winning_procs
