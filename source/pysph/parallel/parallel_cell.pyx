"""
Classes to implement cells and cell manager for a parallel invocation.
"""
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
from pysph.base.point cimport Point, IntPoint, Point_new
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
    #cdef public ParallelCellManager cell_manager
    #cdef public Point origin
    #cdef public dict local_block_map
    #cdef public dict block_map
    #cdef public dict cell_map
    #cdef public list nbr_procs
    #cdef public int pid
    #cdef public double block_size

    def __init__(self, ParallelCellManager cell_manager=None, int pid=0,
                 Point origin=Point(0, 0, 0), double block_size=0.3, 
                 *args, **kwargs):
        self.cell_manager = cell_manager
        self.origin = Point()
        self.local_block_map = {}
        self.block_map = {}
        self.cell_map = {}
        self.nbr_procs = []
        self.pid = pid
        self.conflicts = {}
        
        if cell_manager is not None:
            self.pid = cell_manager.pid
            self.origin.x = cell_manager.origin.x
            self.origin.y = cell_manager.origin.y
            self.origin.z = cell_manager.origin.z
            
        self.block_size = block_size
    
    def __reduce__(self):
        """ Implemented to facilitate pickling of extension types. """
        d = {}
        d['origin'] = self.origin
        d['local_block_map'] = self.local_block_map
        d['block_map'] = self.block_map
        d['cell_map'] = self.cell_map
        d['pid'] = self.pid
        d['conflicts'] = self.conflicts
        d['block_size'] = self.block_size

        return (ProcessorMap, (), d)

    def __setstate__(self, d):
        """ Implemented to facilitate pickling of extension types. """
        self.origin = Point()
        org = d['origin']
        self.origin.x = org.x
        self.origin.y = org.y
        self.origin.z = org.z
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
    
    def update(self):
        """ Update the processor map with local cells maintained by cell manager
        
        Notes:
        ------
      
        The block_map data attribute is a dictionary keyed on block
        index with value being the processor owning that block

        Algorithm:
        ----------

        for all cells in cell managers cell_dict
            find the index of the cell based on binning with block size
            add this to the block_map attribute
            add the cell to the cell_map attribute
        
        """
        cdef ParallelCellManager cm = self.cell_manager
        cdef int pid = self.pid
        cdef dict block_map = {}
        cdef dict cells = cm.cells_dict
        cdef Point centroid = Point()
        cdef Cell cell
        cdef IntPoint cid, bid

        self.cell_map.clear()
        
        for cid, cell in cells.iteritems():
            cell.get_centroid(centroid)
            bid = find_cell_id(self.origin, centroid, self.block_size)
            block_map.setdefault(bid, pid)

            if self.cell_map.has_key(bid):
                self.cell_map[bid].add(cid)
            else:
                self.cell_map[bid] = set([cid])
        
        self.block_map = block_map
        self.local_block_map = copy.deepcopy(block_map)

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
        
        block_list = other_block_map.keys()
        num_blocks = len(block_list)
        
        for i in range(num_blocks):
            other_bid = block_list[i]
            other_proc = other_block_map.get(other_bid)
            
            if other_bid in self.block_map:
                # there may be a conflict
                local_proc = self.block_map.get(other_bid)
                if local_proc < 0:
                    # there is a conflict in self proc_map
                    conflicts = self.conflicts.get(other_bid)
                    if other_proc < 0:
                        # there is also conflict in other proc_map
                        conflicts.update(proc_map.conflicts.get(other_bid))
                    else:
                        conflicts.add(other_proc)
                elif other_proc < 0:
                    # there is conflict in other proc_map
                    self.conflicts[other_bid] = proc_map.conflicts.get(other_bid)
                    merged_block_map[other_bid] = -1
                elif other_proc != local_proc:
                    # conflict created at this level
                    merged_block_map[other_bid] = -1
                    self.conflicts[other_bid] = set([local_proc, other_proc])
                #else:
                # same proc in both proc_maps
                #    pass
            else:
                merged_block_map[other_bid] = other_proc
        
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
        cdef IntPoint nid
        cdef int n_pid

        # for each cell in local_pm
        for bid in local_block_map:
            nids[:] = empty_list
            
            # construct list of neighbor block ids in the proc_map.
            construct_immediate_neighbor_list(bid, nids, False)
            len_nids = len(nids)
            for i in range(len_nids):
                nid = nids[i]

                # if cell exists, collect the occupying processor id
                n_pid = block_map.get(nid, -1)
                if n_pid >= 0:
                    nb.add(n_pid)
        
        self.nbr_procs = sorted(list(nb))

    def __str__(self):
        rep = '\nProcessor Map At proc : %d\n'%(self.pid)
        rep += 'Origin : %s\n'%(self.origin)
        rep += 'Bin size : %s\n'%(self.block_size)
        rep += 'Region neighbors : %s'%(self.nbr_procs)
        return rep 
        

###############################################################################
# `ParallelCell` class.
###############################################################################
cdef class ParallelCell(Cell):
    """ Cell to be used in parallel computations. """

    def __init__(self, IntPoint id, ParallelCellManager cell_manager=None,
                 double cell_size=0.1, int jump_tolerance=1, int pid=-1): 
        cell.Cell.__init__(self, id=id, cell_manager=cell_manager,
                               cell_size=cell_size,
                               jump_tolerance=jump_tolerance)

        self.pid = pid
        self.parallel_cell_info = ParallelCellInfo(cell=self)

    cpdef Cell get_new_sibling(self, IntPoint id):
        cdef ParallelCell cell = ParallelCell(id=id,
                                         cell_manager=self.cell_manager, 
                                         cell_size=self.cell_size,
                                         jump_tolerance=self.jump_tolerance,
                                         pid=self.pid)
        return cell
    
    def __str__(self):
        return 'ParallelCell(pid=%d,id=%s,size=%g,np=%d)' %(self.pid,
                self.id, self.cell_size, self.get_number_of_particles())
    
    def __repr__(self):
        return str(self)
    
###############################################################################
# `ParallelCellRemoteCopy` class.
###############################################################################
cdef class ParallelCellRemoteCopy(ParallelCell):
    """ Cells holding information from another processor."""
    def __init__(self, id, cell_manager=None,
                 cell_size=0.1, jump_tolerance=1, pid=-1):
        ParallelCell.__init__(self, id=id, cell_manager=cell_manager,
                                  cell_size=cell_size,
                                  jump_tolerance=jump_tolerance, pid=pid)
        
        self.particle_start_indices = []
        self.particle_end_indices = []
        self.num_particles = 0
    
    def __str__(self):
        return 'ParallelCellRemoteCopy(pid=%d,id=%s,size=%g,np=%d)' %(self.pid,
                self.id, self.cell_size, self.get_number_of_particles())
    
    def __repr__(self):
        return str(self)
    

###############################################################################
# `ParallelCellManager` class.
###############################################################################
cdef class ParallelCellManager(CellManager):
    """
    Cell manager for parallel invocations.
    """
    def __init__(self, arrays_to_bin=[], min_cell_size=-1.0,
                 max_cell_size=0.5, origin=Point(0, 0, 0),
                 initialize=True, max_radius_scale=2.0,
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
                                  origin=origin,
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

        self.pc = self.parallel_controller
        self.pid = self.pc.rank

        # the processor map

        self.proc_map = ProcessorMap(cell_manager=self)

        # the load balancer

        self.load_balancer = LoadBalancer(parallel_solver=self.solver,
                                          parallel_cell_manager=self)
        self.load_balancing = load_balancing
        
        self.initial_redistribution_done = False

        # the list of remote cells that cells under this root cell are
        # adjoining. The dictionary is indexed on processor ids, and contains a
        # list of cell ids of neighbors from those processors.

        self.adjacent_remote_cells = {}

        # start and end indices of particles from each neighbor processor.

        self.remote_particle_indices = {}

        # a dict containing information about which processor each neighbor cell
        # is located.

        self.nbr_cell_info = {}

        # dict of new particles to be sent to each neighbor proc.

        self.new_particles_for_neighbors = {}
        
        # dict of new particles going into unknown region.

        self.new_region_particles = {}

        # list of new cells that were added in a iteration.

        self.new_cells_added = {}

        # the neighbor share data in a step

        self.neighbor_share_data = {}

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
        setup the origin
        setup the processor map
            
        """
        logger.debug('%s initialize called'%(self))
        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        pc = self.parallel_controller
        # exchange bounds and interaction radii (h) information.

        self.update_global_properties()

        # now setup the origin and cell size to use.

        self.setup_origin()

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

        # update the processor maps now.
        self.glb_update_proc_map()
    
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

    def setup_origin(self):
        """ Setup the origin from the global bounds.

        Notes:
        ------
        
        The origin is set as: bounds_min - max_range

        """
        self.origin = Point()

        pc = self.parallel_controller
        logger.info('(%d) Origin : %s'%(pc.rank,
                                        str(self.origin)))

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
        
        logger.info('(R=%d) cell_size=%g'%(self.pc.rank,
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
        proc_map.cell_manager = self
        
        proc_map.origin.x = self.origin.x
        proc_map.origin.y = self.origin.y
        proc_map.origin.z = self.origin.z

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
        cdef double cell_size = self.cell_size

        cdef double xc = self.local_bounds_min[0] + 0.5 * cell_size
        cdef double yc = self.local_bounds_min[1] + 0.5 * cell_size
        cdef double zc = self.local_bounds_min[2] + 0.5 * cell_size

        id = find_cell_id(self.origin, Point(xc, yc, zc), cell_size)

        cell = ParallelCell(id=id, cell_manager=self, cell_size=cell_size,
                            jump_tolerance=INT_INF(), pid=self.pid)

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

    cpdef glb_update_proc_map(self):
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
        cdef ParallelController pc = self.pc
        cdef MPI.Comm comm = pc.comm
        cdef int c_rank
        cdef ProcessorMap c_proc_map, updated_proc_map
        cdef dict block_particles = {}
        
        self.proc_map.update()

        # merge data from all children proc maps.
        for c_rank in pc.children_proc_ranks:
            c_proc_map = comm.recv(source=c_rank,
                                   tag=TAG_PROC_MAP_UPDATE)
            self.proc_map.merge(c_proc_map)
        
        # we now have partially merged data, send it to parent if not root.
        if pc.parent_rank > -1:
            comm.send(self.proc_map, dest=pc.parent_rank,
                      tag=TAG_PROC_MAP_UPDATE)

            # receive updated proc map from parent
            updated_proc_map = comm.recv(source=pc.parent_rank,
                                         tag=TAG_PROC_MAP_UPDATE)

            # set our proc data with the updated data.
            PyDict_Clear(self.proc_map.block_map)
            PyDict_Update(self.proc_map.block_map, updated_proc_map.block_map)
            
            self.proc_map.conflicts.clear()
            self.proc_map.conflicts.update(updated_proc_map.conflicts)

        # send updated data to children.
        for c_rank in pc.children_proc_ranks:
            comm.send(self.proc_map, dest=c_rank, tag=TAG_PROC_MAP_UPDATE)
        
        # now all procs have same proc_map
        # resolve any conflicts
        if self.proc_map.conflicts:
            # calculate num_blocks per proc
            blocks_per_proc = {}
            recv_procs = set([])
            procs_blocks = {}
            
            for bid in self.proc_map.conflicts:
                for proc in self.proc_map.conflicts[bid]:
                    blocks_per_proc[proc] = blocks_per_proc.get(proc, 0)
            
            # assign block to proc with least blocks then max rank
            for bid in self.proc_map.conflicts:
                candidates = list(self.proc_map.conflicts[bid])
                proc = candidates[0]
                blocks = blocks_per_proc[0]
                
                # find the winning proc for each block
                for i in range(1, len(candidates)):
                    if blocks_per_proc[i] > blocks:
                        proc = candidates[i]
                        blocks = blocks_per_proc[i]
                    elif blocks_per_proc[i] == blocks:
                        if candidates[i] > proc:
                            proc = candidates[i]
                
                blocks_per_proc[proc] += 1
                self.proc_map.block_map[bid] = proc
                
                if proc != self.pid:
                    # send to winning proc
                    if proc in procs_blocks:
                        procs_blocks[proc].append(bid)
                    else:
                        procs_blocks[proc] = [bid]
                    if bid in self.proc_map.local_block_map:
                        del self.proc_map.local_block_map[bid]
                else:
                    # recv from other conflicting procs
                    recv_procs.update(candidates)
            
            logger.info('remote_block_particles: '+str(procs_blocks))
            if self.pid in recv_procs: recv_procs.remove(self.pid)
            self.transfer_blocks_to_procs(procs_blocks, mark_remote=True,
                                          recv_procs=list(recv_procs))
            # remove the transferred particles
            self.remove_remote_particles()
            self.proc_map.conflicts.clear()
        
        # setup the region neighbors.
        self.proc_map.find_region_neighbors()

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

        cdef ParallelCellManager cell_manager = <ParallelCellManager>self

        cell_manager.remove_remote_particles()

        self.delete_empty_cells()

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        logger.debug('++++++++++++++++ UPDATE BEGIN +++++++++++++++++++++')

        # bin the particles and find the new_cells and remote_cells.

        new_block_cells, remote_block_cells = self.bin_particles()

        # create particle copies and mark those particles as remote.

        self.new_particles_for_neighbors = self.create_new_particle_copies(
            remote_block_cells, True)

        # exchange particles moving into another processor

        self.exchange_crossing_particles_with_neighbors(
                                    self.new_particles_for_neighbors)

        # remove any remote particles

        cell_manager.remove_remote_particles()

        # assign new blocks based on the new_block_cells
        
        self.assign_new_blocks(new_block_cells)

        # compute the cell sizes for binning

        self.compute_cell_size()

        # rebin the particles

        self.rebin_particles()

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        # update the processor map and resolve the conflicts

        self.glb_update_proc_map()

        # call a load balancer function.

        if cell_manager.initialized == True:
            if cell_manager.load_balancing == True:
                cell_manager.load_balancer.load_balance()

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
        cdef ProcessorMap pmap = self.proc_map
        cdef int pid
        
        self.nbr_cell_info = {}        

        #find the new configuration of the cells
        
        for cid, cell in self.cells_dict.iteritems():
            if cell.pid == self.pid:
                (<Cell>cell).update(collected_data)

        # if the base cell exists and the initial re-distribution is False,
        # add that to the list of new cells.

        if self.initial_redistribution_done is False:

            # update the global procesosr map

            c = self.cells_dict.values()[0]

            if (<Cell>c).get_number_of_particles() > 0:
                collected_data[(<Cell>c).id] = c
                
            self.cells_dict.clear()
            self.initial_redistribution_done = True
        
        # we have a list of all new cells created by the cell manager.
        for cid, cell in collected_data.iteritems():

            #find the block id to which the newly created cell belongs
            cell.get_centroid(centroid)
            block_id = find_cell_id(pmap.origin, centroid, pmap.block_size)
            
            # get the pid corresponding to the block_id

            pid = pmap.block_map.get(block_id, -1)
            if pid < 0:
                # add to new block particles

                if new_block_cells.has_key(block_id):
                    new_block_cells[block_id].append(cell)
                else:
                    new_block_cells[block_id] = [cell]
            else:
                # find to which remote processor the block belongs to and add

                if not pid == self.pid:
                    if remote_block_cells.has_key(block_id):
                        remote_block_cells[block_id].append(cell)
                    else:
                        remote_block_cells[block_id] = [cell]

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

        num_arrays = len(self.arrays_to_bin)
        
        for bid, parrays in particle_data.iteritems():
            if bid not in self.proc_map.local_block_map:
                self.proc_map.local_block_map[bid] = self.pid
                self.proc_map.block_map[bid] = self.pid
            
            for i in range(num_arrays):
                s_parr = parrays[i]
                d_parr = self.arrays_to_bin[i]

                np = s_parr.get_number_of_particles()

                # set the local property to '1'

                s_parr.local[:] = 1
                d_parr.append_parray(s_parr)

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
            construct_immediate_neighbor_list(bid, block_neighbors, False)
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

                if si != ei:
                    d_parr.copy_properties(s_parr, si, ei)

                else:
                    # this means no particles were added for this array
                    pass

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

        # prepare the data for sharing

        for pid in nbr_procs:
            proc_data[pid] = []
            blocks_to_send[pid] = set()

        # get the list of blocks to send to each processor id

        for bid in local_block_map:
            block_neighbors = []
            construct_immediate_neighbor_list(bid, block_neighbors, False)
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

        # save the shared data for later use in update remote props
    
        self.neighbor_share_data = proc_data

        # sort the processors in increasing order of ranks.

        if nbr_procs.count(self.pid) == 0:
            nbr_procs.append(self.pid)
            nbr_procs = sorted(nbr_procs)
        
        # share data with all processors

        proc_data = share_data(self.pid, nbr_procs, proc_data, comm, 
                               TAG_REMOTE_DATA, True)

        # setup the remote particle indices
        
        self.setup_remote_particle_indices()

        proc_data.pop(self.pid)
        for pid, parray_list in proc_data.iteritems():
            pid_index_info = self.remote_particle_indices[pid]

            for i in range(num_arrays):
                s_parr = parray_list[i]

                index_info = pid_index_info[i]

                d_parr = self.arrays_to_bin[i]
                current_counts.data[i] = d_parr.get_number_of_particles()

                index_info[0] = current_counts.data[i]

                if s_parr.get_number_of_particles() > 0:
                    index_info[1] = index_info[0] + \
                        s_parr.get_number_of_particles() - 1
                else:
                    index_info[1] = index_info[0]

                d_parr.append_parray(s_parr)

                # now insert the indices into the cells
                
                indices = arange_long(index_info[0], index_info[1])
                self.insert_particles(i, indices)

    cpdef list get_cells_in_block(self, IntPoint bid):
        """ return the list of cells in the cells_dict located in block bid """
        cdef list ret = []
        cdef IntPoint p = IntPoint()
        for i in range(bid.x*self.factor, (bid.x+1)*self.factor):
            p.x = i
            for j in range(bid.y*self.factor, (bid.y+1)*self.factor):
                p.y = j
                for k in range(bid.z*self.factor, (bid.z+1)*self.factor):
                    p.z = k
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

    cpdef Cell get_new_cell(self, IntPoint id):
        """get a new parallel cell"""
        return ParallelCell(id=id, cell_manager=self,
                                cell_size=self.cell_size,
                                jump_tolerance=self.jump_tolerance,
                                pid=self.pid)
    
    cpdef Cell get_new_cell_for_copy(self, IntPoint id, int pid):
        """
        """
        return ParallelCellRemoteCopy(id=id, cell_manager=self, 
                                      cell_size=self.cell_size,
                                      jump_tolerance=self.jump_tolerance,
                                      pid=pid)

    def check_jump_tolerance(self, IntPoint myid, IntPoint newid):
        """ Check if the particle has moved more than the jump tolerance """

        cdef ProcessorMap pmap = self.proc_map
        cdef IntPoint block1, block2, pdiff
        cdef Point cent1, cent2

        cent1 = Point()
        cent2 = Point()

        block1 = IntPoint()
        block2 = IntPoint()
        
        cent1.x = self.origin.x + (<double>myid.x + 0.5)*self.cell_size
        cent1.y = self.origin.y + (<double>myid.y + 0.5)*self.cell_size
        cent1.z = self.origin.z + (<double>myid.z + 0.5)*self.cell_size

        cent2.x = self.origin.x + (<double>newid.x + 0.5)*self.cell_size
        cent2.y = self.origin.y + (<double>newid.y + 0.5)*self.cell_size
        cent2.z = self.origin.z + (<double>newid.z + 0.5)*self.cell_size

        block1 = find_cell_id(pmap.origin, cent1, pmap.block_size)
        block2 = find_cell_id(pmap.origin, cent2, pmap.block_size)

        pdiff = block1.diff(block2)        
        
        if (abs(pdiff.x) > self.jump_tolerance or abs(pdiff.y) >
            self.jump_tolerance or abs(pdiff.z) >
            self.jump_tolerance):
            
            msg = 'Particle moved  more than one block width\n'

            msg += 'old id : (%d, %d, %d)\n'%(block1.id.x, block1.id.y,
                                               block1.id.z)

            msg += 'new id  : (%d, %d, %d)\n'%(block2.id.x, block2.id.y, 
                                               block2.id.z)

            msg += 'Block Jump Tolerance is : %s, %d\n'%(self, 
                                                         self.jump_tolerance)
            raise RuntimeError, msg    

    cpdef find_adjacent_remote_cells(self):
        """
        Finds all cells from other processors that are adjacent to cells handled
        by this processor. 
        
        **Note**
            - assumes the neighbor information of all cells to be up-to-date.

        """
        self.adjacent_remote_cells.clear()
        self.remote_particle_indices.clear()
        
        cdef dict arc = {}
        cdef dict cell_dict
        cdef IntPoint cid
        cdef int pid
        cdef ParallelCellInfo pci
        cdef dict info

        for cell in self.cells_dict.values():
            pci = cell.parallel_cell_info
            
            for cid, pid in pci.neighbor_cell_pids.iteritems():
                if pid == self.pid:
                    continue

                if PyDict_Contains(arc, pid) != 1:
                    info = {}
                    arc[pid] = info
                else:
                    info = <dict>PyDict_GetItem(arc, pid)
                # add cellid to the list of cell from processor pid.
                info[cid] = None

        # copy temp data in arc into self.adjacent_remote_cells
        for pid, cell_dict in arc.iteritems():
            self.adjacent_remote_cells[pid] = cell_dict.keys()            

        # setup the remote_particle_indices data.
        # for every adjacent processor, we store 
        # two indices for every parray that is 
        # being binned by the cell manager.
        rpi = self.remote_particle_indices
        for pid in self.proc_map.nbr_procs:
            data = []
            for i in range(len(self.arrays_to_bin)):
                data.append([-1, -1])
            rpi[pid] = data

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
        cdef ProcessorMap p_map = self.proc_map
        cdef list nbr_procs = p_map.nbr_procs
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

        # store to nbr_cell_info for later use.
        self.nbr_cell_info.clear()
        self.nbr_cell_info.update(nbr_cell_info)

        # update the adjacent cell information of the root.
        self.find_adjacent_remote_cells()

    cpdef dict _get_cell_data_for_neighbor(self, list cell_list,
                                           list props=None):
        """
        Return new particle arrays created for particles contained in each of
        the requested cells.

        **Parameters**
        
            - cell_list - the list of cells, whose properties are requested.
            - props - a list whose entries are as follows: for each particle
              array that has been binned, a list of properties required of that
              particle array, or None if all properties are required.

        **Algorithm**

            - Do the following operations for each ParticleArray that is being
              binned by the cell manager.
            - Collect in a LongArray particle indices from all the cells in
              cell_list. Along with this also maintain a LongArray containing
              the number of partilces in each cell (in the same order as the
              cell_list).
            - Extract the required particles (indices collected above) into a
              ParticleArray.
            - Return the ParticleArrays and the LongArrays.
              
        """
        cdef IntPoint cid
        cdef dict data = {}
        cdef int i, num_cells, j, num_arrays
        cdef list index_lists = list()
        cdef list parrays = list()
        cdef int num_particles
        cdef int num_index_lists
        cdef ParticleArray parr, parr_new
        cdef LongArray index_array, ca
         
        cdef list collected_indices = list()
        cdef list particle_counts = list()
        cdef list particle_arrays = list()
        cdef LongArray pcount_temp = LongArray()

        # make sure the properties have been specified properly.
        num_arrays = len(self.arrays_to_bin)
        
        if props is not None:
            if len(props) != len(self.arrays_to_bin):
                msg = 'Need information for each particle array'
                logger.error(msg)
                raise SystemError, msg
        else:
            props = [None]*len(self.arrays_to_bin)

        num_cells = len(cell_list)
        
        # intialize some temp data structures.
        for i in range(len(self.arrays_to_bin)):
            #collected_indices.append(LongArray())
            particle_counts.append(LongArray(num_cells))

        
        # now collect all the particle indices that need to be returned.
        for i in range(num_cells):
            cid = cell_list[i]
            c = self.cells_dict[cid]
            
            # make sure the requested cell is local.
            if c.pid != self.pid:
                msg = 'Data being requested for cell %s which is remote in %d'%(
                    c.id, self.pid)
                logger.error(msg)
                raise SystemError, msg
            c.get_particle_counts_ids(collected_indices, pcount_temp)

            # now find the number of particles that were added to each particle
            # array that is being binned.
            for j from 0 <= j < num_arrays:
                ca = particle_counts[j]
                ca.data[i] = pcount_temp.data[j]
                # reset the value of the next use.
                pcount_temp.data[j] = 0

        # now extract the required particles from the local particle arrays and
        # return. 
        for i in range(num_arrays):
            parr = self.arrays_to_bin[i]
            parr_new = parr.extract_particles(collected_indices[i], props[i])
            parr_new.set_name(parr.name)
            parr_new.local[:] = 0
            parr_new.tag[:] = get_dummy_tag()
            parrays.append(parr_new)

        data['parrays'] = parrays
        data['pcounts'] = particle_counts
        return data
        
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

        if num_particles == 0:
            return 1e20, -1e20

        return min, max

    def compute_neighbor_counts(self):
        """
        Recompute the neighbor counts of all local cells. 
        This does not invovle any global communications. It assumes that the
        data available is up-to-date.

        Note:
        ------

        DO NOT PUT ANY GLOBAL COMMUNICATION CALLS HERE. This function may
        be asynchronously called to update the neighbor counts after say
        transfering cell to another process.

        """
        for c in self.cells_dict.values():
            c.parallel_cell_info.compute_neighbor_counts()

    def update_neighbor_information_local(self):
        """ Update neighbor information locally. """
        for c in self.cells_dict.values():
            c.parallel_cell_info.update_neighbor_information_local()
            c.parallel_cell_info.compute_neighbor_counts()

    cpdef dict _resolve_conflicts(self, dict data):
        """
        Resolve conflicts when multiple processors are competing for a region
        occupied by the same cell.

        **Parameters**
            
            - data - a dictionary indexed on cellids. Each entry contains a
            dictionary indexed on process id, containing the number of particles
            that proc adds to that cell.

        **Algorithm**

            - for each cell
                - if only one pid is occupying that region, that pid is the
                  winner. 
                - sort the competing pids on pid.
                - find the maximum number of particles any processor is
                  contributing to the region.
                - if more than one processor contribute the same number of
                  particles, choose the one with the larger pid.
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

    def update_property_bounds(self):
        """ Updates the min and max values of all properties in all
        particle arrays that are being binned by this cell manager.

        Not sure if this is the correct place for such a function.

        """
        pass

###############################################################################
# `ParallelCellInfo` class.
###############################################################################
cdef class ParallelCellInfo:
    """
    Class to hold information to be maintained with any parallel cell.

    This will be typically used only for those cells that are involved in the
    parallel level. Any cell under this cell behaves just like a serial cell.

    """

    #Defined in the .pxd file
    #cdef ParallelCellManager cell_manager
    #cdef public Cell cell
    #cdef public dict neighbor_cell_pids
    #cdef public dict remote_pid_cell_count
    #cdef public int num_remote_neighbors
    #cdef public int num_local_neighbors

    def __init__(self, cell=None):
        self.cell = cell
        self.cell_manager = self.cell.cell_manager
        self.neighbor_cell_pids = {}
        self.remote_pid_cell_count = {}
        self.num_remote_neighbors = 0
        self.num_local_neighbors = 0

    def update_neighbor_information(self, dict glb_nbr_cell_pids):
        """
        Updates the remote neighbor information from the glb_nbr_cell_pids.

        glb_nbr_cell_pids also contains cell ids from the same processor.

        Notes:
        -------

        The neighbor_cell_pids attribute is a dict keyed on cell ids
        and with a value of the processor id that owns this cell.

        A call to this function finds all neighbors for this cell (27)
        and stores the processor id associated with this cell if it
        exists in the glb_nbr_cell_pids dict passed.

        At the end of this call, the self.neighbor_cell_pids dict has
        all neighbors and their processor ids if any.

        """
        cdef IntPoint id = self.cell.id
        cdef IntPoint nbr_id
        cdef int nbr_pid
        cdef list nbr_ids = []
        construct_immediate_neighbor_list(id, nbr_ids, False)

        # clear previous information.

        self.neighbor_cell_pids.clear()
        
        for nbr_id in nbr_ids:
            if glb_nbr_cell_pids.has_key(nbr_id):
                nbr_pid = glb_nbr_cell_pids.get(nbr_id)
                self.neighbor_cell_pids[nbr_id] = nbr_pid

    def update_neighbor_information_local(self):
        """
        Updates neighbor information using just local data.

        This uses the cell_manager's cells_dict to search for neighbors.

        """
        cdef IntPoint cid
        cdef dict nbr_cell_pids = {}
        nbr_cell_pids.update(self.neighbor_cell_pids)

        for cid in nbr_cell_pids:
            c = self.cell_manager.cells_dict.get(cid)
            if c is not None:
                self.neighbor_cell_pids[cid] = c.pid
        
    def compute_neighbor_counts(self):
        """ Find the number of local and remote neighbors of this cell. """
        self.remote_pid_cell_count.clear()
        self.num_remote_neighbors = 0
        self.num_local_neighbors = 0
        cdef int mypid = self.cell.pid
        cdef int pid
        cdef IntPoint cid

        for cid, pid in self.neighbor_cell_pids.iteritems():
            if pid == mypid:
                self.num_local_neighbors += 1
            else:
                self.num_remote_neighbors += 1
                self.remote_pid_cell_count[pid] = self.remote_pid_cell_count.get(pid,0)+1  


