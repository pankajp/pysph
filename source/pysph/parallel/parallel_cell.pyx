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
from pysph.base.point cimport Point, IntPoint
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
cdef dict share_data(int mypid, list sorted_procs, object data, MPI.Comm comm,
                     int tag=0, bint multi=False):
    """
    Shares the given data among the processors in nbr_proc_list. Returns a
    dictionary containing data from each other processor.

    Parameters:
    -----------
    
    mypid - pid where the function is being called.
    sorted_procs - list of processors to share data with.
    data - the data to be shared.
    comm - the MPI communicator to use.
    tag - a tag for the MPI send/recv calls if needed.
    multi - indicates if 'data' contains data specific to each nbr_proc or not.
            True -> separate data for each nbr proc, 
            False -> Send same data to all nbr_procs.

    Notes:
    ------

    `data` is assumed to be a dictionary keyed on processor id to
     which some message must be sent. The value is the message.

     The function returns the dictionary `proc_data` which is keyed on
     processor id. The values are the messages recieved from the
     respective processor.

     The sorted processor list must contain the calling pid (mypid),
     else no data is shared! (not sure why though)

    """

    cdef int pid, dest, i, num_procs
    num_procs = len(sorted_procs)

    if mypid not in sorted_procs:
        return {}

    cdef dict proc_data = {}
    
    for i in range(num_procs):
        pid = sorted_procs[i]
        proc_data[pid] = {}
    
    for i in range(num_procs):
        pid = sorted_procs[i]

        if pid == mypid:

            for dest in sorted_procs:

                if dest == mypid:
                    continue
                else:

                    if multi:
                        comm.send(data[dest], dest=dest, tag=tag)
                    else:
                        comm.send(data, dest=dest, tag=tag)
        else:
            recv_data = comm.recv(source=pid, tag=tag)
            proc_data[pid] = recv_data

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
    the bin ids just created. For example, if a bin on one prcocessor
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
        """
        Contructor.

        """
        self.cell_manager = cell_manager
        self.origin = Point()
        self.local_block_map = {}
        self.block_map = {}
        self.cell_map = {}
        self.nbr_procs = []
        self.pid = pid

        if cell_manager is not None:
            self.pid = cell_manager.pid
            self.origin.x = cell_manager.origin.x
            self.origin.y = cell_manager.origin.y
            self.origin.z = cell_manager.origin.z
            
        self.block_size = block_size
    
    def __reduce__(self):
        """
        Implemented to facilitate pickling of extension types.
        """
        d = {}
        d['origin'] = self.origin
        d['local_block_map'] = self.local_block_map
        d['block_map'] = self.block_map
        d['cell_map'] = self.cell_map
        d['pid'] = self.pid
        d['block_size'] = self.block_size

        return (ProcessorMap, (), d)

    def __setstate__(self, d):
        """
        """
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
        cdef IntPoint id = IntPoint()
        cdef Cell cell
        cdef IntPoint cid
        
        for cid, cell in cells.iteritems():
            cell.get_centroid(centroid)
            id = find_cell_id(self.origin, centroid, self.block_size)
            block_map.setdefault(id, pid)

            if self.cell_map.has_key(id):
                self.cell_map[id].append(cell)
            else:
                self.cell_map[id] = [cell]
        
        self.block_map = block_map
        self.local_block_map = copy.deepcopy(block_map)

    cpdef merge(self, ProcessorMap proc_map):
        """
        Merge data from other processors proc map into this processor map.

        Parameters:
        -----------

        proc_map -- processor map from another processor.

        Algorithm:
        ----------

        for all bins in the other processor maps bin list
            get the pid of the processor containing that bin
            
            if this cell contains the bin index
                update the set of processors that have this bin index in m_pm
            else 
                add an entry in m_pm for this bin index and this pid

        """
        cdef dict merged_block_map = self.block_map
        cdef dict other_block_map = proc_map.block_map
        cdef IntPoint other_id
        cdef list block_list
        cdef int i, num_blocks, other_proc

        block_list = other_block_map.keys()
        num_blocks = len(block_list)

        for i in range(num_blocks):
            other_id = block_list[i]
            other_proc = <int>PyDict_GetItem(other_block_map, other_id)
                            
            merged_block_map[other_id] = other_proc
        
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
        cdef int num_local_bins, i, len_nids, j
        cdef IntPoint nid
        cdef set n_pids

        # for each cell in local_pm

        for id in local_block_map:
            nids[:] = empty_list
            
            # constructor list of neighbor cell ids in the proc_map.

            construct_immediate_neighbor_list(id, nids)
            len_nids = len(nids)
            for i in range(len_nids):
                nid = nids[i]

                # if cell exists, collect all occupying processor ids

                n_pids = block_map.get(nid)
                if n_pids is not None:
                    nb.update(n_pids)
        
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
    """
    Cell to be used in parallel computations.
    """
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
        self.max_radius_scale = max_radius_scale
        self.pid = 0

        # set the parallel controller.

        if parallel_controller is None:
            self.parallel_controller = ParallelController(cell_manager=self)
        else:
            self.parallel_controller = parallel_controller

        self.pc = self.parallel_controller
        self.pid = self.pc.rank

        self.proc_map = ProcessorMap(cell_manager=self)
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

        # adjacent processors - list of processors that share a cell boundary
        # with this processor.

        self.adjacent_processors = []

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

        find the bounds of the data. - global/parallel.
        set the origin to be used.
        find the max and min cell sizes from the particle radii -
        global/parallel.
        perform a proper update on the data. 
        May have to do a conflict resolutions at this point.
            
        """
        logger.debug('%s initialize called'%(self))
        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        pc = self.parallel_controller

        # exchange bounds and interaction radii (h) information.

        self.initial_info_exchange()

        # now setup the origin and cell size to use.

        self.setup_origin()

        if self.min_cell_size > 0.0:
            self.cell_size = self.min_cell_size
        else:
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

    def initial_info_exchange(self):
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
        bounds_range = [0, 0, 0]
        bounds_range[0] = self.glb_bounds_max[0] - self.glb_bounds_min[0]
        bounds_range[1] = self.glb_bounds_max[1] - self.glb_bounds_min[1]
        bounds_range[2] = self.glb_bounds_max[2] - self.glb_bounds_min[2]
        
        max_range = max(bounds_range)

        self.origin.x = self.glb_bounds_min[0] - max_range
        self.origin.y = self.glb_bounds_min[1] - max_range
        self.origin.z = self.glb_bounds_min[2] - max_range

        pc = self.parallel_controller
        logger.info('(%d) Origin : %s'%(pc.rank,
                                        str(self.origin)))

    cpdef double compute_cell_size(self, double min_cell_size, 
                                   double max_cell_size):
        """ Setup the cell size to use from the 'h' values.
        
        Parameters:
        -----------
        
        min_cell_size -- the minimum cell size to use
        max_cell_size -- the maximum cell size to use

        Notes:
        ------

        The cell size is set to 2*self.max_radius_scale*self.glb_min_h

        """
        self.cell_size = 2*self.max_radius_scale*self.glb_min_h
        logger.info('(R=%d) cell_size=%g'%(self.pc.rank, self.cell_size))
        return self.cell_size

    def setup_processor_map(self):
        """ Setup the processor map 

        Notes:
        ------
        
        The block size for the processor map is set to thrice the cell
        size. This could be incorrect as if different processors have
        diffirent cell sizes, the blocks would be different.
        
        """
        proc_map = self.proc_map
        proc_map.cell_manager = self
        
        proc_map.origin.x = self.origin.x
        proc_map.origin.y = self.origin.y
        proc_map.origin.z = self.origin.z

        proc_map.pid = self.parallel_controller.rank
        
        cell_size = self.cell_size
        proc_map.block_size = cell_size*3.0

    def _build_cell(self):
        """ Build a cell containing all the particles.
        
        Notes:
        ------
        
        This function is similar to the function in the CellManager,
        except that the cells used are the Parallel variants.

        """
        cell_size = self.cell_size

        # create a cell and add to it all the particles.

        cell = ParallelCell(id=IntPoint(0, 0, 0), cell_manager=self,
                                     cell_size=cell_size,
                                     jump_tolerance=INT_INF(),
                                     pid=self.pid)

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
        self.cells_dict[IntPoint(0, 0, 0)] = cell

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
        
        self.proc_map.update()

        # merge data from all children proc maps.

        for c_rank in pc.children_proc_ranks:
            c_proc_map = comm.recv(source=c_rank, 
                                   tag=TAG_PROC_MAP_UPDATE)
            self.proc_map.merge(c_proc_map)

        # we now have partially merged data, send it to parent is not root.

        if pc.parent_rank > -1:
            comm.send(self.proc_map, dest=pc.parent_rank,
                      tag=TAG_PROC_MAP_UPDATE)

            # receive updated proc map from parent

            updated_proc_map = comm.recv(source=pc.parent_rank,
                                         tag=TAG_PROC_MAP_UPDATE)

            # set our proc data with the updated data.

            PyDict_Clear(self.proc_map.block_map)
            PyDict_Update(self.proc_map.block_map, updated_proc_map.block_map)

        # send updated data to children.

        for c_rank in pc.children_proc_ranks:
            comm.send(self.proc_map, dest=c_rank, 
                      tag=TAG_PROC_MAP_UPDATE)

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
  
    cpdef find_adjacent_remote_cells(self):
        """
        Finds all cells from other processors that are adjacent to cells handled
        by this processor. 

        This also updates the adjacent_processors list.
        
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
        
        self.adjacent_processors[:] = self.adjacent_remote_cells.keys()
        # add my pid also into adjacent_processors.
        self.adjacent_processors.append(self.pid)

        self.adjacent_processors[:] = sorted(self.adjacent_processors)

        # setup the remote_particle_indices data.
        # for every adjacent processor, we store 
        # two indices for every parray that is 
        # being binned by the cell manager.
        rpi = self.remote_particle_indices
        for pid in self.adjacent_processors:
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

    cpdef int cells_update(self) except -1:
        """ Update particle information """

        cdef ParallelCellManager cell_manager = <ParallelCellManager>self

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        logger.debug('++++++++++++++++ UPDATE BEGIN +++++++++++++++++++++')

        # bin the particles and find the new_cells and remote_cells.

        new_block_cells, remote_block_cells = self.bin_particles()

        # clear the list of new cells added.

        self.new_cells_added.clear()

        # create a copy of the particles in the new cells and mark those
        # particles as remote.

        self.new_particles_for_neighbors = self.create_new_particle_copies(
            remote_block_cells)

        self.new_region_particles = self.create_new_particle_copies(
            new_block_cells)

        # exchange particles moving into a region assigned to a known
        # processor's region. 

        self.exchange_crossing_particles_with_neighbors(remote_block_cells,
            self.new_particles_for_neighbors
            )
        
        # remove all particles with local flag set to 0.

        cell_manager.remove_remote_particles()

        # assign new blocks based on the new_block_cells

        self.assign_new_blocks(new_block_cells, self.new_region_particles)

        # all new particles entering this processors region and in regions
        # assigned to this processor have been added to the respective particles
        # arrays. The data in the particles arrays is stale and the indices
        # invalid. Perform a top down insertion of all particles

        self.bin_particles_top_down()

        # now update the processor map.

        cell_manager.glb_update_proc_map()
        
        # re-compute the neighbor information
        #self.update_cell_neighbor_information()

        # wait till all processors have reached this point.

        self.parallel_controller.comm.Barrier()

        # call a load balancer function.

        if cell_manager.initialized == True:
            if cell_manager.load_balancing == True:
                cell_manager.load_balancer.load_balance()

        # at this point each processor has all the real particles it is supposed
        # to handle. We can now exchange neighbors particle data with the
        # neighbors.adjacent_remote_cells

        self.exchange_neighbor_particles()

        logger.debug('+++++++++++++++ UPDATE DONE ++++++++++++++++++++')
        return 0

    cpdef bin_particles_top_down(self):
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
        num_arrays = len(self.arrays_to_bin)
        cdef ParticleArray parr
        cdef LongArray indices
        cdef Cell cell
        
        for i in range(len(self.arrays_to_bin)):
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

        This function is callled just after the base cell is built
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
        cdef Cell cell
        cdef ProcessorMap pmap = self.proc_map
        
        self.nbr_cell_info = {}        

        #find the new configuration of the cells
        
        for cid, cell in self.cells_dict.iteritems():
            if cell.pid == self.pid:
                (<Cell>cell).update(collected_data)

        # if the base cell exists and the initial re-distribution is False,
        # add that to the list of new cells.

        if self.initial_redistribution_done is False:

            # update the processor map once - no neighbor information is
            # available at this point.

            self.glb_update_proc_map()

            c = self.cells_dict.values()[0]
            if (<Cell>c).get_number_of_particles() > 0:
                collected_data[(<Cell>c).id] = c

            # remove it from the cell_dict

            self.cells_dict.clear()
            self.initial_redistribution_done = True
            
        # we have a list of all new cells created by the cell manager.
        for cid, cell in collected_data.iteritems():

            #find the block id to which the newly created cell belongs

            block_id = find_cell_id(pmap.origin, cell.centroid, pmap.block_size)
                        
            #get the pid corresponding to the block_id
            
            pid = pmap.block_map.get(block_id)
            if pid is None:

                #add to new block particles

                if new_block_cells.has_key(block_id):
                    new_block_cells[block_id].append(cell)
                else:
                    new_block_cells[block_id] = [cell]
            else:

                #find to which remote processor the block belongs and add
                
                if not pid == self.pid:
                    if remote_block_cells.has_key(block_id):
                        remote_block_cells[block_id].append(cell)
                    else:
                        remote_block_cells[block_id] = [cell]

        return new_block_cells, remote_block_cells
                       
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
    
    cpdef create_new_particle_copies(self, dict cell_dict_to_copy):
        """ Make copies of all particles in the given cell dict.
              
        Parameters:
        -----------

        cell_dict_to_copy -- the new cell dictionary to copy

        Algorithm:
        -----------
        
            - for each cell in cell_dict
                - get indices of all particles in this cell.
                - remove any particles marked as non-local from this set of
                  particles. [not required - infact this is incorrect]
                - create a particle array for each set of particles, including
                  in it only the required particles.
                - mark these particles as remote in the main particle array.

        Notes:
        -------

        The function is called from `cells_update` to make copies for the 
        new and remote block particles.

        For each new or remote block, the function creates a big
        particle array for all particle indices contained in cells
        under the block. Separate such arrays are created for
        different arrays in `arrays_to_bin`.

        The return value is a dictionary keyed on block id with the
        list of particle arrays for that block as value. The lenght if
        this list is of course the number of arrays in `arrays_to_bin`.
                
        """
        cdef dict copies = {}
        cdef IntPoint bid
        cdef Cell c
        cdef list cell_list, index_lists
        cdef int i, num_index_lists
        cdef LongArray index_array

        cdef ParticleArray s_parr, d_parr
        cdef list parray_list

        cdef int num_arrays = len(self.arrays_to_bin)
        
        for bid, cell_list in cell_dict_to_copy.iteritems():

            parray_list = []
            for i in range(num_arrays):
                parray_list.append(ParticleArray())

            for c in cell_list:
                index_lists = []
                c.get_particle_ids(index_lists)

                num_index_lists = len(index_lists)
                for i in range(num_index_lists):
                    s_parr = parray_list[i]

                    index_array = index_lists[i]
                    d_parr = c.arrays_to_bin[i]
                
                    s_parr.append(d_parr.extract_particles(index_array))
                    s_parr.set_name(d_parr.name)
                
                    # mark the particles as remote in the particle array.

                    d_parr.set_flag('local', 0, index_array)

                    # also set them as dummy particles.

                    d_parr.set_tag(get_dummy_tag(), index_array)

            copies[bid] = parray_list

        return copies
        
    cpdef assign_new_blocks(self, dict new_block_dict, dict new_particles):
        """
        Assigns cells created in new regions (i.e. regions not assigned to any
        processor) to some processor. Conflicts are resolved using a
        deterministic scheme which returns the same winner in all processors.
        
        Parameters:
        -----------

        new_block_dict -- a dictionary keyed on block id with a list of
                          cells to be sent to the processor responsible for
                          that block.

        new_particles -- a dictionary keyed on block id with a list of
                         particle arrays, one for each array in
                         arrays_to_bin.

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
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef ProcessorMap proc_map = self.proc_map
        cdef list nbr_procs = self.proc_map.nbr_procs
        cdef int pid
        cdef IntPoint nid, bid
        cdef ParticleArray p
        cdef int np
        cdef dict winning_procs
        cdef list parrays, block_neighbor_list
        cdef dict candidates = {}
        cdef dict proc_data = dict()

        for nbr_pid in nbr_procs:
            proc_data[nbr_pid] = {}

        for bid in new_particles.keys():
            candidates[bid] = set()
            block_neighbor_list = []
            construct_immediate_neighbor_list(bid, block_neighbor_list, False)
            for nid in block_neighbor_list:
                if proc_map.block_map.has_key(nid):
                    candidates[bid].add(proc_map.block_map[nid])

            if len(candidates[bid]) > 1: 

                # need to resolve conflicts
                pid = min(candidates[bid])
                proc_data[pid][bid] = new_particles[bid]

            else:

                #share data with lone neighbor
                pid = candidates[bid].pop()                
                proc_data[pid][bid] = new_particles[bid]

        proc_data = share_data(self.pid, 
                               nbr_procs,
                               proc_data,
                               comm,
                               TAG_NEW_CELL_PARTICLES, 
                               True)

        # now add the particles in the cells assigned to self into
        # corresponding particle arrays.

        for pid, particle_data in proc_data.iteritems():
            self.add_local_particles_to_parray(particle_data)

    cpdef exchange_crossing_particles_with_neighbors(self, dict remote_cells,
                                                     dict particles):
        """
        Send all particles that crossed into a known neighbors region,
        receive particles that got into our region from a neighbors.

        Parameters:
        -----------
        
        remote_cells -- dictionary keyed on block id with a list of cells to
                        send to that block.

        particles    -- dictionary keyed on block id with a list of particle 
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
            - exchange this data with all processors in adjacent_processors.
            - we now have a set of particles (in particle arrays) that entered
              our domain.
            - add these particles as real particles into the corresponding
              particle arrays.

        **Data sent to and received from each processor**

            - 'cell_id' - the cell that they have to create.
            - 'particles' - the particles they have to add to the said cells.
        
        """
        cdef MPI.Comm comm = self.parallel_controller.comm
        cdef dict proc_data = {}
        cdef int proc_id, num_particles
        cdef IntPoint cid, bid
        cdef dict p_data
        cdef list parray_list
        cdef ParticleArray parray
        cdef ProcessorMap proc_map = self.proc_map
        cdef list adjacent_processors

        # create one entry here for each neighbor processor.
        
        adjacent_processors = proc_map.nbr_procs

        for proc_id in adjacent_processors:
            proc_data[proc_id] = {}

        for bid, parray_list in particles.iteritems():

            #find the processor to which the block belongs

            pid = proc_map.block_map.get(bid)            
            
            proc_data[pid][bid] = parray_list

        new_particles = share_data(self.pid, adjacent_processors,
                                   proc_data, comm, TAG_CROSSING_PARTICLES, 
                                   True)
        
        # for each neigbor processor, there is one entry in new_particles
        # containing all new cells that processor sent to us.

        self.add_entering_particles_from_neighbors(new_particles)

    cpdef add_entering_particles_from_neighbors(self, dict new_particles):
        """ Add particles that entered into this processors region.

        Parameters:
        -----------
        
        new_particles - a dictionary keyed on processor ids with value
                        another dictionary of block ids that have a
                        list of particle arrays that have moved into 
                        this processors region.

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
        
        for bid, parrays in particle_data.iteritems():
            count = 0

            num_arrays = len(self.arrays_to_bin)
            for i in range(num_arrays):
                s_parr = parrays[i]
                d_parr = self.arrays_to_bin[i]
                
                # set the local property to '1'
                s_parr.local[:] = 1
                d_parr.append_parray(s_parr)
                count += s_parr.get_number_of_particles()

    def setup_remote_particle_indices(self):
        """ Setup the remote particle indices for this processor.

        For every neighboring processor, store two indices for every
        particle array 

        """
        cdef ProcessorMap proc_map = self.proc_map
        cdef int i, pid
        cdef list data
        
        for pid in proc_map.nbr_procs:
            data = []
            for i in range(len(self.arrays_to_bin)):
                data.append([-1, -1])
            self.remote_particle_indices[pid] = data

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
        cdef dict neighbor_share_data = self.neighbor_share_data
        cdef dict proc_data
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
        
        # share data with all processors

        proc_data = share(self.pid, nbr_procs, proc_data, comm, 
                          TAG_REMOTE_DATA, True)

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

        -- after communication, we recieve from each neighbor, a list
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

        cdef int num_arrays = len(self.arrays_to_bin)
        cdef int i, j, pid, dest, num_nbrs
        cdef IntPoint bid, nid, pid

        cdef list parray_list = []
        cdef list index_lists = [] 
        cdef list block_neighbors = []
        cdef list nbr_procs = proc_map.nbr_procs
        cdef list cell_list, parray_list

        cdef dict remote_particle_data = {}
        cdef dict blocks_to_send = {}
        cdef dict local_block_map = proc_map.local_block_map
        cdef dict global_block_map = proc_map.block_map

        cdef ParticleArray parray, s_parr, d_parr
        cdef Cell cell

        cdef list requested_cells, cell_ids, parrays, particle_counts

        cdef int num_arrays = len(self.arrays_to_bin)
        cdef LongArray current_counts = LongArray(num_arrays)
        cdef LongArray indices
        
        # prepare the data for sharing
        for pid in nbr_procs:
            proc_data[pid] = []
            blocks_to_send[pid] = []

        # get the list of blocks to send to each processor id

        for bid in local_block_map:
            block_neighbors = []
            construct_immediate_neighbor_list(bid, block_neighbors, False)
            for nid in block_neighbors:
                pid = global_block_map.get(nid)
                if pid and not pid == self.pid:
                    blocks_to_send[pid].append(bid)

        # construct the list of particle arrays to be sent to each processor

        for pid, block_list in blocks_to_send.iteritems():

            parray_list = []
            for i in range(num_arrays):
                parray_list.append(ParticleArray())

            for bid in block_list:

                cell_list = proc_map.cell_map[bid]
                for cell in cell_list:

                    index_lists = []
                    cell.get_particle_ids(index_lists)

                    for j in range(num_arrays):
                        s_parr = self.arrays_to_bin[j]
                        d_parr = parray_list[j]
                                
                        parray = s_parr.extract(index_lists[j])
                        parray.local[:] = 0
                        parray.tag[:] = get_dummy_tag()

                        d_parr.append(parray)
                        d_parr.set_name(s_parr.name)

            proc_data[pid] = parray_list

        # save the shared data for later use in update remote props
    
        self.neighbor_share_data = proc_data

        # sort the processors in increasing order of ranks.

        if nbr_procs.count(self.pid) == 0:
            nbr_procs.append(self.pid)
            nbr_procs = sorted(nbr_procs)
        
        # share data with all processors

        proc_data = share(self.pid, nbr_procs, proc_data, comm, 
                          TAG_REMOTE_DATA, True)

        # setup the remote particle indices
        
        self.setup_remote_particle_indices()

        for pid, parray_list in proc_data:
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
                
                indices = LongArray(index_info[0], index_info[1])
                self.insert_particles(i, indices)                    

        #We now have a list of arrays from other processors that we require
        #as neighbors. The particles in these arrays must be binned with
        #respect to the cell size on this processor and must be marked as
        #remote. Moreover, we need to record from where these particles came
        #from as when we update remote particle properties, the sending 
        #processor must update the right properties.

        #num_nbrs = len(nbr_procs)
        #for i in range(num_nbrs):

        #    pid = nbr_procs[i]
        #    if pid == self.pid:

        #        for j in range(num_nbrs):

        #            dest = nbr_procs[j]
        #            if self.pid == dest:
        #                continue

        #            comm.send(proc_data[dest], dest=dest, 
        #                      tag=TAG_REMOTE_CELL_REQUEST)

        #            remote_cell_data[dest] = comm.recv(
        #                source=dest, tah=TAG_REMOTE_CELL_REPLY)
                                                       
        #            #remote_cell_data[dest] = comm.recv(
        #            #    source=dest, tag=TAG_REMOTE_CELL_REPLY)

        #    else:
        #        requested_cells = comm.recv(source=pid,
                                            tag=TAG_REMOTE_CELL_REQUEST)
        #        data = self._get_cell_data_for_neighbor(requested_cells)
        #        comm.send(data, dest=pid, tag=TAG_REMOTE_CELL_REPLY)

        # we now have all the cells we require from the remote processors.
        # create new cells and add the particles to the particle array and the
        # cell.
        #num_arrays = len(self.arrays_to_bin)
        #current_counts = LongArray(num_arrays)

        #for pid, particle_data in remote_cell_data.iteritems():
            # append the new particles to the corresponding local particle
            # arrays. Also get the current number of particles before
            # appending. 
        #    parrays = particle_data['parrays']
        #    particle_counts = particle_data['pcounts']
        #    pid_index_info = self.remote_particle_indices[pid]

            # append the new particles to the corresponding local particle
            # arrays. 
        #    for i in range(num_arrays):
        #        parr = parrays[i]

                # store the start and end indices of the particles being added
                # into this parray from this pid.
        #        index_info = pid_index_info[i]
        #        d_parr = self.arrays_to_bin[i]
        #        current_counts.data[i] = d_parr.get_number_of_particles()
        #        index_info[0] = current_counts.data[i]
        #        if parr.get_number_of_particles() > 0:
        #            index_info[1] = index_info[0] + \
        #                parr.get_number_of_particles() - 1
        #        else:
        #            index_info[1] = index_info[0]
                
                # append the particles from the source parray.
        #        d_parr.append_parray(parr)

            # now insert the particle indices into the cells. The new particles
            # will be appended to the existing particles, as the new particles
            # will be tagged as dummy.
        #    i = 0
        #    cell_ids = arc[pid]
            
        #    for cid in cell_ids:
                # make sure this cell is not already present.
        #        if self.cells_dict.has_key(cid):
        #            msg = 'Cell %s should not be present %d'%(cid, self.pid)
        #            logger.error(msg)
        #            raise SystemError, msg

        #        c = self.get_new_cell_for_copy(cid, pid)

                # add the new particle indices to the cell.
        #        for j in range(num_arrays):
        #            pcount_j = particle_counts[j]
        #            if pcount_j.data[i] == 0:
        #                si = -1
        #                ei = -1
        #            else:
        #                si = current_counts.data[j]
        #                ei = si + pcount_j.data[i] - 1
        #                current_counts.data[j] += pcount_j.data[i]

                    # c.particle_start_indices.append(si)
                    # c.particle_end_indices.append(ei)

                    # insert the indices into the cell.
        #            if si >= 0 and ei >=0:
        #                indices = arange_long(si, ei)
        #                (<Cell>c).insert_particles(j, indices)
                        
                # insert the newly created cell into the cell_dict.
        #        self.cells_dict[cid] = c
        #        i += 1

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


