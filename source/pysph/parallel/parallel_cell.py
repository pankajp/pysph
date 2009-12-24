"""
Classes to implement cells and cell manager for a parallel invocation.
"""
# standard imports
import copy

# numpy imports
import numpy

# logger imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.point import Point, IntPoint
from pysph.base.particle_tags import get_dummy_tag
from pysph.base.particle_tags import *
from pysph.base import cell

from pysph.solver.base import Base
from pysph.solver.fast_utils import arange_long
from pysph.parallel.parallel_controller import ParallelController
from pysph.parallel.load_balancer import LoadBalancer

TAG_PROC_MAP_UPDATE = 1
TAG_CELL_ID_EXCHANGE = 2
TAG_CROSSING_PARTICLES = 3
TAG_NEW_CELL_PARTICLES = 4
TAG_REMOTE_CELL_REQUEST = 5
TAG_REMOTE_CELL_REPLY = 6
TAG_REMOTE_DATA_REQUEST = 7
TAG_REMOTE_DATA_REPLY = 8

################################################################################
# `share_data` function.
################################################################################
def share_data(mypid, sorted_procs, data, comm, tag=0, multi=False):
    """
    Shares the given data among the processors in nbr_proc_list. Returns a
    dictionary containing data from each other processor.

    **Parameters**
    
        - mypid - pid where the function is being called.
        - sorted_procs - list of processors to share data with.
        - data - the data to be shared.
        - comm - the MPI communicator to use.
        - tag - a tag for the MPI send/recv calls if needed.
        - multi - indicates if 'data' contains data specific to each nbr_proc or
          not. True -> separate data for each nbr proc, False -> Send same data
          to all nbr_procs.
    """
    if mypid not in sorted_procs:
        return {}
    proc_data = {}
    
    for pid in sorted_procs:
        proc_data[pid] = {}
    
    for pid in sorted_procs:
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

################################################################################
# `ProcessorMap` class.
################################################################################
class ProcessorMap(Base):
    """
    Class to maintain the assignment of processors to geometric regions.
    """
    def __init__(self, cell_manager=None, pid=0,
                 origin=Point(0, 0, 0), bin_size=0.3, 
                 *args, **kwargs):
        """
        Contructor.
        """
        self.cell_manager = cell_manager
        self.origin = Point()
        self.local_p_map = {}
        self.p_map = {}
        self.nbr_procs = []
        self.pid = pid

        if cell_manager is not None:
            self.pid = cell_manager.pid
            self.origin.x = cell_manager.origin.x
            self.origin.y = cell_manager.origin.y
            self.origin.z = cell_manager.origin.z
            
        # the cell manager may not have been initialized by this time.
        # we just set the bin size of the proc_map to the value given
        # the cell manager may later on setup the cell size.
        self.bin_size = bin_size
        
    def __getstate__(self):
        d = self.__dict__.copy()
        for key in ['cell_manager', 'nbr_procs']:
            d.pop(key, None)
        return d

    def update(self):
        """
        Update the processor map with current cells in the cell manager.

        The cells used are the ones in the 2nd last level in the
        hierarchy_list. This is the level at which cells are required to be
        communicated for a parallel simulation.
        
        """
        cm = self.cell_manager
        pid = self.pid
        pm = {}
        cells = self.cell_manager.root_cell.cell_dict
        centroid = Point()
        id = IntPoint()
        
        for cid, cel in cells.iteritems():
            cel.py_get_centroid(centroid)
            cell.py_find_cell_id(origin=self.origin, pnt=centroid,
                                 cell_size=self.bin_size, outpoint=id)
            pm.setdefault(id.py_copy(), set([pid]))
        
        self.p_map = pm
        self.local_p_map = copy.deepcopy(pm)

    def merge(self, proc_map):
        """
        Merge data from other processors proc map into ours.

        **Parameters**
            - proc_map - processor map from another processor.
        """
        m_pm = self.p_map
        o_pm = proc_map.p_map
        
        for o_id in o_pm:
            if o_id in m_pm:
                m_pm[o_id].update(o_pm[o_id])
            else:
                m_pm[o_id] = set(o_pm[o_id])

    def find_region_neighbors(self):
        """
        Find the processors that are occupying regions in the processor occupied
        by the current pid. These neighbors need not be adjacent i.e sharing
        cells.
        
        """
        pid = self.pid
        nb = set([pid])
        l_pm = self.local_p_map
        pm = self.p_map
        nids = []
        empty_list = []

        # for each cell in local_pm
        for id in l_pm:
            nids[:] = empty_list
            
            # constructor list of neighbor cell ids in the proc_map.
            cell.py_construct_immediate_neighbor_list(cell_id=id,
                                                      neighbor_list=nids)
            for nid in nids:
                # if cell exists, collect all occupying processor ids
                # into the nb set.
                n_pids = pm.get(nid)
                if n_pids is not None:
                    nb.update(n_pids)
        
        self.nbr_procs = sorted(list(nb))

    def __str__(self):
        rep = '\nProcessor Map At proc : %d\n'%(self.pid)
        rep += 'Origin : %s\n'%(self.origin)
        rep += 'Bin size : %s\n'%(self.bin_size)
#         for cid in self.p_map:
#             rep += 'bin %s with processors : %s'%(cid, self.p_map[cid])
#             rep += '\n'
        rep += 'Region neighbors : %s'%(self.nbr_procs)
        return rep


################################################################################
# `ParallelCellInfo` class.
################################################################################
class ParallelCellInfo(Base):
    """
    Class to hold information to be maintained with any parallel cell.

    This will be typically used only for those cells that are involved in the
    parallel level. Any cell under this cell behaves just like a serial cell.

    """
    def __init__(self, cell=None):
        self.cell = cell
        self.root_cell = self.cell.cell_manager.root_cell
        self.neighbor_cell_pids = {}
        self.remote_pid_cell_count = {}
        self.num_remote_neighbors = 0
        self.num_local_neighbors = 0

    def update_neighbor_information(self, glb_nbr_cell_pids):
        """
        Updates the remote neighbor information from the glb_nbr_cell_pids.

        glb_nbr_cell_pids also contains cell ids from the same processor.
        """
        id = self.cell.id
        nbr_ids = []
        cell.py_construct_immediate_neighbor_list(cell_id=id,
                                                  neighbor_list=nbr_ids,
                                                  include_self=False)

        # clear previous information.
        self.neighbor_cell_pids.clear()
        
        for nbr_id in nbr_ids:
            nbr_pid = glb_nbr_cell_pids.get(nbr_id)
            if nbr_pid is not None:
                self.neighbor_cell_pids[nbr_id.py_copy()] = nbr_pid
        
    def compute_neighbor_counts(self):
        """
        Find the number of local and remote neighbors of this cell.
        """
        self.remote_pid_cell_count.clear()
        self.num_remote_neighbors = 0
        self.num_local_neighbors = 0
        mypid = self.cell.pid
        for cid, pid in self.neighbor_cell_pids.iteritems():
            if pid == mypid:
                self.num_local_neighbors += 1
            else:
                self.num_remote_neighbors += 1
                cnt = self.remote_pid_cell_count.get(pid)
                if cnt is None:
                    self.remote_pid_cell_count[pid] = 1
                else:
                    self.remote_pid_cell_count[pid] = cnt + 1

    def update_neighbor_information_local(self):
        """
        Updates neighbor information using just local data.

        This uses the root_cell's cell_dict to search for neighbors.
        """
        nbr_cell_pids = {}
        nbr_cell_pids.update(self.neighbor_cell_pids)

        for cid in nbr_cell_pids:
            c = self.root_cell.cell_dict.get(cid)
            if c is not None:
                self.neighbor_cell_pids[cid.py_copy()] = c.pid        
    
    def is_boundary_cell(self):
        """
        Returns true if this cell is a boundary cell, false otherwise.
        """
        dim = self.cell.cell_manager.dimension
        num_neighbors = len(self.neighbor_cell_pids.keys())
        if dim == 1:
           if num_neighbors < 2:
               return True
           else:
               return False
        elif dim == 2:
            if num_neighbors < 8:
                return True
            else:
                return False
        elif dim == 3:
            if num_neighbors < 26:
                return True
            else:
                return False

        msg = 'Invalid dimension'
        logger.error(msg)
        raise ValueError, msg
        
################################################################################
# `ParallelLeafCell` class.
################################################################################
class ParallelLeafCell(cell.LeafCell):
    """
    Leaf cell to be used in parallel computations.
    """
    def __init__(self, id, cell_manager=None, cell_size=0.1, level=0,
                 jump_tolerance=1, pid=-1):
        cell.LeafCell.__init__(self, id=id, cell_manager=cell_manager,
                               cell_size=cell_size, level=level,
                               jump_tolerance=jump_tolerance)

        self.pid = pid
        self.parallel_cell_info = ParallelCellInfo(cell=self)

    def get_new_sibling(self, id):
        """
        """
        cell = ParallelLeafCell(id=id, cell_manager=self.cell_manager,
                                cell_size=self.cell_size, level=self.level,
                                jump_tolerance=self.jump_tolerance,
                                pid=self.pid)
        return cell

################################################################################
# `ParallelLeafCellRemoteCopy` class.
################################################################################
class ParallelLeafCellRemoteCopy(ParallelLeafCell):
    """
    Leaf cells holding information from another processor.
    """
    def __init__(self, id, cell_manager=None, cell_size=0.1, level=0,
                 jump_tolerance=1, pid=-1):
        ParallelLeafCell.__init__(self, id=id, cell_manager=cell_manager,
                                  cell_size=cell_size, level=level,
                                  jump_tolerance=jump_tolerance, pid=pid)

        self.particle_start_indices = []
        self.particle_end_indices = []
        self.num_particles = 0
################################################################################
# `ParallelNonLeafCell` class.
################################################################################
class ParallelNonLeafCell(cell.NonLeafCell):
    """
    NonLeafCell to be used in parallel computations.
    """
    def __init__(self, id, cell_manager=None, cell_size=0.1, level=1,
                 pid=-1):
        cell.NonLeafCell.__init__(self, id=id, cell_manager=cell_manager,
                                  cell_size=cell_size, level=level)
        self.pid = pid
        self.parallel_cell_info = ParallelCellInfo(cell=self)

    def get_new_sibling(self, id):
        """
        """
        cell = ParallelNonLeafCell(id=id, cell_manager=self.cell_manager,
                                   cell_size=self.cell_size, level=self.level,
                                   pid=self.pid)
        return cell

    def get_new_child(self, id):
        """
        """
        num_levels = self.cell_manager.num_levels
        cell_sizes = self.cell_manager.cell_sizes
        if (num_levels - self.level) == num_levels-1:
            return ParallelLeafCell(id=id, cell_manager=self.cell_manager,
                                    cell_size=cell_sizes.get(self.level-1),
                                    level=self.level-1,
                                    jump_tolerance=self.cell_manager.jump_tolerance
                                    ,pid=self.pid) 
        else:
            return ParallelNonLeafCell(id=id, cell_manager=self.cell_manager,
                                       cell_size=cell_sizes.get(self.level-1),
                                       level=self.level-1, pid=self.pid)

################################################################################
# `ParallelNonLeafCellRemoteCopy` class.
################################################################################
class ParallelNonLeafCellRemoteCopy(ParallelNonLeafCell):
    """
    NonLeafCell holding information from another processor
    """
    def __init__(self, id, cell_manager=None, cell_size=0.1, level=1, pid=-1):
        ParallelNonLeafCell.__init__(self, id=id, cell_manager=cell_manager,
                                     cell_size=cell_size, level=level, pid=pid)
        self.particle_start_indices = []
        self.particle_end_indices = []
        self.num_particles = 0

################################################################################
# `ParallelRootCell` class.
################################################################################
class ParallelRootCell(cell.RootCell):
    """
    Root cell for parallel computations.

    There is one ParallelRootCell per process.
    """
    def __init__(self, cell_manager=None, cell_size=0.1,
                 pid=0, *args, **kwargs):
        cell.RootCell.__init__(self, cell_manager=cell_manager,
                               cell_size=cell_size)

        self.initial_redistribution_done = False

        # the list of remote cells that cells under this root cell are
        # adjoining. The dictionary is indexed on processor ids, and contains a
        # list of cell ids of neighbors from those processors.
        self.adjacent_remote_cells = {}

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

        if cell_manager is None:
            self.pid = pid
        else:
            self.pid = cell_manager.pid
            self.parallel_controller = cell_manager.parallel_controller

    def get_new_child(self, id):
        """
        """
        num_levels = self.cell_manager.num_levels
        cell_sizes = self.cell_manager.cell_sizes

        if num_levels == 1:
            return ParallelLeafCell(id=id, cell_manager=self.cell_manager,
                                    cell_size=cell_sizes.get(0),
                                    level=0,
                                    jump_tolerance=self.cell_manager.jump_tolerance,
                                    pid=self.pid)
        else:
            return ParallelNonLeafCell(id=id, cell_manager=self.cell_manager,
                                       cell_size=cell_sizes.get(self.level-1),
                                       level=self.level-1,
                                       pid=self.pid)

    def get_new_child_for_copy(self, id, pid):
        """
        """
        num_levels = self.cell_manager.num_levels
        cell_sizes = self.cell_manager.cell_sizes

        if num_levels == 1:
            return ParallelLeafCellRemoteCopy(
                id=id,
                cell_manager=self.cell_manager, 
                cell_size=cell_sizes.get(0),
                level=0,
                jump_tolerance=self.cell_manager.jump_tolerance,
                pid=pid)
        else:
            return ParallelNonLeafCellRemoteCopy(
                id=id,
                cell_manager=self.cell_manager,  
                cell_size=cell_sizes.get(self.level-1),
                level=self.level-1,
                pid=pid)

    def find_adjacent_remote_cells(self):
        """
        Finds all cells from other processors that are adjacent to cells handled
        by this processor. 

        This also updates the adjacent_processors list.
        
        **Note**
            - assumes the neighbor information of all cells to be up-to-date.

        """
        self.adjacent_remote_cells.clear()
        arc = {}

        for cell in self.cell_dict.values():
            pci = cell.parallel_cell_info
            
            for cid, pid in pci.neighbor_cell_pids.iteritems():
                if pid == self.pid:
                    continue
                
                info = arc.get(pid)
                if info is None:
                    info = {}
                    arc[pid] = info
                # add cellid to the list of cell from processor pid.
                info[cid.py_copy()] = None

        # copy temp data in arc into self.adjacent_remote_cells
        for pid, cell_dict in arc.iteritems():
            self.adjacent_remote_cells[pid] = cell_dict.keys()
        
        self.adjacent_processors[:] = self.adjacent_remote_cells.keys()
        # add my pid also into adjacent_processors.
        self.adjacent_processors.append(self.pid)

        self.adjacent_processors[:] = sorted(self.adjacent_processors)

        logger.debug('Adjacent processors : %s'%(self.adjacent_processors))
        logger.debug('Adjacent neigbors cells')
        for pid , cell_list in self.adjacent_remote_cells.iteritems():
            for c in cell_list:
                logger.debug('Cell %s with proc %d'%(c, pid))
        
    def update_cell_neighbor_information(self):
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
            - The processor map should be up-to-date before this function is called.
            
        """
        comm = self.parallel_controller.comm
        p_map = self.cell_manager.proc_map
        nbr_procs = p_map.nbr_procs
        
        sorted_nbr_proces = sorted(nbr_procs)
        
        cell_list = self.cell_dict.keys()
        
        nbr_proc_cell_list = share_data(mypid=self.pid,
                                        sorted_procs=sorted_nbr_proces,
                                        data=cell_list, comm=comm, 
                                        tag=TAG_CELL_ID_EXCHANGE,
                                        multi=False)
        
        # add this pids information also here.
        nbr_proc_cell_list[self.pid] = cell_list

        # from this data, construct the nbr_cell_info.
        nbr_cell_info = {}
        for pid, cell_list in nbr_proc_cell_list.iteritems():
            for cid in cell_list:
                n_info = nbr_cell_info.get(cid)
                if n_info is None:
                    nbr_cell_info[cid.py_copy()] = pid
                else:
                    logger.error('Cell %s in more than one processor : %d, %d'%(
                            cid, pid, n_info))
        
        # update the neighbor information of all the children cells.
        for cell in self.cell_dict.values():
            cell.parallel_cell_info.update_neighbor_information(nbr_cell_info)

        # store to nbr_cell_info for later use.
        self.nbr_cell_info.clear()
        self.nbr_cell_info.update(nbr_cell_info)

        # update the adjacent cell information of the root.
        self.find_adjacent_remote_cells()

    def update(self, data):
        """
        Update particle information.
        """
        # wait till all processors have reached this point.
        self.parallel_controller.comm.Barrier()

        logger.debug('++++++++++++++++ UPDATE BEGIN +++++++++++++++++++++')

        # bin the particles and find the new_cells and remote_cells.
        new_cells, remote_cells = self.bin_particles()

        # clear the list of new cells added.
        self.new_cells_added.clear()

        # create a copy of the particles in the new cells and mark those
        # particles as remote.
        self.new_particles_for_neighbors = self.create_new_particle_copies(
            remote_cells)
        self.new_region_particles = self.create_new_particle_copies(
            new_cells)

        # exchange particles moving into a region assigned to a known
        # processor's region. 
        self.exchange_crossing_particles_with_neighbors(
            remote_cells,
            self.new_particles_for_neighbors
            )
        
        # remove all particles with local flag set to 0.
        self.cell_manager.remove_remote_particles()

        # make sure each new cell is with exactly one processor.
        self.assign_new_cells(new_cells, self.new_region_particles)

        # all new particles entering this processors region and in regions
        # assigned to this processor have been added to the respective particles
        # arrays. The data in the particles arrays is stale and the indices
        # invalid. Perform a top down insertion of all particles into the
        # hierarchy tree.
        self.bin_particles_top_down()

        # now update the processor map.
        self.cell_manager.glb_update_proc_map()
        
        # re-compute the neighbor information
        self.update_cell_neighbor_information()

        # wait till all processors have reached this point.
        self.parallel_controller.comm.Barrier()

        # call a load balancer function.
        if self.cell_manager.initialized == True:
            if self.cell_manager.load_balancing == True:
                self.cell_manager.load_balancer.load_balance()

        # at this point each processor has all the real particles it is supposed
        # to handle. We can now exchange neighbors particle data with the
        # neighbors.
        self.exchange_neighbor_particles()

        logger.debug('+++++++++++++++ UPDATE DONE ++++++++++++++++++++')
        return 0

    def bin_particles_top_down(self):
        """
        Clears the tree and re-inserts particles.

        **Algorithm**

            - clear the indices of all the particle arrays that need to be
              binned.
            - reinsert particle indices of all particle arrays that are to be
              binned. 
        """
        for i in range(len(self.cell_manager.arrays_to_bin)):
            self.clear_indices(i)

        # we now have a skeleton tree without any particle indices.
        i = 0
        for parr in self.cell_manager.arrays_to_bin:
            num_particles = parr.get_number_of_particles()
            logger.debug('parray %s with %d particles'%(
                    parr.name, num_particles))
            indices = arange_long(num_particles)
            self.insert_particles(i, indices)
            i += 1

        # delete any empty cells.
        self.delete_empty_cells()

        logger.debug('Top down binning done')
        logger.debug('Cell information as START')
        logger.debug('Num cells : %d'%(len(self.cell_dict)))
        for cid, c in self.cell_dict.iteritems():
            logger.debug(
                'Cell %s with %d particles with pid %d'%(cid,
                                                         c.get_number_of_particles(),
                                                         c.pid))
        logger.debug('Cell information END')
        
    def bin_particles(self):
        """
        Find the cell configurations caused by the particles moving. Returns the
        list of cell created in unassigned regions and those created in regions
        already occupied by some other processor.
        
        """
        new_cells = {}
        remote_cells = {}
        collected_data = {}

        for cid, smaller_cell in self.cell_dict.iteritems():
            if smaller_cell.pid == self.pid:
                smaller_cell.update(collected_data)

        # if the cell from the base hierarchy creation exists and initial
        # re-distribution is yet to be done, add that to the list of new cells.
        if self.initial_redistribution_done is False:

            # update the processor map once - no neighbor information is
            # available at this point.
            self.cell_manager.glb_update_proc_map()

            c = self.cell_dict.values()[0]
            if c.py_get_number_of_particles() > 0:
                collected_data[c.id.py_copy()] = c
            # remove it from the cell_dict
            self.cell_dict.clear()
            self.initial_redistribution_done = True
            
        # we have a list of all new cells created by the smaller cells.
        for cid, smaller_cell in collected_data.iteritems():
            # check if this cell is already there in the list of children cells.
            smaller_cell_1 = self.cell_dict.get(cid)
            if smaller_cell_1 is not None:
                # check if this is a remote cell
                if smaller_cell_1.pid != self.pid:
                    # add it to the remote cells.
                    r_cell = remote_cells.get(cid)
                    if r_cell is None:
                        smaller_cell.pid = smaller_cell_1.pid
                        remote_cells[cid.py_copy()] = smaller_cell
                    else:
                        r_cell.add_particles(smaller_cell)
                else:
                    # add it to the current cells.
                    smaller_cell_1.add_particles(smaller_cell)
            else:
                # check if this cell is in new cells
                smaller_cell_1 = new_cells.get(cid)
                if smaller_cell_1 is None:
                    new_cells[cid.py_copy()] = smaller_cell
                else:
                    smaller_cell_1.add_particles(smaller_cell)

        logger.debug('<<<<<<<<<<<<<<<<<< NEW CELLS >>>>>>>>>>>>>>>>>>>>>')
        for cid in new_cells:
            logger.debug('new cell (%s)'%(cid))
        logger.debug('<<<<<<<<<<<<<<<<<< NEW CELLS >>>>>>>>>>>>>>>>>>>>>')

        logger.debug('<<<<<<<<<<<<<<<<<< REMOTE CELLS >>>>>>>>>>>>>>>>>>>>>')
        for cid in remote_cells:
            logger.debug('remote cell (%s) for proc %d'%(cid, remote_cells[cid].pid))
        logger.debug('<<<<<<<<<<<<<<<<<< REMOTE CELLS >>>>>>>>>>>>>>>>>>>>>')

        return new_cells, remote_cells

    def create_new_particle_copies(self, cell_dict):
        """
        Copies all particles in cell_dict to new particle arrays - one for each
        cell and returns a dictionary indexed on cell_id, containing the newly
        created particles.

        **Algorithm**

            - for each cell in cell_dict
                - get indices of all particles in this cell.
                - remove any particles marked as non-local from this set of
                  particles. [not required - infact this is incorrect]
                - create a particle array for each set of particles, including
                  in it only the required particles.
                - mark these particles as remote in the main particle array.
                
        """
        copies = {}
        for cid, c in cell_dict.iteritems():
            index_lists = []
            parrays = []
            c.get_particle_ids(index_lists)

            for i in range(len(index_lists)):
                index_array = index_lists[i]
                parr = c.arrays_to_bin[i]
                
                parr_new = parr.extract_particles(index_array)
                parr_new.set_name(parr.name)
                parrays.append(parr_new)
                
                # mark the particles as remote in the particle array.
                parr.set_flag('local', 0, index_array)
                # also set them as dummy particles.
                parr.set_tag(get_dummy_tag(), index_array)

            copies[cid.py_copy()] = parrays

        logger.debug('<<<<<<<<<<<<<create_new_particle_copies>>>>>>>>>>>')
        for cid, parrays in copies.iteritems():
            logger.debug('Cell (%s) has :'%(cid))
            for parr in parrays:
                logger.debug('   %s containing - %d particles'%(
                        parr.name, parr.get_number_of_particles()))
        logger.debug('<<<<<<<<<<<<<create_new_particle_copies>>>>>>>>>>>')
        
        return copies
        
    def assign_new_cells(self, new_cell_dict, new_particles):
        """
        Assigns cells created in new regions (i.e. regions not assigned to any
        processor) to some processor. Conflicts are resolved using a
        deterministic scheme which returns the same winner in all processors.
        
        **Parameters**
            
            - new_cell_dict - dictionary of new cells created during a bottom-up
            update of the current tree.
            - new_particles - a set of ParticleArrays for each new cell that was
            created.

        **Algorithm**
        
            - share new particles with all neighbor procs in the proc map.
            - resolve conflict.
            - add data assigned to self into the local particle arrays.

        """
        comm = self.parallel_controller.comm
        nbr_procs = self.cell_manager.proc_map.nbr_procs

        logger.debug('SHARING NEW CELLS WITH : %s'%(nbr_procs))

        proc_data = share_data(mypid=self.pid, 
                               sorted_procs=nbr_procs,
                               data=new_particles,
                               comm=comm,
                               tag=TAG_NEW_CELL_PARTICLES, 
                               multi=False)

        proc_data[self.pid] = new_particles

        logger.debug('<<<<<<<<<<<<<<<new received particles>>>>>>>>>>>>>>>>>')
        for pid, cells in proc_data.iteritems():
            logger.debug('FROM PID %d'%(pid))
            for cid in cells:
                logger.debug('  received : (%s)'%(cid))
        logger.debug('<<<<<<<<<<<<<<<new received particles>>>>>>>>>>>>>>>>>')

        
        # we now have data for all the new cells that were created.
        # invert the data - we have a dictionary indexed on pids, and then on
        # cids. - make it indexed on cid and then pids
        # along with this get the number of particles contributed to each cell
        # by each processor.
        cell_data = {}
        num_particles = {}

        for pid, cell_dict in proc_data.iteritems():
            for cid, parr_list in cell_dict.iteritems():
                c = cell_data.get(cid)
                if c is None:
                    c = {}
                    cell_data[cid.py_copy()] = c
                    num_particles[cid.py_copy()] = {}
                c[pid] = parr_list
                np = 0
                for p in parr_list:
                    np += p.get_number_of_particles()
                num_particles[cid][pid] = np

        winning_procs = self._resolve_conflicts(num_particles)

        # now add the particles in the cells assigned to self into corresponding
        # particle arrays.
        for cid, pid in winning_procs.iteritems():
            if pid != self.pid:
                continue
            
            c_data = cell_data[cid]
            
            for p in c_data.values():
                self.add_local_particles_to_parray({cid.py_copy():p})

    def _resolve_conflicts(self, data):
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
        winning_procs = {}
                
        for cid, p_data in data.iteritems():
            if len(p_data) == 1:
                winning_procs[cid.py_copy()] = p_data.keys()[0]
                continue
            pids = p_data.keys()
            num_particles = p_data.values()
            pids = sorted(pids)
            max_contribution = max(num_particles)
            procs = []
            for pid in pids:
                if p_data[pid] == max_contribution:
                    procs.append(pid)
            winning_procs[cid.py_copy()] = max(procs)

        for cid, proc in winning_procs.iteritems():
            logger.debug('Cell %s assigned to proc %d'%(cid, proc))

        return winning_procs

    def exchange_crossing_particles_with_neighbors(self, remote_cells, particles):
        """
        Send all particles that crossed into a known neighbors region, receive
        particles that got into our region from a neighbors.

        **Parameters**
        
            - remote_cells - dictionary of cells that are to be sent to some
              remote neighbour.
            - particles - particle arrays to be sent to each of these cells.

        **Algorithm**
            
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
        proc_data = {}
        comm = self.parallel_controller.comm

        logger.debug('Exchanging crossing particles with : %s'%(
                self.adjacent_processors))

        # create one entry here for each neighbor processor.
        for proc_id in self.adjacent_processors:
            proc_data[proc_id] = {}

        for cid, c in remote_cells.iteritems():
            p_data = proc_data[c.pid]
            parrays = particles[cid]
            
            p_data[cid.py_copy()] = parrays
            
        logger.debug('Sharing the following data: %s'%(proc_data))

        # for each processor we now have a dictionary indexed on cell id,
        # containing the particles that need to be sent to that processor.
        # share data
        new_particles = share_data(mypid=self.pid,
                                   sorted_procs=self.adjacent_processors,
                                   data=proc_data, comm=comm,
                                   tag=TAG_CROSSING_PARTICLES, multi=True)
        
        logger.debug('Got shared data : %s'%(new_particles))
        # for each neigbor processor, there is one entry in new_particles
        # containing all new cells that processor sent to us.
        self.add_entering_particles_from_neighbors(new_particles)

    def add_entering_particles_from_neighbors(self, new_particles):
        """
        Add particles that entered into our parrays from other processors
        regions.

        **Parameter**
        
            - new_particles - a dictionary having one entry per processor.
              each entry has a dictionary indexed on the cells into which the
              particles are entering.

        **Algorithm**
            
             - for data from each processor
                 - add the particle arrays.
        
        """
        for pid, particle_list in new_particles.iteritems():
            self.add_local_particles_to_parray(particle_list)

    def add_local_particles_to_parray(self, particle_list):
        """
        Adds the given particles to the local parrays as local particles.
                
        """
        for cid, parrays in particle_list.iteritems():
            count = 0
            logger.debug('Adding particles entering cell %s, %s'%(cid, parrays))
            
            for i in range(len(self.cell_manager.arrays_to_bin)):
                s_parr = parrays[i]
                d_parr = self.cell_manager.arrays_to_bin[i]
                
                # set the local property to '1'
                s_parr.local[:] = 1
                d_parr.append_parray(s_parr)
                count += s_parr.get_number_of_particles()

            cnt = self.new_cells_added.get(cid)
            if cnt is None:
                self.new_cells_added[cid.py_copy()] = count
            else:
                self.new_cells_added[cid.py_copy()] += count

    def update_remote_particle_properties(self, props=None):
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
        logger.debug('update_remote_particle_properties')
        nbr_procs = []
        nbr_procs[:] = self.adjacent_processors
        arc = self.adjacent_remote_cells
        remote_cell_data = {}
        comm = self.parallel_controller.comm

        logger.debug('Neighbor procs are : %s'%(nbr_procs))
        for cids in arc.values():
            for cid in cids:
                logger.debug('%s'%(cid))

        if props is None:
            props = [None]*len(self.cell_manager.arrays_to_bin)

        for pid in nbr_procs:
            if pid == self.pid:
                for dest in nbr_procs:
                    if self.pid == dest:
                        continue
                    comm.send(arc[dest], dest=dest, tag=TAG_REMOTE_DATA_REQUEST)
                    remote_cell_data[dest] = comm.recv(
                        source=dest, tag=TAG_REMOTE_DATA_REPLY)
            else:
                requested_cells = comm.recv(source=pid,
                                            tag=TAG_REMOTE_DATA_REQUEST) 
                data = self._get_cell_data_for_neighbor(requested_cells,
                                                        props=props)
                comm.send(data, dest=pid, tag=TAG_REMOTE_DATA_REPLY)

        logger.debug('Data Exchange done')
        # now copy the data received into the appropriate locations.
        for pid, cell_data in remote_cell_data.iteritems():
            for cid, cell_particles in cell_data.iteritems():
                # get the copy of the remote cell with us.
                remote_cell = self.cell_dict.get(cid)
                
                logger.debug('Copy data for cell : %s'%(cid))
                if remote_cell is None:
                    msg = 'Copy of Remote cell %s not found'%(cid)
                    logger.error(msg)
                    raise SystemError, msg
                
                for i in range(len(self.cell_manager.arrays_to_bin)):
                    d_parr = self.cell_manager.arrays_to_bin[i]
                    s_parr = cell_particles[i]

                    si = remote_cell.particle_start_indices[i]
                    ei = remote_cell.particle_end_indices[i]

                    logger.debug('Copying particles of parray :%s'%(
                            d_parr.name))
                    logger.debug('Copying from %d to %d'%(si, ei))
                    logger.debug('Num particles in dest : %d'%(
                            d_parr.get_number_of_particles()))
                    logger.debug('Num particles in source : %d'%(
                            s_parr.get_number_of_particles()))

                    if si == -1 and ei == -1:
                        # there are no particles for this array, continue
                        continue
                    
                    n1 = ei-si+1
                    n2 = s_parr.get_number_of_particles()

                    if n1 != n2:
                        msg = 'Remote data : %d, local copy : %d\n'%(n1, n2)
                        msg += 'Both should have same number of particles'
                        logger.error(msg)
                        raise SystemError, msg
                    
                    # make sure only the required properties are there in the
                    # destination arrays.
                    if props[i] is not None:
                        for prop in s_parr.properties.keys():
                            if prop not in props[i]:
                                s_parr.remove_property(prop)

                    # copy the property values from the source array into
                    # properties of the destination array.
                    d_parr.copy_properties(s_parr, start_index=si, end_index=ei)

    def exchange_neighbor_particles(self):
        """
        Exchange neighbor particles.

        **Algorithm**

            - get all required particles from neighbors.
            - send neighbors their required information.
            
            - for each remote cell just received
                - create a new cell
                - append the particles of this cell as remote and dummy
                  particles to the particle arrays. These particles WILL be
                  APPENDED and will not violate the particle indices.
                - bin the new particles into this cell.

        """
        logger.debug('exchange_neighbor_particles_START')
        nbr_procs = []
        nbr_procs[:] = self.adjacent_processors
        arc = self.adjacent_remote_cells
        remote_cell_data = {}
        comm = self.parallel_controller.comm
        
        if nbr_procs.count(self.pid) == 0:
            nbr_procs.append(self.pid)
            nbr_procs = sorted(nbr_procs)
        
        # get data from all procs and send data to all procs.
        logger.debug('neighbor proces are : %s'%(nbr_procs))
        for pid in nbr_procs:
            if pid == self.pid:
                for dest in nbr_procs:
                    if self.pid == dest:
                        continue
                    # our turn to send request for cells.
                    logger.debug('adjacent remote cells : %s'%(arc))
                    comm.send(arc[dest], dest=dest, tag=TAG_REMOTE_CELL_REQUEST)
                    logger.debug('adjacent cell request send')
                    remote_cell_data[dest] = comm.recv(source=dest,
                                                       tag=TAG_REMOTE_CELL_REPLY)
                    logger.debug('got cells : %s'%(remote_cell_data[dest]))
            else:
                requested_cells = comm.recv(source=pid,
                                            tag=TAG_REMOTE_CELL_REQUEST) 
                data = self._get_cell_data_for_neighbor(requested_cells)
                comm.send(data, dest=pid, tag=TAG_REMOTE_CELL_REPLY)

        logger.debug('Remote data received')

        # we now have all the cells we require from the remote processors.
        # create new cells and add the particles to the particle array and the
        # cell.
        for pid, cell_dict in remote_cell_data.iteritems():
            for cid, cell_particles in cell_dict.iteritems():
                c = self.cell_dict.get(cid)
                
                if c is not None:
                    msg = 'Cell %s should not be present %d'%(cid, self.pid)
                    logger.error(msg)
                    raise SystemError, msg
                
                c = self.get_new_child_for_copy(cid, pid)
                logger.debug('Created cells %s. %s pid :%d'%(c, c.id, pid))

                for i in range(len(self.cell_manager.arrays_to_bin)):
                    d_parr = self.cell_manager.arrays_to_bin[i]
                    s_parr = cell_particles[i]
                    
                    # check if there are no particles for this array.
                    if s_parr.get_number_of_particles() == 0:
                        si = -1
                        ei = -1
                    else:
                        si = d_parr.get_number_of_particles()
                        ei = si + s_parr.get_number_of_particles() - 1
                        logger.debug('SI : %d, EI : %d'%(si, ei))

                    c.particle_start_indices.append(si)
                    c.particle_end_indices.append(ei)
                    
                    d_parr.append_parray(s_parr)
                    
                    if si >=0 and ei >=0:
                        indices = arange_long(start=si, stop=ei)
                        c.insert_particles(i, indices)

                # add this cell to the cell_dict
                self.cell_dict[cid.py_copy()] = c

        logger.debug('exchange_neighbor_particles_DONE')
                
    def _get_cell_data_for_neighbor(self, cell_list, props=None):
        """
        Return new particle arrays created for particles contained in each of
        the requested cells.

        **Parameters**
        
            - cell_list - the list of cells, whose properties are requested.
            - props - a list whose entries are as follows: for each particle
              array that has been binned, a list of properties required of that
              particle array, or None if all properties are required.

        """
        data = {}
        if props is not None:
            if len(props) != len(self.cell_manager.arrays_to_bin):
                msg = 'Need information for each particle array'
                logger.error(msg)
                raise SystemError, msg
        else:
            props = [None]*len(self.cell_manager.arrays_to_bin)
            
        for cid in cell_list:
            logger.debug('Building data for cell : %s'%(cid))
            c = self.cell_dict.get(cid)

            if c is None:
                msg = 'Requested cell (%s) not with proc %d'%(cid, self.pid)
                logger.error(msg)
                raise SystemError, msg

            if c.pid != self.pid:
                msg = 'Data being requested for cell %s which is remote in %d'%(
                    c.id, self.pid)
                logger.error(msg)
                raise SystemError, msg

            index_lists = []
            parrays = []
            
            # get all particles ids of this cell.
            c.get_particle_ids(index_lists)
            num_particles = 0
            for i in range(len(index_lists)):
                index_array = index_lists[i]
                num_particles += index_array.length
                parr = c.cell_manager.arrays_to_bin[i]
                parr_new = parr.extract_particles(index_array,
                                                  props=props[i])
                parr_new.set_name(parr.name)
                parrays.append(parr_new)

                # make the particles as remote in the particle array
                parr_new.local[:] = 0
                parr_new.tag[:] = get_dummy_tag()

            data[cid.py_copy()] = parrays
            
            logger.debug('Requested cell %s has %s particles '%(c.id,
                                                                num_particles))
        return data

    def compute_neighbor_counts(self):
        """
        Recompute the neighbor counts of all local cells. 
        This does not invovle any global communications. It assumes that the
        data available is up-to-date.

        **Note**

            - DO NOT PUT ANY GLOBAL COMMUNICATION CALLS HERE. This function may
            be asynchronously called to update the neighbor counts after say
            transfering cell to another process.
        """
        for c in self.cell_dict.values():
            c.parallel_cell_info.compute_neighbor_counts()

    def update_neighbor_information_local(self):
        """
        Update the neighbor information locally.
        """
        for c in self.cell_dict.values():
            c.parallel_cell_info.update_neighbor_information_local()
            c.parallel_cell_info.compute_neighbor_counts()            

################################################################################
# `ParallelCellManager` class.
################################################################################
class ParallelCellManager(cell.CellManager):
    """
    Cell manager for parallel invocations.
    """
    def __init__(self, arrays_to_bin=[], particle_manager=None,
                 min_cell_size=0.1, max_cell_size=0.5, origin=Point(0, 0, 0),
                 num_levels=2, initialize=True,
                 parallel_controller=None,
                 max_radius_scale=2.0, dimension=3,
                 load_balancing=True,
                 *args, **kwargs):
        """
        Constructor.
        """
        cell.CellManager.__init__(self, arrays_to_bin=arrays_to_bin,
                                  particle_manager=particle_manager,
                                  min_cell_size=min_cell_size,
                                  max_cell_size=max_cell_size,
                                  origin=origin,
                                  num_levels=num_levels,
                                  initialize=False)

        self.dimension = dimension
        self.glb_bounds_min = [0, 0, 0]
        self.glb_bounds_max = [0, 0, 0]
        self.glb_min_h = 0
        self.glb_max_h = 0
        self.max_radius_scale = max_radius_scale
        self.pid = 0
        self.parallel_cell_level = 0

        # set the parallel controller.
        if parallel_controller is None:
            self.parallel_controller = ParallelController(cell_manager=self)
        else:
            self.parallel_controller = parallel_controller

        self.pc = self.parallel_controller
        self.pid = self.pc.rank
        self.root_cell = ParallelRootCell(cell_manager=self)
        self.proc_map = ProcessorMap(cell_manager=self)
        self.load_balancer = LoadBalancer(parallel_cell_manager=self)
        self.load_balancing = load_balancing
                
        if initialize is True:
            self.initialize()

    def initialize(self):
        """
        Initialization function for the cell manager.

        **Algorithm**
            - find the bounds of the data. - global/parallel.
            - set the origin to be used.
            - find the max and min cell sizes from the particle radii -
              global/parallel.
            - perform a proper update on the data. 
            - May have to do a conflict resolutions at this point.
            
        """
        logger.debug('%s initialize called'%(self))
        if self.initialized == True:
            logger.warn('Trying to initialize cell manager more than once')
            return

        pc = self.parallel_controller
        # exchange bounds and interaction radii information.
        self.initial_info_exchange()

        # now setup the origin and cell size to use.
        self.setup_origin()
        self.setup_cell_sizes()

        # setup array indices.
        self.py_rebuild_array_indices()

        # setup the hierarchy list
        self.py_setup_hierarchy_list()

        self.root_cell.level = self.num_levels

        self.py_compute_cell_sizes(self.min_cell_size, self.max_cell_size,
                                   self.num_levels, self.cell_sizes)
        
        # setup the parallel cell level
        self.parallel_cell_level = self.num_levels - 2

        # setup information for the processor map.
        self.setup_processor_map()

        # buid a basic hierarchy with one cell at each level, and all
        # particles in the leaf cell.
        self.build_base_hierarchy()
        
        pc = self.parallel_controller
        logger.info('(%d) cell sizes: %s'%(pc.rank,
                                           self.cell_sizes.get_npy_array()))
        logger.info('(%d) cell step: %s'%(pc.rank,
                                           self.cell_size_step))
        self.update()

        self.py_reset_jump_tolerance()

        self.initialized = True

        # update the processor maps now.
        self.glb_update_proc_map()

    def update_status(self):
        """
        Sets the is_dirty to to true, We cannot decide the dirtyness of this
        cell manager based on just the particle arrays here, as cell managers on
        other processors could have become dirty at the same time.

        We also force an update at this point. Reason being, the delayed updates
        can cause stale neighbor information to be used before the actual update
        happens. 
        """
        self.set_dirty(True)

        self.update()

        return 0

    def build_base_hierarchy(self):
        """
        Build the initial hierachy tree.
        
        This function is similar to the function in the CellManager, except that
        the cells used are the Parallel variants.
        """
        leaf_size = self.cell_sizes.get(0)
        cell_list = []

        # create a leaf with all the particles.
        leaf_cell = ParallelLeafCell(id=IntPoint(0, 0, 0), cell_manager=self,
                                     cell_size=leaf_size, level=0,
                                     jump_tolerance=cell.INT_INF(), pid=self.pid)

        num_arrays = len(leaf_cell.arrays_to_bin)

        for i in range(num_arrays):
            parray = leaf_cell.arrays_to_bin[i]
            num_particles = parray.get_number_of_particles()
            index_arr_source = numpy.arange(num_particles, dtype=numpy.long)
            index_arr = leaf_cell.index_lists[i]
            index_arr.resize(num_particles)
            index_arr.set_data(index_arr_source)

        cell_list.append(leaf_cell)

        # for each intermediate level in the hierarchy create a
        # ParallelNonLeafCell
        for i in range(1, self.num_levels):
            inter_cell = ParallelNonLeafCell(id=IntPoint(0, 0, 0),
                                             cell_manager=self,
                                             cell_size=self.cell_sizes.get(i),
                                             level=i, pid=self.pid)
            # add the previous level cell to this cell.
            inter_cell.cell_dict[IntPoint(0, 0, 0)] = cell_list[i-1]
            cell_list.append(inter_cell)

        self.root_cell.py_clear()
        self.root_cell.cell_dict[IntPoint(0, 0, 0)] = \
            cell_list[self.num_levels-1]

        # build the hierarchy list also
        self.update_cell_hierarchy_list()        
        
    def initial_info_exchange(self):
        """
        Initial information exchange among processors.

        The bounds and h values are exchanged amoung all the processors.

        Based on the bounds and h values, the origin an cell sizes are computed.
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
        """
        Sets up the origin from the global bounds.

        Find the max bound range from x, y and z.
        Use bounds_min - max_range as the origin.

        """
        bounds_range = [0, 0, 0]
        bounds_range[0] = self.glb_bounds_max[0] - self.glb_bounds_min[0]
        bounds_range[1] = self.glb_bounds_max[1] - self.glb_bounds_min[1]
        bounds_range[2] = self.glb_bounds_max[2] - self.glb_bounds_min[2]
        
        max_range = max(bounds_range)

        self.origin.x = self.glb_bounds_min[0] - max_range
        self.origin.y = self.glb_bounds_min[1] - max_range
        self.origin.z = self.glb_bounds_min[2] - max_range

        self.root_cell.origin.x = self.origin.x
        self.root_cell.origin.y = self.origin.y
        self.root_cell.origin.z = self.origin.z

        pc = self.parallel_controller
        logger.info('(%d) Origin : %s'%(pc.rank,
                                        str(self.origin)))

    def setup_cell_sizes(self):
        """
        Sets up the cell sizes to use from the 'h' values.

        The smallest cell size is set to 2*max_radius_scale*min_h
        The larger cell size is set to 2*smallest_cell_size

        Set the number of levels to 2.
        """
        self.min_cell_size = 2*self.max_radius_scale*self.glb_max_h
        self.max_cell_size = 2*self.min_cell_size
        self.num_levels = 2
        pc = self.parallel_controller
        logger.info('(%d) cell sizes : %f %f'%(pc.rank, self.min_cell_size, 
                                               self.max_cell_size))

    def setup_processor_map(self):
        """
        Setup information for the processor map.
        """
        proc_map = self.proc_map
        proc_map.cell_manager = self
        
        proc_map.origin.x = self.origin.x
        proc_map.origin.y = self.origin.y
        proc_map.origin.z = self.origin.z

        proc_map.pid = self.parallel_controller.rank
        
        # use a bin size of thrice the largest cell size
        cell_sizes = self.cell_sizes
        max_size = cell_sizes.get(cell_sizes.length-1)
        proc_map.bin_size = max_size*3.0        

    def glb_update_proc_map(self):
        """
        Brings the processor map up-to-date globally.
        
        After a call to this function, all processors, should have identical
        processor maps.

        **Algorithm**:
        
            - bring local data up to data.
            - receive proc_map from children if any.
            - merge with current p_map.
            - send to parent if not root.
            - receive updated proc_map from root.
            - send updated proc_map to children.

        """
        pc = self.pc
        comm = pc.comm
        self.proc_map.update()

        logger.debug('Local Updated proc map : %s'%(self.proc_map))
        
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
            self.proc_map.p_map.clear()
            self.proc_map.p_map.update(updated_proc_map.p_map)

        # send updated data to children.
        for c_rank in pc.children_proc_ranks:
            comm.send(self.proc_map, dest=c_rank, 
                      tag=TAG_PROC_MAP_UPDATE)

        # setup the region neighbors.
        self.proc_map.find_region_neighbors()

        logger.debug('Updated processor map : %s'%(self.proc_map))
        
    def remove_remote_particles(self):
        """
        Remove all remote particles from the particle arrays.
        These particles are those that have their 'local' flag set to 0.
        
        """
        for parray in self.arrays_to_bin:
            parray.remove_flagged_particles(flag_name='local', flag_value=0)

    def _find_min_max_of_property(self, prop_name):
        """
        Find the min and max values of the property prop_name among all arrays
        that have been binned.
        """
        min = 1e20
        max = -1e20

        num_particles = 0
        
        for arr in self.arrays_to_bin:
            num_particles += arr.get_number_of_particles()
            
            if arr.get_number_of_particles() == 0:
                continue

            min_prop = numpy.min(arr.get(prop_name))
            max_prop = numpy.max(arr.get(prop_name))

            if min > min_prop:
                min = min_prop
            if max < max_prop:
                max = max_prop

        if num_particles == 0:
            return 1e20, -1e20

        return min, max

    def update_remote_particle_properties(self, props=None):
        """
        Update the properties of all copies of remote particles with this
        processor. 
        """
        self.root_cell.update_remote_particle_properties(props)

    def update_property_bounds(self):
        """
        Updates the min and max values of all properties in all particle arrays
        that are being binned by this cell manager.

        Not sure if this is the correct place for such a function.

        """
        pass
        

        
