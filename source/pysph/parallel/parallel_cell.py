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
from pysph.solver.base import Base
from pysph.base import cell
from pysph.parallel.parallel_controller import ParallelController

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
        hl = cm.hierarchy_list
        cells = hl[len(hl)-2]
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
        for cid in self.p_map:
            rep += 'bin %s with processors : %s'%(cid, self.p_map[cid])
            rep += '\n'
        rep += 'Region neighbors : %s'%(self.nbr_procs)
        return rep
            
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
        if cell_manager is None:
            self.pid = pid
        else:
            self.pid = cell_manager.pid
            self.parallel_controller = cell_manager.parallel_controller

    def update(self, data):
        """
        Update particle information.
        """
        collected_data = {}
        new_cells = {}
        remote_cells = {}

        for cid, smaller_cell in self.cell_dict.iteritems():
            smaller_cell.update(collected_data)
            
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
                        remote_cells[cid] = r_cell
                    else:
                        r_cell.add_particles(smaller_cell)
                else:
                    # add it to the current cells.
                    smaller_cell_1.add_particles(smaller_cell)
            else:
                # check if this cell is is new cells
                smaller_cell_1 = new_cells.get(cid)
                if smaller_cell_1 is None:
                    new_cells[cid] = smaller_cell
                else:
                    smaller_cell_1.add_particles(smaller_cell)

        # we now have two lists - remote_cells and new_cells.
        
                    

################################################################################
# `ParallelCellManager` class.
################################################################################
class ParallelCellManager(cell.CellManager):
    """
    Cell manager for parallel invocations.
    """
    TAG_PROC_MAP_UPDATE = 1
    def __init__(self, arrays_to_bin=[], particle_manager=None,
                 min_cell_size=0.1, max_cell_size=0.5, origin=Point(0, 0, 0),
                 num_levels=2, initialize=True,
                 parallel_controller=None,
                 max_radius_scale=2.0):
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
        self.root_cell = ParallelRootCell(cell_manager=self)
        self.proc_map = ProcessorMap(cell_manager=self)
                
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

        self.py_compute_cell_sizes(self.min_cell_size, self.max_cell_size,
                                   self.num_levels, self.cell_sizes)

        # setup information for the processor map.
        self.setup_processor_map()

        # buid a basic hierarchy with one cell at each level, and all
        # particles in the leaf cell.
        self.py_build_base_hierarchy()
        
        pc = self.parallel_controller
        logger.info('(%d) cell sizes: %s'%(pc.rank,
                                           self.cell_sizes.get_npy_array()))
        logger.info('(%d) cell step: %s'%(pc.rank,
                                           self.cell_size_step))
        logger.info('(%d) hierarchy: %s'%(pc.rank,
                                          self.hierarchy_list))
        self.update()

        self.py_reset_jump_tolerance()

        self.initialized = True

        for l in self.hierarchy_list:
            k_list = l.keys()
            logger.info('Num cells in hierarchy : %d'%(len(k_list)))
            for k in k_list:
                logger.info('Cell id : %s'%(l[k].id))
                if l[k].py_is_leaf():
                    logger.info('Particle ids : %s'%(l[k].index_lists[0].get_npy_array()))
                logger.info('Cell num particles : %d'%(
                        l[k].py_get_number_of_particles()))
                logger.info('Cell size : %f'%(l[k].cell_size))

        # update the processor maps now.
        self.glb_update_proc_map()
                
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
        
        glb_min, glb_max = pc.get_glb_min_max_from_dict(data_min, data_max)

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
                                   tag=self.TAG_PROC_MAP_UPDATE)
            self.proc_map.merge(c_proc_map)

        # we now have partially merged data, send it to parent is not root.
        if pc.parent_rank > -1:
            comm.send(self.proc_map, dest=pc.parent_rank,
                      tag=self.TAG_PROC_MAP_UPDATE)
            # receive updated proc map from parent
            updated_proc_map = comm.recv(source=pc.parent_rank,
                                         tag=self.TAG_PROC_MAP_UPDATE)
            # set our proc data with the updated data.
            self.proc_map.p_map.clear()
            self.proc_map.p_map.update(updated_proc_map.p_map)

        # send updated data to children.
        for c_rank in pc.children_proc_ranks:
            comm.send(self.proc_map, dest=c_rank, 
                      tag=self.TAG_PROC_MAP_UPDATE)

        # setup the region neighbors.
        self.proc_map.find_region_neighbors()

        logger.debug('Updated processor map : %s'%(self.proc_map))

        # setup the region neighbors
        
    def force_update(self):
        """
        Update without regard to if particle arrays are dirty or not.
        """
        pass

    def update_neighbor_information(self, *props):
        """
        """
        pass

    def update_neighbor_data(self, *props):
        """
        """
        pass

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
            return None, None

        return min, max
