"""
Class to hold information required for a parallel solver.
"""
# logger imports
import logging
logger = logging.getLogger()

# mpi imports
from mpi4py import MPI

# numpy imports
import numpy

# local imports


class ParallelController:
    """
    Class to hold information requireed for a parallel solver.
    """
    
    # some message tags that will be used while communication appear as class
    # attributes here.
    

    def __init__(self, solver=None, cell_manager=None, *args, **kwargs):
        """
        Constructor.
        """
        self.solver = solver
        
        if solver is not None:
            self.cell_manager = solver.cell_manager
        else:
            self.cell_manager = cell_manager
        
        self.comm = MPI.COMM_WORLD
        comm = self.comm
        self.num_procs = comm.Get_size()
        self.rank = comm.Get_rank()
        
        self.children_proc_ranks = []
        self.parent_rank = -1

        self.setup_control_tree_done = False
        # setup the control tree.
        self.setup_control_tree()
        
    def get_max_min(self, local_max_arr, local_min_arr,
                    result_max_arr, result_min_arr, 
                    pid_arr):
        """
        Find the global max and min values.

        **Parameters**
        
         - local_min_arr,  local_max_arr - array containing local max and min
         values. 
         - result_min_arr, result_max_arr - array in which  the max and min values
         will be returned.

        """
        pass

    def get_glb_min_max(self, local_min_dict, local_max_dict):
        """
        Given various values in the dictionaries, find the global max and min of
        those values. All processors are expected to pass on dictionaries with
        the same set of keys. The resuly min and max dictionaries are returned.

        **Algorithm**

            - if any child exists, receive from each child
            - find max min among the three dicts.
            - if parent exists pass data to parent.
            - receive updated data from parent.
            - pass data to children.

        """
        c_data = {}
        comm = self.comm

        logger.debug('Min dict : %s'%(local_min_dict))
        logger.debug('Max dict : %s'%(local_max_dict))

        for c_rank in self.children_proc_ranks:
            c_data[c_rank] = comm.recv(source=c_rank)
            
        # merge data with data with current processor.
        partial_merged_data = {'min':{}, 'max':{}}
        
        c_data[self.rank] = {'min':local_min_dict, 'max':local_max_dict}

        self._merge_min_max_dicts(c_data, partial_merged_data)

        # if parent exists, send data to parent
        if self.parent_rank > -1:
            comm.send(partial_merged_data, dest=self.parent_rank)
            
            # get the complete merged data from parent
            complete_merged_data = comm.recv(source=self.parent_rank)
        else:
            complete_merged_data = partial_merged_data
            
        # set the output dicts
        result_min_dict = {}
        result_max_dict = {}
        result_min_dict.update(complete_merged_data['min'])
        result_max_dict.update(complete_merged_data['max'])
                              
        # send complete merged data to all children
        for c_rank in self.children_proc_ranks:
            comm.send(complete_merged_data, dest=c_rank)

        return result_min_dict, result_max_dict
        
    def _merge_min_max_dicts(self, collected_dicts, result_dict):
        """
        Merge data from collected from various procs and place result
        in result_dict.

        """
        temp_data = {'min':{}, 'max':{}}
        
        for rank in collected_dicts.keys():
            temp_data['min'][rank] = collected_dicts[rank]['min']
            temp_data['max'][rank] = collected_dicts[rank]['max']

        min_data = temp_data['min']
        max_data = temp_data['max']

        result_max = result_dict['max']
        result_min = result_dict['min']

        for rank in min_data.keys():
            proc_min_data = min_data[rank]
            for prop in proc_min_data.keys():
                if proc_min_data[prop] is None:
                    result_min[prop] = None
                    continue
                curr_min = result_min.get(prop)
                if curr_min is None:
                    result_min[prop] = proc_min_data[prop]
                elif curr_min > proc_min_data[prop]:
                    result_min[prop] = proc_min_data[prop]
        
        for rank in max_data.keys():
            proc_max_data = max_data[rank]
            for prop in proc_min_data.keys():
                if proc_min_data[prop] is None:
                    result_max[prop] = None
                    continue
                curr_max = result_max.get(prop)
                if curr_max is None:
                    result_max[prop] = proc_max_data[prop]
                elif curr_max < proc_max_data[prop]:
                    result_max[prop] = proc_max_data[prop]

    def setup_control_tree(self):
        """
        Setup the processor ids of the parent and children.

        """
        if self.setup_control_tree_done == True:
            return

        # find the child ids.
        l_child_rank = self.rank*2 + 1
        r_child_rank = self.rank*2 + 2

        if l_child_rank >= self.num_procs:
            self.l_child_rank = -1
            self.r_child_rank = -1
        else:
            self.l_child_rank = l_child_rank
            if r_child_rank >= self.num_procs:
                self.r_child_rank = -1
            else:
                self.r_child_rank = r_child_rank

        # find the parent ids.
        if self.rank == 0:
            # root node - no parent
            self.parent_rank = -1
        else:
            if self.rank % 2 == 0:
                # right child
                self.parent_rank = self.rank/2 - 1
            else:
                # left child.
                self.parent_rank = (self.rank-1)/2
        

        if self.l_child_rank != -1:
            self.children_proc_ranks.append(self.l_child_rank)
        if self.r_child_rank != -1:
            self.children_proc_ranks.append(self.r_child_rank)

        logger.info('(%d) Parent %d, L_Child %d, R_Child %d'%(self.rank,
                                                              self.parent_rank,
                                                              self.l_child_rank,
                                                              self.r_child_rank))

        self.setup_control_tree_done = True
