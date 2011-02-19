""" Contains class to perform load balancing using METIS[1]/SCOTCH[2]

[1] METIS: http://glaros.dtc.umn.edu/gkhome/views/metis
[2] SCOTCH: http://www.labri.fr/perso/pelegrin/scotch/

Note: Either of METIS/SCOTCH is acceptable. Installing one of these is enough.
First METIS is attempted to load and if it fails SCOTCH is tried. SCOTCH is
used in the METIS compatibility mode. Only the function `METIS_PartGraphKway`
is used from either of the libraries
"""

# logging imports
import logging
logger = logging.getLogger()

# local imports
from pysph.base.cell import py_construct_immediate_neighbor_list
from load_balancer_mkmeans import LoadBalancerMKMeans

import sys
import ctypes
from ctypes import c_int32 as c_int


if sys.platform.startswith('linux'):
    try:
        libmetis = ctypes.cdll.LoadLibrary('libmetis.so')
    except OSError:
        try:
            libmetis = ctypes.cdll.LoadLibrary('libscotchmetis.so')
        except OSError:
            raise ImportError('could not load METIS library, try installing '
                        'METIS/SCOTCH and ensure it is in LD_LIBRARY_PATH')
elif sys.platform.startswith('win'):
    try:
        libmetis = ctypes.cdll.LoadLibrary('metis')
    except OSError:
        try:
            libmetis = ctypes.cdll.LoadLibrary('scotchmetis')
        except OSError:
            raise ImportError('could not load METIS library, try installing '
                        'METIS/SCOTCH and ensure it is in LD_LIBRARY_PATH')
else:
    raise ImportError('sorry, donno how to use ctypes (for METIS/SCOTCH'
            'load_balancing) on non-linux/win platform, any help appreciated')

METIS_PartGraphKway = libmetis.METIS_PartGraphKway
c_int_p = ctypes.POINTER(c_int)
METIS_PartGraphKway.argtypes = [c_int_p, c_int_p, c_int_p, c_int_p, c_int_p,
                    c_int_p, c_int_p, c_int_p, c_int_p, c_int_p, c_int_p]


def cargs_from_wadj(xadj, adjncy, vwgt, bid_index, nparts):
    """ return the ctype arguments for metis from the adjacency data
    
    Parameters:
    -----------
        - xadj,adjncy,vwgt: lists containing adjacency data in CSR format as
            required by :func:`METIS_PartGraphKway` (check METIS manual)
        - bid_index: dict mapping bid to index in the adjacency data
        - nparts: number of partitions to make of the graph
    
    Returns:
    --------
        - n, xadj, adjncy, vwgt, adjwgt, wgtflag, numflag, nparts, options,
            edgecut, part: the arguments for the :func:`METIS_PartGraphKway`
            functions in ctype data format (all are pointers to c_int32)
    """
    n = len(xadj)-1
    c_n = (c_int*1)(n)
    c_numflag = (c_int*1)()
    c_adjwgt = None
    c_nparts = (c_int*1)(nparts)
    c_options = (c_int*5)()
    c_edgecut = (c_int*1)()
    c_part = (c_int*n)()
    
    c_xadj = (c_int*(n+1))()
    c_xadj[:] = xadj
    
    c_adjncy = (c_int*len(adjncy))()
    c_adjncy[:] = adjncy
    
    if vwgt:
        c_vwgt = (c_int*n)()
        c_vwgt[:] = vwgt
        c_wgtflag = (c_int*1)(2)
    else:
        c_vwgt = None
        c_wgtflag = (c_int*1)(0)
    
    return (c_n, c_xadj, c_adjncy, c_vwgt, c_adjwgt, c_wgtflag, c_numflag,
            c_nparts, c_options, c_edgecut, c_part)

def wadj_from_adj_list(adj_list):
    """ return vertex weights and adjacency information from adj_list
    as returned by :func:`adj_list_from_blocks` """
    bid_index = {}
    xadj = [0]
    adjncy = []
    vwgt = []
    for i,tmp in enumerate(adj_list):
            bid_index[tmp[0]] = i
    for bid, adjl, np in adj_list:
        adjncy.extend((bid_index[b] for b in adjl))
        xadj.append(len(adjncy))
        vwgt.append(np)
    return xadj, adjncy, vwgt, bid_index

def adj_list_from_blocks(block_proc, proc_block_np):
    """ return adjacency list information for use by METIS partitioning
    
    Arguments:
    ----------
        - block_proc: dict mapping bid:proc
        - proc_block_map: list of dict bid:np, in sequence of the process to
            which block belongs
    
    Returns:
    --------
        - adj_list: list of 3-tuples, one for each block in proc-block_np
        The 3-tuple consists of (bid, adjacent bids, num_particles in bid)
    """
    adj_list = []
    nbrs = []
    i = 0
    for blocks in proc_block_np:
        for bid, np in blocks.iteritems():
            nbrs[:] = []
            adjl = []
            py_construct_immediate_neighbor_list(bid, nbrs, False)
            for nbr in nbrs:
                if nbr in block_proc:
                    adjl.append(nbr)
            adj_list.append((bid, adjl, np))
            i += 1
    return adj_list

def lb_metis(block_proc, proc_block_np):
    """ Partition the blocks in proc_block_np using METIS
    
    Arguments:
    ----------
        - block_proc: dict mapping bid:proc
        - proc_block_map: list of dict bid:np, in sequence of the process to
            which block belongs
    
    Returns:
    --------
        - block_proc: dict mapping bid:proc for the new partitioning generated
            by METIS
    """
    adj_list = adj_list_from_blocks(block_proc, proc_block_np)
    xadj, adjncy, vwgt, bid_index = wadj_from_adj_list(adj_list)
    c_args = cargs_from_wadj(xadj, adjncy, vwgt, bid_index, len(proc_block_np))
    
    METIS_PartGraphKway(*c_args)
    ret = c_args[-1]
    
    ret_block_proc = {}
    for bid,bindex in bid_index.iteritems():
        ret_block_proc[bid] = ret[bindex]
    return ret_block_proc

###############################################################################
# `LoadBalancerMetis` class.
###############################################################################
class LoadBalancerMetis(LoadBalancerMKMeans):
    def __init__(self, **args):
        LoadBalancerMKMeans.__init__(self, **args)
        self.method = 'serial_metis'
    
    def load_balance_func_serial_metis(self, **args):
        """ serial load balance function which uses METIS to do the partitioning
        
        calls the :class:Loadbalancer :meth:`load_balance_func_serial`
        """
        self.load_balance_func_serial('metis', **args)
        
    def load_redistr_metis(self, block_proc, proc_block_np, **args):
        """ function to redistribute the cells amongst processes using METIS
        
        This is called by :class:Loadbalancer :meth:`load_balance_func_serial`
        """
        block_proc = lb_metis(block_proc, proc_block_np)
        self.particles_per_proc = [0]*len(proc_block_np)
        block_np = {}
        for b in proc_block_np:
            block_np.update(b)
        for bid,proc in block_proc.iteritems():
            self.particles_per_proc[proc] += block_np[bid]
        self.balancing_done = True
        return block_proc, self.particles_per_proc

###############################################################################

