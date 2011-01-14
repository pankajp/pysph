""" Test the share_data function for various cases

cases to run are chosen based on the size of the MPI.COMM_wORLD

case 1: for 5 processes
Processors arrangement:
  4
0 1 2 3

Nbr lists:
0: 1,4
1: 0,2,4
2: 1,3,4
3: 2
4: 0,1,2

case 2: for 2 processes
both neighbors of each other

case 3,4,5: n processes (n>1 for case 5)
all neighbors of each other

"""

# mpi imports
from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
rank = comm.Get_rank()

from pysph.parallel.parallel_cell import share_data

def case1(multi=True, to_self=False):
    """ 5 processes """
    if num_procs != 5: return
    nbr_lists = [[1,4],
                 [0,2,4],
                 [1,3,4],
                 [2],
                 [0,1,2],
                 ]
    nbr_list = nbr_lists[rank]
    if to_self: nbr_list.append(rank)
    proc_data = {}
    for nbr in nbr_list:
        proc_data[nbr] = (rank, nbr)
    recv_data = share_data(rank, nbr_list, proc_data, comm, multi=multi)
    assert len(recv_data) == len(nbr_list)
    if multi:
        for pid,data in recv_data.iteritems():
            assert data == (pid, rank)
    else:
        for pid,data in recv_data.iteritems():
            for pid2,data2 in data.iteritems():
                assert data2 == (pid, pid2)

def case2():
    """ 2 processes """
    if num_procs != 2: return
    nbr_list = [1-rank]
    proc_data = {1-rank:(rank, 1-rank)}
    recv_data = share_data(rank, nbr_list, proc_data, comm, multi=True)
    print rank, recv_data
    
def case3(multi=True, to_self=False):
    """  all-to-all communication """
    nbr_list = range(num_procs)
    if not to_self: nbr_list.remove(rank)
    proc_data = {}
    for nbr in nbr_list:
        proc_data[nbr] = (rank, nbr)
    recv_data = share_data(rank, nbr_list, proc_data, comm, multi=multi)
    assert len(recv_data) == len(nbr_list)
    if multi:
        for pid,data in recv_data.iteritems():
            assert data == (pid, rank)
    else:
        print rank, recv_data
        for pid,data in recv_data.iteritems():
            for pid2,data2 in data.iteritems():
                assert data2 == (pid, pid2)

def case4(multi=True, to_self=False):
    """ all-to-all oneway communication """
    nbr_list = range(num_procs)
    if not to_self: nbr_list.remove(rank)
    proc_data = {}
    for nbr in nbr_list:
        proc_data[nbr] = (rank, nbr)
    recv_data = share_data(rank, nbr_list, proc_data, comm, multi=multi)
    assert len(recv_data) == len(nbr_list)
    if multi:
        for pid,data in recv_data.iteritems():
            assert data == (pid, rank)
    else:
        print rank, recv_data
        for pid,data in recv_data.iteritems():
            for pid2,data2 in data.iteritems():
                assert data2 == (pid, pid2)

def case5(multi=True, to_self=False):
    """ oneway communication to next two consecutive procs """
    send_procs = [(rank+1)%num_procs, (rank+2)%num_procs]
    recv_procs = [(rank-1)%num_procs, (rank-2)%num_procs]

    proc_data = {}
    for nbr in send_procs:
        proc_data[nbr] = (rank, nbr)
    print rank, send_procs, recv_procs, proc_data
    recv_data = share_data(rank, send_procs, proc_data, comm, multi=multi,
                           recv_procs=recv_procs)
    assert len(recv_data) == len(recv_procs)
    if multi:
        for pid,data in recv_data.iteritems():
            assert data == (pid, rank)
    else:
        print rank, recv_data
        for pid,data in recv_data.iteritems():
            for pid2,data2 in data.iteritems():
                assert data2 == (pid, pid2)


if __name__ == '__main__':
    if num_procs == 2:
        case2()

    for multi in True,False:
        for to_self in True,False:
            if num_procs == 5:
                case1(multi, to_self)
            case3(multi, to_self)
            case4(multi, to_self)
            if num_procs > 1:
                case5(multi, to_self)
