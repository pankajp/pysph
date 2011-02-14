''' Module to run bench modules which need to be run in mpi

This module imports the given module to run, and returns the result
of the bench functions of the modules. Also results are written to
mpirunner.log file

Usage:
1. Print the result in formatted form:
$ mpiexec -n <num_procs> python mpirunner.py <bench_name>

1. Print the result dictionary in pickled form (useful in automation):
$ mpiexec -n <num_procs> python mpirunner.py p <bench_name>
'''

from mpi4py import MPI
import sys
import pickle

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def mpirun(args=None):
    pkl = False
    if args is None:
        comm = MPI.Comm.Get_parent()
        #rank = comm.Get_rank()
        bench_name = comm.bcast('', root=0)
    else:
        if args[0] == 'p':
            pkl = True
            bench_name = args[1]
        else:
            bench_name = args[0]
    logfile = open('mpirunner.log.%d'%rank, 'w')
    stdout_orig = sys.stdout
    stderr_orig = sys.stderr
    sys.stdout = sys.stderr = logfile
    
    bench_mod = __import__(bench_name)
    res = bench_mod.bench()
    sys.stdout = stdout_orig
    sys.stderr = stderr_orig
    logfile.close()
    if rank != 0: return
    
    outtext = ''
    s = bench_name.split('_',1)[1]+' %d\n'%size
    s += '#'*len(s)
    outtext += s + '\n'
    for func in res:
        for k in sorted(func.keys()):
            s = k.ljust(40) + '\t%g'%func[k]
            outtext += s + '\n'
    logfile = open('mpirunner.log', 'w')
    logfile.write(outtext)
    logfile.write('\n')
    logfile.close()
    
    if args is None:
        comm.send(res, 0)
    elif pkl:
        sys.stdout.write(pickle.dumps(res))
    else:
        print outtext

if __name__ == '__main__':
    mpirun(sys.argv[1:])
    