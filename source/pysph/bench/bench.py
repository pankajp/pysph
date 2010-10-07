#! /usr/bin/env python
"""module to run timings test code

Modules to time-test can be specified in two ways
 * bench modules can be automatically collected from directory where this
   file is present if they are cython modules with sources having extension
   `.pyx`. The modules are compiled if they are not already
 * modules (python/cython module name w/o extension) can be passed as
   commandline arguments to time-test the specified modules

The bench modules are special modules having a callable `bench` defined
which returns a list of a dict having string (name of bench) keys and
float (time taken) values. The list is only as a way group different tests.
The modules may implement the bench function in whichever way they deem fit.

To run bench modules which need mpi to execute multiple processes,
name the bench module as "mpi<num_procs>_<bench_name>.pyx",
replacing <num_procs> with the number of processes in which to run the bench
and <bench_name> with the name of you would use for the file.
An easy way to run in different number of processes is to create symlinks with
different names.
The result of a parallel bench is that returned by the bench function
of the root process.

The results of all the bench tests are displayed in a tabular format
Any output from the test modules id redirected to file `bench.log`
Output from mpi runs is redirected to `mpirunner.log.<rank>'
"""

import os
import sys

# local relative import
import setup

def list_pyx_extensions(path):
    """list the files in the path having .pyx extension w/o the extension"""
    ret = [f[:-4] for f in os.listdir(path) if f[-3:]=='pyx' and f[0]!='_']
    ret.sort()
    return ret

def mpirun(bench_name, num_procs):
    from mpi4py import MPI
    comm = MPI.COMM_SELF.Spawn(sys.executable,
                               args=['mpirunner.py'],
                               maxprocs=num_procs)
    
    comm.bcast(bench_name, root=MPI.ROOT)
    ret = comm.recv(0)
    comm.Disconnect()
    return ret
    

def run(extns=None, dirname=None, num_runs=1):
    """run the benchmarks in the modules given
    
    `extns` is names of python modules to benchmark (None => all cython
                extensions in dirname)
    `dirname` is the directory where the modules are found (None implies
                current directory
    `num_runs` is the number of times to run the tests, the minimum value
                is reported over all the runs
    """
    
    if dirname is None:
        dirname = os.path.abspath(os.curdir)
    olddir = os.path.abspath(os.curdir)
    os.chdir(dirname)
    
    if extns is None:
        extns = list_pyx_extensions(os.curdir)
    print 'Running benchmarks:', ', '.join(extns)
    
    # this is needed otherwise setup will take arguments and do something else
    sys.argvold = sys.argv[:]
    sys.argv = sys.argv[:1]
    
    # compile the bench .pyx files
    setup.compile_extns(extns, dirname)#, [os.path.join(dirname,'..','..')])
    
    logfile = open('bench.log', 'w')
    outtext = ''
    
    for bench_name in extns:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = logfile
        mpi = False
        if bench_name.startswith('mpi'):
            mpi = True
            num_procs = int(bench_name.lstrip('mpi').split('_')[0])
        try:
            # bench to be run in mpi
            if mpi:
                res = mpirun(bench_name, num_procs)
            # normal single process bench
            else:
                bench_mod = __import__(bench_name)
                res = bench_mod.bench()
        except:
            stderr_orig.write('Failure running bench %s: %s\n' %(bench_name,
                                    str(sys.exc_info())))
            continue
        # take minimum over `num_runs` runs
        for i in range(num_runs-1):
            # bench to be run in mpi
            if mpi:
                r = mpirun(bench_name, num_procs)
            # normal single process bench
            else:
                r = bench_mod.bench()
            for jn,j in enumerate(res):
                for k,v in j.items():
                    j[k] = min(v, r[jn].get(k, 1e1000))
        
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        if mpi:
            s = bench_name.split('_',2)[1]+' %d\n'%num_procs
            s += '#'*len(s)
            print s
            outtext += s + '\n'
        else:
            s = bench_name + '\n' + '#'*len(bench_name)
            print s
            outtext += s + '\n'
        for func in res:
            for k in sorted(func.keys()):
                s = k.ljust(40) + '\t%g'%func[k]
                print s
                outtext += s + '\n'
            print
            outtext += '\n'
    logfile.write(outtext)
    logfile.close()
    
    sys.argv = sys.argvold
    os.chdir(olddir)

if __name__ == '__main__':
    print sys.argv
    if '-h' in sys.argv or '--help' in sys.argv:
        print '''usage:
        python setup.py [extension1, [extension2, [...]]]
        
        runs the bench extensions present in the current directory
        '''
    elif len(sys.argv) > 1:
        # run specified extensions
        run(sys.argv[1:])
    else:
        # run all extensions found in current directory
        run()
    