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

The results of all the bench tests are displayed in a tabular format
Any output from the test modules id redirected to file `bench.log`
"""

import os
import sys

dirname = os.path.abspath(os.path.dirname(__file__))
olddir = os.path.abspath(os.curdir)
os.chdir(dirname)

def list_pyx_extensions(path):
    """list the files in the path having .pyx extension w/o the extension"""
    return sorted([f[:-4] for f in os.listdir(path) if f.endswith('pyx')])

if len(sys.argv) > 1:
    extns = sys.argv[1:]
else:
    extns = list_pyx_extensions(os.curdir)


def run(extns=extns, num_runs=3):
    """run the benchmarks in the modules given"""
    
    print 'Running benchmarks:', ', '.join(extns)
    
    # this is needed otherwise setup will take arguments and do something else
    sys.argvold = sys.argv[:]
    sys.argv = sys.argv[:1]
    
    # this import actually compiles the bench .pyx files
    import setup
    
    for bench_name in extns:
        bench_mod = __import__(bench_name)
        logfile = open('bench.log', 'w')
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = logfile
        try:
            res = bench_mod.bench()
        except:
            stderr_orig.write('Failure running bench %s: %s\n' %(bench_name,
                                    str(sys.exc_info())))
            continue
        # take minimum over `num_runs` runs
        for i in range(num_runs-1):
            r = bench_mod.bench()
            for jn,j in enumerate(res):
                for k,v in j.items():
                    j[k] = min(v, r[jn].get(k, 1e1000))
        
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        logfile.close()
        
        print bench_name
        print '#'*len(bench_name)
        for func in res:
            for k in sorted(func.keys()):
                print k.ljust(40), '\t', func[k]
            print
        
        sys.argv = sys.argvold

if __name__ == '__main__':
    run()
    
# this is outside main to revert the value os.curdir
os.chdir(olddir)

