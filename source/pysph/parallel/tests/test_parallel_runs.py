"""
Module to run the parallel files in pysph.parallel.tests using mpiexec and
report their success/failure results

Add a function to the ParalelTest class corresponding to a parallel script to
be tested.
This is done till better strategy for parallel testing is implemented
"""

import unittest
from subprocess import Popen, PIPE
from threading import Timer
import os
import sys

directory = os.path.dirname(os.path.abspath(__file__))

# seconds after which the spawned process will be killed
timeout = 5.0

# number of processes to start
nprocs = 2

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()
    #time.sleep(0.5)
    #process.terminate()

def run_mpi_script(filename, path=None):
    if path is None:
        path = directory
    path = os.path.join(path, filename)
    cmd = ['mpiexec','-n', str(nprocs), sys.executable, path]
    print 'running test:', cmd
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    timer = Timer(timeout, kill_process, [process])
    timer.start()
    out, err = process.communicate()
    timer.cancel()
    retcode = process.returncode
    if retcode:
        msg = 'test ' + filename + ' failed with returncode ' + str(retcode)
        print out
        print err
        print '#'*80
        print msg
        print '#'*80
        raise RuntimeError, msg
    return retcode, out, err

class ParallelTest(unittest.TestCase):
    """ Testcase to run all parallel test scripts using mpiexec. """
    def test_controller_check(self):
        run_mpi_script('controller_check.py')
    
    def test_lb_check_1d(self):
        run_mpi_script('lb_check_1d.py')
    
    def test_lb_check_2d(self):
        run_mpi_script('lb_check_2d.py')
    
    def test_remote_data_copy(self):
        run_mpi_script('remote_data_copy.py')
    
    def test_parallel_cell_check(self):
        run_mpi_script('parallel_cell_check.py')

#for filename in tests:
#    test_name = 'test_%s' %(filename)
#    testfunc = gen_func(filename)
#    setattr(ParallelTest, test_name, testfunc)

if __name__ == "__main__":
    unittest.main()
    
