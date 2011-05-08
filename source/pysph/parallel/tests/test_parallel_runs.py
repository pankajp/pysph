"""
Module to run the parallel files in pysph.parallel.tests using mpiexec and
report their success/failure results

Add a function to the ParallelTest class corresponding to a parallel script to
be tested.
This is done till better strategy for parallel testing is implemented
"""

import unittest
from subprocess import Popen, PIPE
from threading import Timer
import os
import sys

directory = os.path.dirname(os.path.abspath(__file__))

def kill_process(process):
    print 'KILLING PROCESS ON TIMEOUT'
    process.kill()

def run_mpi_script(filename, nprocs=2, timeout=5.0, path=None):
    """ run a file python script in mpi
    
    Parameters:
    -----------
    filename - filename of python script to run under mpi
    nprocs - (2) number of processes of the script to run
    timeout - (5) time in seconds to wait for the script to finish running,
        else raise a RuntimeError exception
    path - the path under which the script is located
        Defaults to the location of this file (__file__), not curdir
    
    """
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
    
    def test_lb_check_parallel(self):
        run_mpi_script('lb_check_parallel.py', 2)
    
    def test_remote_data_copy(self):
        run_mpi_script('remote_data_copy.py')
    
    def test_parallel_cell_check(self):
        for i in range(1,5):
            run_mpi_script('parallel_cell_check.py', i)
    
    def test_parallel_cell_check2(self):
        run_mpi_script('parallel_cell_check2.py', 2)
    
    def test_parallel_cell_check3(self):
        for i in range(1,6):
            run_mpi_script('parallel_cell_check3.py', i)
    
    def test_share_data(self):
        for i in range(1,6):
            run_mpi_script('share_data.py', i)

    def test_particle_array_pickling(self):
        for i in range(2,3):
            run_mpi_script('particle_array_pickling.py', i)
        

#for filename in tests:
#    test_name = 'test_%s' %(filename)
#    testfunc = gen_func(filename)
#    setattr(ParallelTest, test_name, testfunc)

if __name__ == "__main__":
    unittest.main()
    
