""" Test the communication of particle arrays with MPI

Run this test with two processes like so:

mpirun -n 2 <path to python> test_particle_array_pickling.py

Each process creates a partile array with default properties and some
temporary variables that are prefixed with an underscore eg:

_rho, _p

These variables should not be communicated with MPI  

"""

import pysph.base.api as base
import numpy

from mpi4py import MPI
comm = MPI.COMM_WORLD
num_procs = comm.Get_size()
pid = comm.Get_rank()

x = numpy.linspace(0,1,11)
pa = base.get_particle_array(name='test'+str(pid), x=x)

pa.add_property({'name':'_rho'})
pa.add_property({'name':'_p'})

if pid == 0:

    # make sure the sending data defines the underscored variables
    assert pa.properties.has_key('_rho')
    assert pa.properties.has_key('_p')

    comm.send(obj=pa, dest=1)

if pid == 1:

    pa2 = comm.recv(source=0)

    # make sure that the underscored variables are not there
    assert not pa2.properties.has_key('_rho')
    assert not pa2.properties.has_key('_p')

    # now append the received array to the local array
    pa.append_parray(pa2)

    print "OK"
