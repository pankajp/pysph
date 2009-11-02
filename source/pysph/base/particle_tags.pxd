"""
Declares various tags for particles, and functions to check them.

Note that these tags are the ones set in the 'tag' property of the particles, in
a particle array. To define additional discrete properties, one can add another
integer property to the particles in the particle array while creating them.

These tags could be considered as 'system tags' used internally to distinguish
among different kinds of particles. If more tags are needed for a particular
application, add them as mentioned above.

The is_* functions defined below are to be used in Python for tests etc. Cython
modules can directly use the enum name.

"""

cdef enum ParticleTag:
    LocalReal = 0
    LocalDummy
    RemoteReal
    RemoteDummy

cpdef bint is_local_real(long tag)
cpdef bint is_local_dummy(long tag)
cpdef bint is_remote_real(long tag)
cpdef bint is_remote_dummy(long tag)

cpdef long get_local_real_tag()
cpdef long get_local_dummy_tag()
cpdef long get_remote_real_tag()
cpdef long get_remote_dummy_tag()
