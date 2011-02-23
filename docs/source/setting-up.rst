****************************
Setting up your system 
****************************

To install PySPH on your system, you need to ensure that the following
dependencies are met.


*  Numpy


All array support for PySPH is provided by numpy
http://numpy.scipy.org/. This may be installed using the script::

    sudo apt-get install python-numpy

*  MPI


To run PySPH across distributed processors via the `Message Passing
Interface`_, you need a working implementation of MPI::

    sudo apt-get install openmpi-bin libopenmpi-dev

.. _Message Passing Interface: http://www.open-mpi.org/

*  setuptools and virtualenv


This is not strictly a dependency but we recommend it especially if
you do not have root access on the machine you wish to run PySPH.::

    sudo apt-get install python-setuptools python-virtualenv


Once you have `virtualenv`_ installed, you can create an
environment within which PySPH can be installed.::

    $ cd ~
    $ virtualenv envs/pysph
    $ source envs/pysph/bin/activate
    $(pysph)

Within this environment, you may install all python related
dependencies as simply as::

    $(pysph) easy_install <package_name>

.. _virtualenv: http://virtualenv.openplans.org/


*  Cython


`Cython`_ is essential to build the source code.::

    sudo apt-get install python-dev
    $(pysph) easy_install cython

.. _Cython: http://cython.org/

*  Sphinx


This is not strictly a dependency.  `Sphinx`_ is used for building the
documentation.::

    $(pysph) easy_install sphinx

..  _Sphinx: http://sphinx.pocoo.org/

*  Nose


This is not strictly a dependency. `nose`_ is used for running PySPH tests.::

    $(pysph) easy_install nose

.. _nose: http://code.google.com/p/python-nose/

*  mpi4py


To run the parallel version of PySPH, you need `mpi4py`_ ::

   $(pysph) easy_install mpi4py

Note that mpi4py requires a working implementation of MPI on your
machine. 

.. _mpi4py: http://mpi4py.scipy.org