Installation
=============

:Release: |version|
:Date: |today|

There are several ways to install the PySPH package. 

Installation on Ubuntu 10.10
-----------------------------

Assuming you have administrative rights, install the following
essential dependencies

* `Numpy`_: >= 1.0.4
* `Cython`_: >= 0.14
* `setuptools`_

To install these::

    sudo apt-get install python-numpy python-dev cython python-setuptools

The following are non essential dependencies

* `MPI`_ and `mpi4py`_ >=1.2: Message Passing Interface for parallel runs
* `sphinx`_: >= 0.5 to generate documentation.
* `nose`_: to run tests

To install ::

    sudo apt-get install openmpi-bin libopenmpi-dev python-mpi4py python-sphinx python-nose

.. _Numpy: http://numpy.scipy.org/
.. _Cython: http://cython.org/
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _MPI: http://www.open-mpi.org/
.. _mpi4py: http://mpi4py.scipy.org
.. _sphinx: http://sphinx.pocoo.org/
.. _nose: http://code.google.com/p/python-nose/


Download and untar the latest source tarball from
http://pysph.googlecode.com ::

    $(pysph) tar -xvzf <tarball>

Alternatively, you make a clone of the repository as::

    hg clone https://devel.pysph.googlecode.com/hg pysph

Build and install the package as ::

    $(pysph) make develop; make install 

Once the package is installed you can test if everything is OK by
running the test suite like so::

  $ make test 

If all goes well, you have successfully installed PySPH!

To build the documentation using `Sphinx`_ do the following::

    $ cd docs
    $ make html

The docs should now be available in the ``build/html`` directory.
Point your browser at ``build/html/index.html``.


Setting up PySPH using virtualenv
----------------------------------

`Virtualenv`_ lets you set up a python environment within which you may
install python packages locally and without administrative rights. 

Ensure all dependencies are met::

       sudo apt-get install python-numpy python-dev openmpi-bin libopenmpi-dev python-virtualenv

Once you have virtualenv installed, you can create an environment
within which PySPH can be installed.::

    $ cd ~
    $ virtualenv envs/pysph
    $ source envs/pysph/bin/activate
    $(pysph)

Within this environment, you may install all python related
dependencies as simply as::

    $(pysph) easy_install cython mpi4py python-sphinx python-nose

.. _Virtualenv: http://virtualenv.openplans.org/

Download the source code from https://pysph.googlecode.com and install::
 
    $(pysph) make install
    $(pysph) make test

If all the tests pass, you have successfully installed PySPH!




