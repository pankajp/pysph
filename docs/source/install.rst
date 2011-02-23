Installation
=============

:Release: |version|
:Date: |today|

There are several ways to install the PySPH package. All the methods
described pertain to the installation on Ubuntu 10.10

For each of the installation methods, you need to ensure that the
following dependencies are installed.

.. toctree::
   :maxdepth: 2
   :numbered:

   setting-up.rst
   install_from_source.rst

Dependencies
-------------

The following dependencies are absolutely essential:

  * virtualenv:

  * numpy_: version 1.0.4 and above will work.

  * Cython_: version 0.12 and above.  This implies that your system have
    a working C compiler.

  * mpi4py_: version 1.2 and above.  This will require working MPI libs
    on your computer

  * setuptools_: version 0.6c9 is tested to work.  This is needed to
    build and install the package.  A later version will work better.

The following dependencies are optional but recommended:

  * nose_: This package is used for running any tests.

  * Sphinx_: version 0.5 will work.  This package is used for building
    the documentation.



  * Mayavi_: version 3.x and above. This is convenient for visualization
    and generation of VTK files.  This is entirely optional though.

Dependencies on Debian/Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Installing the dependencies on a recent flavor of Ubuntu or Debian is
relatively easy.  Just do the following::

  $ sudo apt-get install python-numpy python-setuptools python-dev gcc \
    python-pyrex python-nose mayavi2 libopenmpi-dev

Cython and Sphinx may not be readily available, in which case you can
install them using either the usual ``python setup.py`` dance for the
respective projects or using the more convenient::

    $ easy_install Cython 
    $ easy_install mpi4py


Once you have the essential dependencies installed you can easily build
the package.


.. _Cython: http://www.cython.org
.. _numpy: http://numpy.scipy.org
.. _setuptools: http://pypi.python.org/pypi/setuptools
.. _mpi2py: http://mpi4py.scipy.org
.. _nose: http://pypi.python.org/pypi/nose
.. _Sphinx: http://pypi.python.org/pypi/Sphinx/w
.. _VTK: http://www.vtk.org
.. _Mayavi: http://code.enthough.com/projects/mayavi

Install from Sources
---------------------

If you have a tarball or a hg checkout/clone of the project you can
untar the tarball or check it out from the repository and then cd into
the directory and then issue::

    $ python setup.py install

If you'd like a developer install, so you can develop as you go do the
following::

    $ python setup.py develop

Once this is done, you can use the package and then when you change a
file in your branch you can either recompile inplace or re-issue the
develop command as::

    $ python setup.py build_ext --inplace

Install from PyPI
------------------

Once this package is officially released you may install it via easy
install also like so::

    $ easy_install PySPH

.. warning::
    This does not work yet since the package isn't released!


Installation in a virtualenv_
-----------------------------

Installing PySPH in a virtualenv_ is a very convenient way to install
and use PySPH.  Assuming you have Python installed along with
virtualenv_, numpy_, and a working MPI setup you can do the following::

  $ virtualenv pysph
  $ source pysph/bin/activate
  $ easy_install Cython
  $ easy_install mpi4py
  $ cd PySPH-source-directory
  $ python setup.py install
 
The advantage with this approach is that your entire PySPH install and
its dependencies are isolated inside the ``pysph`` directory created by
virtualenv and uninstalling the packages is as easy as deleting the
directory.

You may also install the optional dependencies like Mayavi, like so::

  $ easy_install Mayavi[app]


.. _virtualenv: http://pypi.python.org/pypi/virtualenv

Running the tests
-----------------

Once the package is installed you can test if everything is OK by
running the test suite like so::

  $ cd source
  $ nosetests

If this runs without error you are all set.  If not please contact the
developers.


Building the documentation
--------------------------

To build the documentation using Sphinx_ do the following::

    $ cd docs
    $ make html

If you don't have ``make``, you can do the following::

    $ mkdir -p build/html
    $ sphinx-build -b html -E source build/html

The docs should now be available in the ``build/html`` directory.
Point your browser at ``build/html/index.html``.

