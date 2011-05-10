*************************
Install from source
*************************

Ensure that your system is properly setup as described `here`_

.. _here: setting-up.html



Running the tests
-----------------

Once the package is installed you can test if everything is OK by
running the test suite like so::

  $ make test 

If this runs without error you are all set.  If not please contact the
developers.


Building the documentation
--------------------------

To build the documentation using `Sphinx`_ do the following::

    $ cd docs
    $ make html

The docs should now be available in the ``build/html`` directory.
Point your browser at ``build/html/index.html``.

.. _Sphinx: http://pypi.python.org/pypi/Sphinx/
