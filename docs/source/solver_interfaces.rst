Solver Interfaces
=================

Interfaces are a way to control, gather data and execute commands on a running
solver instance. This can be useful for example to pause/continue the solver,
get the iteration count, get/set the dt or final time or simply to monitor the
running of the solver.


.. py:currentmodule:: pysph.solver.controller

CommandManager
--------------

The :py:class:`CommandManager` class provides functionality to control the solver in
a restricted way so that adding multiple interfaces to the solver is possible
in a simple way.

The figure :ref:`image_controller` shows an overview of the classes and objects
involved in adding an interface to the solver.

.. _image_controller:
.. figure:: ../images/controller.png
	:align: center
	:width: 900
	
	Overview of the Solver Interfaces

The basic design of the controller is as follows:

#. :py:class:`~pysph.solver.solver.Solver` has a method
   :py:meth:`~pysph.solver.solver.Solver.set_command_handler` takes a callable
   and a command_interval, and calls the callable with self as an argument
   every `command_interval` iterations
 
#. The method :meth:`CommandManager.execute_commands` of `CommandManager` object is
   set as the command_handler for the solver. Now `CommandManager` can do any operation
   on the solver
 
#. Interfaces are added to the `CommandManager` by the :meth:`CommandManager.add_interface`
   method, which takes a callable (Interface) as an argument and calls the callable
   in a separate thread with a new :class:`Controller` instance as an argument
 
#. A `Controller` instance is a proxy for the `CommandManager` which redirects
   its methods to call :meth:`CommandManager.dispatch` on the `CommandManager`, which is
   synchronized in the `CommandManager` class so that only one thread (Interface)
   can call it at a time. The `CommandManager` queues the commands and sends them to
   all procs in a parallel run and executes them when the solver calls its
   :meth:`execute_commands` method
 
#. Writing a new Interface is simply writing a function/method which calls
   appropriate methods on the :class:`Controller` instance passed to it.



Controller
----------

The :py:class:`Controller` class is a convenience class which has various methods
which redirect to the :py:meth:`Controller.dispatch` method to do the actual
work of queuing the commands. This method is synchronized so that multiple
controllers can operate in a thread-safe manner. It also restricts the operations
which are possible on the solver through various interfaces. This enables adding
multiple interfaces to the solver convenient and safe. Each interface gets a
separate Controller instance so that the various interfaces are isolated.


.. py:currentmodule:: pysph.solver.solver_interfaces

Interfaces
----------

Interfaces are functions which are called in a separate thread and receive a
:py:class:`Controller` instance so that they can query the solver, get/set
various properties and execute commands on the solver in a safe manner.

Here's the example of a simple interface which simply prints out the iteration
count every second to monitor the solver

.. _simple_interface:

::

	import time
	
   	def simple_interface(controller):
   	    while True:
   	        print controller.get_count()
   	        time.sleep(1)

You can use ``dir(controller)`` to find out what methods are available on the
controller instance.

A few simple interfaces are implemented in the :py:mod:`~pysph.solver.solver_interfaces`
module, namely :py:class:`CommandlineInterface`, :py:class:`XMLRPCInterface`
and :py:class:`MultiprocessingInterface`, and also in `examples/controller_elliptical_drop_client.py`.
You can check the code to see how to implement various kinds of interfaces.


Usage
-----

To add interfaces to a plain solver (not created using :py:class:`~pysph.solver.application.Application`),
the following steps need to be taken:

- Set :py:class:`~pysph.solver.controller.CommandManager` for the solver (it is not setup by default)
- Add the interface to the CommandManager

The following code demonstrates how the the simple interface created above :ref:`simple_interface`
can be added to a solver::

    # add CommandManager to solver
    command_manager = CommandManager(solver)
    solver.set_command_handler(self.command_manager.execute_commands)
    
    # add the interface
    command_manager.add_interface(simple_interface)


