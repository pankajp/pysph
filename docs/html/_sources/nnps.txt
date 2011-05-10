Nearest Neighbor Particle Searching
====================================
The interaction of particles in SPH is limited to neighboring particles that are within a certain *cutoff*
radius from the target particle as shown in the figure below.

.. _image_particles:
.. figure:: ../images/particles.png
   :align: center
   :width: 175
   :height: 250
   
   Interaction radius for a target particle
   

The all-pair search is very inefficient :math:`O(n^2)` and can be reduced to :math:`O(nlogn)` with the
use of an efficient indexing scheme.

The cell structure
--------------------
PySPH imposes a cell structure over the particle distribution as shown in the figure. The cell size is
fixed at the start of each iteration and cells are only created where particles exist.

.. _image_cells:
.. figure:: ../images/cells.png
   :align: center
   :width: 175
   :height: 250

Using this cell structure and the cell to which the target particle belongs, we need only consider
neighboring cells to search for neighbors.