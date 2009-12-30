"""
Class to write entites into VTK files in a parallel simulation.
"""

# logger imports
import logging
logger = logging.getLogger()

# standard imports
import numpy

# local imports
from pysph.base.particle_array import ParticleArray
from pysph.base.particle_tags import *
from pysph.solver.vtk_writer import *

################################################################################
# `ParallelVTKWriter` class.
################################################################################
class ParallelVTKWriter(VTKWriter):
    """
    Component to write entities into VTK files in a parallel simulation.
    """
    PARALLEL_VTK_WRITE_DATA=201
    def __init__(self,
                 name='',
                 solver=None,
                 entity_list=[],
                 file_name_prefix='',
                 xml_output=True,
                 scalars=[],
                 vectors=[],
                 coords=['x', 'y', 'z'],
                 only_real_particles=True,
                 write_local=False,
                 *args, **kwargs):
        """
        Constructor.
        """
        VTKWriter.__init__(self, 
                           name=name,
                           solver=solver, 
                           entity_list=entity_list,
                           file_name_prefix=file_name_prefix,
                           xml_output=xml_output,
                           scalars=scalars,
                           vectors=vectors,
                           coords=coords,
                           only_real_particles=only_real_particles)
        
        self.write_local=write_local
        self.parallel_controller = None
        self.only_real_particles=only_real_particles

    def write(self):
        """
        The write function.
        
        **Algorithm**

            - If write_local is True:
                - generate appropriate file name.
                - write file locally.
            - If write_local is False:
                - If any children are present, get all particle arrays from
                  them.
                - Merge these with data with self.
                - IF root node, write the data.
                - IF not root node, send the merged data to parent node.
                
        """
        # if local write is needed, just call the parent class write function.
        if self.write_local:
            return VTKWriter.write(self)
        else:
            self._write_parallel()

        return 0

    def update_property_requirements(self):
        """
        Set the property requirements on this component.
        """
        in_types = self.information.get_dict(self.INPUT_TYPES)
        
        for t in in_types.keys():
            self.add_write_prop_requirement(t, ['pid'])

        return 0

    def setup_component(self):
        """
        Perform any setup for this component.

        Makes sure only entities that provide particle arrays, are there in the
        entity_list, others are removed before execute is called.
        
        Gets the parallel controller from the solver.

        """
        if self.setup_done == True:
            return 0

        to_remove = []
        for e in self.entity_list:
            if e.get_particle_array() is None:
                to_remove.append(e)
        
        for e in to_remove:
            self.entity_list.remove(e)

        self.parallel_controller = self.solver.parallel_controller
        
        if self.parallel_controller is None:
            logger.warn(
                'Solver does not seem to have created its parallel controller')
            self.parallel_controller = ParallelController()

        logger.info('Only real particles : %s'%(self.only_real_particles))

        self.setup_done = True

        return 0

    def _write_parallel(self):
        """
        Collects data at node 0 and writes the data at node 0.

        **Algorithm**
            - If any children are present, get all particle arrays from them.
            - Merge these with the data in self.
            - If parent is there, send merged data to parent.
            - Else write data to file.
        
        """
        pc = self.parallel_controller
        comm = pc.comm
        data = [None]*len(pc.children_proc_ranks)

        i = 0
        # receive data from all children - Each child will send a list of
        # particle arrays, one for each entity in entity_list.
        for pid in pc.children_proc_ranks:
            data[i] = comm.recv(source=pid, tag=self.PARALLEL_VTK_WRITE_DATA)
            i += 1
                    
        # add self data into the merged output.
        merged_data = []
        for e in self.entity_list:
            merged_parray = ParticleArray()
            my_parray = e.get_particle_array()
            merged_parray.append_parray(my_parray)

            if merged_parray.get_number_of_particles() > 0:
                merged_parray.pid[:] = pc.rank
            
            if self.only_real_particles:
                merged_parray.remove_flagged_particles(flag_name='local',
                                                       flag_value=0)
            merged_data.append(merged_parray)
            
        # merge data of all chilren and send data to parent.
        for i in range(len(pc.children_proc_ranks)):
            d = data[i]
            pid = pc.children_proc_ranks[i]
            for j in range(len(merged_data)):
                merged_parray = merged_data[j]
                c_parray = d[j]
                if c_parray.get_number_of_particles() > 0:
                    c_parray.pid[:] = pid
                if self.only_real_particles:
                    c_parray.remove_flagged_particles(flag_name='local',
                                                      flag_value=0)
                merged_parray.append_parray(c_parray)

        # now send this data to parent if there is one, else write data.
        if pc.rank == 0:
            self._write_merged_data(merged_data)
        else:
            comm.send(merged_data, dest=pc.parent_rank,
                      tag=self.PARALLEL_VTK_WRITE_DATA) 
        
        return 0
            
    def _write_merged_data(self, parray_list):
        """
        Writes merged parray data to file at root node.
        """
        for i in range(len(self.entity_list)):
            e = self.entity_list[i]
            parr = parray_list[i]
            file_num = str(self.write_count)
            file_name = self.file_name_prefix + '_' + e.name
            file_name += '_' + file_num + '.' + self.file_ext
            self._write(parr, file_name)

        self.write_count += 1

        return 0
