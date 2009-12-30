"""
Component to write particle data into vtk files.
"""

# tvtk imports
from enthought.tvtk.api import tvtk, write_data

# standard impotrs
import numpy

# local imports
from pysph.solver.base import Base
from pysph.solver.file_writer_component import *
from pysph.solver.fast_utils import make_nd_array

class ScalarInfo(Base):
    """
    Information about a scalar to be written.
    """
    def __init__(self, scalar_name='', array_name='', d_type='float'):
        self.scalar_name = scalar_name
        self.array_name = array_name
        self.d_type = d_type

class VectorInfo(Base):
    """
    Information about a vector to be written.
    """
    def __init__(self, vector_name='', array_names=[]):
        """
        """
        self.array_names = []
        self.array_names[:] = array_names
        self.vector_name = vector_name

################################################################################
# `VTKWriter` class.
################################################################################
class VTKWriter(FileWriterComponent):
    """
    Component to write particle data into files.
    """
    def __init__(self, name='',
                 solver=None,
                 component_manager=None,
                 entity_list=[],
                 file_name_prefix='',
                 xml_output=True,
                 scalars=[],
                 vectors={},
                 coords=['x', 'y', 'z'],
                 only_real_particles=False,
                 *args, **kwargs):
        """
        Constructor.
        """
        FileWriterComponent.__init__(self,
                                     name=name,
                                     solver=solver,
                                     component_manager=component_manager,
                                     entity_list=entity_list,
                                     *args, **kwargs)

        self.scalars = []
        self.vectors = []
        self.coords = ['x', 'y', 'z']
        self.write_count = 0
        self.only_real_particles = only_real_particles
        self.file_name_prefix=file_name_prefix

        self.xml_output = xml_output
        self.enable_xml_output(xml_output)
        
        # setup the input scalars and vecors
        for s in scalars:
            self.add_scalar(s, s)

        for v_name in vectors.keys():
            v_arrays = vectors[v_name]
            self.add_vector(v_name, v_arrays)

    def enable_xml_output(self, enable=True):
        self.xml_output = enable
        if self.xml_output:
            self.file_ext = 'xml'
        else:
            self.file_ext = 'vtk'
        
    def add_scalar(self, scalar_name='', array_name='', d_type='float'):
        """
        Add a scalar to be written.
        """
        if array_name == '' and scalar_name == '':
            logger.warn('Invalid name specified')
            return

        if scalar_name == '':
            scalar_name = array_name
        if array_name == '':
            array_name = scalar_name

        si = ScalarInfo(scalar_name=scalar_name,
                        array_name=array_name,
                        d_type=d_type)

        self.scalars.append(si)

    def add_vector(self, vector_name='', array_names=[]):
        """
        Add a vector to be written.
        """
        if array_names == [] or vector_name == '':
            logger.warn('Invalid info %s %s'%(array_names, vector_name))
            return

        vi = VectorInfo(vector_name=vector_name,
                        array_names=array_names)
        
        self.vectors.append(vi)

    def set_coord_info(self, coord_arrays=['x', 'y', 'z']):
        """
        """
        self.coords[:] = coords

    def compute(self):
        self.write()

    def write(self):
        """
        Function to write.
        """
        for e in self.entity_list:
            parr = e.get_particle_array()
            if parr is None:
                continue
            else:
                file_num = str(self.write_count)
                file_name = self.file_name_prefix + '_' + e.name
                file_name += '_' + file_num + '.' + self.file_ext
                self._write(parr, file_name)
        
        self.write_count += 1

        return 0

    def _write(self, particle_array, file_name):
        """
        """
        x = y = z = None
        if particle_array.has_array(self.coords[0]):
            x = particle_array.get(self.coords[0], 
                                   only_real_particles=self.only_real_particles)
        if particle_array.has_array(self.coords[1]):
            y = particle_array.get(self.coords[1],
                                   only_real_particles=self.only_real_particles)
        if particle_array.has_array(self.coords[2]):
            z = particle_array.get(self.coords[2],
                                   only_real_particles=self.only_real_particles)
        
        # get the max length of coords - some can be zero
        if x is None and y is None and z is None:
            return

        np = 0

        if x is None:
          if y is None:
              np = len(z)
          else:
              np = len(y)
        else:
            np = len(x)

        if x is None:
            x = numpy.zeros(np, dtype=float)
        if y is None:
            y = numpy.zeros(np, dtype=float)
        if z is None:
            z = numpy.zeros(np, dtype=float)

        coords = make_nd_array([x, y, z])
        pd = tvtk.PolyData(points=coords)
        
        # write vertex index information
        polys = numpy.arange(np)
        polys.shape = (np, 1)
        pd.verts = polys

        # add the scalar to the polydata
        for s_info in self.scalars:
            if particle_array.has_array(s_info.array_name):
                s_arr = particle_array.get(
                    s_info.array_name,
                    only_real_particles=self.only_real_particles)

            else:
                # if the array was missing, do not write
                continue

            if s_info.d_type == 'int':
                arr = tvtk.IntArray()
            else:
                arr = tvtk.DoubleArray()

            arr.name = s_info.scalar_name
            arr.from_array(s_arr)
            pd.point_data.add_array(arr)

        # add the vectors to the polydata
        for v_info in self.vectors:
            v_name = v_info.vector_name
            v_arrs = []
            for arr in v_info.array_names:
                if particle_array.has_array(arr):
                    a = particle_array.get(
                        arr, only_real_particles=self.only_real_particles)
                    v_arrs.append(a)
                else:
                    v_arrs.append(None)

            # if some array was missing do not write.
            if v_arrs.count(None) > 0:
                continue
            
            # to make a 2d array into 3d.
            if len(v_arrs) < 3:
                v_arrs.append(numpy.zeros(np, dtype=float))
            if len(v_arrs) < 2:
                v_arrs.append(numpy.zeros(np, dtype=float))
            
            arr = make_nd_array(v_arrs)

            parr = tvtk.DoubleArray()
            parr.name = v_name
            parr.from_array(arr)
            pd.point_data.add_array(parr)

        write_data(pd, file_name)
