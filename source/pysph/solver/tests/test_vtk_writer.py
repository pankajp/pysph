"""
Tests for the vtk_writer module
"""

# standard imports
import unittest

# local imports
from pysph.solver.fluid import Fluid
from pysph.solver.vtk_writer import *

################################################################################
# `TestVTKWriter` class.
################################################################################
class TestVTKWriter(unittest.TestCase):
    """
    Tests for the VTKWriter class.
    """
    def test_constructor(self):
        """
        Tests for the constructor.
        """
        vw = VTKWriter()

        self.assertEqual(vw.scalars, [])
        self.assertEqual(vw.vectors, [])
        self.assertEqual(vw.xml_output, True)
        self.assertEqual(vw.file_ext, 'xml')
        self.assertEqual(vw.only_real_particles, False)
        self.assertEqual(vw.coords, ['x', 'y', 'z'])
        self.assertEqual(vw.file_name_prefix, '')

    def test_add_functions(self):
        """
        Tests the add_scalar and add_vector functions.
        """
        vw = VTKWriter()

        vw.add_scalar(array_name='t')
        vw.add_scalar(array_name='d')
        
        self.assertEqual(vw.scalars[0].scalar_name, 't')
        self.assertEqual(vw.scalars[0].array_name, 't')

        self.assertEqual(vw.scalars[1].scalar_name, 'd')
        self.assertEqual(vw.scalars[1].array_name, 'd')

        vw.add_vector(vector_name='v', array_names=['u', 'v'])
        self.assertEqual(vw.vectors[0].vector_name, 'v')
        self.assertEqual(vw.vectors[0].array_names, ['u', 'v'])

    def test_enable_xml_output(self):
        """
        Tests the enable_xml_output function.
        """
        vw = VTKWriter()
        
        vw.enable_xml_output(False)
        self.assertEqual(vw.file_ext, 'vtk')
        vw.enable_xml_output(True)
        self.assertEqual(vw.file_ext, 'xml')

    def test_write(self):
        """
        Tests the write function.
        """
        vw = VTKWriter(file_name_prefix='/tmp/test',
                       xml_output=True)

        f = Fluid(name='f1')
        fp = f.get_particle_array()
        fp.add_property({'name':'x', 'data':[1, 2, 3, 4]})
        fp.add_property({'name':'u', 'data':[1, 1, 1, 1]})
        fp.add_property({'name':'v', 'data':[1, 1, 1, 1]})
        fp.add_property({'name':'w', 'data':[1, 1, 1, 1]})
        fp.add_property({'name':'t', 'data':[0, 1, 2, 3]})
        fp.align_particles()
        
        vw.entity_list.append(f)

        s = Fluid(name='f2')
        sp = s.get_particle_array()
        sp.add_property({'name':'y', 'data':[1, 2, 3, 4]})
        sp.add_property({'name':'t', 'data':[3, 4, 5, 6]})
        sp.align_particles()

        vw.entity_list.append(s)
        vw.add_scalar(scalar_name='t')
        vw.add_vector(vector_name='velocity', array_names=['u'])
        vw.write()

        # TODO
        # read the file and make sure it has the written data.
        
if __name__ == '__main__':
    unittest.main()
