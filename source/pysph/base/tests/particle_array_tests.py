"""
Tests for the particle array module.
"""

# standard imports
import unittest
import numpy

# local imports
import pysph
from pysph.base import particle_array
from pysph.base.carray import LongArray

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

################################################################################
# `ParticleArrayTest` class.
################################################################################
class ParticleArrayTest(unittest.TestCase):
    """
    Tests for the particle array class.
    """
    def test_constructor(self):
        """
        Test the constructor.
        """
        # Default constructor test.
        p = particle_array.ParticleArray(particle_manager=None,
                                         name='test_particle_array')  

        self.assertEqual(p.particle_manager, None)
        self.assertEqual(p.name, 'test_particle_array')
        self.assertEqual(p.temporary_arrays == {}, True)
        self.assertEqual(p.is_dirty, True)
        self.assertEqual(p.properties['tag'] == 0, True)
        self.assertEqual(p.property_arrays[0].length == 0, True)

        # Constructor with some properties.
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        self.assertEqual(p.particle_manager, None)
        self.assertEqual(p.name, '')

        self.assertEqual(p.properties.has_key('x'), True)
        self.assertEqual(p.properties.has_key('y'), True)
        self.assertEqual(p.properties.has_key('z'), True)
        self.assertEqual(p.properties.has_key('m'), True)
        self.assertEqual(p.properties.has_key('h'), True)

        # get the properties are check if they are the same
        x_id = p.properties['x']
        xarr = p.property_arrays[x_id].get_npy_array()
        self.assertEqual(check_array(xarr, x), True)
        
        y_id = p.properties['y']
        yarr = p.property_arrays[y_id].get_npy_array()
        self.assertEqual(check_array(yarr, y), True)

        z_id = p.properties['z']
        zarr = p.property_arrays[z_id].get_npy_array()
        self.assertEqual(check_array(zarr, z), True)

        m_id = p.properties['m']
        marr = p.property_arrays[m_id].get_npy_array()
        self.assertEqual(check_array(marr, m), True)

        h_id = p.properties['h']
        harr = p.property_arrays[h_id].get_npy_array()
        self.assertEqual(check_array(harr, h), True)
        
        # check if the 'tag' array was added.
        t_id = p.properties['tag']
        self.assertEqual(t_id, 0)
        self.assertEqual(p.property_arrays[0].length == len(x), True)

        # Constructor with tags
        tags = [0, 1, 0, 1]
        p = particle_array.ParticleArray(x=x, y=y, z=z, tag=tags)
        
        tag_id = p.properties['tag']
        tagarr = p.property_arrays[tag_id].get_npy_array()
        self.assertEqual((tagarr == tags).all(), True)

    def test_get_number_of_particles(self):
        """
        Tests the get_number_of_particles of particles.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        self.assertEqual(p.get_number_of_particles(), 4)

    def test_get(self):
        """
        Tests the get function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)
        
        self.assertEqual(check_array(x, p.get('x')), True)
        self.assertEqual(check_array(y, p.get('y')), True)
        self.assertEqual(check_array(z, p.get('z')), True)
        self.assertEqual(check_array(m, p.get('m')), True)
        self.assertEqual(check_array(h, p.get('h')), True)

    def test_set(self):
        """
        Tests the set function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)
        
        # set the x array with new values
        p.set(**{'x':[4., 3, 2, 1], 'h':[0.2, 0.2, 0.2, 0.2]})
        self.assertEqual(check_array(p.get('x'), [4., 3, 2, 1]), True)
        self.assertEqual(check_array(p.get('h'), [0.2, 0.2, 0.2, 0.2]), True)
        
        # trying to set the tags
        p.set(**{'tag':[0, 1, 1, 1]})
        self.assertEqual(check_array(p.get('tag'), [0, 1, 1, 1]), True)

        # try setting array with different length.
        self.assertRaises(ValueError, p.set, **{'x':[1., 2, 3]})

    def test_add_temporary_array(self):
        """
        Tests the add_temporary_array function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        # make sure the temporary_arrays dict is empty.
        self.assertEqual(p.temporary_arrays, {})
        
        # now add some temporary arrays.
        p.add_temporary_array('temp1')
        p.add_temporary_array('temp2')
        # get the arrays and make sure they are of correct size.
        self.assertEqual(p.get('temp1').size == 4, True)
        self.assertEqual(p.get('temp2').size == 4, True)
        
        # try to add temporary array with name as some property.
        self.assertRaises(ValueError, p.add_temporary_array, 'x')

        # try setting a temporary array.
        p.set(**{'temp1':[2, 4, 3, 1]})
        self.assertEqual(check_array(p.get('temp1'), [2, 4, 3, 1]), True)
                
    def test_clear(self):
        """
        Tests the clear function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        p.clear()

        self.assertEqual(p.properties, {'tag':0})
        self.assertEqual(p.is_dirty, True)
        self.assertEqual(p.temporary_arrays, {})

    def test_getattr(self):
        """
        Tests the __getattr__ function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        self.assertEqual(check_array(x, p.x), True)
        self.assertEqual(check_array(y, p.y), True)
        self.assertEqual(check_array(z, p.z), True)
        self.assertEqual(check_array(m, p.m), True)
        self.assertEqual(check_array(h, p.h), True)

        # try getting an non-existant attribute
        self.assertRaises(AttributeError, p.__getattr__, 'a')

    def test_setattr(self):
        """
        Tests the __setattr__ function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)

        p.x = p.x*2.0

        self.assertEqual(check_array(p.get('x'), [2., 4, 6, 8]), True)
        p.x = p.x + 3.0*p.x
        self.assertEqual(check_array(p.get('x'), [8., 16., 24., 32.]), True)

    def test_remove_particles(self):
        """
        Tests the remove_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)
        p.add_temporary_array('tmp1')

        remove_arr = LongArray(0)
        remove_arr.append(0)
        remove_arr.append(1)
        
        p.remove_particles(remove_arr)

        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)
        self.assertEqual(len(p.tmp1), 2)

        # now try invalid operatios to make sure errors are raised.
        remove_arr.resize(10)
        self.assertRaises(ValueError, p.remove_particles, remove_arr)

        # now try to remove a particle with index more that particle
        # length.
        remove_arr.resize(1)
        remove_arr[0] = 2

        p.remove_particles(remove_arr)
        # make sure no change occurred.
        self.assertEqual(p.get_number_of_particles(), 2)
        self.assertEqual(check_array(p.x, [3., 4.]), True)
        self.assertEqual(check_array(p.y, [2., 3.]), True)
        self.assertEqual(check_array(p.z, [0., 0.]), True)
        self.assertEqual(check_array(p.m, [1., 1.]), True)
        self.assertEqual(check_array(p.h, [.1, .1]), True)
        self.assertEqual(len(p.tmp1), 2)
        
    def test_add_particles(self):
        """
        Tests the add_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]

        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h)
        p.add_temporary_array('tmp1')
        p.set_dirty(False)
        
        new_particles = {}
        new_particles['x'] = numpy.array([5., 6, 7])
        new_particles['y'] = numpy.array([4., 5, 6])
        new_particles['z'] = numpy.array([0., 0, 0])

        p.add_particles(**new_particles)

        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(p.is_dirty, True)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)
        self.assertEqual(len(p.tmp1), 7)

        p.set_dirty(False)

        # try adding an empty particle list 
        p.add_particles(**{})
        self.assertEqual(p.get_number_of_particles(), 7)
        self.assertEqual(check_array(p.x, [1., 2, 3, 4, 5, 6, 7]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2, 3, 4, 5, 6]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0, 0, 0, 0, 0]), True)
        self.assertEqual(p.is_dirty, False)

        # make sure the other arrays were resized
        self.assertEqual(len(p.h), 7)
        self.assertEqual(len(p.m), 7)
        self.assertEqual(len(p.tmp1), 7)

    def test_remove_tagged_particles(self):
        """
        Tests the remove_tagged_particles function.
        """
        x = [1, 2, 3, 4.]
        y = [0., 1., 2., 3.]
        z = [0., 0., 0., 0.]
        m = [1., 1., 1., 1.]
        h = [.1, .1, .1, .1]
        tag = [1, 1, 1, 0]
        
        p = particle_array.ParticleArray(x=x, y=y, z=z, m=m, h=h, tag=tag)
        p.add_temporary_array('tmp1')

        p.remove_tagged_particles(0)

        self.assertEqual(p.get_number_of_particles(), 3)
        self.assertEqual(check_array(p.x, [1, 2, 3.]), True)
        self.assertEqual(check_array(p.y, [0., 1, 2]), True)
        self.assertEqual(check_array(p.z, [0., 0, 0]), True)
        self.assertEqual(check_array(p.h, [.1, .1, .1]), True)
        self.assertEqual(check_array(p.m, [1., 1., 1.]), True)
        self.assertEqual(len(p.tmp1), 3)
        
if __name__ == '__main__':
    unittest.main()
