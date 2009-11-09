"""
Tests for the carray module.

Only the LongArray is tested. As the code in carray.pyx is auto-generated, tests
for one class hould suffice.
"""


# standard imports
import unittest
import numpy

# local imports
from pysph.base.carray import LongArray

class TestLongArray(unittest.TestCase):
    """
    Tests for the LongArray class.
    """
    def test_constructor(self):
        """
        Test the constructor.
        """
        l = LongArray(10)
        
        self.assertEqual(l.length, 10)
        self.assertEqual(l.alloc, 10)
        self.assertEqual(len(l.get_npy_array()), 10)
        
        l = LongArray()

        self.assertEqual(l.length, 0)
        self.assertEqual(l.alloc, 16)
        self.assertEqual(len(l.get_npy_array()), 0)

    def test_get_set_indexing(self):
        """
        Test get/set and [] operator.
        """
        l = LongArray(10)
        l.set(0, 10)
        l.set(9, 1)

        self.assertEqual(l.get(0), 10)
        self.assertEqual(l.get(9), 1)
        
        l[9] = 2
        self.assertEqual(l[9], 2)

    def test_append(self):
        """
        Test the append function.
        """
        l = LongArray(0)
        l.append(1)
        l.append(2)
        l.append(3)

        self.assertEqual(l.length, 3)
        self.assertEqual(l[0], 1)
        self.assertEqual(l[1], 2)
        self.assertEqual(l[2], 3)
    
    def test_reserve(self):
        """
        Tests the reserve function.
        """
        l = LongArray(0)
        l.reserve(10)

        self.assertEqual(l.alloc, 16)
        self.assertEqual(l.length, 0)
        self.assertEqual(len(l.get_npy_array()), 0)
        
        l.reserve(20)
        self.assertEqual(l.alloc, 20)
        self.assertEqual(l.length,  0)
        self.assertEqual(len(l.get_npy_array()), 0)

    def test_resize(self):
        """
        Tests the resize function.
        """
        l = LongArray(0)
        
        l.resize(20)
        self.assertEqual(l.length, 20)
        self.assertEqual(len(l.get_npy_array()), 20)
        self.assertEqual(l.alloc >= l.length, True)
    
    def test_get_npy_array(self):
        """
        Tests the get_npy_array array.
        """
        l = LongArray(3)
        l[0] = 1
        l[1] = 2
        l[2] = 3
        
        nparray = l.get_npy_array()
        self.assertEqual(len(nparray), 3)
        
        for i in range(3):
            self.assertEqual(nparray[0], l[0])
        
    def test_set_data(self):
        """
        Tests the set_data function.
        """
        l = LongArray(5)
        np = numpy.arange(5)
        l.set_data(np)

        for i in range(5):
            self.assertEqual(l[i], np[i])

        self.assertRaises(ValueError, l.set_data, numpy.arange(1))

    def test_squeeze(self):
        """
        Tests the squeeze function.
        """
        l = LongArray(5)
        l.append(4)

        self.assertEqual(l.alloc > l.length, True)

        l.squeeze()

        self.assertEqual(l.length, 6)
        self.assertEqual(l.alloc == l.length, True)
        self.assertEqual(len(l.get_npy_array()), 6)
    
    def test_reset(self):
        """
        Tests the reset function.
        """
        l = LongArray(5)
        l.reset()
        
        self.assertEqual(l.length, 0)
        self.assertEqual(l.alloc, 5)
        self.assertEqual(len(l.get_npy_array()), 0)
    
    def test_extend(self):
        """
        Tests the extend function.
        """
        l1 = LongArray(5)
        
        for i in range(5):
            l1[i] = i
        
        l2 = LongArray(5)
        
        for i in range(5):
            l2[i] = 5 + i

        l1.extend(l2.get_npy_array())

        self.assertEqual(l1.length, 10)
        self.assertEqual(numpy.allclose(l1.get_npy_array(), numpy.arange(10)), True)

    def test_remove(self):
        """
        Tests the remove function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))
        rem = [0, 4, 3]
        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 7)
        self.assertEqual(numpy.allclose([7, 1, 2, 8, 9, 5, 6],
                                        l1.get_npy_array()), True)
        
        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 4)
        self.assertEqual(numpy.allclose([6, 1, 2, 5], l1.get_npy_array()), True)

        rem = [0, 1, 3]
        l1.remove(numpy.array(rem, dtype=numpy.int))
        self.assertEqual(l1.length, 1)
        self.assertEqual(numpy.allclose([2], l1.get_npy_array()), True)

        l1.remove(numpy.array([0], dtype=numpy.int))
        self.assertEqual(l1.length, 0)
        self.assertEqual(len(l1.get_npy_array()), 0)

    def test_align_array(self):
        """
        Test the align_array function.
        """
        l1 = LongArray(10)
        l1.set_data(numpy.arange(10))
        
        new_indices = LongArray(10)
        new_indices.set_data(numpy.asarray([1, 5, 3, 2, 4, 7, 8, 6, 9, 0]))
        
        l1.align_array(new_indices)
        self.assertEqual(numpy.allclose([1, 5, 3, 2, 4, 7, 8, 6, 9, 0],
                                        l1.get_npy_array()), True)
if __name__ == '__main__':
    unittest.main()
        
        
