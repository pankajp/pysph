"""
Tests for classes in the point.pyx module.
"""

# standard imports
import unittest

# numpy import
import numpy

# local imports
from pysph.base.point import *

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)

###############################################################################
# Tests for the `Point` class.
###############################################################################
class TestPoint(unittest.TestCase):
    def test_constructor(self):
        """Test the constructor."""
        p = Point()
        self.assertEqual(p.x, 0.0)
        self.assertEqual(p.y, 0.0)
        self.assertEqual(p.z, 0.0)
        p = Point(1.0, 1.0)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 1.0)
        self.assertEqual(p.z, 0.0)
        p = Point(y=1.0)
        self.assertEqual(p.x, 0.0)
        self.assertEqual(p.y, 1.0)
        self.assertEqual(p.z, 0.0)
        p = Point()
        p.set(1,1,1)
        self.assertEqual(p.x, 1.0)
        self.assertEqual(p.y, 1.0)
        self.assertEqual(p.z, 1.0)


    def test_special(self):
        """Test the special methods."""
        p = Point(1.0, 1.0, 1.0)
        # __str__ 
        self.assertEqual(str(p), '(%f, %f, %f)'%(p.x, p.y, p.z))
        p1 = Point(1.0, 2.0, 3.0)
        # __add__
        p2 = p + p1
        self.assertEqual(p2.x, 2.0)
        self.assertEqual(p2.y, 3.0)
        self.assertEqual(p2.z, 4.0)
        # __sub__
        p2 = p1 - p
        self.assertEqual(p2.x, 0.0)
        self.assertEqual(p2.y, 1.0)
        self.assertEqual(p2.z, 2.0)
        # __mul__
        p2 = p*2.0
        self.assertEqual(p2.x, 2.0)
        self.assertEqual(p2.y, 2.0)
        self.assertEqual(p2.z, 2.0)
        # __div__
        p2 = p/2.0
        self.assertEqual(p2.x, 0.5)
        self.assertEqual(p2.y, 0.5)
        self.assertEqual(p2.z, 0.5)
        # __neg__
        p2 = -p
        self.assertEqual(p2.x, -1.0)
        self.assertEqual(p2.y, -1.0)
        self.assertEqual(p2.z, -1.0)
        # __abs__
        self.assertEqual(abs(p), numpy.sqrt(3.0))
        # __iadd__
        p2 = Point(1.0, 1.0, 1.0)
        p2 += p1
        self.assertEqual(p2.x, 2.0)
        self.assertEqual(p2.y, 3.0)
        self.assertEqual(p2.z, 4.0)
        # __isub__
        p2 -= p1
        self.assertEqual(p2.x, 1.0)
        self.assertEqual(p2.y, 1.0)
        self.assertEqual(p2.z, 1.0)
        # __imul__
        p2 *= 2.0
        self.assertEqual(p2.x, 2.0)
        self.assertEqual(p2.y, 2.0)
        self.assertEqual(p2.z, 2.0)
        # __div__
        p2 /= 2.0
        self.assertEqual(p2.x, 1.0)
        self.assertEqual(p2.y, 1.0)
        self.assertEqual(p2.z, 1.0)
        # __richcmp__
        self.assertEqual(p==p2, True)
        self.assertEqual(p==p1, False)
        self.assertEqual(p!= p1, True)
        try:
            p < p1
        except TypeError:
            pass
        try:
            p <= p1
        except TypeError:
            pass
        try:
            p > p1
        except TypeError:
            pass
        try:
            p >= p1
        except TypeError:
            pass

    def test_methods(self):
        """Test the other methods of the point class."""
        p = Point(1.0, 2.0, 3.0)
        # asarray
        x = p.asarray()
        self.assertEqual(type(x) is numpy.ndarray, True)
        self.assertEqual(check_array(x, [1.0, 2.0, 3.0]), True)
        # norm
        self.assertEqual(p.norm(), 14.0)
        # length
        self.assertEqual(p.length(), numpy.sqrt(14.0))
        # dot
        p1 = Point(0.5, 1.0, 1.5)
        self.assertEqual(p.dot(p1), 7.0)
        # cross
        self.assertEqual(p.cross(p1), Point())



###############################################################################
# `TestIntPoint` class.
###############################################################################
class TestIntPoint(unittest.TestCase):
    """
    Tests for the IntPoint class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        p = IntPoint()
       
        self.assertEqual(p.x, 0)
        self.assertEqual(p.y, 0)
        self.assertEqual(p.z, 0)

        p = IntPoint(1, 4, 6)
        
        self.assertEqual(p.x, 1)
        self.assertEqual(p.y, 4)
        self.assertEqual(p.z, 6)

    def test_set(self):
        """
        Tests the set function.
        """
        p = IntPoint()
        p.set(-1, -2, -3)

        self.assertEqual(p.x, -1)
        self.assertEqual(p.y, -2)
        self.assertEqual(p.z, -3)

    def test_is_equal(self):
        """
        Test the is_equal function.
        """
        p1 = IntPoint()
        self.assertEqual(p1.py_is_equal(IntPoint()), True)

        p1.x = 1
        p1.y = 2
        p1.z = 3
        
        self.assertEqual(IntPoint(1, 2, 3).py_is_equal(p1), True)
        self.assertEqual(IntPoint(1, 2, 3) == p1, True)

    def test_hash(self):
        """
        Tests if the IntPoints are mapped properly into a dictionary.
        """
        p1 = IntPoint(1, 5, 2)
        p2 = IntPoint(1, 5, 2)
        p3 = IntPoint(3, 4, 1)

        d = {}
        
        d[p1] = 'p1'
        self.assertEqual(d.has_key(p2), True)
        self.assertEqual(d.has_key(p1), True)
        self.assertEqual(d.has_key(p3), False)

        self.assertEqual(d[p1] == d[p2], True)
        self.assertEqual(d[p1] == 'p1', True)

        d[p3] = 'p3'
        self.assertEqual(d[p3] == 'p3', True)

    def test_diff(self):
        """
        Tests the diff function.
        """
        
        p1 = IntPoint()
        p2 = IntPoint(1, 1, 1)
        p3 = p2.py_diff(p1)

        self.assertEqual(p3.py_is_equal(p2), True)
        
        p3 = p1.py_diff(p2)

        self.assertEqual(p3.py_is_equal(IntPoint(-1, -1, -1)), True)

    def test_asarray(self):
        """
        Tests the asarray function.
        """
        p1 = IntPoint(-1, 33, 1)
        arr = p1.asarray()

        self.assertEqual(arr.dtype, numpy.int)
        self.assertEqual(arr[0], -1)
        self.assertEqual(arr[1], 33)
        self.assertEqual(arr[2], 1)
    
