"""
Tests for the information_base module.
"""

# Standard imports
import unittest

# local imports
from pysph.solver.typed_dict import TypedDict

class TestTypedDict(unittest.TestCase):
    """
    Tests the TypedDict class.
    """
    def test_constructor(self):
        """
        Tests the constructor.
        """
        i = TypedDict()
        self.assertEqual(i.get_number_of_keys(), 0)

    def test_set_get_int(self):
        """
        Test the set/get _int functions.
        """
        i = TypedDict()

        i.set_int('i1', 1)
        self.assertEqual(i.get_int('i1'), 1)

        self.assertRaises(KeyError, i.get_int, 'i2')
        self.assertRaises(TypeError, i.get_float, 'i1')
        self.assertRaises(TypeError, i.get_double, 'i1')
        self.assertRaises(TypeError, i.get_str, 'i1')
        self.assertRaises(TypeError, i.get_list, 'i1')
        self.assertRaises(TypeError, i.get_dict, 'i1')
        self.assertRaises(TypeError, i.get_object, 'i1')

    def test_set_get_float(self):
        """
        Test the set/get _float functions.
        """
        
        i = TypedDict()
        i.set_float('i1', 1.31)
        self.assertEqual(abs(i.get_float('i1')-1.31)<1e-07, True)

        self.assertRaises(KeyError, i.get_float, 'i2')
        self.assertRaises(TypeError, i.get_int, 'i1')
        self.assertRaises(TypeError, i.get_double, 'i1')
        self.assertRaises(TypeError, i.get_str, 'i1')
        self.assertRaises(TypeError, i.get_list, 'i1')
        self.assertRaises(TypeError, i.get_dict, 'i1')
        self.assertRaises(TypeError, i.get_object, 'i1')

    def test_set_get_double(self):
        """
        Test the set/get _double functions.
        """
        
        i = TypedDict()

        i.set_double('i1', 1.31)
        self.assertEqual(i.get_double('i1'), 1.31)

        self.assertRaises(KeyError, i.get_double, 'i2')
        self.assertRaises(TypeError, i.get_int, 'i1')
        self.assertRaises(TypeError, i.get_float, 'i1')
        self.assertRaises(TypeError, i.get_str, 'i1')
        self.assertRaises(TypeError, i.get_list, 'i1')
        self.assertRaises(TypeError, i.get_dict, 'i1')
        self.assertRaises(TypeError, i.get_object, 'i1')

    def test_set_get_str(self):
        """
        Test the set/get str functions.
        """
        i = TypedDict()

        i.set_str('s1', 'abcd')
        self.assertEqual(i.get_str('s1'), 'abcd')
        
        self.assertRaises(KeyError, i.get_str, 's2')

        self.assertRaises(TypeError, i.get_int, 's1')
        self.assertRaises(TypeError, i.get_float, 's1')
        self.assertRaises(TypeError, i.get_double, 's1')
        self.assertRaises(TypeError, i.get_list, 's1')
        self.assertRaises(TypeError, i.get_dict, 's1')
        self.assertRaises(TypeError, i.get_object, 's1')

    def test_set_get_list(self):
        """
        Test the set/get list functions.
        """
        l = [1,2,3,4]

        i = TypedDict()
        i.set_list('l1', l)

        self.assertEqual(i.get_list('l1'), [1,2,3,4])

        self.assertRaises(KeyError, i.get_list, 's2')

        self.assertRaises(TypeError, i.get_int, 'l1')
        self.assertRaises(TypeError, i.get_float, 'l1')
        self.assertRaises(TypeError, i.get_double, 'l1')
        self.assertRaises(TypeError, i.get_str, 'l1')
        self.assertRaises(TypeError, i.get_dict, 'l1')
        self.assertRaises(TypeError, i.get_object, 'l1')

    def test_set_get_dict(self):
        """
        Test the set/get dict functions.
        """
        d = {'a':1, 'b':2}
        
        i = TypedDict()
        i.set_dict('d1', d)
        
        self.assertEqual(i.get_dict('d1'), {'b':2, 'a':1})
        
        self.assertRaises(KeyError, i.get_dict, 's2')

        self.assertRaises(TypeError, i.get_int, 'd1')
        self.assertRaises(TypeError, i.get_float, 'd1')
        self.assertRaises(TypeError, i.get_double, 'd1')
        self.assertRaises(TypeError, i.get_str, 'd1')
        self.assertRaises(TypeError, i.get_list, 'd1')
        self.assertRaises(TypeError, i.get_object, 'd1')
        
    def test_set_get_object(self):
        """
        Test the set/get object functions.
        """
        l = [[1,2,3], {'a':1}]

        i = TypedDict()
        i.set_object('o1', l)

        self.assertEqual(i.get_object('o1'), [[1,2,3], {'a':1}])
        
        self.assertRaises(KeyError, i.get_dict, 's2')

        self.assertRaises(TypeError, i.get_int, 'o1')
        self.assertRaises(TypeError, i.get_float, 'o1')
        self.assertRaises(TypeError, i.get_double, 'o1')
        self.assertRaises(TypeError, i.get_str, 'o1')
        self.assertRaises(TypeError, i.get_list, 'o1')
        self.assertRaises(TypeError, i.get_dict, 'o1')

    def test_get_number_of_keys(self):
        """
        Test the get_number_of_keys function.
        """
        d = TypedDict()

        d.set_float('f', 10.)
        d.set_double('d', 11)
        d.set_str('s', 'abf')

        self.assertEqual(d.get_number_of_keys(), 3)

        d.set_list('l', [])
        self.assertEqual(d.get_number_of_keys(), 4)

    def test_has_key(self):
        """
        Tests the has_key function.
        """
        d = TypedDict()
        
        d.set_float('f', 10.)
        self.assertEqual(d.has_key('f'), True)
        self.assertEqual(d.has_key('f', 'float'), True)
        self.assertEqual(d.has_key('f', 'double'), False)
        self.assertEqual(d.has_key('f', 'object'), False)
        self.assertEqual(d.has_key('f', 'list'), False)
        self.assertEqual(d.has_key('f', 'str'), False)
        self.assertEqual(d.has_key('f', 'dict'), False)
        
    def test_remove_key(self):
        """
        Test the remove_key function.
        """
        d = TypedDict()
        d.set_float('f', 10.)
        d.set_double('d', 11)
        d.set_str('s', 'abf')
        d.set_list('l', [])

        self.assertEqual(d.has_key('f'), True)
        self.assertEqual(d.has_key('d'), True)
        self.assertEqual(d.has_key('s'), True)
        self.assertEqual(d.has_key('l'), True)

        d.remove_key('f')
        self.assertEqual(d.has_key('f'), False)

        d.remove_key('d')
        d.remove_key('s')
        d.remove_key('l')

        self.assertEqual(d.has_key('d'), False)
        self.assertEqual(d.has_key('s'), False)
        self.assertEqual(d.has_key('l'), False)
        
if __name__ == '__main__':
    unittest.main()
