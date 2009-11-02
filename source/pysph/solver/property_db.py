"""
Module to hold information about various properties and array names assigned to
them. 
"""

class PropertyDb(object):
    """
    Class to hold information about various properties.

    A dictionary indexed on property name is maintained. For each property the
    dictionary entry holds the following information:

        - type - indicates if the property is a scalar or a vector.
        - arrays - for vectors, the arrays making up the vector.
        - integrate_to - property got by integrating this property.
        - differentiate_to - property got by differentiating this property.
        
    Note that only common properties - properties which are generally used in
    most components will be listed here. Any property that is purely private to
    a component need not be listed here. Put differently, 'standard' property
    names *should* appear here.

    These properties are just a convenient reference to properties from
    anywhere in the code. The array names here will be used directly from
    anywhere in the code. Thus changing array names here will require major
    changes through out the code. It is hence advised NOT to change the array
    names for vectors and property names for scalars.

    To add a new property, add an entry to the 'init' function.

    """
    # a few class attributes.
    init_done = False
    property_dict = {}

    def __init__(self):
        """
        Constructor.

        We do not allow objects of this to be created. Only the class methods
        should be used to access this class.
        """
        raise SystemError, 'Do not instantiate the PropertyDb class'


    @staticmethod
    def get_property(prop_name):
        """
        Return the dictionary for the property.
        """
        PropertyDb.init()
            
        return PropertyDb.property_dict[prop_name]

    @staticmethod
    def init():
        """
        Sets up the property dict.
        """
        pdb = PropertyDb

        if pdb.init_done == True:
            return
                
        # add various properties here.
        pdict = pdb.property_dict

        ##########################################
        # density
        ##########################################
        d = pdb.get_dict()
        pdict['rho'] = d
        d['type'] = 'scalar'
        d['differentiate_to'] = 'rho_rate'
        ##########################################
        # density_rate
        ##########################################
        d = pdb.get_dict()
        pdict['rho_rate'] = d
        d['type'] = 'scalar'
        d['integrate_to'] = 'rho'
        ##########################################
        # pressure - p
        ##########################################
        d = pdb.get_dict()
        pdict['p'] = d
        d['type'] = 'scalar'
        ##########################################
        # interaction radius - h
        ##########################################
        d = pdb.get_dict()
        pdict['h'] = d
        d['type'] = 'scalar'
        ##########################################
        # x
        ##########################################
        d = pdb.get_dict()
        pdict['x'] = d
        d['type'] = 'scalar'
        d['differentiate_to'] = 'u'
        ##########################################
        # u
        ########################################## 
        d = pdb.get_dict()
        pdict['u'] = d
        d['type'] = 'scalar'
        d['integrate_to'] = 'x'
        d['differentiate_to'] = 'ax'
        ##########################################
        # ax
        ##########################################
        d = pdb.get_dict()
        pdict['ax'] = d
        d['type'] = 'scalar'
        d['integrate_to'] = 'u'
        ##########################################
        # position2d
        ########################################## 
        d = pdb.get_dict()
        pdict['position2d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['x', 'y']
        d['differentiate_to'] = 'velocity2d'
        ##########################################
        # velocity2d
        ########################################## 
        d = pdb.get_dict()
        pdict['velocity2d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['u', 'v']
        d['integrate_to'] = 'position2d'
        d['differentiate_to'] = 'acceleration2d'
        ##########################################
        # acceleration2d
        ########################################## 
        d = pdb.get_dict()
        pdict['acceleration2d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['ax', 'ay']
        d['integrate_to'] = 'velocity2d'
        ##########################################
        # position3d
        ########################################## 
        d = pdb.get_dict()
        pdict['position3d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['x', 'y', 'z']
        d['differentiate_to'] = 'velocity3d'
        ##########################################
        # velocity3d
        ########################################## 
        d = pdb.get_dict()
        pdict['velocity3d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['u', 'v', 'w']
        d['integrate_to'] = 'position3d'
        d['differentiate_to'] = 'acceleration3d'
        ##########################################
        # acceleration3d
        ########################################## 
        d = pdb.get_dict()
        pdict['acceleration3d'] = d
        d['type'] = 'vector'
        d['arrays'] = ['ax', 'ay', 'az']
        d['integrate_to'] = 'velocity3d'
        
        pdb.init_done = True

    @staticmethod
    def get_dict():
        """
        Returns an empty dictionary with required keys.
        """
        return {'type':'scalar', 'arrays':[], 'integrate_to':'',
                'differentiate_to':''}
