class ParticleType:
    """
    Empty class to emulate an enum for all entity types.

    Add any new entity as a class attribute.
    """
    Fluid = 0
    Solid = 1
    DummyFluid = 2
    
    def __init__(self):
        """
        Constructor.

        We do not allow this class to be instantiated. Only the class attributes
        are directly accessed. Instantiation will raise an error.
        """
        raise SystemError, 'Do not instantiate the EntityTypes class'



    
