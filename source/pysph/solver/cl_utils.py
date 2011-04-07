HAS_CL = True
try:
    import pyopencl as cl
except ImportError:
    HAS_CL=False

# Return all available devices on the host
def get_cl_devices():
    """ Return a dictionary keyed on device type for all devices """

    _devices = {'CPU':[], 'GPU':[]}

    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices()
        for device in devices:
            if device.type == cl.device_type.CPU:
                _devices['CPU'].append(device)
            elif device.type == cl.device_type.GPU:
                _devices['GPU'].append(device)
                
                
    return _devices

def get_cl_include():
    """ Include directories for OpenCL definitions """

    inc_dir = [ "-I/home/kunalp/pysph/source/pysph/base",
                "-I/home/kunalp/pysph/source/pysph/solver",
                ]

    return inc_dir

     
