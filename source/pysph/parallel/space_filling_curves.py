""" Module to implement various space filling curves for load balancing """

import numpy
from pysph.base.point import IntPoint
try:
    from hilbert import Hilbert_to_int
except ImportError:
    # TODO: implement Hilbert's SFC
    pass

def morton_sfc(cell_id, maxlen=20, dim=3):
    """Returns key of indices using Morton's space filling curve """
    if isinstance(cell_id, IntPoint):
        cell_id = (cell_id.x,cell_id.y,cell_id.z)
        if dim==2:
            cell_id = (cell_id[0],cell_id[1])
        elif dim==1:
            cell_id = (cell_id[0])
    binary_repr = numpy.binary_repr
    s = 2**maxlen
    #x_bin = binary_repr(cell_id.x+s)
    #y_bin = binary_repr(cell_id.y+s)
    #z_bin = binary_repr(cell_id.z+s)
    binr = [binary_repr(i+s) for i in cell_id]
    #maxlen = len(binary_repr(2**self.level))
    bins = []
    for bin in binr:
        if len(bin) < maxlen:
            bin = '0'*(maxlen-len(bin)) + bin
        bins.append(bin)
    #x_bin ,y_bin,z_bin = bins
    key = ''
    for i in range(maxlen):
        for bin in bins:
            key += bin[i]
    key = int(key)
    return key

def hilbert_sfc(cell_id, maxlen=20, dim=3):
    """Returns key of indices using Hilbert space filling curve """
    if isinstance(cell_id, IntPoint):
        cell_id = (cell_id.x,cell_id.y,cell_id.z)
        if dim==2:
            cell_id = (cell_id[0],cell_id[1])
        elif dim==2:
            cell_id = (cell_id[0])
    s = 2**maxlen
    return Hilbert_to_int([int(i+s) for i in cell_id])

sfc_func_dict = {'morton':morton_sfc,
                 'hilbert':hilbert_sfc}
