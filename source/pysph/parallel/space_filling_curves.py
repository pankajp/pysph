""" Module to implement various space filling curves for load balancing """

import numpy
from pysph.base.point import IntPoint
try:
    from hilbert import Hilbert_to_int
    have_hilbert = True
except ImportError:
    # TODO: implement Hilbert's SFC
    have_hilbert = False

def morton_sfc(cell_id, maxlen=20, dim=3):
    """Returns key of indices using Morton's space filling curve """
    if isinstance(cell_id, IntPoint):
        cell_id = (cell_id.x,cell_id.y,cell_id.z)
    cell_id = cell_id[:dim]
    binary_repr = numpy.binary_repr
    s = 2**maxlen
    #x_bin = binary_repr(cell_id.x+s)
    #y_bin = binary_repr(cell_id.y+s)
    #z_bin = binary_repr(cell_id.z+s)
    binr = [binary_repr(i+s) for i in cell_id]
    #maxlen = len(binary_repr(2**self.level))
    bins = []
    for bin in binr:
        if len(bin) < maxlen+1:
            bin = '0'*(maxlen-len(bin)) + bin
        bins.append(bin)
    #x_bin ,y_bin,z_bin = bins
    key = 0
    for i in range(maxlen+1):
        for bin in bins:
            key = 2*key + (bin[i] == '1')
    return key

def hilbert_sfc(cell_id, maxlen=20, dim=3):
    """Returns key of indices using Hilbert space filling curve """
    if isinstance(cell_id, IntPoint):
        cell_id = (cell_id.x,cell_id.y,cell_id.z)
    cell_id = cell_id[:dim]
    s = 2**maxlen
    return Hilbert_to_int([int(i+s) for i in cell_id])

sfc_func_dict = {'morton':morton_sfc}
if have_hilbert:
    sfc_func_dict['hilbert'] = hilbert_sfc
