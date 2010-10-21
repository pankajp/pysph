"""
Module contains some common functions.
"""

# standard imports
import numpy, sys, os, format
import zipfile
import tempfile

def check_array(x, y):
    """Check if two arrays are equal with an absolute tolerance of
    1e-16."""
    return numpy.allclose(x, y, atol=1e-16, rtol=0)


def zipfile_factory(*args, **kwargs):
    import zipfile
    if sys.version_info >= (2, 5):
        kwargs['allowZip64'] = True
    return zipfile.ZipFile(*args, **kwargs)

def savez(file, *args, **kwds):
    """
    Save several arrays into a single file in uncompressed ``.npz`` format.

    If arguments are passed in with no keywords, the corresponding variable
    names, in the .npz file, are 'arr_0', 'arr_1', etc. If keyword arguments
    are given, the corresponding variable names, in the ``.npz`` file will
    match the keyword names.

    Parameters
    ----------
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the data will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    *args : Arguments, optional
        Arrays to save to the file. Since it is not possible for Python to
        know the names of the arrays outside `savez`, the arrays will be saved
        with names "arr_0", "arr_1", and so on. These arguments can be any
        expression.
    **kwds : Keyword arguments, optional
        Arrays to save to the file. Arrays will be saved in the file with the
        keyword names.

    Returns
    -------
    None

    See Also
    --------
    save : Save a single array to a binary file in NumPy format.
    savetxt : Save an array to a file as plain text.

    Notes
    -----
    The ``.npz`` file format is a zipped archive of files named after the
    variables they contain.  The archive is not compressed and each file
    in the archive contains one variable in ``.npy`` format. For a
    description of the ``.npy`` format, see `format`.

    When opening the saved ``.npz`` file with `load` a `NpzFile` object is
    returned. This is a dictionary-like object which can be queried for
    its list of arrays (with the ``.files`` attribute), and for the arrays
    themselves.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()
    >>> x = np.arange(10)
    >>> y = np.sin(x)

    Using `savez` with *args, the arrays are saved with default names.

    >>> np.savez(outfile, x, y)
    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['arr_1', 'arr_0']
    >>> npzfile['arr_0']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    Using `savez` with **kwds, the arrays are saved with the keyword names.

    >>> outfile = TemporaryFile()
    >>> np.savez(outfile, x=x, y=y)
    >>> outfile.seek(0)
    >>> npzfile = np.load(outfile)
    >>> npzfile.files
    ['y', 'x']
    >>> npzfile['x']
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    See Also
    --------
    numpy.savez_compressed : Save several arrays into a compressed .npz file
    format

    """
    _savez(file, args, kwds, False)

def savez_compressed(file, *args, **kwds):
    """
    Save several arrays into a single file in compressed ``.npz`` format.

    If keyword arguments are given, then filenames are taken from the keywords.
    If arguments are passed in with no keywords, then stored file names are
    arr_0, arr_1, etc.

    Parameters
    ----------
    file : string
        File name of .npz file.
    args : Arguments
        Function arguments.
    kwds : Keyword arguments
        Keywords.

    See Also
    --------
    numpy.savez : Save several arrays into an uncompressed .npz file format

    """
    _savez(file, args, kwds, True)

def _savez(file, args, kwds, compress):
    # Import is postponed to here since zipfile depends on gzip, an optional
    # component of the so-called standard library.
    import zipfile
    # Import deferred for startup time improvement
    import tempfile

    if isinstance(file, basestring):
        if not file.endswith('.npz'):
            file = file + '.npz'

    namedict = kwds
    for i, val in enumerate(args):
        key = 'arr_%d' % i
        if key in namedict.keys():
            msg = "Cannot use un-named variables and keyword %s" % key
            raise ValueError, msg
        namedict[key] = val

    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED

    zip = zipfile_factory(file, mode="w", compression=compression)

    # Stage arrays in a temporary file on disk, before writing to zip.
    fd, tmpfile = tempfile.mkstemp(suffix='-numpy.npy')
    os.close(fd)
    try:
        for key, val in namedict.iteritems():
            fname = key + '.npy'
            fid = open(tmpfile, 'wb')
            try:
                format.write_array(fid, numpy.asanyarray(val))
                fid.close()
                fid = None
                zip.write(tmpfile, arcname=fname)
            finally:
                if fid:
                    fid.close()
    finally:
        os.remove(tmpfile)

    zip.close()

#############################################################################


def get_distributed_particles(pa, comm, cell_size):
    
    from pysph.parallel.load_balancer import LoadBalancer
    num_procs=comm.Get_size()
    rank = comm.Get_rank()

    n = len(pa.properties)

    if rank == 0:
        lb = LoadBalancer.distribute_particles(pa, num_procs=num_procs, 
                                               cell_size=0.025)

        for dest in range(1, len(lb)):
            parr = lb[dest]
            comm.send(parr, dest=dest)
            for key, arr in parr.properties.iteritems():
                comm.send((key, arr), dest=dest)

        pa = lb[0]

    else:
        pa = comm.recv(source=0)
        data = {}
        for i in range(n):
            key, arr = comm.recv(source=0)
            data[key] = arr
            pa.set(**data)

    return pa
