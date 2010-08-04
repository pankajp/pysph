#!/usr/bin/python
'''
Module to collect and generate source files from `cog` template files

`cog`: http://nedbatchelder.com/code/cog/
The template files must have an extension '.src'.
The generated files have the name same as the src file but with the '.src'
extension removed and the last underscore '_' replace with a dot '.'
Example: `carray_pyx.src` is generated into `carray.pyx`
'''

from cogapp import Cog
import os

def get_src_files(dirname):
    '''returns all files in directory having and extension `.src`'''
    ls = os.listdir(dirname)
    ls = [os.path.join(dirname,f) for f in ls if f.endswith('.src')]
    return ls

def generate_files(src_files):
    '''generates source files from the template cog files with extension `.src`
    '''
    for filename in src_files:
        outfile = '.'.join(filename[:-4].rsplit('_',1))
        print 'generating file %s from %s' %(outfile, filename)
        Cog().main([__file__, '-d', '-o', outfile, filename])

def main(paths=None):
    '''generates source files using cog template files

    `args` is a list of `.src` cog template files to convert
    if `args` is `None` all src files in this file's directory are converted
    if `args` is an empty list all src files in current directory are converted
    '''
    if paths is None:
        files = [os.path.dirname(__file__)]
    elif len(paths)>0:
        files = paths
    else:
        files = get_src_files(os.path.curdir)
    generate_files(files)

if __name__ == '__main__':
    import sys
    if '--help' in sys.argv or '-h' in sys.argv:
        print 'usage:'
        print '    generator.py [filenames]'
        print
        print ('    Convert `cog` template files with extension `.src` into '
        'source files')
        print ('    If filenames is omitted all `.src` files in current '
        'directory will be converted')

    else:
        main(sys.argv[1:])

