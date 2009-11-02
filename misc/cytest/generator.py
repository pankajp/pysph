#!/usr/bin/python
"""
takes as argument 
1. a c-typename (int, char, long etc.)
2. template source filename
3. output filename
4. the 'ARRAY_TYPE' variable

Generates an output file with every occurance of
the 'ARRAY_TYPE' variable in the template source with 
the c-typename
"""

# header to be inserted into the pxd file
pxd_header_file = 'pxd_header.src'
# file containing the templates pxd source
pxd_source_file = 'pxd_source.src'
# header to be inserted into the pyx file
pyx_header_file = 'pyx_header.src'
# file containing the templates pyx source
pyx_source_file = 'pyx_source.src'
# file containing the test cases for the template
test_souce = ''

# filenames to write to, when writing all types
# to a single file.
single_pyx_file_name='carray.pyx'
single_pxd_file_name='carray.pxd'

# a dictionary containing the types that we want to generate
# and some metadata for each. 
# Currencty the list contains the name of the class that will
# be generated (the CLASSNAME variable will be substituted with 
# this value). This will also be the name of the file that will
# be generated.
c_types_info = {'int':['IntArray', "int_array", "NPY_INT",[]],
                'double':['DoubleArray', "double_array", "NPY_DOUBLE",[]],
                'long':['LongArray', "long_array", "NPY_LONG",[]],
                'float':['FloatArray', "float_array", "NPY_FLOAT",[]]
                }

from special_imports import special_imports
import sys

usage = 'generator template_source_file outputfile tpye_variable_name c-typename'

def make_header():
    header_message = '# This file has been generated using the command :\n'
    header_message += '# '
    for argv in sys.argv:
        header_message +=  argv 
        header_message += ' '
    header_message += '\n'
    header_message += '# Do not change this file.\n# Modify the source to make changes\n'
    return header_message

def get_current_date_time():
    """
    get the current date as a nice string.
    """
    import datetime
    now = datetime.datetime.now()

    return now.ctime()

def make_header_new():
    """
    Generates a header to be put at the top of the generated
    files.
    """
    curr_time = get_current_date_time()
    header_message = '# This file has been generated automatically on\n'
    header_message += '# %s\n'%(get_current_date_time())

    header_message += '# DO NOT modify this file\n'
    header_message += '# To make changes modify the source templates and'
    header_message += ' regenerate\n'
    
    return header_message    

def check_args():
    if len(sys.argv) != 5:
        print usage
        sys.exit(-1)

def generate(source, output, typevar, c_type):
    """
    the actual generator
    """
    f = open(source)
    data = f.read()
    op = open(output,'w')
    code = data.replace(typevar, c_type)
    op.write(make_header())
    op.write(code)

def generate_for_type(typename):
    """
    generate the pxd, pyx and test files for the given type
    """

    class_name   = c_types_info[typename][0]
    pyx_filename = c_types_info[typename][1] + '.pyx'
    pxd_filename = c_types_info[typename][1] + '.pxd'
    np_type_name = c_types_info[typename][2]
    print np_type_name
    
    f = open(pxd_source_file, 'r')
    data = f.read()
    f.close()

    code = data.replace('CLASSNAME', class_name)
    code = code.replace('ARRAY_TYPE', typename)
    code = code.replace('NUMPY_TYPENAME', np_type_name)


    f = open(pxd_filename,'w')
    f.write(make_header())

    if special_imports.has_key(typename):
        f.write(special_imports[typename])

    f.write(code)
    
    f.close()
    
    # now for the pyx file
    f = open(pyx_source_file, 'r')
    data = f.read()
    f.close()
    
    code = data.replace('CLASSNAME', class_name)
    code = code.replace('ARRAY_TYPE', typename)
    code = code.replace('NUMPY_TYPENAME', np_type_name)

    f = open(pyx_filename, 'w')
    f.write(make_header())
    
    if special_imports.has_key(typename):
        f.write(special_imports[typename])

    f.write(code)
    f.close()

    # now write the test files
    
    return

def generate_one_file():
    """
    Generate one file for all types.
    """
    pxd_f = open(single_pxd_file_name, 'w')
    pyx_f = open(single_pyx_file_name, 'w')

    warn_header = make_header_new()
    pxd_f.write(warn_header)
    pyx_f.write(warn_header)

    # read the pyx and pxd headers and write 
    # to output file.
    f = open(pxd_header_file, 'r')
    d = f.read()
    pxd_f.write(d)
    f.close()

    f = open(pyx_header_file, 'r')
    d = f.read()
    pyx_f.write(d)
    f.close()

    # read the pxd and pyx source from template files.
    f = open(pyx_source_file, 'r')
    pyx_source = f.read()
    f.close()
    f = open(pxd_source_file, 'r')
    pxd_source = f.read()
    f.close()

    for key in c_types_info.keys():

        typename = key
        class_name = c_types_info[typename][0]
        np_type_name = c_types_info[typename][2]

        # replace templates in pxd file.
        code = pxd_source.replace('CLASSNAME', class_name)
        code = code.replace('ARRAY_TYPE', typename)
        code = code.replace('NUMPY_TYPENAME', np_type_name)

        pxd_f.write(code)

        # replace templates in the pyx file.
        code = pyx_source.replace('CLASSNAME', class_name)
        code = code.replace('ARRAY_TYPE', typename)
        code = code.replace('NUMPY_TYPENAME', np_type_name)

        pyx_f.write(code)

        pyx_f.write('\n\n')
        pxd_f.write('\n\n')

    pxd_f.close()
    pyx_f.close()
    
def generate_separate_files():
    """
    Generate separate files for each type.
    """

    for key in c_types_info.keys():
        print 'Generaing sources for :', key 
        generate_for_type(key)

    return 0


if __name__ == "__main__":
    generate_one_file()
    #generate_separate_files()




#for every_type in c_type:
    # get the file name and the classname for this type
    # write the header
    # write the special imports if any for this type
    # write the template file
    
    # write the testcase files

    
