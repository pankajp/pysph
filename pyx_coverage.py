''' module to find coverage of cython functions

Usage: This file adds cython function coverage analysis ability to
Ned Batchelder's coverage.py http://nedbatchelder.com/code/coverage/ ,
you need to have it installed.
All commands and options for coverage.py are applicable here too.

To find coverage of cython files (pyx extension) you need to do following:
1. compile cython code to 'c' with directive profile=True
2. keep source pyx files in same locations as the compiled .so files
    i.e. use 'python setup.py build_ext --inplace' or 'python setup.py develop'
3. run coverage (this file) with the option timid enables
    i.e. 'python pyx_coverage.py run --timid my_module.py'

You can use nose test collector as follows:
python pyx_coverage.py run /path/to/nosetests /path/to/source
'''
import sys
import os
import re
from StringIO import StringIO

from coverage.files import FileLocator
from coverage.collector import PyTracer
from coverage.parser import CodeParser

FileLocator.canonical_filename_orig = FileLocator.canonical_filename
def FileLocator_canonical_filename(FileLocator_self, filename):
    self = FileLocator_self
    cf = self.canonical_filename_orig(filename)
    if cf.endswith('.so'):
        if os.path.exists(cf[:-3] + '.pyx'):
            cf = cf[:-3] + '.pyx'
        else:
            # this is so that parser does not crash on .so files
            cf = None
        self.canonical_filename_cache[filename] = cf
    return self.canonical_filename_cache[filename]
FileLocator.canonical_filename = FileLocator_canonical_filename


def PyTracer__profile(PyTracer_self, frame, event, func):
    """The profile function passed to sys.setprofile."""
    self = PyTracer_self
    filename = frame.f_code.co_filename
    if filename.endswith('pyx'):
        #self._trace(frame, event, func)
        if event == 'call':
            tracename = self.should_trace_cache.get(filename)
            if tracename is None:
                tracename = self.should_trace(filename, frame)
                self.should_trace_cache[filename] = tracename
            if tracename:
                if tracename not in self.data:
                    self.data[tracename] = {}
                self.data[tracename][frame.f_lineno] = None
    return self._profile
PyTracer._profile = PyTracer__profile

PyTracer.start_orig = PyTracer.start
def PyTracer_start(PyTracer_self):
    """Start this Tracer.

    Return a Python function suitable for use with sys.settrace().

    """
    self = PyTracer_self
    sys.settrace(self._trace)
    sys.setprofile(self._profile)
    return self._trace
PyTracer.start = PyTracer_start

PyTracer.stop_orig = PyTracer.stop
def PyTracer_stop(PyTracer_self):
    """Stop this Tracer."""
    sys.settrace(None)
    sys.setprofile(None)
PyTracer.stop = PyTracer_stop


CodeParser._raw_parse_orig = CodeParser._raw_parse
def CodeParser__raw_parse(CodeParser_self):
    self = CodeParser_self
    if self.filename.endswith('.pyx'):
        self._parse_pyx()
    else:
        self._raw_parse_orig()
CodeParser._raw_parse = CodeParser__raw_parse

def CodeParser__parse_pyx(CodeParser_self):
    """Parse cython pyx files to find the functions defined.

    A handful of member fields are updated.

    """
    self = CodeParser_self
    # Find lines which match an exclusion pattern.
    if self.exclude:
        re_exclude = re.compile(self.exclude)
        for i, ltext in enumerate(self.lines):
            if re_exclude.search(ltext):
                self.excluded.add(i + 1)
    
    # cython function pattern
    # [c[p]]def [inline] rettype fname (<matching parenthesis>) :
    func_pattern = re.compile(r'(?P<def>[c]?[p]?def)\s*(inline)?\s*' + 
        r'(?P<rtype>\S*?)\s*?(?P<name>\S+?)(?P<args>' + 
        r'(?<!\\)\((\\\(|\\\)|[^\(\)]|(?<!\\)\(.*(?<!\\)\))*(?<!\\)\))\s*:')
    
    linestarts = [0]
    for line in StringIO(self.text):
        linestarts.append(linestarts[-1] + len(line))
    classdefs = range(1, len(linestarts))
    lno = 0
    func_starts = []
    for match in re.finditer(func_pattern, self.text):
        while linestarts[lno] <= match.start():
            lno += 1
        if match.group('rtype') == 'class':
            lno -= 1
            continue
        func_starts.append(lno)
        classdefs.remove(lno)
        lno -= 1
    self.classdefs = classdefs
    self.statement_starts.update(func_starts)
CodeParser._parse_pyx = CodeParser__parse_pyx

CodeParser.arcs_orig = CodeParser.arcs
def CodeParser_arcs(CodeParser_self):
    self = CodeParser_self
    if self.filename.endswith('.pyx'):
        return []
    else:
        return self.arcs_orig()
CodeParser.arcs = CodeParser_arcs

if __name__ == '__main__':
    import sys
    from pkg_resources import load_entry_point    
    sys.exit(load_entry_point('coverage', 'console_scripts', 'coverage')())
