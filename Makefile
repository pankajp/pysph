ROOT = $(shell pwd)
MAKEFILE = $(ROOT)/Makefile
SRC = $(ROOT)/source
PKG = $(SRC)/pysph
SUBPKG = base sph solver parallel sph/funcs 
DIRS := $(foreach dir,$(SUBPKG),$(PKG)/$(dir))

# this is used for cython files on recursive call to make
PYX = $(wildcard *.pyx)

# set this variable to value to limit benchmark to those benches only
# example: BENCH="point kernels"
# empty BENCH runs all benchmarks
BENCH = 

MPI4PY_INCL = $(shell python -c "import mpi4py; print mpi4py.get_include()")

# the default target to make
all : build

.PHONY : $(DIRS) bench build

build :
	python setup.py build_ext --inplace

$(DIRS) : 
	cd $@;  python $(ROOT)/source/pysph/base/generator.py
	$(MAKE) -f $(MAKEFILE) -C $@ cythoncpp ROOT=$(ROOT)

%.c : %.pyx
	python `which cython` -I$(SRC) -I$(MPI4PY_INCL) $<

%.cpp : %.pyx
	python `which cython` --cplus -I$(SRC) -I$(MPI4PY_INCL) $<

%.html : %.pyx
	python `which cython` --cplus -I$(SRC) -I$(MPI4PY_INCL) -a $<

cython : $(PYX:.pyx=.c)

cythoncpp : $(PYX:.pyx=.cpp)

_annotate : $(PYX:.pyx=.html)

annotate :
	for f in $(DIRS); do $(MAKE) -f $(MAKEFILE) -C $${f} _annotate ROOT=$(ROOT); done

clean : 
	python setup.py clean
	-for dir in $(DIRS); do rm -f $$dir/*.c; done
	-for dir in $(DIRS); do rm -f $$dir/*.cpp; done

cleanall : clean
	-for dir in $(DIRS) bench; do rm -f $$dir/*.so; done
#	-rm $(patsubst %.pyx,%.c,$(wildcard $(PKG)/*/*.pyx))

test :
	python `which nosetests` --exe $(PKG)

bench :
	# try '$$ make bench BENCH="point kernels"' etc to limit the benchmarks run
	# AVAILABLE: carray cell kernels nnps parallel_timings point serial_timings
	# '$$ make bench BENCH=' runs all benchmarks
	#
	#####################################################################
	#
	#-$(MAKE) -f $(MAKEFILE) -i -C $(PKG)/bench/ cython ROOT=$(ROOT)
	cd bench; python bench.py $(BENCH)

epydoc :
	python cython-epydoc.py --config epydoc.cfg pysph

doc :
	cd docs; make html

develop :
	python setup.py develop

install :
	python setup.py install

clang :
	python $(ROOT)/source/pysph/base/generator.py
	for f in $(DIRS); do $(MAKE) -f $(MAKEFILE) -C $${f} cythoncpp ROOT=$(ROOT); done
	cd source/pysph/; for f in */*.cpp */*/*.cpp; do clang++ -g -O2 -shared -fPIC -o $${f%.*}.so $$f -I /usr/include/python2.7/ $(shell mpicxx --showme:compile) $(shell mpicxx --showme:link); done


