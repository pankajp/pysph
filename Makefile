ROOT = $(shell pwd)
MAKEFILE = $(ROOT)/Makefile
SRC = $(ROOT)/source
PKG = $(SRC)/pysph
SUBPKG = base sph solver parallel
DIRS := $(foreach dir,$(SUBPKG),$(PKG)/$(dir))

# this is used for cython files on recursive call to make
PYX = $(wildcard *.pyx)

# set this variable to value to limit benchmark to those benches only
# example: BENCH="point kernels"
# empty BENCH runs all benchmarks
BENCH = 

MPI4PY_INCL = $(shell python -c "import mpi4py; print mpi4py.get_include()")

# the default target to make
all : $(DIRS) extn

.PHONY : $(DIRS)

$(DIRS) : 
	cd $@;  python $(ROOT)/source/pysph/base/generator.py
	$(MAKE) -f $(MAKEFILE) -C $@ cython ROOT=$(ROOT)

%.c : %.pyx
	python `which cython` --directive profile=True -I$(SRC) -I$(MPI4PY_INCL) -a $<

cython : $(PYX:.pyx=.c)

extn : $(DIRS)
	python setup.py build_ext --inplace

clean : 
	python setup.py clean
	-for dir in $(DIRS) $(PKG)/bench; do rm -f $$dir/*.so; done

cleanall : clean
	-for dir in $(DIRS) $(PKG)/bench; do rm -f $$dir/*.c; done
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
	-$(MAKE) -f $(MAKEFILE) -i -C $(PKG)/bench/ cython ROOT=$(ROOT)
	cd $(PKG)/bench; python bench.py $(BENCH)

coverage :
	python `which nosetests` --exe --cover-erase --with-coverage --cover-html-dir=cover/ --cover-html --cover-package=pysph source/pysph/

epydoc :
	python cython-epydoc.py --config epydoc.cfg pysph

doc :
	cd docs; make html

develop : $(DIRS)
	python setup.py develop

install : $(DIRS)
	python setup.py install

