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
all : $(DIRS) extn

.PHONY : $(DIRS) bench

$(DIRS) : 
	cd $@;  python $(ROOT)/source/pysph/base/generator.py
	$(MAKE) -f $(MAKEFILE) -C $@ cython ROOT=$(ROOT)

%.c : %.pyx
	python `which cython` -I$(SRC) -I$(MPI4PY_INCL) -a $<

%.cpp : %.pyx
	python `which cython` --cplus -I$(SRC) -I$(MPI4PY_INCL) -a $<

cython : $(PYX:.pyx=.c)

cythoncpp : $(PYX:.pyx=.cpp)

extn : $(DIRS)
	python setup.py build_ext --inplace

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

coverage2 :
	python `which nosetests` --exe --cover-erase --with-coverage --cover-html-dir=htmlcov/ --cover-html --cover-package=pysph source/pysph/

coverage :
	python pyx_coverage.py erase
	-python pyx_coverage.py run `which nosetests` --exe source/pysph/
	python pyx_coverage.py html

epydoc :
	python cython-epydoc.py --config epydoc.cfg pysph

doc :
	cd docs; make html

develop : $(DIRS)
	python setup.py develop

install : $(DIRS)
	python setup.py install

clang :
	python $(ROOT)/source/pysph/base/generator.py
	for f in $(DIRS); do $(MAKE) -f $(MAKEFILE) -C $${f} cythoncpp ROOT=$(ROOT); done
	cd source/pysph/; for f in */*.cpp; do clang++ -g -shared -fPIC -o $${f%.*}.so $$f -I /usr/include/python2.7/ $(shell mpicc --showme:compile) $(shell mpicc --showme:link); done
	cd source/pysph/; for f in */*/*.cpp; do clang++ -g -shared -fPIC -o $${f%.*}.so $$f -I /usr/include/python2.7/ $(shell mpicxx --showme:compile) (shell mpicxx --showme:link); done


