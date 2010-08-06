ROOT = $(PWD)
MAKEFILE = $(ROOT)/Makefile
ROOT = $(PWD)
PKG = $(ROOT)/source/pysph
DIRS = $(PKG)/base $(PKG)/sph $(PKG)/solver $(PKG)/parallel
PYX = $(wildcard *.pyx)
BENCH = 
MPI4PY_INCL = $(shell python -c "import mpi4py; print mpi4py.get_include()")
all : $(DIRS) extn
.PHONY : $(DIRS)
$(DIRS) : 
	cd $@;  python $(ROOT)/source/pysph/base/generator.py
	$(MAKE) -f $(MAKEFILE) -C $@ cython ROOT=$(ROOT)
%.c : %.pyx
	cython -I../.. -I$(MPI4PY_INCL) -a $<
cython : $(PYX:.pyx=.c)
extn : $(DIRS)
	python setup.py build_ext --inplace
clean :
	python setup.py clean
	-rm $(PKG)/*/*.so
cleanall : clean
	-rm $(patsubst %.pyx,%.c,$(PKG)/*/*.pyx)
test :
	nosetests --exe $(PKG)
bench :
	# can also try '$$ make bench BENCH="point kernels"' etc to limit the benchmarks run
	$(MAKE) -f $(MAKEFILE) -C $(PKG)/bench/ cython
	python setup.py build_ext --inplace
	python $(PKG)/bench/bench.py $(BENCH)
coverage :
	nosetests --exe --cover-erase --with-coverage --cover-html-dir=cover/ --cover-html --cover-package=pysph source/pysph/
epydoc :
	python cython-epydoc.py --config epydoc.cfg pysph
doc :
	cd docs; make html

