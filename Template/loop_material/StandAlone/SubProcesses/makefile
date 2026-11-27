
ifeq ($(wildcard ../Source/make_opts),)
    ifeq ($(wildcard ../../Source/make_opts),)
       ROOT = ../../..
    else
       ROOT = ../..
    endif
else
    ROOT = ..
endif
LIBDIR = $(abspath $(ROOT))/lib/

PROG   = check
all : $(PROG)

HERE := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
ROOTNAME = $(notdir $(abspath $(ROOT)))

# For the compilation of the MadLoop file polynomial.f it makes a big difference to use -O3 and
# to turn off the bounds check. These can however be modified here if really necessary.
POLYNOMIAL_OPTIMIZATION = -O3
POLYNOMIAL_BOUNDS_CHECK =
# The other MadLoop sources (loop_matrix.f, the Born, CT and coefficient
# construction calls, the reduction interfaces) are otherwise compiled without
# optimization, as standalone make_opts sets no -O flag. It comes before
# FFLAGS, so that an -O flag given there (e.g. -O0 -g) still takes precedence.
MADLOOP_OPTIMIZATION = -O2

include $(ROOT)/Source/make_opts
FFLAGS += -I$(ROOT)/Source/MODEL -I$(ROOT)/Source/DHELAS
include $(ROOT)/SubProcesses/MadLoop_makefile_definitions
SHELL = /bin/bash

OLP    = OLP
STABCHECKDRIVER = StabilityCheckDriver
CHECK_SA_BORN_SPLITORDERS = check_sa_born_splitOrders
LINKLIBS = -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS)
LIBS =  $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext) $(LOOP_LIBS)
DYLIBS =  $(LIBDIR)libdhelas.$(dylibext) $(LIBDIR)libmodel.$(dylibext) $(LOOP_LIBS)

PROCESS= MadLoopParamReader.o MadLoopCommons.o \
 $(patsubst $(DOTF),$(DOTO),$(wildcard polynomial.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard jamp?_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_born_amps_and_wfs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_compute_loop_coefs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard loop_CT_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mp_coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard TIR_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard GOLEM_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard COLLIER_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard TIRUST_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard compute_color_flows.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard onia_loop.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard mg5_citation.f))

OLP_PROCESS= MadLoopParamReader.o MadLoopCommons.o \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/polynomial.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/improve_ps.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/born_matrix.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/CT_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_num.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/jamp?_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_born_amps_and_wfs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_compute_loop_coefs.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_helas_calls*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/loop_CT_calls_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/mp_coef_construction_*.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/TIR_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/GOLEM_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/COLLIER_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/TIRUST_interface.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/compute_color_flows.f)) \
 $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/onia_loop.f))

POLYNOMIAL = $(patsubst $(DOTF),$(DOTO),$(wildcard polynomial.f))
OLP_POLYNOMIAL = $(patsubst $(DOTF),$(DOTO),$(wildcard $(LOOP_PREFIX)*/polynomial.f))



$(PROG): check_sa.o $(PROCESS) makefile $(LIBS) libcollier.$(dylibext)
	$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)

$(STABCHECKDRIVER):  StabilityCheckDriver.o $(PROCESS) makefile $(LIBS)
	$(FC) $(FFLAGS) -o $(STABCHECKDRIVER) StabilityCheckDriver.o $(PROCESS) $(LINKLIBS)

# The program below is not essential but just an helpful one to run the born only
$(CHECK_SA_BORN_SPLITORDERS):  check_sa_born_splitOrders.o $(patsubst $(DOTF),$(DOTO),$(wildcard *born_matrix.f)) makefile $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)
	$(FC) $(FFLAGS) -o $(CHECK_SA_BORN_SPLITORDERS) check_sa_born_splitOrders.o $(patsubst $(DOTF),$(DOTO),$(wildcard *born_matrix.f)) -L$(LIBDIR) -ldhelas -lmodel

# This is the core of madloop computationally wise, so make sure to turn optimizations on and bound checks off.
# We use %olynomial.o and not directly polynomial.o because we want it to match when both doing make check here
# or make OLP one directory above
%oloop_matrix.o : %olynomial.o %oloop_matrix.f
%olynomial.o : %olynomial.f
	$(FC) $(patsubst -O%,, $(subst -fbounds-check,,$(FFLAGS))) $(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) -c $< -o $@ $(LOOP_INCLUDE)

%/%oloop_matrix.o : %/polynomial.o %/%oloop_matrix.f
	$(FC) $(patsubst -O%,,$(subst -fbounds-check,,$(FFLAGS))) \
		$(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) \
		-c $< -o $@ $(LOOP_INCLUDE)

%/polynomial.o : %/polynomial.f
	$(FC) $(patsubst -O%,,$(subst -fbounds-check,,$(FFLAGS))) \
		$(POLYNOMIAL_OPTIMIZATION) $(POLYNOMIAL_BOUNDS_CHECK) \
		-c $< -o $@ $(LOOP_INCLUDE)


$(DOTO) : $(DOTF) $(POLYNOMIAL) $(OLP_POLYNOMIAL)
	$(FC) $(MADLOOP_OPTIMIZATION) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)

$(DOTO) : $(DOTF)
	$(FC) $(MADLOOP_OPTIMIZATION) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)

$(OLP): $(OLP_PROCESS) $(LIBS) mg5_citation.o
	$(FC) -shared $(OLP_PROCESS) mg5_citation.o -o libMadLoop.$(dylibext) $(LINKLIBS)

$(OLP)_static: $(OLP_PROCESS)
	ar rcs libMadLoop.$(libext) $(OLP_PROCESS)
	mv libMadLoop.$(libext) $(MADLOOP_LIB)

../$(OLP):
	ln -sf ../libMadLoop.$(dylibext) libMadLoop.$(dylibext)
	cd $(ROOT)/SubProcesses; make $(OLP)

../$(OLP)_static:
	cd $(ROOT)/SubProcesses; make $(OLP)_static

# ROOT is .. only when this makefile runs from SubProcesses itself; from a
# P<i>_* subdirectory it is ../.. . The ../$(OLP) route symlinks to the
# SubProcesses-level library and recurses into $(ROOT)/SubProcesses, which
# from SubProcesses is this very directory: that second make rebuilds the
# same objects as the local $(OLP) rule, so under -j the two collide (torn
# .o files, or "ln: File exists"). Build libMadLoop here directly instead.
ifeq ($(ROOT),..)
libMadLoop.$(dylibext): $(OLP)
else
libMadLoop.$(dylibext): ../$(OLP)
endif

WRAPPER_SRCS := $(wildcard */f2py_wrapper.f)
WRAPPER_OBJS := $(patsubst %.f,%.o,$(wildcard */f2py_wrapper.f))

%/f2py_wrapper.o: %/f2py_wrapper.f
	$(MAKE) -C $* f2py_wrapper.o




ALL_DOTF := $(wildcard */polynomial.f */loop_matrix.f */improve_ps.f */born_matrix.f */CT_interface.f \
             */loop_num.f \
             */helas_calls*.f */mp_compute_loop_coefs.f */mp_helas_calls*.f */coef_construction_*.f \
             */loop_CT_calls_*.f */mp_coef_construction_*.f */TIR_interface.f */COLLIER_interface.f */TIRUST_interface.f \
             MadLoopParamReader.f MadLoopCommons.f mg5_citation.f */GOLEM_interface.f */compute_color_flows.f \
             */mp_born_amps_and_wfs.f */jamp?_calls_*.f)

# Convert .f to .o
ALL_DOTO := $(patsubst %.f,%.o,$(ALL_DOTF))

ifeq ($(UNAME), Darwin)
  LIBALLME_DYNFLAG = -install_name @rpath/$(notdir $@) -undefined dynamic_lookup
  WHOLE_ARCH=-Wl,-force_load,
  NOWHOLE_ARCH=
  STAT_LIB = $(WHOLE_ARCH)$(LIBDIR)libcts.$(libext) $(WHOLE_ARCH)$(LIBDIR)libiregi.$(libext)
  LD_F2PY=
else
  LIBALLME_DYNFLAG =
  WHOLE_ARCH=  -Wl,--whole-archive
  NOWHOLE_ARCH= -Wl,--no-whole-archive
  STAT_LIB = $(WHOLE_ARCH) $(LIBDIR)libcts.$(libext) $(LIBDIR)libiregi.$(libext) $(NOWHOLE_ARCH)
  LD_F2PY=-lgfortran -lquadmath
endif

ifeq ($(origin MENUM),undefined)
  MENUM=2
endif

# name of the f2py python module. A caller importing a rebuild of this
# directory in a python process that already imported the previous build has
# to give it (and ROOTNAME) a new value: a module cannot be imported twice.
F2PYMOD= all_matrix$(MENUM)py

# f2py links the shared libraries below with -l, and its meson backend names a
# variable after that name: a '-' (output of a restricted model,
# PROC_sm-no_b_mass_0) becomes '_'.
LIBALLME = all$(subst -,_,$(ROOTNAME))_$(MENUM)me

lib$(LIBALLME).$(dylibext): all_matrix.o libMadLoop.$(dylibext) $(WRAPPER_OBJS) $(LIBS)  $(OLP)
	$(CXX) $(DYNLIBFLAG)  $(LIBALLME_DYNFLAG)  $(STDLIB_FLAG)  -o $@ all_matrix.o */f2py_wrapper.o ../Source/DHELAS/*.o ../Source/MODEL/*.o $(STAT_LIB) $(RPATH_LIBS) $(LINK_LOOP_LIBS) $(ALL_DOTO) $(STDLIB) $(LINK_LOOP_LIBS)



libcollier.$(dylibext):
	ln -s $(LIBDIR)/collier_lib/libcollier.$(dylibext) || echo "libcolier already linked" 



shared: lib$(LIBALLME).$(dylibext)




# The three f2py modules are built the same way. A shared library, linked by
# the C++ compiler, carries the Fortran objects (cite() included, through
# mg5_citation.o), the reduction libraries and the C++ runtime. f2py then only
# compiles the wrapper and links that library through LDFLAGS and -l, so no
# -lc++ or -lstdc++ reaches f2py's command line: its meson backend (the only
# one on Python >= 3.12) names a variable after each -l library, and c++ or
# stdc++ is not a valid name for one. meson also ignores --f77exec and takes
# the Fortran compiler from FC.
# matrix and allmatrix wrap routines already compiled into their library: f2py
# builds them from a signature file (f2py -h), so no Fortran source is copied
# to meson's build directory, where INCLUDE files are not found and the
# P*/f2py_wrapper.f of different subprocesses would overwrite each other.
# Of the loop libraries, LDFLAGS takes only the library paths and linker
# options (run paths): the libraries themselves are in the wrapped library.
# meson also links LDFLAGS into its compiler sanity check, which runs in its
# build directory, where a relative run path ($ORIGIN, as for TIRust with
# internal dependencies) does not find a shared library.
F2PY_ENV = FC="$(FC)" LDFLAGS="-Wl,-rpath,$(HERE) -L$(HERE) $(RPATH_LIBS) $(filter -L% -Wl%,$(LINK_LOOP_LIBS)) $(LD_F2PY)"

# library of matrix$(MENUM)py, named after the output and subprocess
# directories, like liball after the output one: f2py modules of several
# subprocesses can be loaded in one process.
MATRIXME = me$(subst -,_,$(ROOTNAME))_$(notdir $(abspath .))_$(MENUM)

lib$(MATRIXME).$(dylibext): f2py_wrapper.o $(PROCESS) $(LIBS) makefile
	$(CXX) $(DYNLIBFLAG) $(LIBALLME_DYNFLAG) $(STDLIB_FLAG) -o $@ f2py_wrapper.o $(PROCESS) $(ROOT)/Source/DHELAS/*.o $(ROOT)/Source/MODEL/*.o $(STAT_LIB) $(RPATH_LIBS) $(LINK_LOOP_LIBS) $(STDLIB)

matrix$(MENUM)py.so: lib$(MATRIXME).$(dylibext) f2py_wrapper.f makefile
	$(F2PY) -m matrix$(MENUM)py -h matrix$(MENUM)py.pyf --overwrite-signature f2py_wrapper.f
	$(F2PY_ENV) $(F2PY) -c matrix$(MENUM)py.pyf -L$(HERE) -l$(MATRIXME)
	touch __init__.py

ALLMATRIX_WRAPPERS = $(wildcard $(LOOP_PREFIX)*/f2py_wrapper.f)

allmatrix$(MENUM)py.so: lib$(LIBALLME).$(dylibext) all_matrix.f $(ALLMATRIX_WRAPPERS) makefile
	$(F2PY) -m allmatrix$(MENUM)py -h allmatrix$(MENUM)py.pyf --overwrite-signature all_matrix.f $(ALLMATRIX_WRAPPERS)
	$(F2PY_ENV) $(F2PY) -c allmatrix$(MENUM)py.pyf -L$(HERE) -l$(LIBALLME)
	touch __init__.py


all_matrix$(MENUM)py.so:  lib$(LIBALLME).$(dylibext) f2py_wrapper.f makefile
	$(F2PY_ENV) $(F2PY)  -c f2py_wrapper.f -L$(HERE) -l$(LIBALLME) -m $(F2PYMOD)
	touch all_matrix$(MENUM)py.so
	touch __init__.py


clean:
	@rm -f *.o *.so *.$(libext) *.$(dylibext) *.pyf
