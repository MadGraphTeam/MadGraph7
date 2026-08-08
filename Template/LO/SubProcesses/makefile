include ../../Source/make_opts
FFLAGS+= -w -I ../../Source/DHELAS/ -I ../../Source/MODEL

# Load additional dependencies of the bias module, if present
ifeq (,$(wildcard ../bias_dependencies))
BIASDEPENDENCIES =
else
include ../bias_dependencies
endif

# Definitions

LIBDIR = ../../lib/
BINDIR = ../../bin/
PROG   = madevent

ifneq ("$(wildcard ../MadLoop_makefile_definitions)","")
    include ../MadLoop_makefile_definitions
else
    LINK_LOOP_LIBS =
    LOOP_LIBS =
    LOOP_INCLUDE =
    LINK_MADLOOP_LIB =
    MADLOOP_LIB =
endif

LINKLIBS = $(LINK_MADLOOP_LIB) $(LINK_LOOP_LIBS) -L../../lib/ -ldhelas -ldsample -lmodel -lgeneric -lpdf -lgammaUPC -lcernlib $(llhapdf) -lbias 

# what the batched color sum is linked against, empty unless a matrix
# element was generated with one
BLASLIBS =
LINKLIBS += $(BLASLIBS)

LIBS = $(LIBDIR)libbias.$(libext) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libdsample.$(libext) $(LIBDIR)libgeneric.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libgammaUPC.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext) $(MADLOOP_LIB) $(LOOP_LIBS)

ifneq ("$(wildcard ../../Source/RUNNING)","")
    LINKLIBS += -lrunning
    LIBS += $(LIBDIR)librunning.$(libext) 
endif


# Source files

MATRIX_HEL = $(patsubst %.f,%.o,$(wildcard matrix*_orig.f))
MATRIX = $(patsubst %.f,%.o,$(wildcard matrix*_optim.f))
# Crossing-symmetry routers: matrix*_router.f share a base subprocess's matrix
# element and are never recycled, so they are compiled into both the forhel and
# the optimized binaries alongside the recycled bases. When recycling is off the
# matrix*.f glob below already covers them.
ROUTER = $(patsubst %.f,%.o,$(wildcard matrix*_router.f))
# Amplitude chunks: at high multiplicity the HELAS call sequence of a matrix
# element is emitted as its own set of files, one subroutine each, so that it
# is not one enormous basic block (gfortran's cost on which grows faster than
# linearly) and so that it can carry its own optimisation flag -- see AMP_FLAG
# below. matrix<i>_origamp<k>.f goes into both binaries, because a matrix
# element that has no helicity to recycle is reused as its own optimized copy.
ORIGAMP = $(patsubst %.f,%.o,$(wildcard matrix*_origamp*.f))
OPTIMAMP = $(patsubst %.f,%.o,$(wildcard matrix*_optimamp*.f))
AMPCHUNK = $(ORIGAMP) $(OPTIMAMP)
ifeq ($(strip $(MATRIX_HEL)),)
        MATRIX = $(patsubst %.f,%.o,$(wildcard matrix*.f))
        AMPCHUNK =
else
        MATRIX += $(ROUTER) $(AMPCHUNK)
        MATRIX_HEL += $(ROUTER) $(ORIGAMP)
endif
# every matrix object except the amplitude chunks, which get a rule of their own
MATRIX_CORE = $(filter-out $(AMPCHUNK),$(sort $(MATRIX) $(MATRIX_HEL)))


PROCESS= driver.o myamp.o genps.o unwgt.o setcuts.o \
         cuts.o cluster.o reweight.o initcluster.o addmothers.o setscales.o \
	 idenparts.o dummy_fct.o \
         $(patsubst %.f,%.o,$(wildcard auto_dsig*.f)) \

SYMMETRY = symmetry.o idenparts.o 

# Binaries

$(PROG): $(PROCESS) auto_dsig.o $(LIBS) $(MATRIX)
	$(FC) -o $(PROG) $(PROCESS) $(MATRIX) $(LINKLIBS) $(LDFLAGS) $(BIASDEPENDENCIES) -fopenmp

$(PROG)_forhel: $(PROCESS) auto_dsig.o $(LIBS) $(MATRIX_HEL)
	$(FC) -o $(PROG)_forhel $(PROCESS) $(MATRIX_HEL) $(LINKLIBS) $(LDFLAGS) $(BIASDEPENDENCIES) -fopenmp

libcollier.$(dylibext):
	ln -s $(LIBDIR)/collier_lib/libcollier.$(dylibext) || echo 'already done'

gensym: $(SYMMETRY) configs.inc $(LIBS) libcollier.$(dylibext)
	$(FC) -o gensym $(SYMMETRY) -L../../lib/ $(LINKLIBS) $(LDFLAGS)

$(LIBDIR)libmodel.$(libext): ../../Cards/param_card.dat
	cd ../../Source/MODEL; make

$(LIBDIR)libgeneric.$(libext): ../../Cards/run_card.dat
	cd ../../Source; make

$(LIBDIR)libpdf.$(libext): 
	cd ../../Source/PDF; make

$(LIBDIR)libgammaUPC.$(libext):
	cd ../../Source/PDF/gammaUPC; make

# Add source so that the compiler finds the DiscreteSampler module.
$(MATRIX_CORE): %.o: %.f
	$(FC) $(FFLAGS) $(MATRIX_FLAG) -c $< -I../../Source/ -I../../Source/PDF/gammaUPC
# The amplitude chunks carry their own flag, which lands after FFLAGS and so
# wins over the GLOBAL_FLAG. It is empty by default: splitting the sequence up
# is what makes it compilable, and dropping it to -O0 on top buys about 1.5x
# more on the compile but costs 19% of the run time at g g > t t~ 3g and 61% at
# g g > 5g. Set amp_flag in the run_card when the compile is the problem.
$(AMPCHUNK): %.o: %.f
	$(FC) $(FFLAGS) $(MATRIX_FLAG) $(AMP_FLAG) -c $< -I../../Source/ -I../../Source/PDF/gammaUPC
%.o: %.f
	$(FC) $(FFLAGS) -c $< -I../../Source/ -I../../Source/PDF/gammaUPC

# Dependencies

driver.f: genps.inc
symmetry.o: genps.inc nexternal.inc configs.inc run_config.inc ../../Source/run_card.inc
genps.o: genps.inc nexternal.inc configs.inc
dummy_fct.0: run.inc genps.inc
cuts.o: genps.inc nexternal.inc pmass.inc
setcuts.o: genps.inc run_config.inc
invarients.o: genps.inc nexternal.inc
myamp.o: props.inc genps.inc nexternal.inc
reweight.o: sudakov.inc cluster.inc sudakov.inc run.inc message.inc \
	    run_config.inc
cluster.o: cluster.inc genps.inc nexternal.inc message.inc
addmothers.o: genps.inc nexternal.inc symswap.inc message.inc
unwgt.o: genps.inc nexternal.inc symswap.inc cluster.inc run.inc message.inc \
	 run_config.inc
initcluster.o: message.inc

clean:
	$(RM) *.o gensym madevent madevent_forhel

# Track B cross-group crossing: present only in a dependent P directory, it makes
# the base group's matrix<b> object(s) be symlinked from the base directory
# instead of recompiled here (see write_crossgroup_mk). Included LAST so its
# specific rules override the makefile's %.o pattern / $(MATRIX) static-pattern
# rules. Absent (a no-op) elsewhere.
-include crossgroup.mk
