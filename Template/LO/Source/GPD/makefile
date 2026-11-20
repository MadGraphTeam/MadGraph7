# ----------------------------------------------------------------------------
# Makefile of the GPD library (generalised parton distributions and the
# x integration of GPD processes). The dd model needs the PDF library:
# 'make' builds it (madevent), 'make standalone' builds the dummy.
# ----------------------------------------------------------------------------
LIBRARY = libgpd.$(libext)
LIBDIR  = ../../lib/

include ../make_opts

GPD_COMMON = gpd_models.o gpd_xint.o

all: $(LIBDIR)$(LIBRARY)

standalone: $(GPD_COMMON) gpd_dd_dummy.o
	rm -f $(LIBDIR)$(LIBRARY)
	$(call CREATELIB, $(LIBDIR)$(LIBRARY), $^)

# nuclear PDF sets of the dd model (gpd_npdf) through LHAPDF, if linked
ifdef lhapdf
GPD_NPDF = gpd_npdf_lhapdf.o
else
GPD_NPDF = gpd_npdf_dummy.o
endif

$(LIBDIR)$(LIBRARY): $(GPD_COMMON) gpd_dd.o $(GPD_NPDF)
	rm -f $(LIBDIR)$(LIBRARY)
	$(call CREATELIB, $@, $^)

gpd_models.o: gpd_models.f gpd.inc
gpd_xint.o: gpd_xint.f gpd.inc
gpd_dd.o: gpd_dd.f gpd.inc
gpd_npdf_lhapdf.o: gpd_npdf_lhapdf.f
gpd_npdf_dummy.o: gpd_npdf_dummy.f

clean:
	$(RM) *.o *.mod $(LIBDIR)$(LIBRARY)
