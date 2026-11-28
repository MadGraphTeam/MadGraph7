# Copyright (C) 2020-2026 CERN and UCLouvain.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created originally by: A. Valassi (Sep 2021) for the MadGraph7 CUDACPP plugin.
# Further modified by: S. Hageboeck, O. Mattelaer, S. Roiser, J. Teig, A. Valassi, Z. Wettersten (2021-2024).
# Integrated with the MadGraph7 project in Feb 2026.

import shutil
import os
import sys
import subprocess
import json
import copy
import re

PLUGIN_NAME = __name__.rsplit('.',1)[0]
PLUGINDIR = os.path.dirname( __file__ )

# AV - model_handling includes the custom FileWriter, ALOHAWriter, UFOModelConverter, OneProcessExporter and HelasCallWriter, plus additional patches
from . import model_handling
from . import hel_recycling_filter

# AV - create a plugin-specific logger
import logging
logger = logging.getLogger('madgraph.%s.output'%PLUGIN_NAME)
from madgraph import MG5DIR
#------------------------------------------------------------------------------------

from os.path import join as pjoin
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.export_cpp as export_cpp
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.various.misc as misc



NLO_REAL_MANIFEST_VERSION = 2
NLO_REAL_MANIFEST_TOP_LEVEL_KEYS = (
    'version', 'library_model', 'process_id', 'nexternal', 'split_order_names',
    'global_squared_orders', 'real_matrix_elements', 'fks_rows')
NLO_REAL_MANIFEST_REAL_ME_KEYS = (
    'id', 'library_process_id', 'local_flavours',
    'local_squared_orders', 'backend_capabilities', 'process_fingerprint')
NLO_REAL_MANIFEST_FKS_ROW_KEYS = (
    'fks_row', 'topology', 'real_me_id', 'fortran_flavour_index',
    'madmatrix_flavour_index', 'pdgs', 'local_to_global')

def relative_path_list(relative_path, files_list):
    return list(map(lambda f: pjoin(relative_path, f), files_list))

def template_sources(dirpath, extensions=('.h', '.cc')):
    """The C++ sources of a template directory, sorted. Only regular files with
    a source extension: whatever else lies there (.DS_Store, editor backups
    such as foo.cc~ or .#foo.cc, merge leftovers) must not be copied into every
    generated output."""
    return sorted(f for f in os.listdir(dirpath)
                  if f.endswith(extensions) and not f.startswith('.')
                  and os.path.isfile(pjoin(dirpath, f)))

# AV - define the plugin's process exporter
# (NB: this is the plugin's main class, enabled in the new_output dictionary in __init__.py)
class ProcessExporterMadMatrix(export_cpp.ProcessExporterMG7):
    # Class structure information
    #  - object
    #  - VirtualExporter(object) [in madgraph/iolibs/export_v4.py]
    #  - ProcessExporterCPP(VirtualExporter) [in madgraph/iolibs/export_cpp.py]
    #  - ProcessExporterMG7(ProcessExporterCPP) [in madgraph/iolibs/export_cpp.py]
    #  - ProcessExporterMadMatrix(ProcessExporterMG7)
    #      This class

    # Below are the class variable that are defined in export_v4.VirtualExporter
    # AV - keep defaults from export_v4.VirtualExporter
    # Check status of the directory. Remove it if already exists
    ###check = True
    # Output type: [Template/dir/None] copy the Template (via copy_template), just create dir or do nothing
    ###output = 'Template'

    # If sa_symmetry is true, generate fewer matrix elements
    # AV - keep OM's default for this plugin (using grouped_mode=False, "can decide to merge uu~ and u~u anyway")
    sa_symmetry = True

    # The name this exporter is reached by on the 'output' line, for the error
    # messages that have to name it back to the user.
    format_name = 'mg7'

    # The color sum can run on the (n-2)! Del Duca-Dixon-Maltoni basis for a
    # multi-gluon process, but a color flow still has to be picked among the
    # (n-1)! trace structures, so the trace basis is built alongside and the
    # trace jamps are rebuilt from the DDM ones through the Kleiss-Kuijf
    # relations (see set_color_flow_lines_cpp in model_handling.py).
    ddm_color_basis = 'auto'
    ddm_needs_flow_basis = True

    # Below are the class variable that are defined in export_cpp.ProcessExporterGPU
    # AV - keep defaults from export_cpp.ProcessExporterGPU
    # Decide which type of merging is used [madevent/madweight]
    grouped_mode = False
    # Other options
    default_opt = {'clean': False, 'complex_mass':False, 'export_format':'madevent', 'mp': False, 'v5_model': True }

    # AV - keep defaults from export_cpp.ProcessExporterGPU
    # AV - used in MadGraphCmd.do_output to assign export_cpp.ExportCPPFactory to MadGraphCmd._curr_exporter (if cpp or gpu)
    # AV - used in MadGraphCmd.export to assign helas_call_writers.(CPPUFO|GPUFO)HelasCallWriter to MadGraphCmd._curr_helas_model (if cpp or gpu)
    # Language type: 'v4' for f77, 'cpp' for C++ output
    exporter = 'gpu'

    # AV - use a custom OneProcessExporter
    oneprocessclass = model_handling.OneProcessExporterMadMatrix

    # Crossing symmetry (extended flavor id) is supported by the madmatrix
    # backends, cpu/simd and GPU (gated by --use_crossing, default on,
    # and by the matrix element recording a crossed subprocess: see
    # ProcessExporterMG7.generate_subprocess_directory). The MG7 (pure-cpp
    # mg7_v5) exporter keeps supports_crossing=False. When the machinery is off
    # the generated output is byte-identical to before.
    supports_crossing = True

    # Information to find the template files to include from madgraph
    templates_path = pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files')
    mg7_templates = pjoin(templates_path, 'mg7')
    madmatrix_templates = pjoin(templates_path, 'madmatrix')
    home_path = pjoin(MG5DIR, "madmatrix")

    from_template = {'.': relative_path_list(home_path, ['COPYRIGHT', 'COPYING', 'COPYING.LESSER']),
                     'src': relative_path_list(madmatrix_templates, [
                         'read_slha.h', 'read_slha.cc',
                         # --hel_recycling: the per-thread tables, shared by the subprocess libraries
                         'HelRecyclingScratch.h', 'HelRecyclingScratch.cc'
                     ]),
                     # Backend-owned skeleton files live only under backend/<variant>/ now
                     # (see backend_variants below); only genuinely backend-agnostic files
                     # (no backend/ counterpart) are copied flat into SubProcesses/. umami.h
                     # is the only one needed outside standalone mode too (it's the header
                     # for backend/<variant>/umami.cc's UMAMI API); nvtx.h is
                     # standalone-driver-only (see _standalone_extra_files below). The
                     # rambo/random-number sources live once in src/rambo/
                     'SubProcesses': relative_path_list(madmatrix_templates, ['umami.h']),
                     # run_card.toml is generated in finalize() (ProcessExporterMG7.create_run_card)
                     # from the template, not copied verbatim.
                     # Default cards for the optional post-processing tools
                     # (Pythia8/Delphes/MadSpin/reweight/analysis) so that
                     # bin/generate_events can offer to enable and edit them.
                     # The reweight card is the MG7 one: it documents both
                     # reweighting backends (classic module, MadtRex).
                     'Cards': relative_path_list(pjoin(MG5DIR, 'Template', 'Common', 'Cards'),
                                  ['madspin_card_default.dat',
                                   'density_card_default.dat',
                                   'delphes_card_default.dat']) +
                              relative_path_list(mg7_templates, ['reweight_card_default.dat']) +
                              relative_path_list(pjoin(MG5DIR, 'Template', 'LO', 'Cards'),
                                  ['pythia8_card_default.dat',
                                   'madanalysis5_parton_card_default.dat',
                                   'madanalysis5_hadron_card_default.dat',
                                   'rivet_card_default.dat']),
                     # the script do_delphes launches (from Events/): without
                     # it Delphes failed with "No such file or directory"
                     'bin/internal': relative_path_list(pjoin(MG5DIR, 'Template', 'LO', 'bin', 'internal'),
                                  ['run_delphes3'])}

    # Backend split: mirror template_files/madmatrix/backend/{common,cpu,simd,gpu}/
    # as a top-level backend/ dir, sibling of SubProcesses/src/lib; madmatrix.mk
    # picks the variant (BACKENDDIR) at make time.
    # backend/common/ holds the files that are identical for every variant; it
    # is searched after backend/<variant>/ and no file name is in both, since a
    # quoted #include resolves in the including file's own directory first.
    backend_variants = ('cpu', 'simd', 'gpu')
    backend_template_dir = pjoin(madmatrix_templates, 'backend')
    for _backend_variant in ('common',) + backend_variants:  # plain loop: comprehension wouldn't see the locals above
        from_template[pjoin('backend', _backend_variant)] = relative_path_list(
            pjoin(backend_template_dir, _backend_variant),
            template_sources(pjoin(backend_template_dir, _backend_variant)))
    del _backend_variant

    # Rambo/random-number files copy in src/rambo/
    rambo_template_dir = pjoin(madmatrix_templates, 'src', 'rambo')
    from_template['src/rambo'] = relative_path_list(
        rambo_template_dir, template_sources(rambo_template_dir))

    # Backend-owned skeleton files (GpuRuntime.h, color_sum.{h,cc}, the
    # MemoryAccess*.h family, umami.cc, etc.) are NOT linked into P* at all: they are read straight from the
    # top-level backend/<variant>/ and backend/common/ dirs via the Makefile's
    # INCFLAGS/vpath (see BACKENDDIR in madmatrix.mk). They are still compiled
    # once per P* directory, into its own build.<BACKEND>_<FPTYPE>/, and have to be:
    # they include that directory's generated ProcessData.h, ColorData.h and
    # ProcessTables.h. Only files with no backend/ counterpart - genuinely
    # backend-agnostic - stay here.
    to_link_in_P = ['umami.h']

    template_src_make = pjoin(madmatrix_templates, 'madmatrix_src.mk')
    # SubProcesses/makefile is only a dispatcher over the P* directories: it is
    # what makes 'make -j N' in SubProcesses build all the subprocesses with a
    # single, shared pool of N jobs.
    template_Sub_make = pjoin(madmatrix_templates, 'madmatrix_subprocesses.mk')

    # The actual build rules, rendered into SubProcesses/. Each P* directory
    # links one of them (p_makefile, see OneProcessExporterMadMatrix) as its own
    # 'makefile'.
    p_makefiles = ['madmatrix.mk']

    dirs_to_create = ['bin', 'bin/internal', 'src', 'src/rambo', 'lib', 'Cards', 'SubProcesses',
                      'backend',
                      'backend/common',
                      'backend/cpu',
                      'backend/simd',
                      'backend/gpu']

    # AV - use a custom UFOModelConverter (model/aloha exporter)
    create_model_class = model_handling.MadMatrixUFOModelConverter

    # AV - use a custom GPUFOHelasCallWriter
    # (NB: use "helas_exporter" - see class MadGraphCmd in madgraph_interface.py - not "aloha_exporter" that is never used!)
    ###helas_exporter = None
    helas_exporter = model_handling.MadMatrixUFOHelasCallWriter # this is one of the main fixes for issue #341!

    # AV (default from OM's tutorial) - add a debug printout
    def __init__(self, *args, **kwargs):
        self.in_madevent_mode = False # see MR #747
        # {model} is resolved by ProcessExporterMG7.generate_subprocess_directory()
        # alongside {process_id}, from the matrix element's own model -- same
        # sanitized name (get_model_name()) used for MADMATRIX_LIB/
        # MADMATRIX_COMMONLIB's %(model)s in madmatrix.mk. {{device}}/{{fptype}}
        # stay literal ("{device}"/"{fptype}") in the resulting me_path: which
        # concrete BACKEND/FPTYPE a given process directory was compiled for is
        # a per-build choice made when the libraries are compiled and loaded
        # (see compile_matrix_elements() in mg7/launch.py), not something
        # subprocesses.json commits to at generation time.
        args[1]["me_lib_format"] = pjoin(
            "lib", "libmadmatrix_{model}_{process_id}_{{device}}_{{fptype}}.so"
        )
        super().__init__(*args, **kwargs)
        # Honor the output command's --mask=True|False (flavor-mask
        # optimization for grouped/merged flavors). Default: enabled.
        self.use_flavor_mask = self._parse_flavor_mask_option()
        # --axial_gauge is read per matrix element by the helas call writer;
        # say here, once per output, if the FD gauge drops it
        helas_call_writers.axial_gauge_requested(
                            self.opt.get('output_options'), warn=True)
        # --hel_recycling: a new output starts with nothing deferred of its own;
        # what an earlier one left registered (one that died before its
        # finalize) must never be recorded for this one
        del model_handling.OneProcessExporterMadMatrix._hr_warmups[:]

    def _parse_flavor_mask_option(self):
        """Read --mask=True|False from the output command line (default True)."""
        out_opts = self.opt.get('output_options', {}) if hasattr(self, 'opt') else {}
        val = out_opts.get('mask', True)
        if isinstance(val, str):
            return val.strip().lower() not in ('false', '0', 'no', 'off')
        return bool(val)

    def get_makefile_replace_dict(self, model):
        """Add what madmatrix.mk needs to know about a host BLAS for the C++
        color sum. Whether a given process actually takes it is decided when
        that process is written out (see cpp_blas_wanted); this only settles
        whether one could be linked at all. Only Accelerate has established
        evidence for enabling it by default: a generic linkable BLAS can be
        scalar Netlib, slower than the generated SIMD colour sum. Keep its
        flags available for an explicit CPPBLAS=hasBlas override."""

        replace_dict = super().get_makefile_replace_dict(model)
        flags = self.oneprocessclass.blas_available_flags()
        replace_dict['cpp_blas_default'] = (
            'hasBlas' if flags == '-framework Accelerate' else 'hasNoBlas')
        replace_dict['cpp_blas_libflags'] = flags
        return replace_dict

    # AV - overload the default version: create CMake directory, do not create lib directory
    def copy_template(self, model):
        if not getattr(self, 'has_madmatrix', True):
            # only the cards and the launcher: no src/, backend/ or makefiles
            self.dirs_to_create = ['bin', 'bin/internal', 'lib', 'Cards', 'SubProcesses']
            self.from_template = {key: self.from_template[key]
                                  for key in ('.', 'Cards', 'bin/internal')}
            self.template_src_make = self.template_Sub_make = None
            return super().copy_template(model)
        super().copy_template(model)
        # Copy Arithmetics headers for the double-word expansion (FPTYPE=e)
        arithmetics_src = pjoin(self.madmatrix_templates, 'Arithmetics')
        if os.path.isdir(arithmetics_src):
            arithmetics_dst = pjoin(self.dir_path, 'src', 'Arithmetics')
            try:
                os.makedirs(arithmetics_dst, exist_ok=True)
            except os.error:
                pass
            for f in ['Double.h', 'basicOPs.h', 'errorFreeOPs.h', 'Invariant.h']:
                files.cp(pjoin(arithmetics_src, f), arithmetics_dst)

        # Rename Makefile to makefile
        if self.template_src_make:
            shutil.move(os.path.join(self.dir_path, "src", "Makefile"), os.path.join(self.dir_path, "src", "makefile"))
        if self.template_Sub_make:
            shutil.move(os.path.join(self.dir_path, "SubProcesses", "Makefile"), os.path.join(self.dir_path, "SubProcesses", "makefile"))
        self.write_p_makefiles(model)

    def write_p_makefiles(self, model):
        """Render the build rules shared by all the P* directories into
        SubProcesses/ (they are linked from there as each P*/makefile)."""
        # through the hook, not an inline dict: madmatrix.mk also carries the
        # host-BLAS placeholders that get_makefile_replace_dict fills in
        replace_dict = self.get_makefile_replace_dict(model)
        # kept to render the --hel_recycling probe makefile at finalize
        self._p_makefile_replace = replace_dict
        for name in self.p_makefiles:
            rendered = self.read_template_file(pjoin(self.madmatrix_templates, name)) % replace_dict
            open(pjoin(self.dir_path, 'SubProcesses', name), 'w').write(rendered)

    def check_split_orders(self, matrix_element):
        """Report what a squared-order constraint will produce here.

        The jamps carry an amplitude-order index and the color sum pairs them
        (color_sum_cpu_splitorders in backend/{cpu,simd}/color_sum.cc and
        color_sum_kernel in backend/gpu/color_sum.cc, the Fortran GET_MATRIX
        contract), so a '^2' constraint that keeps only some squared orders gets
        the contribution it asked for rather than the total. That is what makes
        the interference case work -- `u u~ > u u~ QED^2==2` keeps every diagram
        and wants the QCD-EW cross term alone, which no amount of dropping
        diagrams at generation can produce.

        The BLAS color sums (CPU: compile time, GPU: per run)
        cannot pair split orders and are not used for such a process.
        """

        so = export_v4.split_order_tables(matrix_element)
        if not so or so['nampso'] <= 1:
            return
        process = matrix_element.get('processes')[0]
        kept = [n for n, k in zip(so['names'], so['chosen']) if k]
        dropped = [n for n, k in zip(so['names'], so['chosen']) if not k]
        logger.info(
            "%s: '%s' has %d squared-order components (%s); keeping %s%s. "
            "The jamps are split over %d amplitude orders and the color sum "
            "pairs them.",
            self.__class__.format_name,
            process.nice_string().replace('Process: ', ''),
            so['nsqampso'], ', '.join(so['names']),
            ', '.join(kept) if kept else 'nothing',
            '' if not dropped else ', dropping %s' % ', '.join(dropped),
            so['nampso'])

    # AV - add debug printouts (in addition to the default one from OM's tutorial)
    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer, proc_number=None):
        self.check_split_orders(matrix_element)
        # Propagate the --mask toggle to the helas call writer that emits the
        # guarded wavefunction/amplitude calls, and the output command line as
        # a whole for the --jamp_optim toggle of the color-flow optimisation.
        if cpp_helas_call_writer is not None:
            cpp_helas_call_writer.use_flavor_mask = self.use_flavor_mask
            cpp_helas_call_writer.cmd_options = self.opt.get('output_options', {})
        return super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)

    def write_aloha_library(self, model, wanted_lorentz, gauge='', **opts):
        """The HELAS header of the other gauge of an output whose processes
        are in two gauges, src/HelAmps_<model>_fd.h (called under its
        aloha.gauge_scope; the parameters are those of the session model)."""

        builder = self.create_model_class(model, pjoin(self.dir_path, 'src'),
                                          wanted_lorentz, [])
        builder.helamps_suffix = '_' + gauge.lower()
        builder.write_aloha_routines()

    # AV (default from OM's tutorial) - add a debug printout
    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[], **opts):
        # **opts: an option meant for one exporter (npwave, for the dual HELAS
        # libraries of P-wave bound states) is passed by keyword to all of them
        if not getattr(self, 'has_madmatrix', True):
            return
        if hasattr(model , 'madmatrix_wanted_ordered_couplings'):
            wanted_couplings = model.madmatrix_wanted_ordered_couplings
            del model.madmatrix_wanted_ordered_couplings
        return super().convert_model(model, wanted_lorentz, wanted_couplings, **opts)

    # AV (default from OM's tutorial) - overload settings and add a debug printout
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
        # Irrelevant here since group_mode=False so this function is never called
        misc.sprint('Entering ProcessExporterMadMatrix.modify_grouping')
        return False, matrix_element

    def finalize(self, *args, **kwargs):
        super().finalize(*args, **kwargs)
        # --hel_recycling: once every P* directory is complete, record it as
        # unmeasured (the launch measures it, CPU/SIMD only)
        self._hel_recycling_at_output()

    # The --hel_recycling probe runs through the standalone driver check_sa.exe,
    # which only `output standalone` ships. Any other output keeps a copy of it
    # (with the standalone makefile, under another name) in SubProcesses/hrprobe,
    # which the probe tree lends each P* directory it measures at launch
    # (madmatrix/hel_recycling_filter.py).
    _hr_probe_files = ['check_sa.cc', 'nvtx.h', 'timer.h', 'timermap.h']
    _hr_probe_makefile = hel_recycling_filter.PROBE_MAKEFILE

    def _write_hr_probe_sources(self):
        """SubProcesses/hrprobe: the driver and makefile of the probe build."""
        target = pjoin(self.dir_path, 'SubProcesses', hel_recycling_filter.PROBE_SOURCES)
        os.makedirs(target, exist_ok=True)
        for name in self._hr_probe_files:
            files.cp(pjoin(self.madmatrix_templates, name), pjoin(target, name))
        rendered = self.read_template_file(pjoin(self.madmatrix_templates,
                        'madmatrix_standalone.mk')) % self._p_makefile_replace
        with open(pjoin(target, self._hr_probe_makefile), 'w') as fsock:
            fsock.write(rendered)

    # Whether an output measures the --hel_recycling filter itself. An mg7
    # output does not: its launch measures it at the parameters of the run
    # (each scan point at its own card). A standalone output is never launched,
    # so it measures at its own param_card (see ProcessExporterMadMatrixStandalone).
    _hr_measure_at_output = False

    def _hel_recycling_at_output(self):
        """For each --hel_recycling P* directory of this output, either record
        that its filter is not measured yet and keep the plain code
        (hel_recycling_filter.defer_at_output; the first launch measures it at
        its own parameters, and each scan point at its own, through
        hel_recycling_filter.update from MadgraphProcess.update_hel_recycling),
        or, for an output that is never launched, measure it now at the
        output's param_card and write the recycled build
        (hel_recycling_filter.measure_at_output).

        The probe is built in a mirror of the output (.hrprobe), never in the
        P* directory itself. On any failure the directory keeps the plain
        code: correct but not recycled."""
        exporters = model_handling.OneProcessExporterMadMatrix._hr_warmups
        # only this output's directories, whatever happens: the list is shared
        # by every exporter of the session
        mine = [e for e in exporters if self._is_own_dir(e.path)]
        del exporters[:]
        if not mine:
            return
        self._write_hr_probe_sources()
        if self._hr_measure_at_output:
            hel_recycling_filter.measure_at_output(
                self.dir_path, mine, pjoin(self.dir_path, 'Cards', 'param_card.dat'))
        else:
            hel_recycling_filter.defer_at_output(self.dir_path, mine)

    def _is_own_dir(self, path):
        """Whether path lies in this output's directory."""
        own = os.path.realpath(self.dir_path)
        return os.path.commonpath([own, os.path.realpath(path)]) == own

    @staticmethod
    def _parse_hel_warmup(output):
        """See hel_recycling_filter.parse_probe_output."""
        return hel_recycling_filter.parse_probe_output(output)


class ProcessExporterMadMatrixNLOReal(ProcessExporterMadMatrix):
    """MadMatrix's tree-only companion to the ordinary FKS exporter.

    This class deliberately does not own, copy or finalize an LO output tree.
    It writes one uniquely named process directory for every distinct real
    ``N_ME`` while reusing the same OneProcess exporter and model converter as
    the LO exporter.
    """

    format_name = 'mg7 NLO real'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._nlo_model_prepared = False
        self._wanted_lorentz = []
        self._wanted_couplings = []

    @staticmethod
    def _append_unique(destination, values):
        for value in values:
            if value not in destination:
                destination.append(value)

    def prepare_nlo_tree(self, model):
        """Install only the common files required by generated real MEs."""
        if self._nlo_model_prepared:
            return
        self._nlo_tree_model = model
        for directory in ('src', 'lib', 'SubProcesses'):
            os.makedirs(pjoin(self.dir_path, directory), exist_ok=True)
        destinations = ('src', 'src/rambo', 'SubProcesses', 'backend/common',
                        'backend/cpu', 'backend/simd', 'backend/gpu')
        for destination in destinations:
            os.makedirs(pjoin(self.dir_path, destination), exist_ok=True)
            for source in self.from_template.get(destination, []):
                files.cp(source, pjoin(self.dir_path, destination,
                                       os.path.basename(source)))
        replacements = self.get_makefile_replace_dict(model)
        self.lib_model_name = replacements['model']
        src_make = self.read_template_file(self.template_src_make) % replacements
        with open(pjoin(self.dir_path, 'src', 'makefile'), 'w') as stream:
            stream.write(src_make)
        self.write_p_makefiles(model)
        arithmetics_src = pjoin(self.madmatrix_templates, 'Arithmetics')
        arithmetics_dst = pjoin(self.dir_path, 'src', 'Arithmetics')
        if os.path.isdir(arithmetics_src):
            os.makedirs(arithmetics_dst, exist_ok=True)
            for name in ('Double.h', 'basicOPs.h', 'errorFreeOPs.h', 'Invariant.h'):
                files.cp(pjoin(arithmetics_src, name),
                         pjoin(arithmetics_dst, name))
        self._nlo_model_prepared = True

    def nlo_tree_dependencies(self):
        """Return worker-local tree dependencies for low-memory generation."""
        return {
            'lorentz': self._wanted_lorentz,
            'couplings': self._wanted_couplings,
        }

    def merge_nlo_tree_dependencies(self, dependencies):
        """Merge dependencies returned by an isolated low-memory worker."""
        if not dependencies:
            return
        self._append_unique(self._wanted_lorentz,
                            dependencies.get('lorentz', []))
        self._append_unique(self._wanted_couplings,
                            dependencies.get('couplings', []))

    def _configure_nlo_process_exporter(self, matrix_element,
                                        cpp_helas_call_writer):
        """Create an NLO-real process writer with the production settings."""
        self.check_split_orders(matrix_element)
        cpp_helas_call_writer.use_flavor_mask = self.use_flavor_mask
        cpp_helas_call_writer.physical_flavor_indices = True
        process = matrix_element.get('processes')[0]
        model = process.get('model')
        cpp_helas_call_writer.zero_width_parameters = set(
            model.get_particle(leg.get('id')).get('width')
            for leg in process.get('legs_with_decays')) - {'ZERO'}
        cpp_helas_call_writer.cmd_options = self.opt.get('output_options', {})
        return self.oneprocessclass(
            matrix_element, cpp_helas_call_writer,
            merge_same_topologies=self.opt.get('merge_same_topologies', True),
            physical_flavor_indices=True, nlo_real_order_blocks=True)

    def prime_nlo_tree_dependencies(self, fks_matrix_element,
                                    cpp_helas_call_writer):
        """Preseed global coupling indices before low-memory workers fork.

        Every low-memory worker owns a copy of the HELAS writer. Without this
        ordered, one-matrix-element-at-a-time pass each copy would number its
        first coupling from zero while the shared model library retains one
        global coupling table. Building the C++ replacement dictionary is
        sufficient to discover the exact production order and does not write a
        second process tree.
        """
        for fksreal in fks_matrix_element.real_processes:
            matrix_element = fksreal.matrix_element
            self._append_unique(self._wanted_lorentz,
                                matrix_element.get_used_lorentz())
            process_exporter = self._configure_nlo_process_exporter(
                matrix_element, cpp_helas_call_writer)
            try:
                process_exporter.write_process_cc_file(False)
            finally:
                process_exporter.restore_original_numbering()
            for attribute in ('wanted_ordered_dep_couplings',
                              'wanted_ordered_indep_couplings',
                              'wanted_ordered_flv_couplings'):
                self._append_unique(
                    self._wanted_couplings,
                    getattr(cpp_helas_call_writer, attribute, []))

    def _generate_real_directory(self, matrix_element, cpp_helas_call_writer,
                                 born_number, real_number):
        process_exporter = self._configure_nlo_process_exporter(
            matrix_element, cpp_helas_call_writer)
        directory = 'PMM_B%d_R%d' % (born_number + 1, real_number)
        dirpath = pjoin(self.dir_path, 'SubProcesses', directory)
        if os.path.exists(dirpath):
            raise RuntimeError('duplicate MadMatrix NLO real directory %s' %
                               dirpath)
        os.mkdir(dirpath)
        with misc.chdir(dirpath):
            logger.info('Creating NLO real files in directory %s', dirpath)
            process_exporter.path = dirpath
            process_exporter.generate_process_files()
            for name in self.to_link_in_P:
                files.ln(pjoin(self.dir_path, 'SubProcesses', name), '.',
                         name=name)
        return directory, process_exporter

    def generate_real_processes(self, fks_matrix_element,
                                cpp_helas_call_writer, born_number,
                                process_id, global_squared_orders):
        """Write all distinct real MEs and the exact physical-row manifest."""
        reals = []
        local_orders = {}
        for real_number, fksreal in enumerate(
                fks_matrix_element.real_processes, start=1):
            matrix_element = fksreal.matrix_element
            # Capture tree dependencies before OneProcess generation: HELAS
            # writers are allowed to annotate the shared model for their own
            # later conversion pass.
            self._append_unique(self._wanted_lorentz,
                                matrix_element.get_used_lorentz())
            directory, process_exporter = self._generate_real_directory(
                matrix_element, cpp_helas_call_writer, born_number,
                real_number)
            for attribute in ('wanted_ordered_dep_couplings',
                              'wanted_ordered_indep_couplings',
                              'wanted_ordered_flv_couplings'):
                self._append_unique(
                    self._wanted_couplings,
                    getattr(cpp_helas_call_writer, attribute, []))
            squared_orders = [list(order) for order in
                              matrix_element.get_split_orders_mapping()[0]]
            if not squared_orders:
                squared_orders = [[]]
            local_orders[real_number] = squared_orders
            split = export_v4.split_order_tables(matrix_element)
            reals.append({
                'id': real_number,
                'library_process_id': directory,
                'local_flavours': process_exporter.physical_flavor_rows(),
                'local_squared_orders': squared_orders,
                'backend_capabilities': {
                    'scalar': True,
                    'simd': True,
                    'cuda': not split or split['nampso'] <= 1,
                    'hip': not split or split['nampso'] <= 1,
                },
                'process_fingerprint': process_exporter.process_fingerprint(),
            })

        global_orders = [list(order) for order in global_squared_orders]
        rows = []
        flavor_map = fks_matrix_element.get_fks_flavor_map(
            resolve_virtual=False)
        for row_number, entry in enumerate(flavor_map, start=1):
            real_number = entry['n_me']
            mapping = []
            for order in local_orders[real_number]:
                try:
                    mapping.append(global_orders.index(order) + 1)
                except ValueError:
                    raise RuntimeError(
                        'local real squared order %s is absent from global '
                        'AMP_SPLIT for %s' % (order, process_id))
            rows.append({
                'fks_row': row_number,
                'topology': entry['fks_config_index'],
                'real_me_id': real_number,
                'fortran_flavour_index': entry['real_flavor_index'],
                'madmatrix_flavour_index': entry['real_flavor_index'] - 1,
                'pdgs': list(entry['real_pdgs']),
                'local_to_global': mapping,
            })

        process = fks_matrix_element.born_me.get('processes')[0]
        manifest = {
            'version': NLO_REAL_MANIFEST_VERSION,
            'library_model': self.lib_model_name,
            'process_id': process_id,
            'nexternal': fks_matrix_element.get_nexternal_ninitial()[0],
            'split_order_names': list(process.get('split_orders') or []),
            'global_squared_orders': global_orders,
            'real_matrix_elements': reals,
            'fks_rows': rows,
        }
        manifest_path = pjoin(self.dir_path, 'SubProcesses', process_id,
                              'nlo_real_manifest.json')
        with open(manifest_path, 'w') as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write('\n')
        self._write_bridge_config(manifest, os.path.dirname(manifest_path))
        self._write_fortran_routing(manifest, os.path.dirname(manifest_path))
        return manifest

    def _write_bridge_config(self, manifest, process_path):
        """Materialize checked compile-time identities for the generic bridge."""
        lines = [
            '#ifndef MG7_NLO_REAL_BRIDGE_CONFIG_H',
            '#define MG7_NLO_REAL_BRIDGE_CONFIG_H 1',
            'static const char MG7_NLO_LIBRARY_MODEL[] = %s;' %
            json.dumps(manifest['library_model']),
            'struct MG7NLORealConfig {',
            '  const char* library_process_id;',
            '  const char* fingerprint;',
            '  int squared_order_count;',
            '  int flavour_count;',
            '};',
            'static const int MG7_NLO_PARTICLE_COUNT = %d;' %
            manifest['nexternal'],
            'static const char MG7_NLO_COUPLINGS[] = %s;' %
            json.dumps(','.join(manifest['split_order_names'])),
            'static const int MG7_NLO_COUPLING_COUNT = %d;' %
            len(manifest['split_order_names']),
            'static const int MG7_NLO_REAL_COUNT = %d;' %
            len(manifest['real_matrix_elements']),
            'static const MG7NLORealConfig MG7_NLO_REAL_CONFIGS[] = {']
        for real in manifest['real_matrix_elements']:
            lines.append('  {"%s", "%s", %d, %d},' % (
                real['library_process_id'], real['process_fingerprint'],
                len(real['local_squared_orders']),
                len(real['local_flavours'])))
        lines.extend(['};', '#endif'])
        with open(pjoin(process_path, 'nlo_real_bridge_config.h'), 'w') as stream:
            stream.write('\n'.join(lines) + '\n')
        for name in ('nlo_real_bridge.h', 'nlo_real_bridge.cc', 'umami.h'):
            files.ln(pjoin(self.madmatrix_templates, name), process_path,
                     name=name)
        directories = ' '.join(
            real['library_process_id']
            for real in manifest['real_matrix_elements'])
        gpu_directories = ' '.join(
            real['library_process_id']
            for real in manifest['real_matrix_elements']
            if real['backend_capabilities']['cuda'])
        makefile = [
            '# Generated MadMatrix NLO-real build dispatcher.',
            'CXX ?= g++',
            'BACKEND ?= scalar',
            'FPTYPE ?= d',
            'LIBTAG ?=',
            'HELINL ?= 0',
            'HRDCOD ?= 0',
            'NLO_REAL_DIRS := %s' % directories,
            'NLO_REAL_GPU_DIRS := %s' % gpu_directories,
            'NLO_REAL_BUILD_DIRS := $(NLO_REAL_DIRS)',
            'ifneq (,$(filter $(BACKEND),cuda hip))',
            'NLO_REAL_BUILD_DIRS := $(NLO_REAL_GPU_DIRS)',
            'endif',
            'NLO_REAL_BRIDGE := libnlo_real_bridge.so',
            'ifeq ($(shell uname -s),Darwin)',
            'NLO_REAL_BRIDGE_SONAME := -Wl,-install_name,@rpath/libnlo_real_bridge.so',
            'else',
            'NLO_REAL_BRIDGE_SONAME :=',
            'endif',
            '',
            '.PHONY: all libraries bridge clean commonlib',
            'all: libraries bridge',
            '',
            '# The common library (../../src) is shared by every P* directory.',
            '# The run interface builds it once with this target, before',
            '# compiling the P* directories in parallel with',
            '# MADMATRIX_COMMONLIB_EXTERNAL=1, so that they do not all rebuild',
            '# it at the same time. Any real-emission directory can build it.',
            'commonlib:',
            '\t@set -e; for directory in $(firstword $(NLO_REAL_BUILD_DIRS)); do \\',
            '\t  $(MAKE) -C ../$$directory USEBUILDDIR=1 BACKEND=$(BACKEND) '
            'FPTYPE=$(FPTYPE) LIBTAG=$(LIBTAG) HELINL=$(HELINL) HRDCOD=$(HRDCOD) '
            'NLO_REAL_STRICT=1 commonlib; \\',
            '\tdone',
            '',
            'libraries:',
            '\t@set -e; for directory in $(NLO_REAL_BUILD_DIRS); do \\',
            '\t  $(MAKE) -C ../$$directory USEBUILDDIR=1 BACKEND=$(BACKEND) '
            'FPTYPE=$(FPTYPE) LIBTAG=$(LIBTAG) HELINL=$(HELINL) HRDCOD=$(HRDCOD) '
            'NLO_REAL_STRICT=1; \\',
            '\tdone',
            '',
            'bridge: $(NLO_REAL_BRIDGE)',
            '',
            *self._bridge_build_identity_rules(),
            '',
            '$(NLO_REAL_BRIDGE): nlo_real_bridge.cc nlo_real_bridge.h '
            'nlo_real_bridge_config.h nlo_real_build_identity.h umami.h',
            '\t$(CXX) -O2 -std=c++17 -Wall -Wextra -fPIC -shared -I. '
            'nlo_real_bridge.cc -ldl $(NLO_REAL_BRIDGE_SONAME) -o $@',
            '',
            'clean:',
            '\trm -f $(NLO_REAL_BRIDGE) nlo_real_build_identity.h',
            '\t@set -e; for directory in $(NLO_REAL_DIRS); do \\',
            '\t  $(MAKE) -C ../$$directory cleanall; \\',
            '\tdone',
        ]
        with open(pjoin(process_path, 'nlo_real.mk'), 'w') as stream:
            stream.write('\n'.join(makefile) + '\n')

    @staticmethod
    def _bridge_build_identity_rules(fptype='$(FPTYPE)', libtag='$(LIBTAG)',
                                     backend='$(BACKEND)'):
        # Both dispatchers check the active content. A previous per-option
        # stamp can still exist when toggling A -> B -> A; it cannot prove the
        # fixed-name bridge and process DSOs currently implement A.
        definitions = [
            '#define MG7_NLO_REAL_FPTYPE "'+fptype+'"',
            '#define MG7_NLO_REAL_LIBTAG "'+libtag+'"',
            '// backend='+backend+' CPPBLAS=$(CPPBLAS) HELINL=$(HELINL) HRDCOD=$(HRDCOD) NLO_REAL_STRICT=1']
        quoted = ' '.join("'"+line+"'" for line in definitions)
        return ['.PHONY: nlo_real_identity_force',
                'nlo_real_build_identity.h: nlo_real_identity_force',
                "\t@printf '%s\\n' "+quoted+' > $@.new',
                '\t@cmp -s $@.new $@ || mv $@.new $@; rm -f $@.new']

    def _write_fortran_routing(self, manifest, process_path):
        """Route scalar Fortran real calls through the bridge with fallback."""
        real_by_id = {
            real['id']: real for real in manifest['real_matrix_elements']}
        max_local = max(
            len(real['local_squared_orders'])
            for real in manifest['real_matrix_elements'])
        real_count = len(manifest['real_matrix_elements'])

        capability_lines = []
        for backend in ('cuda', 'hip'):
            capability_lines.append(
                '    if (trim(backend) == "%s") then' % backend)
            for real in manifest['real_matrix_elements']:
                if real['backend_capabilities'][backend]:
                    continue
                rows = [str(row['fks_row']) for row in manifest['fks_rows']
                        if row['real_me_id'] == real['id']]
                capability_lines.extend([
                    '      real_unavailable(%d) = .true.' % real['id'],
                    '      write(*,\'(A,I0,A)\') &',
                    '        "MG7 NLO real GPU fallback: real_me_id=", %d, &' %
                    real['id'],
                    '        ", fks_rows=%s, reason=multiple squared orders"' %
                    ','.join(rows),
                ])
            capability_lines.append('    end if')

        cases = []
        for row in manifest['fks_rows']:
            nlocal = len(real_by_id[row['real_me_id']][
                'local_squared_orders'])
            mapping = ', '.join(str(index)
                                for index in row['local_to_global'])
            cases.extend([
                '  case (%d)' % row['fks_row'],
                '    real_me_id = %d' % row['real_me_id'],
                '    flavour = %d' % row.get('fortran_flavour_index', 1),
                '    nlocal = %d' % nlocal,
                '    local_to_global(1:nlocal) = (/ %s /)' % mapping,
            ])

        source = [
            'module mg7_nlo_real_offload_state',
            '  use, intrinsic :: iso_c_binding',
            '  implicit none',
            '  type(c_ptr), save :: context = c_null_ptr',
            '  logical, save :: initialization_attempted = .false.',
            '  logical, save :: bridge_available = .false.',
            '  logical, save :: real_unavailable(%d) = .false.' % real_count,
            '  logical, save :: real_evaluated(%d) = .false.' % real_count,
            '  logical, save :: trace_batches = .false., trace_ownership = .false.',
            '  ! The process-local bridge consumes one synchronous batch at a time.',
            '  ! Reuse packing storage; the logical count remains the order stride.',
            '  real(c_double), allocatable, save :: packed_momenta(:), packed_g(:)',
            '  real(c_double), allocatable, target, save :: packed_orders(:)',
            '  integer(c_int32_t), allocatable, save :: packed_flavour(:)',
            '  integer, allocatable, save :: packed_lanes(:)',
            "  include 'nlo_real_backend.inc'",
            '  character(len=64), save :: backend_name = "uninitialized"',
            '',
            '  interface',
            '    integer(c_int) function mg7_nlo_real_initialize(handle, &',
            '        param_card, library_dir, backend) bind(C)',
            '      import :: c_int, c_ptr, c_char',
            '      type(c_ptr) :: handle',
            '      character(c_char), intent(in) :: param_card(*)',
            '      character(c_char), intent(in) :: library_dir(*)',
            '      character(c_char), intent(in) :: backend(*)',
            '    end function',
            '    integer(c_int) function mg7_nlo_real_available(handle, real_id) bind(C)',
            '      import :: c_int, c_ptr',
            '      type(c_ptr), value :: handle',
            '      integer(c_int), value :: real_id',
            '    end function',
            '    integer(c_int) function mg7_nlo_real_evaluate_ew(handle, &',
            '        real_me_id, event_count, momenta, g_strong, g_ew, flavour, &',
            '        squared_orders) bind(C)',
            '      import :: c_int, c_int32_t, c_size_t, c_ptr, c_double',
            '      type(c_ptr), value :: handle',
            '      integer(c_int), value :: real_me_id',
            '      integer(c_size_t), value :: event_count',
            '      real(c_double), intent(in) :: momenta(*)',
            '      real(c_double), intent(in) :: g_strong(*)',
            '      real(c_double), value :: g_ew',
            '      integer(c_int32_t), intent(in) :: flavour(*)',
            '      real(c_double), intent(out) :: squared_orders(*)',
            '    end function',
            '    subroutine mg7_nlo_real_set_trace(handle,input,output) bind(C)',
            '      import :: c_ptr, c_funptr',
            '      type(c_ptr), value :: handle',
            '      type(c_funptr), value :: input, output',
            '    end subroutine',
            '    type(c_ptr) function mg7_nlo_real_last_error(handle) bind(C)',
            '      import :: c_ptr',
            '      type(c_ptr), value :: handle',
            '    end function',
            '  end interface',
            '',
            'contains',
            '',
            '  subroutine print_bridge_error(prefix)',
            '    character(len=*), intent(in) :: prefix',
            '    character(kind=c_char), pointer :: chars(:)',
            '    character(len=1024) :: message',
            '    type(c_ptr) :: pointer',
            '    integer :: index',
            '    message = ""',
            '    pointer = mg7_nlo_real_last_error(context)',
            '    if (c_associated(pointer)) then',
            '      call c_f_pointer(pointer, chars, (/ 1024 /))',
            '      do index = 1, 1024',
            '        if (chars(index) == c_null_char) exit',
            '        message(index:index) = chars(index)',
            '      end do',
            '    end if',
            '    write(*,\'(A,A,A)\') trim(prefix), ": ", trim(message)',
            '  end subroutine',
            '',
            '  subroutine abort_bridge_evaluation(real_me_id)',
            '    ! The backend of a real ME is fixed before it produces its first',
            '    ! value: a real ME that fails on its first evaluation (for',
            '    ! example a library that could not be loaded) uses Fortran',
            '    ! from the start, but a later failure must not switch to',
            '    ! Fortran in the middle of a run.',
            '    integer, intent(in) :: real_me_id',
            '    character(len=64) :: prefix',
            '    if (.not. real_evaluated(real_me_id)) then',
            '      real_unavailable(real_me_id) = .true.',
            '      call print_bridge_error(&',
            '        "MG7 NLO real ME disabled; using Fortran fallback")',
            '      return',
            '    end if',
            '    write(prefix,\'(A,I0)\') &',
            '      "MG7 NLO real evaluation failed for real ME ", real_me_id',
            '    call print_bridge_error(trim(prefix))',
            '    stop 1',
            '  end subroutine',
            '',
            '  subroutine initialize_bridge()',
            '    use mc_batch_trace, only: batch_trace_unit,trace_umami_input,trace_umami_output,real_profile_active',
            '    character(kind=c_char, len=1024) :: param_card, library_dir',
            '    character(kind=c_char, len=64) :: backend',
            '    character(len=1024) :: env_value',
            '    integer(c_int) :: status',
            '    integer :: length, env_status',
            '    if (initialization_attempted) return',
            '    initialization_attempted = .true.',
            '    param_card = "auto"',
            '    library_dir = "auto"',
            '    ! The backend is the run-card nlo_real_backend compiled into',
            '    ! nlo_real_backend.inc; there is no run-time override.',
            '    backend = default_backend',
            '    env_value = ""',
            '    call get_environment_variable("MG7_NLO_REAL_PARAM_CARD", &',
            '      env_value, length=length, status=env_status)',
            '    if (env_status == 0 .and. length > 0) &',
            '      param_card = env_value(1:length)',
            '    env_value = ""',
            '    call get_environment_variable("MG7_NLO_REAL_LIBRARY_DIR", &',
            '      env_value, length=length, status=env_status)',
            '    if (env_status == 0 .and. length > 0) &',
            '      library_dir = env_value(1:length)',
            '    env_value = ""',
            '    call get_environment_variable("MG7_NLO_REAL_TRACE", &',
            '      env_value, length=length, status=env_status)',
            '    trace_batches = .false.',
            '    if (env_status == 0 .and. length > 0) &',
            '      trace_batches = trim(env_value(1:length)) /= "0"',
            '    backend_name = trim(backend)',
            '    if (trim(backend) == "fortran") then',
            '      write(*,\'(A)\') &',
            '        "MG7 NLO real offload disabled; using Fortran fallback"',
            '      return',
            '    end if',
        ]
        source.extend(capability_lines)
        source.extend([
            '    if (all(real_unavailable)) then',
            '      write(*,\'(A,A,A)\') &',
            '        "MG7 NLO real backend ", trim(backend), &',
            '        " has no supported real libraries; using Fortran fallback"',
            '      return',
            '    end if',
            '    status = mg7_nlo_real_initialize(context, &',
            '      trim(param_card)//c_null_char, &',
            '      trim(library_dir)//c_null_char, trim(backend)//c_null_char)',
            '    if (status /= 0_c_int) then',
            '      call print_bridge_error("MG7 NLO real bridge unavailable")',
            '      return',
            '    end if',
            '    trace_ownership = batch_trace_unit() /= -1',
            '    if (trace_ownership .or. real_profile_active) call mg7_nlo_real_set_trace(context, &',
            '      c_funloc(trace_umami_input),c_funloc(trace_umami_output))',
            '    bridge_available = .true.',
            '    write(*,\'(A,A,A,I0)\') &',
            '      "MG7 NLO real offload initialized: backend=", &',
            '      trim(backend), ", real_libraries=", &',
            '      count(.not. real_unavailable)',
            '  end subroutine',
            '',
            '  subroutine resolve_real_row(row, real_me_id, flavour, nlocal, local_to_global)',
            '    integer, intent(in) :: row',
            '    integer, intent(out) :: real_me_id, flavour, nlocal, local_to_global(%d)' % max_local,
            '    real_me_id = 0',
            '    flavour = 0',
            '    nlocal = 0',
            '    local_to_global = 0',
            '    select case (row)',
            *cases,
            '    end select',
            '  end subroutine',
            '',
            'end module',
            '',
            'subroutine mg7_nlo_real_try(p, ret_amp_split, wgt, &',
            '    nfksprocess, real_flav_idx, g_input, success)',
            '  use, intrinsic :: iso_c_binding',
            '  use mg7_nlo_real_offload_state',
            '  use mc_batch_trace, only: trace_mapping',
            '  implicit none',
            "  include 'nexternal.inc'",
            "  include 'orders.inc'",
            '  real(c_double), intent(in) :: p(0:3,nexternal)',
            '  real(c_double), intent(out) :: ret_amp_split(amp_split_size)',
            '  real(c_double), intent(out) :: wgt',
            '  integer, intent(in) :: nfksprocess, real_flav_idx',
            '  real(c_double), intent(in) :: g_input',
            '  logical, intent(out) :: success',
            '  real(c_double) :: momenta(4*nexternal), g_strong(1)',
            '  real(c_double) :: local(%d), ans_max' % max_local,
            '  integer(c_int32_t) :: flavour(1)',
            '  integer :: local_to_global(%d)' % max_local,
            '  integer :: real_me_id, row_flavour, nlocal, ipart, imu, index',
            '  integer :: component_orders(nsplitorders)',
            '  complex(c_double_complex) :: gal(2)',
            '  common /weak/ gal',
            '  integer(c_int) :: status',
            '  success = .false.',
            '  ret_amp_split(:) = 0d0',
            '  wgt = 0d0',
            '  call resolve_real_row(nfksprocess, real_me_id, row_flavour, nlocal, local_to_global)',
            '  if (real_me_id == 0) return',
            '  call initialize_bridge()',
            '  if (.not. bridge_available) return',
            '  if (real_unavailable(real_me_id)) return',
            '  do imu = 0, 3',
            '    do ipart = 1, nexternal',
            '      momenta(imu*nexternal + ipart) = p(imu,ipart)',
            '    end do',
            '  end do',
            '  g_strong(1) = g_input',
            '  flavour(1) = int(real_flav_idx - 1, c_int32_t)',
            '  local(:) = 0d0',
            '  status = mg7_nlo_real_evaluate_ew(context, &',
            '    int(real_me_id,c_int), 1_c_size_t, momenta, g_strong, &',
            '    real(gal(1),c_double), flavour, local)',
            '  if (status /= 0_c_int) then',
            '    call abort_bridge_evaluation(real_me_id)',
            '    return',
            '  end if',
            '  real_evaluated(real_me_id) = .true.',
            '  ans_max = maxval(abs(local(1:nlocal)))',
            '  do index = 1, nlocal',
            '    call amp_split_pos_to_orders(local_to_global(index), component_orders)',
            '    if (trace_ownership) call trace_mapping(index,local_to_global(index),all(component_orders <= nlo_orders))',
            '    if (any(component_orders > nlo_orders)) cycle',
            '    wgt = wgt + local(index)',
            '    if (abs(local(index)) > ans_max*1d-12) &',
            '      ret_amp_split(local_to_global(index)) = local(index)',
            '  end do',
            '  if (abs(wgt) < ans_max*1d-12) wgt = 0d0',
            '  success = .true.',
            'end subroutine',
        ])
        source.extend([
            '',
            'subroutine mg7_nlo_real_try_batch(p, g_values, ew, &',
            '    ret_amp_split, wgt, active, vector_size, nfksprocess, &',
            '    real_flav_idx, success)',
            '  use, intrinsic :: iso_c_binding',
            '  use mg7_nlo_real_offload_state',
            '  use mc_batch_trace, only: trace_mapping',
            '  implicit none',
            "  include 'nexternal.inc'",
            "  include 'orders.inc'",
            '  integer, intent(in) :: vector_size, nfksprocess',
            '  integer, intent(in) :: real_flav_idx',
            '  real(c_double), intent(in) :: p(0:3,nexternal,vector_size)',
            '  real(c_double), intent(in) :: g_values(vector_size), ew',
            '  real(c_double), intent(out) :: &',
            '    ret_amp_split(amp_split_size,vector_size)',
            '  real(c_double), intent(out) :: wgt(vector_size)',
            '  logical, intent(in) :: active(vector_size)',
            '  logical, intent(out) :: success',
            '  real(c_double), pointer, contiguous :: local(:,:)',
            '  integer :: local_to_global(%d)' % max_local,
            '  integer :: real_me_id, row_flavour, nlocal, event_count',
            '  integer :: lane, event, ipart, imu, index',
            '  integer :: component_orders(nsplitorders)',
            '  real(c_double) :: ans_max',
            '  integer(c_int) :: status',
            '  success = .false.',
            '  ret_amp_split(:,:) = 0d0',
            '  wgt(:) = 0d0',
            '  call resolve_real_row(nfksprocess, real_me_id, row_flavour, nlocal, local_to_global)',
            '  if (real_me_id == 0) return',
            '  call initialize_bridge()',
            '  if (.not. bridge_available) return',
            '  if (real_unavailable(real_me_id)) return',
            '  event_count = count(active)',
            '  if (event_count == 0) then',
            '    success = .true.',
            '    return',
            '  end if',
            '  if (trace_batches) write(*,\'(A,A,A,I0,A,I0,A,I0)\') &',
            '    "MG7 NLO real batch: backend=", trim(backend_name), &',
            '    ", real_me_id=", real_me_id, ", vector_size=", &',
            '    vector_size, ", event_count=", event_count',
            '  if (allocated(packed_lanes)) then',
            '    if (size(packed_lanes) < event_count) &',
            '      deallocate(packed_lanes,packed_momenta,packed_g,packed_flavour,packed_orders)',
            '  end if',
            '  if (.not. allocated(packed_lanes)) then',
            '    allocate(packed_lanes(event_count),packed_momenta(event_count*4*nexternal), &',
            '      packed_g(event_count),packed_flavour(event_count),packed_orders(event_count*%d))' % max_local,
            '  end if',
            '  local(1:event_count,1:nlocal) => packed_orders(1:event_count*nlocal)',
            '  event = 0',
            '  do lane = 1, vector_size',
            '    if (.not. active(lane)) cycle',
            '    event = event + 1',
            '    packed_lanes(event) = lane',
            '    packed_g(event) = g_values(lane)',
            '    packed_flavour(event) = int(real_flav_idx - 1, c_int32_t)',
            '    do imu = 0, 3',
            '      do ipart = 1, nexternal',
            '        packed_momenta(event_count*(imu*nexternal+ipart-1)+event) = &',
            '          p(imu,ipart,lane)',
            '      end do',
            '    end do',
            '  end do',
            '  local(:,:) = 0d0',
            '  status = mg7_nlo_real_evaluate_ew(context, &',
            '    int(real_me_id,c_int), int(event_count,c_size_t), packed_momenta, &',
            '    packed_g, ew, packed_flavour, local)',
            '  if (status /= 0_c_int) then',
            '    call abort_bridge_evaluation(real_me_id)',
            '    return',
            '  end if',
            '  real_evaluated(real_me_id) = .true.',
            '  do event = 1, event_count',
            '    lane = packed_lanes(event)',
            '    ans_max = maxval(abs(local(event,1:nlocal)))',
            '    do index = 1, nlocal',
            '      call amp_split_pos_to_orders(local_to_global(index), component_orders)',
            '      if (trace_ownership .and. event == 1) call trace_mapping(index,local_to_global(index),all(component_orders <= nlo_orders))',
            '      if (any(component_orders > nlo_orders)) cycle',
            '      wgt(lane) = wgt(lane) + local(event,index)',
            '      if (abs(local(event,index)) > ans_max*1d-12) &',
            '        ret_amp_split(local_to_global(index),lane) = &',
            '          local(event,index)',
            '    end do',
            '    if (abs(wgt(lane)) < ans_max*1d-12) wgt(lane) = 0d0',
            '  end do',
            '  success = .true.',
            'end subroutine',
        ])
        source.extend([
            '',
            'logical function mc_provider_can_batch(row, flav)',
            '  use mg7_nlo_real_offload_state',
            '  implicit none',
            '  integer, intent(in) :: row, flav',
            '  integer :: real_me_id, row_flavour, nlocal, mapping(%d)' % max_local,
            '  mc_provider_can_batch = .false.',
            '  call resolve_real_row(row, real_me_id, row_flavour, nlocal, mapping)',
            '  if (real_me_id == 0 .or. flav /= row_flavour) return',
            '  call initialize_bridge()',
            '  if (.not. bridge_available) return',
            '  if (real_unavailable(real_me_id)) return',
            '  mc_provider_can_batch = mg7_nlo_real_available(context, int(real_me_id,c_int)) /= 0_c_int',
            'end function',
            '',
            'subroutine mc_evaluate_provider_batch(p, g, ew, row, flav, n, wgt, amps, ok)',
            '  implicit none',
            "  include 'nexternal.inc'",
            "  include 'orders.inc'",
            '  integer, intent(in) :: row, flav, n',
            '  real(8), intent(in) :: p(0:3,nexternal,n), g(n), ew',
            '  real(8), intent(out) :: wgt(n), amps(amp_split_size,n)',
            '  logical, intent(out) :: ok',
            '  logical :: active(n)',
            '  active = .true.',
            '  call mg7_nlo_real_try_batch(p, g, ew, amps, wgt, active, n, row, flav, ok)',
            'end subroutine',
            '',
            'logical function mg7_nlo_real_is_fortran()',
            '  use mg7_nlo_real_offload_state, only: default_backend',
            '  implicit none',
            "  mg7_nlo_real_is_fortran = trim(default_backend).eq.'fortran'",
            'end function',
        ])
        with open(pjoin(process_path, 'nlo_real_offload.f90'), 'w') as stream:
            stream.write('\n'.join(source) + '\n')

        chooser_path = pjoin(process_path, 'real_me_chooser.f')
        with open(chooser_path) as stream:
            chooser = stream.read()
        # Keep the plain exporter's constant query under an unused name;
        # production routing must inspect the compiled selector, even before
        # initialization or when an accelerated backend falls back to Fortran.
        chooser, backend_query_count = re.subn(
            r'\bMG7_NLO_REAL_IS_FORTRAN\b',
            'MG7_NLO_REAL_IS_FORTRAN_DEFAULT', chooser, flags=re.I)
        if backend_query_count != 2:
            raise RuntimeError('cannot install NLO real backend query in %s' %
                               chooser_path)
        for stub in ('mc_provider_can_batch', 'mc_evaluate_provider_batch'):
            chooser = re.sub(r'\b%s\b' % stub, stub + '_default', chooser, flags=re.I)
        scalar = re.compile(r'(SUBROUTINE\s+SMATRIX_REAL)\s*\(', re.I)
        vector = re.compile(
            r'(RECURSIVE\s+SUBROUTINE\s+SMATRIX_REAL_VEC)\s*\(', re.I)
        vector_batch = re.compile(
            r'(RECURSIVE\s+SUBROUTINE\s+SMATRIX_REAL_VEC_BATCH)\s*\(',
            re.I)
        chooser, scalar_count = scalar.subn(
            r'\1_FORTRAN(', chooser, count=1)
        chooser, vector_count = vector.subn(
            r'\1_FORTRAN(', chooser, count=1)
        chooser, vector_batch_count = vector_batch.subn(
            r'\1_FORTRAN(', chooser, count=1)
        chooser, fallback_call_count = re.subn(
            r'(CALL\s+)SMATRIX_REAL_VEC\s*\(',
            r'\1SMATRIX_REAL_VEC_FORTRAN(', chooser, flags=re.I)
        if (scalar_count != 1 or vector_count != 1 or
                vector_batch_count != 1 or fallback_call_count != 1):
            raise RuntimeError('cannot install NLO real Fortran routing in %s' %
                               chooser_path)
        wrappers = """

      SUBROUTINE SMATRIX_REAL(P,RET_AMP_SPLIT,WGT)
      USE MC_BATCH_TRACE, ONLY: TRACE_FORTRAN_INPUT,TRACE_FORTRAN_RESULT
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'fks_info.inc'
      INCLUDE 'real_matrix_weights.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL),RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE PRECISION WGT
      INTEGER NFKSPROCESS
      LOGICAL SUCCESS,MG7_NLO_REAL_IS_FORTRAN
      DOUBLE COMPLEX GAL(2)
      COMMON/WEAK/GAL
      DOUBLE PRECISION G,ALL_G
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      COMMON/STRONG/G,ALL_G
C     The hard-channel partition needs all diagram weights. UMAMI's
C     component result does not carry this native scalar side effect.
      IF (GET_REAL_DIAGRAM_WEIGHTS) THEN
         CALL TRACE_FORTRAN_INPUT(NFKSPROCESS,
     $    REAL_FLAVOR_INDEX_D(NFKSPROCESS),P,G,DBLE(GAL(1)))
         CALL SMATRIX_REAL_FORTRAN(P,RET_AMP_SPLIT,WGT)
         CALL TRACE_FORTRAN_RESULT(WGT,RET_AMP_SPLIT,2)
         RETURN
      ENDIF
      CALL MG7_NLO_REAL_TRY(P,RET_AMP_SPLIT,WGT,NFKSPROCESS,
     $ REAL_FLAVOR_INDEX_D(NFKSPROCESS),G,SUCCESS)
      IF (.NOT.SUCCESS) THEN
         CALL TRACE_FORTRAN_INPUT(NFKSPROCESS,
     $    REAL_FLAVOR_INDEX_D(NFKSPROCESS),P,G,DBLE(GAL(1)))
         CALL SMATRIX_REAL_FORTRAN(P,RET_AMP_SPLIT,WGT)
         CALL TRACE_FORTRAN_RESULT(WGT,RET_AMP_SPLIT,
     $                         MERGE(1,3,MG7_NLO_REAL_IS_FORTRAN()))
      ENDIF
      RETURN
      END

      RECURSIVE SUBROUTINE SMATRIX_REAL_VEC(P,RET_AMP_SPLIT,WGT,
     $ IVEC,NFKSPROCESS,REAL_FLAV_IDX)
      USE COUPLINGS, ONLY: G_VEC
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL),RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE PRECISION WGT
      INTEGER IVEC,NFKSPROCESS,REAL_FLAV_IDX
      LOGICAL SUCCESS
      DOUBLE PRECISION G_INPUT,G,ALL_G
      COMMON/STRONG/G,ALL_G
      G_INPUT=G
      IF (ALLOCATED(G_VEC)) G_INPUT=G_VEC(IVEC)
      CALL MG7_NLO_REAL_TRY(P,RET_AMP_SPLIT,WGT,NFKSPROCESS,
     $ REAL_FLAV_IDX,G_INPUT,SUCCESS)
      IF (.NOT.SUCCESS) CALL SMATRIX_REAL_VEC_FORTRAN(P,RET_AMP_SPLIT,
     $ WGT,IVEC,NFKSPROCESS,REAL_FLAV_IDX)
      RETURN
      END

      RECURSIVE SUBROUTINE SMATRIX_REAL_VEC_BATCH(P,G_STRONG,
     $ RET_AMP_SPLIT,WGT,ACTIVE,COUP_INDEX,VECTOR_SIZE,
     $ NFKSPROCESS,REAL_FLAV_IDX)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INTEGER VECTOR_SIZE,NFKSPROCESS,REAL_FLAV_IDX
      INTEGER COUP_INDEX(VECTOR_SIZE)
      LOGICAL ACTIVE(VECTOR_SIZE),SUCCESS
      DOUBLE PRECISION P(0:3,NEXTERNAL,VECTOR_SIZE)
      DOUBLE PRECISION G_STRONG(VECTOR_SIZE)
      DOUBLE PRECISION RET_AMP_SPLIT(AMP_SPLIT_SIZE,VECTOR_SIZE)
      DOUBLE PRECISION WGT(VECTOR_SIZE)
      DOUBLE COMPLEX GAL(2)
      COMMON/WEAK/GAL
      CALL MG7_NLO_REAL_TRY_BATCH(P,G_STRONG,DBLE(GAL(1)),
     $ RET_AMP_SPLIT,WGT,ACTIVE,VECTOR_SIZE,NFKSPROCESS,
     $ REAL_FLAV_IDX,SUCCESS)
      IF (.NOT.SUCCESS) CALL SMATRIX_REAL_VEC_BATCH_FORTRAN(P,
     $ G_STRONG,RET_AMP_SPLIT,WGT,ACTIVE,COUP_INDEX,VECTOR_SIZE,
     $ NFKSPROCESS,REAL_FLAV_IDX)
      RETURN
      END
"""
        with open(chooser_path, 'w') as stream:
            stream.write(chooser.rstrip() + wrappers)

        self._write_nlo_build_fragment(manifest, process_path)

    def _write_nlo_build_fragment(self, manifest, process_path):
        """Bind the production build to its current selected library identity."""
        directories = ' '.join(
            real['library_process_id']
            for real in manifest['real_matrix_elements'])
        fragment = [
            '# Generated NLO-real production routing.',
            '# NLO_REAL_RUNTIME_BACKEND is the run-card nlo_real_backend',
            '# (set in Source/make_opts by the run interface); it is compiled',
            '# into the wrapper as its default. NLO_REAL_BACKEND is the',
            '# MadMatrix backend built (scalar when the runtime is fortran).',
            'NLO_REAL_RUNTIME_BACKEND ?= fortran',
            'NLO_REAL_BACKEND ?= $(if $(filter fortran,'
            '$(NLO_REAL_RUNTIME_BACKEND)),scalar,$(NLO_REAL_RUNTIME_BACKEND))',
            'NLO_REAL_FPTYPE ?= d',
            'HELINL ?= 0',
            'HRDCOD ?= 0',
            'NLO_REAL_LIBTAG ?= $(LIBTAG)',
            'NLO_REAL_DIRS := %s' % directories,
            'NLO_REAL_STAMP := .nlo_real_$(NLO_REAL_BACKEND)_$(NLO_REAL_FPTYPE)$(NLO_REAL_LIBTAG)_cpp$(CPPBLAS).stamp',
            'NLO_REAL_INPUTS := nlo_real_offload.mk nlo_real.mk nlo_real_bridge.cc '
            'nlo_real_bridge.h nlo_real_bridge_config.h umami.h '
            '$(foreach directory,$(NLO_REAL_DIRS),$(wildcard ../$(directory)/*)) '
            '$(wildcard ../../src/*.h ../../src/*.cc ../../src/Arithmetics/*.h '
            '../../src/rambo/*.h ../../src/rambo/*.cc '
            '../../backend/common/* ../../backend/cpu/* '
            '../../backend/simd/* ../../backend/gpu/*)',
            'FILES += nlo_real_offload.o',
            'FKSSA += nlo_real_offload.o mc_batch_trace.o',
            'ifeq ($(shell uname -s),Darwin)',
            'NLO_REAL_RUNTIME_LIBS := -lc++ -Wl,-rpath,@loader_path',
            'else',
            "NLO_REAL_RUNTIME_LIBS := -lstdc++ -Wl,-rpath,'$$ORIGIN'",
            'endif',
            'NLO_REAL_LINKLIBS := -L$(HERE) -lnlo_real_bridge $(NLO_REAL_RUNTIME_LIBS) -ldl',
            'LINKLIBS += $(NLO_REAL_LINKLIBS)',
            'LINKLIBSSUD += $(NLO_REAL_LINKLIBS)',
            '',
            *self._bridge_build_identity_rules(
                '$(NLO_REAL_FPTYPE)', '$(NLO_REAL_LIBTAG)', '$(NLO_REAL_BACKEND)'),
            '',
            '$(NLO_REAL_STAMP): $(NLO_REAL_INPUTS) nlo_real_build_identity.h',
            '\t$(MAKE) -f nlo_real.mk BACKEND=$(NLO_REAL_BACKEND) '
            'FPTYPE=$(NLO_REAL_FPTYPE) LIBTAG=$(NLO_REAL_LIBTAG) all',
            '\ttouch $@',
            '',
            '.PHONY: nlo_real_commonlib',
            'nlo_real_commonlib:',
            '\t$(MAKE) -f nlo_real.mk BACKEND=$(NLO_REAL_BACKEND) '
            'FPTYPE=$(NLO_REAL_FPTYPE) LIBTAG=$(NLO_REAL_LIBTAG) commonlib',
            '',
            '.PHONY: nlo_real_backend_force',
            'nlo_real_backend.inc: nlo_real_backend_force',
            '\t@echo \'character(len=*), parameter :: default_backend = '
            '"$(NLO_REAL_RUNTIME_BACKEND)"\' > $@.new',
            '\t@cmp -s $@.new $@ || mv $@.new $@; rm -f $@.new',
            '',
            'nlo_real_offload.o: nlo_real_offload.f90 $(NLO_REAL_STAMP) '
            'nlo_real_backend.inc mc_batch_trace.o',
            '\t$(FC) $(FFLAGS) -ffree-line-length-none -c $< -o $@',
            '',
            '.PHONY: nlo_real_clean',
            'clean: nlo_real_clean',
            'nlo_real_clean:',
            '\t$(MAKE) -f nlo_real.mk clean',
            '\trm -f .nlo_real_*.stamp nlo_real_backend.inc',
        ]
        with open(pjoin(process_path, 'nlo_real_offload.mk'), 'w') as stream:
            stream.write('\n'.join(fragment) + '\n')

    def finalize_nlo_model(self, model):
        """Run the one shared tree-level model/common conversion pass."""
        # MadLoop's preceding virtual generation leaves loop-specific ALOHA
        # variables in the process-global symbolic kernel. A tree-only second
        # exporter must start from a clean kernel, just as a fresh LO output
        # does, or names such as P1 can retain an incompatible loop type.
        import aloha.aloha_lib as aloha_lib
        aloha_lib.KERNEL.clean()
        # The primary MadLoop exporter may cache its ordered loop/UV coupling
        # set on the shared UFO model. ProcessExporterMadMatrix honors that
        # cache for an ordinary dual export, but this lifecycle is deliberately
        # tree-only: use only dependencies collected from the real MEs above.
        # Snapshot only after the real HELAS writers have registered generated
        # FLV_Coupling objects on the grouped model. Exact coupling filtering
        # below excludes all loop/UV cache content from this copy.
        tree_model = copy.deepcopy(model)
        if hasattr(tree_model, 'madmatrix_wanted_ordered_couplings'):
            del tree_model.madmatrix_wanted_ordered_couplings
        tree_couplings = []
        for coupling in self._wanted_couplings:
            if isinstance(coupling, str):
                tree_couplings.append(coupling)
            else:
                # Generated FLV_Coupling objects are not registered in the
                # UFO model's ordinary coupling dictionary. Give the isolated
                # conversion pass its own copy of the exact generated object.
                tree_couplings.append(copy.deepcopy(coupling))
        self.convert_model(tree_model, self._wanted_lorentz,
                           tree_couplings)


MADMATRIX_EXPORTER_REGISTRY = {
    'lo': ProcessExporterMadMatrix,
    'nlo_real': ProcessExporterMadMatrixNLOReal,
}


class MadMatrixExporterFactory(object):
    """Explicit lifecycle selection; exporter construction remains ordinary."""

    @classmethod
    def get_exporter_class(cls, lifecycle):
        try:
            return MADMATRIX_EXPORTER_REGISTRY[lifecycle]
        except KeyError:
            raise ValueError('unknown MadMatrix exporter lifecycle %r' %
                             lifecycle)


# Standalone mode: in addition to the normal madmatrix exports, this writes
# an additional wrapper makefile (madmatrix_standalone.mk) on top of madmatrix.mk,
# so that when running `make` in a P* folder, it builds check_sa.exe as well as the process library (predicatable behaviour)
class ProcessExporterMadMatrixStandalone(ProcessExporterMadMatrix):

    format_name = 'standalone'

    # No subprocesses.json (see finalize): the crossed subprocesses are reached
    # through the extended flavor id alone, and the crossings recorded inside
    # a decay chain fold as well (the standalone crossing tables span the
    # decay leaves).
    writes_subprocess_info = False
    folds_decay_chain_crossings = True

    # Each P* directory links madmatrix_standalone.mk (which itself includes
    # madmatrix.mk) as its 'makefile'; both have to be rendered in SubProcesses/
    p_makefiles = ProcessExporterMadMatrix.p_makefiles + ['madmatrix_standalone.mk']

    # Standalone-only template files needed to build check_sa.exe
    _standalone_extra_files = ['check_sa.cc', 'nvtx.h', 'timer.h', 'timermap.h']

    from_template = dict(ProcessExporterMadMatrix.from_template)
    from_template['SubProcesses'] = (ProcessExporterMadMatrix.from_template['SubProcesses']
                                     + relative_path_list(ProcessExporterMadMatrix.madmatrix_templates,
                                                          _standalone_extra_files))

    # We don't need the run_card.toml
    from_template['Cards'] = []
    # nor, without the tool cards, the Delphes script
    from_template['bin/internal'] = []

    # Symlink the madmatrix.mk file to each P* folder (madmatrix_standalone.mk,
    # linked there as 'makefile', includes it by name)
    to_link_in_P = ProcessExporterMadMatrix.to_link_in_P + _standalone_extra_files + ['madmatrix.mk']

    # P*/makefile points at the standalone wrapper, so that a plain 'make' in a
    # P* directory (or from the SubProcesses dispatcher) also builds check_sa.exe
    oneprocessclass = model_handling.OneProcessExporterMadMatrixStandalone

    def copy_template(self, model):
        super().copy_template(model)

        # Write another custom bin/generate_events to orchestrate the standalone mode
        gen_events = pjoin(self.dir_path, 'bin', 'generate_events')
        if os.path.exists(gen_events):
            os.remove(gen_events)
        files.cp(pjoin(self.madmatrix_templates, 'generate_events_standalone'),
                 gen_events)
        os.chmod(gen_events, 0o755)

        # no run_card here (so no create_run_card), but `launch` still needs
        # the model, e.g. for the widths set to 'auto' in the param_card
        self.write_model_reference(model)

    # What a caller evaluating the libraries one phase-space point at a time
    # (madmatrix.density, used by MadSpin) needs to find the library and the
    # flavor index of a given process
    processes_file = 'madmatrix_processes.json'

    def generate_subprocess_directory(self, matrix_element, cpp_helas_call_writer, proc_number=None):
        out = super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)
        if not hasattr(self, 'processes_of_subprocess'):
            self.processes_of_subprocess = []
        crossed, crossed_flavors = self.crossed_flavor_ids(
            matrix_element, os.path.basename(self.process_info[-1]['path']))
        self.processes_of_subprocess.append((
            [[leg.get('id') for leg in process.get_legs_with_decays()]
             for process in matrix_element.get('processes')],
            # the legs marked off-shell ('*'), evaluated at the invariant mass of
            # their momentum (offshell_mass, backend/*/SigmaKin.cc)
            sorted(wf.get('number_external')
                   for wf in matrix_element.get_external_wavefunctions()
                   if wf.get('offshell')),
            # the crossed processes folded onto this one (--use_crossing):
            # listed nowhere else, and evaluated through the extended flavor
            # index only
            crossed, crossed_flavors,
            self.scale_dependence(matrix_element),
            # the massive fermions whose spin is quantised along a reference
            # (--axial_gauge): their states are not helicity states
            helas_call_writers.get_spin_reference_legs(matrix_element,
                helas_call_writers.get_output_axial_gauge_refs(matrix_element,
                                             self.opt.get('output_options')))))
        return out

    def crossed_flavor_ids(self, matrix_element, proc_dir_name):
        """The crossed processes folded onto matrix_element, as their leg ids
        (decays expanded to their leaves, merged labels as they are), and
        [extended flavor id, pdgs] for every pdg combination of theirs, the pdgs
        in the leg order of the crossed process: the rows of the crossing table
        (the one ProcessTables.h was written from, in the coupling-class
        convention of the C++ flavor index, export_cpp get_crossing_table) that
        serve them. No flavor id when the directory has no crossing machinery."""
        folded = self.folded_crossed_processes(matrix_element)
        legs = [[leg.get('id') for leg in process.get_legs_with_decays()]
                for process in folded]
        if not folded:
            return legs, []
        tables = pjoin(self.dir_path, 'SubProcesses', proc_dir_name, 'ProcessTables.h')
        try:
            with open(tables) as fsock:
                if 'constexpr bool use_crossing = true;' not in fsock.read():
                    return legs, []
        except IOError:
            return legs, []
        crossing_all = str((self.opt.get('output_options') or {}).get(
            'crossing_table', '')).lower() == 'all'
        table = export_v4.ProcessExporterFortran.get_crossing_table(
            self, matrix_element, 'classes', all_applicable=crossing_all)
        nflav = len(matrix_element.get_external_flavors_with_iden())
        # a decay-chain base records its crossings at the production level:
        # the table's records carry their leaf ids
        legs = [list(record.labels) for record in table.records]
        return legs, [[a.index(nflav), list(a.pdgs)] for a in table.assignments()]

    @staticmethod
    def scale_dependence(matrix_element):
        """What makes the matrix element depend on the renormalisation scale
        beyond alpha_s: the couplings and the masses/widths it uses that depend
        on MU_R, and the running elements of the model. The libraries compute
        their couplings from G alone and keep MU_R at its param-card value,
        while the Fortran output sets MU_R from the event scale
        (UPDATE_AS_PARAM2): a caller passing per-event scales needs the
        Fortran one when this is not empty."""
        model = matrix_element.get('processes')[0].get('model')
        scale_keys = [key for key in model.get('couplings')
                      if 'MU_R' in key]
        scale_coups = set(coup.name for key in scale_keys
                          for coup in model.get('couplings')[key])
        used = set()
        for entry in matrix_element.get_used_couplings():
            for coup in (entry if isinstance(entry, list) else [entry]):
                if isinstance(coup, str):
                    used.add(coup.lstrip('-'))
                else:
                    used.add(coup.get('name'))
                    used.update(str(c).lstrip('-')
                                for c in coup.get('flavors').values())
        reasons = sorted(used & scale_coups)
        scale_params = set(param.name for key, params in model.get('parameters').items()
                           if 'MU_R' in key for param in params)
        for wf in matrix_element.get_all_wavefunctions():
            for name in (wf.get('mass'), wf.get('width')):
                if name in scale_params and name not in reasons:
                    reasons.append(name)
        if model.get('running_elements'):
            reasons.append('running elements of the model')
        return reasons

    def finalize(self, *args, **kwargs):
        # no subprocesses.json (nor anything else the mg7 launcher needs), only
        # the --hel_recycling warm-up, which can only run once every P*
        # directory is complete and buildable (measured here: a standalone
        # output is never launched), and the process table of the standalone
        # libraries
        self._hel_recycling_at_output()
        self.write_processes_file()

    # never launched: nothing but the output would measure the filter. A user
    # who edits the param_card relies on the build's run-time check
    # (hr_check_stale), which falls back to the plain code
    _hr_measure_at_output = True

    def _write_hr_probe_sources(self):
        # check_sa.exe is part of a standalone output already
        pass

    def write_processes_file(self):
        """SubProcesses/madmatrix_processes.json: per subprocess, its library
        (with the make variables LIBTAG, BACKEND and FPTYPE left as '{libtag}',
        '{backend}' and '{fptype}', as for the common library, common_library),
        its legs, the processes it evaluates, its off-shell legs, the legs
        whose two states are not helicity states (spin_reference, output
        --axial_gauge=True; numbered as the legs of the subprocess), the
        crossed processes folded onto it, per flavor index, every pdg
        combination (in the leg order of the process) that this index stands
        for and, for the crossed processes, [extended flavor id, pdgs] for
        each of their pdg combinations (crossed_flavors).

        A crossed process is evaluated with an extended flavor index (crossing
        row K of ProcessTables.h, K*nmaxflavor + flavor). Its selected
        helicity is the crossed process's own code, but its colour-flow and
        diagram indices number the BASE subprocess's flows and diagrams: decode
        them through row K (see the note above umami_matrix_element in
        umami.cc), as `output mg7` does.

        It also lists what of a subprocess depends on the renormalisation scale
        beyond alpha_s (scale_dependence)."""
        import json
        subprocesses = []
        for info, (processes, offshell, crossed, crossed_flavors, scale,
                   spin_reference) in zip(
                self.process_info, getattr(self, 'processes_of_subprocess', [])):
            path = os.path.basename(info['path'])
            flavors = []
            for flavor in info['flavors']:
                while len(flavors) <= flavor['index']:
                    flavors.append([])
                flavors[flavor['index']].extend(list(pdgs) for pdgs in flavor['options'])
            subprocesses.append({
                'path': path,
                'library': 'libmadmatrix{libtag}_%s_%s_{backend}_{fptype}.so'
                           % (self.lib_model_name, path),
                'ninitial': len(info['incoming']),
                'legs': list(info['incoming']) + list(info['outgoing']),
                'processes': processes,
                'offshell': offshell,
                'crossed': crossed,
                'spin_reference': spin_reference,
                # what depends on MU_R besides alpha_s (scale_dependence)
                'scale_dependence': scale,
                'flavors': flavors,
                'crossed_flavors': crossed_flavors})
        with open(pjoin(self.dir_path, 'SubProcesses', self.processes_file), 'w') as fsock:
            json.dump({'common_library': 'libmadmatrix_common{libtag}_%s_{backend}_{fptype}.so'
                                         % self.lib_model_name,
                       'subprocesses': subprocesses}, fsock, indent=1)
