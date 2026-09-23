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

# AV - create a plugin-specific logger
import logging
logger = logging.getLogger('madgraph.%s.output'%PLUGIN_NAME)
from madgraph import MG5DIR
#------------------------------------------------------------------------------------

from os.path import join as pjoin
import madgraph.iolibs.files as files
import madgraph.iolibs.export_v4 as export_v4
import madgraph.iolibs.export_cpp as export_cpp
import madgraph.various.misc as misc

from . import launch_plugin


NLO_REAL_MANIFEST_VERSION = 1
NLO_REAL_MANIFEST_TOP_LEVEL_KEYS = (
    'version', 'process_id', 'nexternal', 'split_order_names',
    'global_squared_orders', 'real_matrix_elements', 'fks_rows')
NLO_REAL_MANIFEST_REAL_ME_KEYS = (
    'id', 'library_process_id', 'local_flavours',
    'local_squared_orders', 'backend_capabilities', 'process_fingerprint')
NLO_REAL_MANIFEST_FKS_ROW_KEYS = (
    'fks_row', 'topology', 'real_me_id', 'fortran_flavour_index',
    'madmatrix_flavour_index', 'pdgs', 'local_to_global')

def relative_path_list(relative_path, files_list):
    return list(map(lambda f: pjoin(relative_path, f), files_list))

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
    support_ddm_color_basis = True
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

    # Information to find the template file that we want to include from madgraph
    # you can include additional file from the plugin directory as well
    # AV - use template files from PLUGINDIR instead of MG5DIR and add gpu/mgOnGpuVectors.h
    # [NB: mgOnGpuConfig.h, check_sa.cc and fcheck_sa.f are handled through dedicated methods]
    ###s = MG5DIR + '/madgraph/iolibs/template_files/'

    templates_path = pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files')
    mg7_templates = pjoin(templates_path, 'mg7')
    madmatrix_templates = pjoin(templates_path, 'madmatrix')
    home_path = pjoin(MG5DIR, "madmatrix")

    from_template = {'.': relative_path_list(home_path, ['COPYRIGHT', 'COPYING', 'COPYING.LESSER']),
                     'src': relative_path_list(madmatrix_templates, [
                         'mgOnGpuFptypes.h', 'mgOnGpuCxtypes.h', 'mgOnGpuVectors.h',
                         'constexpr_math.h', 'read_slha.h', 'read_slha.cc'
                     ]),
                     'SubProcesses': relative_path_list(madmatrix_templates, ['nvtx.h', 'GpuRuntime.h', 'GpuAbstraction.h', 'color_sum.h', 'color_sum.cc',
                                      'MemoryAccessHelpers.h', 'MemoryAccessVectors.h',
                                      'MemoryAccessMatrixElements.h', 'MemoryAccessMomenta.h',
                                      'MemoryAccessRandomNumbers.h', 'MemoryAccessWeights.h',
                                      'MemoryAccessAmplitudes.h', 'MemoryAccessWavefunctions.h',
                                      'MemoryAccessGs.h', 'MemoryAccessCouplingsFixed.h',
                                      'MemoryAccessNumerators.h', 'MemoryAccessDenominators.h',
                                      'MemoryAccessChannelIds.h', 'MemoryAccessIflavorVec.h',
                                      'CrossSectionKernels.cc', 'CrossSectionKernels.h',
                                      'MatrixElementKernels.cc', 'MatrixElementKernels.h',
                                      'EventStatistics.h',
                                      'umami.h', 'umami.cc', 'rambo.h']),
                     # run_card.toml is generated in finalize() (ProcessExporterMG7.create_run_card)
                     # from the template, not copied verbatim.
                     # Default cards for the optional post-processing tools
                     # (Pythia8/Delphes/MadSpin/reweight/analysis) so that
                     # bin/generate_events can offer to enable and edit them.
                     'Cards': relative_path_list(pjoin(MG5DIR, 'Template', 'Common', 'Cards'),
                                  ['madspin_card_default.dat', 'reweight_card_default.dat',
                                   'density_card_default.dat',
                                   'delphes_card_default.dat']) +
                              relative_path_list(pjoin(MG5DIR, 'Template', 'LO', 'Cards'),
                                  ['pythia8_card_default.dat',
                                   'madanalysis5_parton_card_default.dat',
                                   'madanalysis5_hadron_card_default.dat',
                                   'rivet_card_default.dat'])}

    to_link_in_P = ['nvtx.h', 'GpuRuntime.h', 'GpuAbstraction.h', 'color_sum.h',
                    'MemoryAccessHelpers.h', 'MemoryAccessVectors.h',
                    'MemoryAccessMatrixElements.h', 'MemoryAccessMomenta.h',
                    'MemoryAccessRandomNumbers.h', 'MemoryAccessWeights.h',
                    'MemoryAccessAmplitudes.h', 'MemoryAccessWavefunctions.h',
                    'MemoryAccessGs.h', 'MemoryAccessCouplingsFixed.h',
                    'MemoryAccessNumerators.h', 'MemoryAccessDenominators.h',
                    'MemoryAccessChannelIds.h', 'MemoryAccessIflavorVec.h',
                    'CrossSectionKernels.cc', 'CrossSectionKernels.h',
                    'MatrixElementKernels.cc', 'MatrixElementKernels.h',
                    'EventStatistics.h',
                    'MemoryBuffers.h', # this is generated from a template in Subprocesses but we still link it in P1
                    'MemoryAccessCouplings.h', # this is generated from a template in Subprocesses but we still link it in P1
                    'umami.h', 'umami.cc', 'rambo.h']

    template_src_make = pjoin(madmatrix_templates, 'madmatrix_src.mk')
    # SubProcesses/makefile is only a dispatcher over the P* directories: it is
    # what makes 'make -j N' in SubProcesses build all the subprocesses with a
    # single, shared pool of N jobs.
    template_Sub_make = pjoin(madmatrix_templates, 'madmatrix_subprocesses.mk')

    # The actual build rules, rendered into SubProcesses/. Each P* directory
    # links one of them (p_makefile, see OneProcessExporterMadMatrix) as its own
    # 'makefile'.
    p_makefiles = ['madmatrix.mk']

    dirs_to_create = ['bin', 'src', 'lib', 'Cards', 'SubProcesses']

    # AV - use a custom UFOModelConverter (model/aloha exporter)
    create_model_class = model_handling.MadMatrixUFOModelConverter

    # AV - use a custom GPUFOHelasCallWriter
    # (NB: use "helas_exporter" - see class MadGraphCmd in madgraph_interface.py - not "aloha_exporter" that is never used!)
    ###helas_exporter = None
    helas_exporter = model_handling.MadMatrixUFOHelasCallWriter # this is one of the main fixes for issue #341!

    # AV (default from OM's tutorial) - add a debug printout
    def __init__(self, *args, **kwargs):
        self.in_madevent_mode = False # see MR #747
        args[1]["me_lib_format"] = pjoin("lib", "libmadmatrix_{process_id}_{{device}}.so")
        super().__init__(*args, **kwargs)
        # Honor the output command's --mask=True|False (flavor-mask
        # optimization for grouped/merged flavors). Default: enabled.
        self.use_flavor_mask = self._parse_flavor_mask_option()

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
        whether one could be linked at all."""

        replace_dict = super().get_makefile_replace_dict(model)
        flags = self.oneprocessclass.blas_available_flags()
        replace_dict['cpp_blas_default'] = 'hasBlas' if flags else 'hasNoBlas'
        replace_dict['cpp_blas_libflags'] = flags
        return replace_dict

    # AV - overload the default version: create CMake directory, do not create lib directory
    def copy_template(self, model):
        super().copy_template(model)
        # Copy Arithmetics headers for the double-word expansion (FPTYPE=e)
        arithmetics_src = pjoin(self.madmatrix_templates, 'Arithmetics')
        if os.path.isdir(arithmetics_src):
            arithmetics_dst = pjoin(self.dir_path, 'src', 'Arithmetics')
            try:
                os.makedirs(arithmetics_dst, exist_ok=True)
            except os.error:
                pass
            for f in ['Double.h', 'basicOPs.h', 'errorFreeOPs.h']:
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
        for name in self.p_makefiles:
            rendered = self.read_template_file(pjoin(self.madmatrix_templates, name)) % replace_dict
            open(pjoin(self.dir_path, 'SubProcesses', name), 'w').write(rendered)

    def check_split_orders(self, matrix_element):
        """Report what a squared-order constraint will produce here.

        Supported: the jamps carry an amplitude-order index and the color sum
        pairs them (color_sum_splitorders.cc, the Fortran GET_MATRIX contract),
        so a '^2' constraint that keeps only some squared orders gets the
        contribution it asked for rather than the total. That is what makes the
        interference case work -- `u u~ > t t~ QED^2==2` keeps all three
        diagrams and wants the QCD-EW cross term alone, which no amount of
        dropping diagrams at generation can produce.

        Not supported: a GPU build of such a process. The device jamp buffers
        are sized for one jamp vector per helicity (ncolor, not njampso), and
        the backend is a make-time choice rather than an output-time one, so
        the refusal cannot live here: color_sum_splitorders.cc #errors under
        MGONGPUCPP_GPUIMPL instead. Say so now rather than let a GPU build be
        the first the user hears of it.
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
            "pairs them; CPU backends only (a GPU build of this process will "
            "not compile, by design).",
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
        out = super().generate_subprocess_directory(matrix_element, cpp_helas_call_writer, proc_number)
        return out

    # AV (default from OM's tutorial) - add a debug printout
    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[], **opts):
        # **opts: an option meant for one exporter (npwave, for the dual HELAS
        # libraries of P-wave bound states) is passed by keyword to all of them
        if hasattr(model , 'cudacpp_wanted_ordered_couplings'):
            wanted_couplings = model.cudacpp_wanted_ordered_couplings
            del model.cudacpp_wanted_ordered_couplings
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
        for destination in ('src', 'SubProcesses'):
            for source in self.from_template.get(destination, []):
                files.cp(source, pjoin(self.dir_path, destination,
                                       os.path.basename(source)))
        src_make = self.read_template_file(self.template_src_make) % \
            self.get_makefile_replace_dict(model)
        with open(pjoin(self.dir_path, 'src', 'makefile'), 'w') as stream:
            stream.write(src_make)
        self.write_p_makefiles(model)
        arithmetics_src = pjoin(self.madmatrix_templates, 'Arithmetics')
        arithmetics_dst = pjoin(self.dir_path, 'src', 'Arithmetics')
        if os.path.isdir(arithmetics_src):
            os.makedirs(arithmetics_dst, exist_ok=True)
            for name in ('Double.h', 'basicOPs.h', 'errorFreeOPs.h'):
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
            physical_flavor_indices=True)

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
            'struct MG7NLORealConfig {',
            '  const char* library_process_id;',
            '  const char* fingerprint;',
            '  int squared_order_count;',
            '  int flavour_count;',
            '};',
            'static const int MG7_NLO_PARTICLE_COUNT = %d;' %
            manifest['nexternal'],
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
        makefile = [
            '# Generated MadMatrix NLO-real build dispatcher.',
            'CXX ?= g++',
            'BACKEND ?= scalar',
            'FPTYPE ?= d',
            'HELINL ?= 0',
            'HRDCOD ?= 0',
            'NLO_REAL_DIRS := %s' % directories,
            'NLO_REAL_BRIDGE := libnlo_real_bridge.so',
            '',
            '.PHONY: all libraries bridge clean',
            'all: libraries bridge',
            '',
            'libraries:',
            '\t@set -e; for directory in $(NLO_REAL_DIRS); do \\',
            '\t  $(MAKE) -C ../$$directory BACKEND=$(BACKEND) '
            'FPTYPE=$(FPTYPE) HELINL=$(HELINL) HRDCOD=$(HRDCOD); \\',
            '\tdone',
            '',
            'bridge: $(NLO_REAL_BRIDGE)',
            '',
            '$(NLO_REAL_BRIDGE): nlo_real_bridge.cc nlo_real_bridge.h '
            'nlo_real_bridge_config.h umami.h',
            '\t$(CXX) -O2 -std=c++17 -Wall -Wextra -fPIC -shared -I. '
            'nlo_real_bridge.cc -ldl -o $@',
            '',
            'clean:',
            '\trm -f $(NLO_REAL_BRIDGE)',
            '\t@set -e; for directory in $(NLO_REAL_DIRS); do \\',
            '\t  $(MAKE) -C ../$$directory clean; \\',
            '\tdone',
        ]
        with open(pjoin(process_path, 'nlo_real.mk'), 'w') as stream:
            stream.write('\n'.join(makefile) + '\n')

    def _write_fortran_routing(self, manifest, process_path):
        """Route scalar Fortran real calls through the bridge with fallback."""
        real_by_id = {
            real['id']: real for real in manifest['real_matrix_elements']}
        max_local = max(
            len(real['local_squared_orders'])
            for real in manifest['real_matrix_elements'])
        real_count = len(manifest['real_matrix_elements'])

        cases = []
        for row in manifest['fks_rows']:
            nlocal = len(real_by_id[row['real_me_id']][
                'local_squared_orders'])
            mapping = ', '.join(str(index)
                                for index in row['local_to_global'])
            cases.extend([
                '  case (%d)' % row['fks_row'],
                '    real_me_id = %d' % row['real_me_id'],
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
            '  logical, save :: trace_batches = .false.',
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
            '    integer(c_int) function mg7_nlo_real_evaluate(handle, &',
            '        real_me_id, event_count, momenta, g_strong, flavour, &',
            '        squared_orders) bind(C)',
            '      import :: c_int, c_int32_t, c_size_t, c_ptr, c_double',
            '      type(c_ptr), value :: handle',
            '      integer(c_int), value :: real_me_id',
            '      integer(c_size_t), value :: event_count',
            '      real(c_double), intent(in) :: momenta(*)',
            '      real(c_double), intent(in) :: g_strong(*)',
            '      integer(c_int32_t), intent(in) :: flavour(*)',
            '      real(c_double), intent(out) :: squared_orders(*)',
            '    end function',
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
            '  subroutine initialize_bridge()',
            '    character(kind=c_char, len=1024) :: param_card, library_dir',
            '    character(kind=c_char, len=64) :: backend',
            '    character(len=1024) :: env_value',
            '    integer(c_int) :: status',
            '    integer :: length, env_status',
            '    if (initialization_attempted) return',
            '    initialization_attempted = .true.',
            '    param_card = "auto"',
            '    library_dir = "auto"',
            '    backend = "scalar"',
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
            '    call get_environment_variable("MG7_NLO_REAL_BACKEND", &',
            '      env_value, length=length, status=env_status)',
            '    if (env_status == 0 .and. length > 0) &',
            '      backend = env_value(1:length)',
            '    env_value = ""',
            '    call get_environment_variable("MG7_NLO_REAL_TRACE", &',
            '      env_value, length=length, status=env_status)',
            '    trace_batches = .false.',
            '    if (env_status == 0 .and. length > 0) &',
            '      trace_batches = trim(env_value(1:length)) /= "0"',
            '    backend_name = trim(backend)',
            '    if (trim(backend) == "fortran" .or. &',
            '        trim(backend) == "off" .or. trim(backend) == "none") then',
            '      write(*,\'(A)\') &',
            '        "MG7 NLO real offload disabled; using Fortran fallback"',
            '      return',
            '    end if',
            '    status = mg7_nlo_real_initialize(context, &',
            '      trim(param_card)//c_null_char, &',
            '      trim(library_dir)//c_null_char, trim(backend)//c_null_char)',
            '    if (status /= 0_c_int) then',
            '      call print_bridge_error("MG7 NLO real bridge unavailable")',
            '      return',
            '    end if',
            '    bridge_available = .true.',
            '    write(*,\'(A,A,A,I0)\') &',
            '      "MG7 NLO real offload initialized: backend=", &',
            '      trim(backend), ", real_libraries=", %d' % real_count,
            '  end subroutine',
            '',
            'end module',
            '',
            'subroutine mg7_nlo_real_try(p, ret_amp_split, wgt, &',
            '    nfksprocess, real_flav_idx, g_input, success)',
            '  use, intrinsic :: iso_c_binding',
            '  use mg7_nlo_real_offload_state',
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
            '  integer :: real_me_id, nlocal, ipart, imu, index',
            '  integer(c_int) :: status',
            '  success = .false.',
            '  ret_amp_split(:) = 0d0',
            '  wgt = 0d0',
            '  real_me_id = 0',
            '  nlocal = 0',
            '  local_to_global(:) = 0',
            '  select case (nfksprocess)',
        ]
        source.extend(cases)
        source.extend([
            '  case default',
            '    return',
            '  end select',
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
            '  status = mg7_nlo_real_evaluate(context, &',
            '    int(real_me_id,c_int), 1_c_size_t, momenta, g_strong, &',
            '    flavour, local)',
            '  if (status /= 0_c_int) then',
            '    real_unavailable(real_me_id) = .true.',
            '    call print_bridge_error(&',
            '      "MG7 NLO real ME disabled; using Fortran fallback")',
            '    return',
            '  end if',
            '  ans_max = maxval(abs(local(1:nlocal)))',
            '  do index = 1, nlocal',
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
            'subroutine mg7_nlo_real_try_batch(p, g_values, &',
            '    ret_amp_split, wgt, active, vector_size, nfksprocess, &',
            '    real_flav_idx, success)',
            '  use, intrinsic :: iso_c_binding',
            '  use mg7_nlo_real_offload_state',
            '  implicit none',
            "  include 'nexternal.inc'",
            "  include 'orders.inc'",
            '  integer, intent(in) :: vector_size, nfksprocess',
            '  integer, intent(in) :: real_flav_idx',
            '  real(c_double), intent(in) :: p(0:3,nexternal,vector_size)',
            '  real(c_double), intent(in) :: g_values(vector_size)',
            '  real(c_double), intent(out) :: &',
            '    ret_amp_split(amp_split_size,vector_size)',
            '  real(c_double), intent(out) :: wgt(vector_size)',
            '  logical, intent(in) :: active(vector_size)',
            '  logical, intent(out) :: success',
            '  real(c_double), allocatable :: momenta(:), g_strong(:)',
            '  real(c_double), allocatable :: local(:,:)',
            '  integer(c_int32_t), allocatable :: flavour(:)',
            '  integer, allocatable :: lanes(:)',
            '  integer :: local_to_global(%d)' % max_local,
            '  integer :: real_me_id, nlocal, event_count',
            '  integer :: lane, event, ipart, imu, index',
            '  real(c_double) :: ans_max',
            '  integer(c_int) :: status',
            '  success = .false.',
            '  ret_amp_split(:,:) = 0d0',
            '  wgt(:) = 0d0',
            '  real_me_id = 0',
            '  nlocal = 0',
            '  local_to_global(:) = 0',
            '  select case (nfksprocess)',
        ])
        source.extend(cases)
        source.extend([
            '  case default',
            '    return',
            '  end select',
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
            '  if (allocated(momenta)) deallocate(momenta)',
            '  if (allocated(g_strong)) deallocate(g_strong)',
            '  if (allocated(flavour)) deallocate(flavour)',
            '  if (allocated(local)) deallocate(local)',
            '  if (allocated(lanes)) deallocate(lanes)',
            '  allocate(momenta(event_count*4*nexternal))',
            '  allocate(g_strong(event_count), flavour(event_count))',
            '  allocate(local(event_count,nlocal), lanes(event_count))',
            '  event = 0',
            '  do lane = 1, vector_size',
            '    if (.not. active(lane)) cycle',
            '    event = event + 1',
            '    lanes(event) = lane',
            '    g_strong(event) = g_values(lane)',
            '    flavour(event) = int(real_flav_idx - 1, c_int32_t)',
            '    do imu = 0, 3',
            '      do ipart = 1, nexternal',
            '        momenta(event_count*(imu*nexternal+ipart-1)+event) = &',
            '          p(imu,ipart,lane)',
            '      end do',
            '    end do',
            '  end do',
            '  local(:,:) = 0d0',
            '  status = mg7_nlo_real_evaluate(context, &',
            '    int(real_me_id,c_int), int(event_count,c_size_t), momenta, &',
            '    g_strong, flavour, local)',
            '  if (status /= 0_c_int) then',
            '    real_unavailable(real_me_id) = .true.',
            '    call print_bridge_error(&',
            '      "MG7 NLO real ME disabled; using Fortran fallback")',
            '    deallocate(momenta,g_strong,flavour,local,lanes)',
            '    return',
            '  end if',
            '  do event = 1, event_count',
            '    lane = lanes(event)',
            '    ans_max = maxval(abs(local(event,1:nlocal)))',
            '    do index = 1, nlocal',
            '      wgt(lane) = wgt(lane) + local(event,index)',
            '      if (abs(local(event,index)) > ans_max*1d-12) &',
            '        ret_amp_split(local_to_global(index),lane) = &',
            '          local(event,index)',
            '    end do',
            '    if (abs(wgt(lane)) < ans_max*1d-12) wgt(lane) = 0d0',
            '  end do',
            '  deallocate(momenta,g_strong,flavour,local,lanes)',
            '  success = .true.',
            'end subroutine',
        ])
        with open(pjoin(process_path, 'nlo_real_offload.f90'), 'w') as stream:
            stream.write('\n'.join(source) + '\n')

        chooser_path = pjoin(process_path, 'real_me_chooser.f')
        with open(chooser_path) as stream:
            chooser = stream.read()
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
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INCLUDE 'orders.inc'
      INCLUDE 'fks_info.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL),RET_AMP_SPLIT(AMP_SPLIT_SIZE)
      DOUBLE PRECISION WGT
      INTEGER NFKSPROCESS
      LOGICAL SUCCESS
      DOUBLE PRECISION G,ALL_G
      COMMON/C_NFKSPROCESS/NFKSPROCESS
      COMMON/STRONG/G,ALL_G
      CALL MG7_NLO_REAL_TRY(P,RET_AMP_SPLIT,WGT,NFKSPROCESS,
     $ REAL_FLAVOR_INDEX_D(NFKSPROCESS),G,SUCCESS)
      IF (.NOT.SUCCESS) CALL SMATRIX_REAL_FORTRAN(P,RET_AMP_SPLIT,WGT)
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
      CALL MG7_NLO_REAL_TRY_BATCH(P,G_STRONG,RET_AMP_SPLIT,WGT,
     $ ACTIVE,VECTOR_SIZE,NFKSPROCESS,REAL_FLAV_IDX,SUCCESS)
      IF (.NOT.SUCCESS) CALL SMATRIX_REAL_VEC_BATCH_FORTRAN(P,
     $ G_STRONG,RET_AMP_SPLIT,WGT,ACTIVE,COUP_INDEX,VECTOR_SIZE,
     $ NFKSPROCESS,REAL_FLAV_IDX)
      RETURN
      END
"""
        with open(chooser_path, 'w') as stream:
            stream.write(chooser.rstrip() + wrappers)

        directories = ' '.join(
            real['library_process_id']
            for real in manifest['real_matrix_elements'])
        fragment = [
            '# Generated scalar NLO-real production routing.',
            'NLO_REAL_BACKEND ?= scalar',
            'NLO_REAL_FPTYPE ?= d',
            'NLO_REAL_DIRS := %s' % directories,
            'NLO_REAL_STAMP := .nlo_real_$(NLO_REAL_BACKEND)_$(NLO_REAL_FPTYPE).stamp',
            'NLO_REAL_INPUTS := nlo_real.mk nlo_real_bridge.cc '
            'nlo_real_bridge.h nlo_real_bridge_config.h umami.h '
            '$(foreach directory,$(NLO_REAL_DIRS),$(wildcard ../$(directory)/*))',
            'FILES += nlo_real_offload.o',
            'FKSSA += nlo_real_offload.o',
            'NLO_REAL_LINKLIBS := -L$(HERE) -lnlo_real_bridge '
            "-Wl,-rpath,'$$ORIGIN' -lstdc++ -ldl",
            'LINKLIBS += $(NLO_REAL_LINKLIBS)',
            'LINKLIBSSUD += $(NLO_REAL_LINKLIBS)',
            '',
            '$(NLO_REAL_STAMP): $(NLO_REAL_INPUTS)',
            '\t$(MAKE) -f nlo_real.mk BACKEND=$(NLO_REAL_BACKEND) '
            'FPTYPE=$(NLO_REAL_FPTYPE) all',
            '\ttouch $@',
            '',
            'nlo_real_offload.o: nlo_real_offload.f90 $(NLO_REAL_STAMP)',
            '\t$(FC) $(FFLAGS) -ffree-line-length-none -c $< -o $@',
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
        if hasattr(tree_model, 'cudacpp_wanted_ordered_couplings'):
            del tree_model.cudacpp_wanted_ordered_couplings
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

    # Each P* directory links madmatrix_standalone.mk (which itself includes
    # madmatrix.mk) as its 'makefile'; both have to be rendered in SubProcesses/
    p_makefiles = ProcessExporterMadMatrix.p_makefiles + ['madmatrix_standalone.mk']

    # Standalone-only template files needed to build check_sa.exe
    _standalone_extra_files = ['check_sa.cc',
                               'RamboSamplingKernels.cc', 'RamboSamplingKernels.h',
                               'CommonRandomNumberKernel.cc', 'CommonRandomNumbers.h',
                               'RandomNumberKernels.h',
                               'massless_rambo.h', 'timer.h', 'timermap.h']

    from_template = dict(ProcessExporterMadMatrix.from_template)
    from_template['SubProcesses'] = (ProcessExporterMadMatrix.from_template['SubProcesses']
                                     + relative_path_list(ProcessExporterMadMatrix.madmatrix_templates,
                                                          _standalone_extra_files))

    # We don't need the run_card.toml
    from_template['Cards'] = []

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

    def finalize(self, *args, **kwargs):
        # We disable this since we don't need subprocesses.json either
        pass
