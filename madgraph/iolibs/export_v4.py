################################################################################
#
# Copyright (c) 2009 The MadGraph5_aMC@NLO Development team and Contributors
#
# This file is a part of the MadGraph5_aMC@NLO project, an application which 
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph5_aMC@NLO license which should accompany this 
# distribution.
#
# For more information, visit madgraph.phys.ucl.ac.be and amcatnlo.web.cern.ch
#
################################################################################
from __future__ import absolute_import, division
from madgraph.iolibs.helas_call_writers import HelasCallWriter
from madgraph.core import base_objects
"""Methods and classes to export matrix elements to v4 format."""

import bisect
import copy
import math, cmath
from io import StringIO
import itertools
import fractions
import glob
import logging
import math
import os
import io
import re
import shutil
import subprocess
import sys
import time
import traceback
import  collections

import aloha

import madgraph
import models
import madgraph.core.base_objects as base_objects
import madgraph.core.color_algebra as color
import madgraph.core.color_amp as color_amp
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.drawing_eps as draw
import madgraph.iolibs.files as files
import madgraph.iolibs.group_subprocs as group_subprocs
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gen_infohtml as gen_infohtml
import madgraph.iolibs.jamp_optimiser as jamp_optimiser
import madgraph.iolibs.template_files as template_files
import madgraph.iolibs.ufo_expression_parsers as parsers
import madgraph.iolibs.helas_call_writers as helas_call_writers
import madgraph.interface.common_run_interface as common_run_interface
import madgraph.various.diagram_symmetry as diagram_symmetry
import madgraph.various.misc as misc
import madgraph.various.banner as banner_mod
import madgraph.various.process_checks as process_checks
import madgraph.loop.loop_diagram_generation as loop_diagram_generation
import madgraph
import aloha.create_aloha as create_aloha
import models.import_ufo as import_ufo
import models.write_param_card as param_writer
import models.check_param_card as check_param_card
from models import UFOError


from madgraph import MadGraph5Error, MG5DIR, ReadWrite
from madgraph.iolibs.files import cp, ln, mv

from madgraph import InvalidCmd


pjoin = os.path.join

_file_path = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0] + '/'
logger = logging.getLogger('madgraph.export_v4')
if madgraph.ordering:
    set = misc.OrderedSet


default_compiler= {'fortran': 'gfortran',
                       'f2py': 'f2py',
                       'cpp':'g++'}


# Number of fortran statements per amplitude-chunk file. The HELAS call
# sequence of a high-multiplicity matrix element is one enormous basic block
# and gfortran's cost on it grows faster than linearly, so it is emitted as its
# own set of files (matrix<i>_origamp<k>.f / matrix<i>_optimamp<k>.f), one
# subroutine each, called in sequence. A matrix element whose sequence is
# shorter than this is written inline exactly as before, which keeps every
# small process byte-identical to the unchunked output.
# Overridable at output time with 'output madevent --amp_chunk_size=N' and, for
# the helicity-recycled copy, with the 'amp_chunk_size' run_card parameter.
# 0 (or a negative value) disables the split entirely.
AMP_CHUNK_SIZE_DEFAULT = 2000


_AMP_COMMENT_RE = re.compile(r"^(\s*#|c\$|c$|(c\s+([^=]|$))|cf2py|c\-\-|c\*\*|\s*!|!\$)",
                             re.IGNORECASE)
_AMP_CONTINUATION_RE = re.compile(r"^(?:     )[$&]")


def chunk_fortran_statements(lines, chunk_size, fixed_form=True):
    """Group *lines* into slices of about *chunk_size* statements each, and
    return the list of slices.

    With fixed_form=True the lines are already column-formatted fortran (what
    hel_recycle produces); with fixed_form=False they are the raw HELAS calls
    the exporter hands to the FortranWriter, one statement per entry and with
    '#' comments.

    A slice boundary may only fall where a new statement starts at nesting
    depth zero: continuation lines (5 blanks then '$' or '&') stay with their
    statement, comments attach to the statement that follows them, and an
    IF(...)THEN / DO block -- hel_recycle emits those around a flavor-masked
    split amplitude -- is never cut in half.
    """

    def is_continuation(line):
        return fixed_form and bool(_AMP_CONTINUATION_RE.match(line))

    def is_comment(line):
        if not line.strip():
            return True
        if fixed_form:
            return bool(_AMP_COMMENT_RE.search(line))
        return line.lstrip().startswith('#')

    def depth_change(line):
        code = line.upper().split('!')[0].strip()
        if code.startswith('IF') and code.endswith('THEN'):
            return 1
        if code.startswith('DO ') or code == 'DO':
            return 1
        if code.startswith('END IF') or code.startswith('ENDIF') or \
           code.startswith('END DO') or code.startswith('ENDDO'):
            return -1
        return 0

    chunks = []
    current = []
    pending = []          # comments waiting for the statement they annotate
    nb_statements = 0
    depth = 0
    for line in lines:
        if is_comment(line):
            pending.append(line)
            continue
        if is_continuation(line):
            # a continuation can only follow a statement already in flight
            (current if current else pending).append(line)
            continue
        if depth == 0 and nb_statements >= chunk_size and current:
            chunks.append(current)
            current = []
            nb_statements = 0
        current.extend(pending)
        pending = []
        current.append(line)
        nb_statements += 1
        depth += depth_change(line)
        if depth < 0:
            depth = 0
    if pending:
        (current if current else chunks[-1] if chunks else current).extend(pending)
    if current:
        chunks.append(current)
    return chunks


class VirtualExporter(object):
    
    #exporter variable who modified the way madgraph interacts with this class
    
    grouped_mode = 'madevent'  
    # This variable changes the type of object called within 'generate_subprocess_directory'
    #functions. 
    # False to avoid grouping (only identical matrix element are merged)
    # 'madevent' group the massless quark and massless lepton
    # 'madweight' group the gluon with the massless quark
    sa_symmetry = False
    # If no grouped_mode=False, uu~ and u~u will be called independently. 
    #Putting sa_symmetry generates only one of the two matrix-element.
    check = True
    # Ask madgraph to check if the directory already exists and propose to the user to 
    #remove it first if this is the case
    output = 'Template'
    # [Template, None, dir]
    #    - Template, madgraph will call copy_template
    #    - dir, madgraph will just create an empty directory for initialisation
    #    - None, madgraph do nothing for initialisation
    exporter = 'v4'
    # language of the output 'v4' for Fortran output
    #                        'cpp' for C++ output
    support_ddm_color_basis = False
    # True for the output formats which can use the (n-2)! Del Duca-Dixon-
    # Maltoni basis for the color sum of multi-gluon processes.
    ddm_needs_flow_basis = False
    # True when the format also needs a color flow per event: the trace basis
    # is then built next to the DDM one, and the trace JAMPs are obtained from
    # the DDM ones through the Kleiss-Kuijf relations.

    default_vector_size = 0
    
    
    def __init__(self, dir_path = "", opt=None):
        # cmd_options is a dictionary with all the optional argurment passed at output time
        
        # Activate some monkey patching for the helas call writer.
        helas_call_writers.HelasCallWriter.customize_argument_for_all_other_helas_object = \
                self.helas_call_writer_custom
        
        self.has_second_exporter = None
        

    # helper function for customise helas writter
    @staticmethod
    def custom_helas_call(call, arg):
        """static method to customise the way aloha function call are written
        call is the default template for the call
        arg are the dictionary used for the call
        """
        return call, arg
    
    helas_call_writer_custom = lambda x,y,z: x.custom_helas_call(y,z)


    def copy_template(self, model):
        return

    def generate_subprocess_directory(self, subproc_group, helicity_model, me=None, **opt):
    #    generate_subprocess_directory(self, matrix_element, helicity_model, me_number) [for ungrouped]
        return 0 # return an integer stating the number of call to helicity routine

    def generate_subprocess_directory_end(self, **opt):
        """ This is called only if the class is used as a second exporter. (like simd plugin)
            in that case opt contains all the local variable defined in the upstream class.
            so if multiple option exists this can lead to variable existing in some setup and not other
        """
        return 

    def convert_model(self, model, wanted_lorentz=[], wanted_couplings=[]):
        return
    
    def finalize(self,matrix_element, cmdhistory, MG5options, outputflag, second_exporter=None):
        return
    
    
    def pass_information_from_cmd(self, cmd):
        """pass information from the command interface to the exporter.
           Please do not modify any object of the interface from the exporter.
        """
        return

    def expand_merged_particle_legs(self, proc_defs):
        """Return a copy of the process definitions in which merged-flavor beam
        PDG codes (81/82/83 = _quark/_lepton/_neutrino, and their conjugates)
        are expanded to the concrete signed flavour PDGs, using the model's
        ``merged_particles`` map.

        Tools that reverse-map a process leg's ids to a multiparticle name
        (e.g. MadAnalysis5) do not understand the merged codes, which is why
        they must be expanded before the process is handed to them."""
        if not proc_defs:
            return proc_defs
        try:
            merged = proc_defs[0].get('model').get('merged_particles') or {}
        except Exception:
            merged = {}
        if not merged:
            return proc_defs

        import copy
        # deep-copy the definitions but keep the (large) model object shared
        memo = {}
        try:
            model = proc_defs[0].get('model')
            memo[id(model)] = model
        except Exception:
            pass
        proc_defs = copy.deepcopy(proc_defs, memo)
        for procdef in proc_defs:
            for leg in procdef.get('legs'):
                expanded = []
                for pid in leg.get('ids'):
                    base = abs(pid)
                    if base in merged:
                        sign = 1 if pid > 0 else -1
                        expanded.extend(sign * real for real in merged[base])
                    else:
                        expanded.append(pid)
                leg.set('ids', sorted(set(expanded)))
        return proc_defs

    def modify_grouping(self, matrix_element):
        return False, matrix_element
           
    def export_model_files(self, model_v4_path):
        raise Exception("V4 model not supported by this type of exporter. Please use UFO model")
        return
    
    def export_helas(self, HELAS_PATH):
        raise Exception("V4 model not supported by this type of exporter. Please use UFO model")
        return

#===============================================================================
# ColorReflectionFolding
#===============================================================================
class ColorReflectionFolding(object):
    """Reversing every color basis element maps the basis onto itself, and for
    a pure gluon process the two flows of a pair only differ by one overall
    sign. Half the color flows then carry nothing of their own and |M|^2 can be
    summed over one flow per pair, against a color matrix folded onto them.

    Shared by the fortran exporters and by the madmatrix (C++/GPU) one, which
    only differ in when a folding is worth taking (jamp_fold_worthwhile)."""

    # sum |M|^2 over one color flow per reversal pair instead of over every one
    # Folding the color matrix onto one line per reversal pair only works
    # where the template sums over NCOLORFOLD. get_color_data_lines is shared
    # by every fortran exporter, so this stays off unless the template agrees.
    jamp_fold = False

    # Above this many entries the folded color matrix is not written out but
    # rebuilt at run time, which only the sign +1 case can do (see
    # jamp_fold_worthwhile).
    color_fold_max_written = 300000

    @staticmethod
    def jamp_color_rows(matrix_element):
        """The color coefficient of every amplitude, one dictionary per color
        basis line. Same numbers get_JAMP_lines works from."""

        rows = []
        powers = {}
        for coeff_list in matrix_element.get_color_amplitudes():
            row = {}
            for coefficient, amp in coeff_list:
                if not coefficient:
                    continue
                try:
                    power = powers[coefficient[3]]
                except KeyError:
                    power = fractions.Fraction(3) ** coefficient[3]
                    powers[coefficient[3]] = power
                value = (1j if coefficient[2] else 1) * coefficient[0] * \
                        coefficient[1] * power
                row[amp] = row.get(amp, 0) + value
            rows.append(dict((amp, complex(v)) for amp, v in row.items() if v))
        return rows

    def get_jamp_reflection(self, matrix_element):
        """Reversing every color basis element maps the basis onto itself, and
        for a pure gluon process the color coefficients of a line and of its
        reverse differ by one overall sign, so half the color flows carry no
        information of their own:

            JAMP[reverse(i)] = sign * JAMP[i]

        Return (reverse, sign) or None. The relation is read off the color
        coefficients themselves rather than assumed, so a process where it does
        not hold -- a quark line, where reversing does not commute with the
        fermion flow -- simply gets None."""

        if not isinstance(matrix_element, helas_objects.HelasMatrixElement):
            return None
        color_basis = matrix_element.get('color_basis')
        if not color_basis or len(color_basis) < 2:
            return None
        keys = sorted(color_basis.keys())
        position = dict((key, i) for i, key in enumerate(keys))

        reverse = []
        for key in keys:
            other = color_amp.reverse_immutable(key)
            if other is None or other not in position:
                return None
            reverse.append(position[other])
        if any(reverse[reverse[i]] != i for i in range(len(keys))):
            return None

        columns = self.jamp_color_rows(matrix_element)

        sign = None
        for i in range(len(keys)):
            here, there = columns[i], columns[reverse[i]]
            if set(here) != set(there):
                return None
            for amp, value in here.items():
                ratio = there[amp] / value
                if ratio not in (1, -1):
                    return None
                if sign is None:
                    sign = int(ratio.real)
                elif sign != int(ratio.real):
                    return None
        if sign is None:
            return None
        return reverse, sign

    @staticmethod
    def jamp_reflection_representatives(reverse):
        """One line per pair, and for every line the pair it belongs to."""

        representatives = [i for i in range(len(reverse)) if i <= reverse[i]]
        slot = {}
        for index, line in enumerate(representatives):
            slot[line] = index
            slot[reverse[line]] = index
        return representatives, slot

    def jamp_fold_worthwhile(self, sign, nb_pairs):
        """Whether a folding is taken once it has been found.

        With sign +1 every line of a pair enters with the same weight, so the
        permutations leaving the color basis invariant carry over to the pairs
        unchanged and the folded matrix can still be rebuilt at run time from
        one line per orbit. With sign -1 a permutation may send a line onto its
        own partner, which flips the weight, and the rebuilt form would need a
        sign of its own; there the folded matrix is written out instead, which
        is only affordable while it stays small."""

        return sign > 0 or \
            nb_pairs * (nb_pairs + 1) // 2 <= self.color_fold_max_written

    def get_jamp_folding(self, matrix_element):
        """Whether to sum |M|^2 over one line per reversal pair, and the
        (reverse, sign, representatives, slot) that goes with it."""

        if not self.jamp_fold:
            return None
        found = self.get_jamp_reflection(matrix_element)
        if not found:
            return None
        reverse, sign = found
        representatives, slot = self.jamp_reflection_representatives(reverse)
        if not self.jamp_fold_worthwhile(sign, len(representatives)):
            return None
        return {'reverse': reverse, 'sign': sign,
                'representatives': representatives, 'slot': slot}

    def jamp_folded_color_matrix(self, matrix_element, reverse, sign):
        """The color matrix over one line per reversal pair. Summing |M|^2 over
        the pairs instead of over every line gives the same number, since the
        two lines of a pair only differ by the overall sign:

            C'[a][b] = sum over the two lines of a and the two of b, each
                       weighted by its sign relative to the line kept

        Returns (denominator, rows) with rows[a][b] integer, a and b indexing
        the representatives."""

        color_matrix = matrix_element.get('color_matrix')
        representatives, _slot = self.jamp_reflection_representatives(reverse)
        denominator = max(color_matrix.get_line_denominators())
        full = [color_matrix.get_line_numerators(i, denominator)
                for i in range(len(reverse))]

        def pair(a):
            return [(a, 1)] if reverse[a] == a else [(a, 1), (reverse[a], sign)]

        rows = []
        for a in representatives:
            row = []
            for b in representatives:
                total = 0
                for i, ci in pair(a):
                    for j, cj in pair(b):
                        total += ci * cj * full[i][j]
                assert int(total) == total
                row.append(int(total))
            rows.append(row)
        return denominator, rows

#===============================================================================
# ProcessExporterFortran
#===============================================================================
class ProcessExporterFortran(ColorReflectionFolding, VirtualExporter,
                             jamp_optimiser.JampOptimiser):
    """Class to take care of exporting a set of matrix elements to
    Fortran (v4) format."""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{}
                        }
    grouped_mode = False
    # jamp_fold (sum |M|^2 over one color flow per reversal pair) comes from
    # ColorReflectionFolding and stays off unless the template sums over
    # NCOLORFOLD: get_color_data_lines is shared by every fortran exporter.
    # jamp_optim, myjamp_count and jamp_integer_walk come from JampOptimiser.
    # BLAS-3 for the color sum: all helicities at once as one right hand side.
    # None means take it when the library is there and the process is big
    # enough for it to pay.
    blas = None
    blas_min_ncolor = 100
    # write the JAMP definitions as one recipe per orbit of the permutations
    # leaving the color basis invariant, instead of one line per definition
    jamp_orbit = False
    # How the definitions reach memory: 'recipes' rebuilds them at the first
    # call from one recipe per orbit, 'tables' writes the operand indices out
    # as DATA. Both run the very same loop, and both start from the orbit
    # equivariant optimisation, so they only differ in the source they need.
    jamp_emit = 'tables'
    # finish with the plain scan once the orbit rounds have nothing left to
    # take as a whole (only used by the table emission, see below)
    jamp_greedy_tail = True
    # Read the amplitudes of the current helicity into a buffer before running
    # the definitions over it, instead of holding the definitions at the end of
    # AMP. Needed where AMP is indexed by helicity, which is what madevent does
    # once it rewrites the matrix element for helicity recycling.
    jamp_gather = False
    # up to this many entries in the matrix, both optimisations are run and the
    # shorter result kept (see optimise_jamp_best)
    jamp_compare_max_size = 20000
    # Below this many definitions writing them out is both smaller and faster:
    # the lines still fit in the instruction cache, while the loop reading the
    # operands from a table pays for the two indirections whatever the size.
    # Measured on g g > n g, the two cost the same at about five thousand
    # definitions (795 definitions: 0.47 us written out against 1.40 us;
    # 9990 definitions: 39.4 us against 26.4 us).
    jamp_orbit_min_def = 5000
    # how much smaller the compressed color matrix has to be before it is used
    # instead of writing every entry out (see get_color_matrix_encoding)
    color_encoding_margin = 4
    run_card_class = None
    use_flavor_mask = True
    # Whether this exporter can honor the --use_crossing of the generate/add
    # command, i.e. emit a matrix element whose FLAV_IDX carries a crossing.
    # Only the fortran standalone implements the machinery, so every other
    # exporter must refuse the request rather than silently write code that
    # cannot answer a crossed FLAV_IDX (see _check_crossing_support).
    supports_crossing = False

    def __init__(self,  dir_path = "", opt=None):
        """Initiate the ProcessExporterFortran with directory information"""
        self.mgme_dir = MG5DIR
        self.dir_path = dir_path
        self.model = None
        self.beam_polarization = [True,True]

        self.opt = dict(self.default_opt)
        if opt:
            self.opt.update(opt)
        self.cmd_options = self.opt['output_options']
        self._configure_flavor_mask_from_cmd_options()
        self._check_crossing_support()
        
        #place holder to pass information to the run_interface
        self.proc_characteristic = banner_mod.ProcCharacteristic()
        # call mother class
        super(ProcessExporterFortran,self).__init__(dir_path, opt)

    @staticmethod
    def _get_broken_symmetry_data(process, ninitial):
        """Return decay-aware symmetry metadata for broken_sym generation."""

        def sort_decay_chains_by_leg(proc):
            decay_chains = copy.copy(proc.get('decay_chains'))
            sorted_decay_chains = []
            for leg in proc.get_final_legs():
                init_ids = [d.get('legs')[0].get('id') for d in decay_chains]
                if leg.get('id') in init_ids:
                    sorted_decay_chains.append(decay_chains.pop(init_ids.index(leg.get('id'))))
            return sorted_decay_chains

        def recurse(proc, next_flav_index):
            components = []
            current_entries = []
            decay_chains = sort_decay_chains_by_leg(proc)
            for leg in proc.get_final_legs():
                decay = None
                for i, candidate in enumerate(decay_chains):
                    if candidate.get('legs')[0].get('id') == leg.get('id'):
                        decay = decay_chains.pop(i)
                        break
                if decay:
                    start = next_flav_index
                    child_components, child_fingerprints, next_flav_index = \
                        recurse(decay, next_flav_index)
                    end = next_flav_index - 1
                    components.extend(child_components)
                    # Fingerprint encodes the entire decay sub-tree so that
                    # two entries with the same PID but different decay
                    # products are not counted as identical when computing
                    # the symmetry factor COMP_OLD.
                    fingerprint = (leg.get('id'), tuple(child_fingerprints))
                else:
                    start = next_flav_index
                    end = next_flav_index
                    next_flav_index += 1
                    fingerprint = (leg.get('id'),)
                current_entries.append({
                    'pid': leg.get('id'),
                    'start': start,
                    'length': end - start + 1,
                    'fingerprint': fingerprint,
                })
            components.insert(0, current_entries)
            current_fingerprints = [e['fingerprint'] for e in current_entries]
            return components, current_fingerprints, next_flav_index

        components, _, _ = recurse(process, ninitial + 1)

        comp_starts = []
        comp_ends = []
        comp_old_factors = []
        pid_list = []
        block_starts = []
        block_lengths = []
        entry_idx = 1
        for entries in components:
            comp_starts.append(entry_idx)
            # Count identical entries by (pid, full-decay-tree fingerprint).
            # Two entries with the same top-level PID but different decay
            # sub-trees are NOT identical and must not contribute to the
            # over-counting factor.  Using only the PID (old behaviour) was
            # wrong: e.g. two Z bosons decaying to _quark and _lepton were
            # both counted as PID=23 giving COMP_OLD=2, even though the base
            # IDEN already treats them as distinguishable.
            fp_counts = {}
            old_factor = 1
            for entry in entries:
                key = entry['fingerprint']
                fp_counts[key] = fp_counts.get(key, 0) + 1
                pid_list.append(entry['pid'])
                block_starts.append(entry['start'])
                block_lengths.append(entry['length'])
                entry_idx += 1
            for multiplicity in fp_counts.values():
                old_factor *= math.factorial(multiplicity)
            comp_old_factors.append(old_factor)
            comp_ends.append(entry_idx - 1)

        return {
            'ncomponents': len(components),
            'nentries': len(pid_list),
            'component_starts': comp_starts,
            'component_ends': comp_ends,
            'component_old_factors': comp_old_factors,
            'pid_list': pid_list,
            'block_starts': block_starts,
            'block_lengths': block_lengths,
            # Per-component list of entries (pid, start, length, fingerprint).
            # Two entries within a component that share a fingerprint describe
            # identical, interchangeable decay sub-trees; used to enumerate the
            # flavor permutations that GET_FLAVOR_INDEX must also resolve.
            'components': components,
        }

    @staticmethod
    def _fill_broken_sym_replace_dict(replace_dict, sym_data):
        """Populate *replace_dict* with the eight broken_sym_* Fortran DATA
        keys that are consumed by the BROKEN_SYM function in every matrix
        template.  Centralised here so that callers never drift out of sync.
        Also used by the C++ and Python exporters which keep the individual
        DATA keys in their own templates.
        """
        replace_dict['broken_sym_ncomponents'] = sym_data['ncomponents']
        replace_dict['broken_sym_nentries'] = sym_data['nentries']
        replace_dict['broken_sym_component_starts'] = \
            ",".join(str(v) for v in sym_data['component_starts'])
        replace_dict['broken_sym_component_ends'] = \
            ",".join(str(v) for v in sym_data['component_ends'])
        replace_dict['broken_sym_component_old_factors'] = \
            ",".join(str(v) for v in sym_data['component_old_factors'])
        replace_dict['broken_sym_pid_list'] = \
            ",".join(str(v) for v in sym_data['pid_list'])
        replace_dict['broken_sym_block_starts'] = \
            ",".join(str(v) for v in sym_data['block_starts'])
        replace_dict['broken_sym_block_lengths'] = \
            ",".join(str(v) for v in sym_data['block_lengths'])

    @staticmethod
    def _make_broken_sym_fortran_function(func_name, sym_data,
                                          nexternal_decl='include'):
        """Return the complete Fortran BROKEN_SYM function as a string.

        This single implementation is shared by all Fortran matrix templates
        via the %(broken_sym_function)s placeholder, eliminating copy-paste
        duplication.

        Args:
            func_name      : full Fortran function name, e.g. 'BROKEN_SYM1'
                             or 'MYSMATRIX_BROKEN_SYM'.
            sym_data       : dict returned by _get_broken_symmetry_data.
            nexternal_decl : 'include' (default) to emit
                             "include 'nexternal.inc'", or an integer to emit
                             an explicit PARAMETER (NEXTERNAL=N) declaration
                             (used by templates that lack the include file).
        """
        template_path = pjoin(_file_path, 'iolibs', 'template_files',
                              'fortran_matrix_broken_sym_fct.inc')
        template = open(template_path).read()

        if nexternal_decl == 'include':
            nexternal_lines = "      include 'nexternal.inc'"
        else:
            nexternal_lines = ('      INTEGER NEXTERNAL\n'
                               '      PARAMETER (NEXTERNAL=%d)' % int(nexternal_decl))

        replace_dict = {
            'func_name': func_name,
            'nexternal_decl': nexternal_lines,
        }
        ProcessExporterFortran._fill_broken_sym_replace_dict(replace_dict, sym_data)
        return template % replace_dict

    def _build_flav_table_flat(self, matrix_element):
        """Return (n_flavors, flav_table_flat) for this matrix element.

        Return type: a 2-tuple (int, list[int]) -- n_flavors is the number of
        allowed flavors (>= 1) and flav_table_flat is a flat list of ints.

        flav_table_flat is the column-major (leg-fastest) flattening of
        FLAV_TABLE(NEXTERNAL, NFLAV), where column f holds the per-leg group
        position (1..N within each merged-particle group) of the f-th allowed
        flavor. This is the single source of the flavor table consumed both by
        GET_FLAVOR_INDEX (forward FLAVOR->index lookup) and by the FLAVOR
        rebuild inside MATRIX/GET_AMP.

        The flavor machinery is now always present (consistent API): an ME with
        no merged-particle variants is treated as a single flavor whose group
        position is 1 on every leg, i.e. n_flavors=1 and an all-ones table row.
        This matches the FLAVOR(:)=1 convention the drivers/callers use and the
        flv=1 argument HELAS expects for an unmerged leg.
        """

        allowed_flavors = matrix_element.compute_flavor_masks()
        n_flavors = len(allowed_flavors)
        if n_flavors == 0:
            nexternal = matrix_element.get_nexternal_ninitial()[0]
            return (1, [1] * nexternal)
        model = matrix_element.get('processes')[0].get('model')
        pdg_to_group_pos, max_group_size = self._build_flavor_group_lookup(model)
        # FLAV_TABLE laid out column-major: FLAV_TABLE(NEXTERNAL, NMASK_FLAV).
        # Fortran DATA without explicit indices iterates fastest in the first
        # dimension, so we emit (leg, flavor) tuples with leg-first ordering.
        flav_table_flat = []
        for flavor in allowed_flavors:
            for p in flavor:
                flav_table_flat.append(self._map_flavor_to_group_pos(
                    p, pdg_to_group_pos, max_group_size))
        return (n_flavors, flav_table_flat)

    def _build_flav_pdg_tables(self, matrix_element):
        """Return (n_flavors, pdg_flat, antipdg_flat) for this matrix element.

        The FLAVOR array threaded through matrix.f holds unsigned group
        *positions* (see _build_flav_table_flat), which is all the matrix
        element needs: every member of a flavor group shares the couplings, so
        the position alone selects the mask. A caller working in PDG codes --
        the f2py layer -- cannot use that: a position means nothing without
        knowing which group and which leg it belongs to, and nothing in the
        generated code maps one back to a PDG. These tables are that missing
        map, and they are the only thing standing between an f2py caller and
        being able to ask for a process by its PDG codes.

        Two tables are emitted rather than one, both column-major
        (leg-fastest, matching FLAV_TABLE):

        - pdg_flat:     the signed PDG of each leg for each flavor.
        - antipdg_flat: the PDG of the *antiparticle* of that same leg.

        The antiparticle table exists because a crossing conjugates every leg
        that swaps between the initial and the final state, and conjugation is
        NOT "negate the PDG": a self-conjugate particle (the gluon, 21) must
        stay itself. Tabulating both here lets the generated fortran pick one
        or the other by the sign of SGN(k) -- which GET_CROSS_PERM already
        computes -- instead of trying to re-derive the model's conjugation rule
        at runtime. It is the same get_anti_pdg_code() that
        get_iden_cross_lines uses to build BASEPID_CROSS_TABLE, so the two stay
        consistent by construction.

        The per-leg sign comes from the process's own leg id (e.g. -81 for an
        incoming anti-quark), while the magnitude comes from the group member
        sitting at that position; a leg that is not part of a merged group
        (a gluon) keeps its own PDG whatever the flavor.
        """

        allowed_flavors = matrix_element.compute_flavor_masks()
        process = matrix_element.get('processes')[0]
        model = process.get('model')
        # compute_flavor_masks() is indexed by the FULL external legs, so for a
        # decay chain (p p > w+ w-, w+ > j j, w- > j j) the flavor tuple spans the
        # 6 decay leaves, not the 4 core legs of process.get('legs'). Expand the
        # decays so leg_ids lines up with the flavor tuple (a no-op without decays).
        legs = process.get_legs_with_decays() if hasattr(process, 'get_legs_with_decays') \
            else process.get('legs')
        leg_ids = [leg.get('id') for leg in legs]
        nexternal = len(leg_ids)

        if not allowed_flavors:
            allowed_flavors = [tuple([1] * nexternal)]

        merged_particles = (model.get('merged_particles') or {}) if model else {}

        def leg_pdg(leg_id, pos):
            """The signed PDG of a leg whose flavor sits at group position pos."""
            members = merged_particles.get(abs(leg_id))
            if not members:
                # Not a merged leg: its PDG does not depend on the flavor.
                return int(leg_id)
            try:
                magnitude = int(members[int(pos) - 1])
            except (IndexError, ValueError, TypeError):
                return int(leg_id)
            # The group id carries the particle/antiparticle sign of the leg.
            return magnitude if leg_id > 0 else -magnitude

        pdg_flat = []
        antipdg_flat = []
        for flavor in allowed_flavors:
            for leg, pos in enumerate(flavor):
                pdg = leg_pdg(leg_ids[leg], pos)
                pdg_flat.append(pdg)
                try:
                    antipdg_flat.append(
                        model.get('particle_dict')[pdg].get_anti_pdg_code())
                except KeyError:
                    # No such particle in the model (should not happen): fall
                    # back to the naive conjugation rather than crash the
                    # export. A wrong entry here can only mis-*match* a PDG
                    # request, never corrupt a matrix element.
                    antipdg_flat.append(-pdg)

        return (len(allowed_flavors), pdg_flat, antipdg_flat)

    def _build_flav_index_lookup(self, matrix_element, n_flavors, flav_table_flat):
        """Build the expanded GET_FLAVOR_INDEX lookup for decay-chain MEs.

        The mask/goodhel FLAV_TABLE keeps a single representative per
        identical-decay-block permutation class: get_external_flavors dedups
        flavors that differ only by swapping identical, identically-decaying
        particles (e.g. the two Z systems of `z z, z>l+l-, z>l+l-`). A caller
        such as MadSpin can legitimately present such a swapped ordering, which
        is NOT a column of FLAV_TABLE; an exact lookup would then return 0
        (-> |M|=0 -> the MadSpin unweighting loop spins forever).

        Returns (lookup_flat, index_map): a column-major table of every
        interchangeable-block permutation of every representative flavor, plus a
        parallel 1-based index_map giving, for each permutation, the
        representative's FLAV_TABLE column it resolves to. Returns (None, None)
        when there are no interchangeable blocks (the common, non-decay case),
        leaving the simple one-row-per-flavor lookup -- and the goldens --
        unchanged.
        """
        if n_flavors <= 1:
            return (None, None)
        process = matrix_element.get('processes')[0]
        if not process.get('decay_chains'):
            return (None, None)
        nexternal = len(flav_table_flat) // n_flavors
        ninitial = matrix_element.get_nexternal_ninitial()[1]
        try:
            sym = self._get_broken_symmetry_data(process, ninitial)
        except Exception:
            return (None, None)
        # Groups of interchangeable position-blocks: entries within a component
        # that share a fingerprint (same PID and same decay sub-tree).
        swap_groups = []
        for entries in sym.get('components', []):
            by_fp = {}
            for entry in entries:
                by_fp.setdefault(entry['fingerprint'], []).append(
                    (entry['start'], entry['length']))
            for blocks in by_fp.values():
                if len(blocks) > 1:
                    swap_groups.append(blocks)
        if not swap_groups:
            return (None, None)
        reps = [tuple(flav_table_flat[i * nexternal:(i + 1) * nexternal])
                for i in range(n_flavors)]
        perms_per_group = [list(itertools.permutations(blocks))
                           for blocks in swap_groups]
        lookup = {}  # group-position tuple -> 1-based representative index
        for ridx, rep in enumerate(reps, 1):
            for combo in itertools.product(*perms_per_group):
                v = list(rep)
                for blocks, perm in zip(swap_groups, combo):
                    # Destination block i receives the content the representative
                    # has at perm[i] (same fingerprint => identical block length).
                    for (dst_s, dst_l), (src_s, _src_l) in zip(blocks, perm):
                        for k in range(dst_l):
                            v[dst_s - 1 + k] = rep[src_s - 1 + k]
                lookup.setdefault(tuple(v), ridx)
        if len(lookup) == n_flavors:
            return (None, None)
        lookup_flat = []
        index_map = []
        for gpos, ridx in lookup.items():
            lookup_flat.extend(gpos)
            index_map.append(ridx)
        return (lookup_flat, index_map)

    def _make_flavor_index_fortran_function(self, func_name, n_flavors,
                                            flav_table_flat, nexternal_decl='include',
                                            lookup_flat=None, index_map=None):
        """Return the complete Fortran GET_FLAVOR_INDEX function as a string.

        Emitted via the %(flavor_index_function)s placeholder and used to
        resolve the input FLAVOR(NEXTERNAL) vector to its 1-based allowed-flavor
        index exactly once per phase-space point. The resulting FLAV_IDX is then
        threaded into MATRIX/GET_AMP and used to index the per-flavor
        good-helicity filter.

        Args mirror _make_broken_sym_fortran_function: *func_name* is the full
        Fortran name (e.g. 'MG5_1_GET_FLAVOR_INDEX'); *nexternal_decl* is
        'include' for "include 'nexternal.inc'" or an integer for an explicit
        PARAMETER declaration.

        *lookup_flat*/*index_map* (from _build_flav_index_lookup) request the
        permutation-aware variant: the lookup table holds every interchangeable
        decay-block permutation of each flavor and index_map maps each back to
        its representative FLAV_TABLE column. When None (the common case), the
        plain one-row-per-flavor lookup is emitted (byte-identical to before).
        """
        if nexternal_decl == 'include':
            nexternal_lines = "      include 'nexternal.inc'"
        else:
            nexternal_lines = ('      INTEGER NEXTERNAL\n'
                               '      PARAMETER (NEXTERNAL=%d)' % int(nexternal_decl))

        if lookup_flat is not None:
            template_path = pjoin(_file_path, 'iolibs', 'template_files',
                                  'fortran_matrix_flavor_index_fct_perm.inc')
            template = open(template_path).read()
            return template % {
                'func_name': func_name,
                'nexternal_decl': nexternal_lines,
                'nlookup': len(index_map),
                'flav_table_data': ', '.join(str(v) for v in lookup_flat),
                'flav_index_map': ', '.join(str(v) for v in index_map),
            }

        template_path = pjoin(_file_path, 'iolibs', 'template_files',
                              'fortran_matrix_flavor_index_fct.inc')
        template = open(template_path).read()
        return template % {
            'func_name': func_name,
            'nexternal_decl': nexternal_lines,
            'nflav': n_flavors,
            'flav_table_data': ', '.join(str(v) for v in flav_table_flat),
        }

    def _make_flavor_array_fortran_function(self, func_name, n_flavors,
                                            flav_table_flat, nexternal_decl='include'):
        """Return the complete Fortran GET_FLAVOR(FLAV_IDX, FLAVOR) function as a
        string: the reverse of GET_FLAVOR_INDEX, filling FLAVOR from the table.
        Emitted via the %(flavor_array_function)s placeholder and used by the
        outer entry points (SMATRIX, ...) which now receive FLAV_IDX but still
        need the FLAVOR array (e.g. for BROKEN_SYM). Same args/convention as
        _make_flavor_index_fortran_function."""
        template_path = pjoin(_file_path, 'iolibs', 'template_files',
                              'fortran_matrix_flavor_array_fct.inc')
        template = open(template_path).read()

        if nexternal_decl == 'include':
            nexternal_lines = "      include 'nexternal.inc'"
        else:
            nexternal_lines = ('      INTEGER NEXTERNAL\n'
                               '      PARAMETER (NEXTERNAL=%d)' % int(nexternal_decl))

        return template % {
            'func_name': func_name,
            'nexternal_decl': nexternal_lines,
            'nflav': n_flavors,
            'flav_table_data': ', '.join(str(v) for v in flav_table_flat),
        }

    def _make_flavor_pdg_fortran_function(self, func_name, n_flavors, pdg_flat,
                                          antipdg_flat, cross_snippets,
                                          nexternal_decl='include'):
        """Return the complete Fortran GET_PDG_FOR_FLAVOR routine as a string.

        Emitted via the %(flavor_pdg_function)s placeholder. It is the inverse
        of the GET_FLAVOR/GET_FLAVOR_INDEX pair in the PDG vocabulary: those two
        only ever speak group positions, so without this an f2py caller has no
        way to learn which physical process a FLAV_IDX denotes -- let alone
        which one a *crossed* FLAV_IDX denotes.

        *cross_snippets* is the (decl, decode, apply) triple filled by
        fill_crossing_replace_dict: with crossing on it defers to
        GET_CROSS_PERM so the permutation/conjugation follows exactly the same
        code path the matrix element itself uses; with crossing off there is no
        crossing to decode and the plain table lookup is emitted.
        Same args/convention as _make_flavor_index_fortran_function.
        """
        template_path = pjoin(_file_path, 'iolibs', 'template_files',
                              'fortran_matrix_flavor_pdg_fct.inc')
        template = open(template_path).read()

        if nexternal_decl == 'include':
            nexternal_lines = "      include 'nexternal.inc'"
        else:
            nexternal_lines = ('      INTEGER NEXTERNAL\n'
                               '      PARAMETER (NEXTERNAL=%d)' % int(nexternal_decl))

        decl, decode, apply_block = cross_snippets
        return template % {
            'func_name': func_name,
            'nexternal_decl': nexternal_lines,
            'nflav': n_flavors,
            'pdg_table_data': ', '.join(str(v) for v in pdg_flat),
            'antipdg_table_data': ', '.join(str(v) for v in antipdg_flat),
            'pdg_cross_decl': decl,
            'pdg_cross_decode': decode,
            'pdg_cross_apply': apply_block,
        }

    #===========================================================================
    # process exporter fortran switch between group and not grouped
    #===========================================================================
    def export_processes(self, matrix_elements, fortran_model, second_exporter=None, second_helas=None):
        """Make the switch between grouped and not grouped output"""

        calls = 0
        self._crossgroup = {}   # (group_idx, me_idx) -> base info; Track B below
        self._router_base_mes = set()  # id(me) of the within-group (Track A) bases
        self._crossgroup_dirs = []  # (dependent_dir, base_dir) for the parallel makefile
        self._crossgroup_helperms = {}  # base_dir -> {base_proc_id -> [hel perms]}
        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):
            # check handling for the polarization
            for m in matrix_elements:
                for me in m.get('matrix_elements'):
                    for p in me.get('processes'):
                        for beamid in [1,2]:
                            for pid in p.get_initial_ids(beamid):
                                spin = p.get('model').get_particle(pid).get('spin')
                                if spin != 2:
                                    self.beam_polarization[beamid-1] = False
                                    break

            # Cross-group crossing (Track B): a group whose matrix element is a
            # crossing of another group's reuses (symlinks) that base group's
            # compiled matrix element. Detect it here, where every group is
            # visible, and hand the per-group routing to generate_subprocess_
            # directory (keyed by the same enumerate index it receives).
            self._crossgroup = self.compute_crossgroup_routing(matrix_elements)
            # The MEs that serve as a cross-group base must publish their per-flow
            # JAMP2 (so dependents can reselect colour natively); gate that emission
            # to these MEs only, keeping every other madevent ME byte-identical.
            self._crossgroup_base_mes = set(
                id(cg['base_me']) for cg in self._crossgroup.values())
            if self._crossgroup:
                logger.info('Cross-group crossing: %d subprocess(es) will reuse '
                            'a base group\'s matrix element via crossing.'
                            % len({k[0] for k in self._crossgroup}))
                # A shared matrix element now spans physically distinct (crossed)
                # initial states, so a per-beam property is ill-defined. Tag it so
                # check_card_consistency blocks beam polarisation / EVA (same guard
                # as the within-group case; see fill of 'limitations' there).
                if 'crossing' not in self.proc_characteristic['limitations']:
                    self.proc_characteristic['limitations'].append('crossing')

            for (group_number, me_group) in enumerate(matrix_elements):
                calls = calls + self.generate_subprocess_directory(\
                                          me_group, fortran_model, group_number,
                                          second_exporter=second_exporter, second_helas=second_helas
                                          )
            if self._crossgroup_dirs:
                self.write_crossgroup_parallel_makefile(
                    pjoin(self.dir_path, 'SubProcesses'))
            if self._crossgroup_helperms:
                self.write_crossgroup_helunion(
                    pjoin(self.dir_path, 'SubProcesses'))
        else:
             # check handling for the polarization
            self.beam_polarization = [True,True]
            for me in matrix_elements.get_matrix_elements():
                for p in me.get('processes'):
                    for beamid in [1,2]:
                        for pid in p.get_initial_ids(beamid):
                            spin = p.get('model').get_particle(pid).get('spin')
                            if spin != 2:
                                self.beam_polarization[beamid-1] = False
                                break
            for me_number, me in enumerate(matrix_elements.get_matrix_elements()):
                calls = calls + self.generate_subprocess_directory(\
                                                   me, fortran_model, me_number,
                                                   second_exporter=second_exporter, second_helas=second_helas)    

        return calls    
        

    #===========================================================================
    #  create the run_card 
    #===========================================================================
    def create_run_card(self, matrix_elements, history):
        """ """


        # bypass this for the loop-check
        import madgraph.loop.loop_helas_objects as loop_helas_objects
        if isinstance(matrix_elements, loop_helas_objects.LoopHelasMatrixElement):
            matrix_elements = None


        run_card = banner_mod.RunCard(self.run_card_class)
        
        default=True
        if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):            
            processes = [me.get('processes')  for megroup in matrix_elements 
                                        for me in megroup['matrix_elements']]
        elif matrix_elements:
            processes = [me.get('processes') 
                                 for me in matrix_elements['matrix_elements']]
        else:
            default =False
    
        if default:
            run_card.create_default_for_process(self.proc_characteristic, 
                                            history,
                                            processes)
        
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'))
        shutil.copyfile(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'),
                        pjoin(self.dir_path, 'Cards', 'run_card.dat'))
        
        
    #===========================================================================
    # write_vector_size
    #===========================================================================
    def write_vector_size(self, fsock):
        """Write the vector.inc which indicates how many event are handle in parralel."""

        try:
            vector_size = self.opt['output_options']['vector_size']
        except KeyError:
            vector_size = 1
        vector_size = banner_mod.ConfigFile.format_variable(vector_size, int, name='vector_size')
        vector_size = max(1, vector_size)

        try:
            nb_warp = self.opt['output_options']['nb_warp']
        except KeyError:
            nb_warp = 1
        nb_warp = banner_mod.ConfigFile.format_variable(nb_warp, int, name='nb_warp')
        nb_warp = max(1, nb_warp)

        text=["""C
C If VECSIZE_MEMMAX is greater than 1, a vector API is used:
C this is designed for offloading MEs to GPUs or vectorized C++,
C but it can also be used for computing MEs in Fortran.
C If VECSIZE_MEMMAX equals 1, the old scalar API is used:
C this can only be used for computing MEs in Fortran.
C
C Fortran arrays in the vector API can hold up to VECSIZE_MEMMAX
C events and are statically allocated at compile time.
C The constant value of VECSIZE_MEMMAX is fixed at codegen time
C (output madevent ... --vector_size=<VECSIZE_MEMMAX>).
C
C While the arrays can hold up to VECSIZE_MEMMAX events,
C only VECSIZE_USED (<= VECSIZE_MEMAMX) are used in Fortran loops.
C The value of VECSIZE_USED can be chosen at runtime
C (typically 8k-16k for GPUs, 16-32 for vectorized C++).
C
C The value of VECSIZE_USED represents the number of events
C handled by one call to the Fortran/cudacpp "bridge".
C This is not necessarily the number of events which are
C processed in lockstep within a single SIMD vector on CPUs
C or within a single "warp" of threads on GPUs. These parameters
C are internal to the cudacpp bridge and need not be exposed
C to the Fortran program which calls the cudacpp bridge.
C
C NB: THIS FILE CANNOT CONTAIN #ifdef DIRECTIVES
C BECAUSE IT DOES NOT GO THROUGH THE CPP PREPROCESSOR
C (see https://github.com/madgraph5/madgraph4gpu/issues/458).
C
      INTEGER WARP_SIZE
      PARAMETER (WARP_SIZE=%i)
      INTEGER NB_WARP
      PARAMETER (NB_WARP=%i)
      INTEGER VECSIZE_MEMMAX
      PARAMETER (VECSIZE_MEMMAX=%i)
              
              """ % (vector_size,nb_warp, vector_size*nb_warp)]
        fsock.writelines(text)
        return vector_size        

    #===========================================================================
    # copy the Template in a new directory.
    #===========================================================================
    def copy_template(self, model):
        """create the directory run_name as a copy of the MadEvent
        Template, and clean the directory
        """

        #First copy the full template tree if dir_path doesn't exit
        if not os.path.isdir(self.dir_path):
            assert self.mgme_dir, \
                     "No valid MG_ME path given for MG4 run directory creation."
            logger.info('initialize a new directory: %s' % \
                        os.path.basename(self.dir_path))
            misc.copytree(pjoin(self.mgme_dir, 'Template/LO'),
                            self.dir_path, True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/Common'), 
                               self.dir_path)
            # copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards',card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to copy " + card + ".dat to default")
        elif os.getcwd() == os.path.realpath(self.dir_path):
            logger.info('working in local directory: %s' % \
                                                os.path.realpath(self.dir_path))
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/LO'), 
                               self.dir_path)
#            for name in misc.glob('Template/LO/*', self.mgme_dir):
#                name = os.path.basename(name)
#                filname = pjoin(self.mgme_dir, 'Template','LO',name)
#                if os.path.isfile(filename):
#                    files.cp(filename, pjoin(self.dir_path,name))
#                elif os.path.isdir(filename):
#                     misc.copytree(filename, pjoin(self.dir_path,name), True)
            # misc.copytree since dir_path already exists
            misc.copytree(pjoin(self.mgme_dir, 'Template/Common'), 
                               self.dir_path)
            # Copy plot_card
            for card in ['plot_card']:
                if os.path.isfile(pjoin(self.dir_path, 'Cards',card + '.dat')):
                    try:
                        shutil.copy(pjoin(self.dir_path, 'Cards', card + '.dat'),
                                   pjoin(self.dir_path, 'Cards', card + '_default.dat'))
                    except IOError:
                        logger.warning("Failed to copy " + card + ".dat to default")            
        elif not os.path.isfile(pjoin(self.dir_path, 'TemplateVersion.txt')):
            assert self.mgme_dir, \
                      "No valid MG_ME path given for MG4 run directory creation."
        try:
            shutil.copy(pjoin(self.mgme_dir, 'MGMEVersion.txt'), self.dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'MGMEVersion.txt'), 'w').write(MG5_version['version'])

        #Ensure that the Template is clean
        if self.opt['clean']:
            logger.info('remove old information in %s' % \
                                                  os.path.basename(self.dir_path))
            if 'MADGRAPH_BASE' in os.environ:
                misc.call([pjoin('bin', 'internal', 'clean_template'),
                                 '--web'], cwd=self.dir_path)
            else:
                try:
                    misc.call([pjoin('bin', 'internal', 'clean_template')], \
                                                                       cwd=self.dir_path)
                except Exception as why:
                    raise MadGraph5Error('Failed to clean correctly %s: \n %s' \
                                                % (os.path.basename(self.dir_path),why))

            #Write version info
            MG_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'SubProcesses', 'MGVersion.txt'), 'w').write(
                                                              MG_version['version'])

        # add the makefile in Source directory 
        # now moved to finalize

        self.write_vector_size(writers.FortranWriter(pjoin(self.dir_path, 'Source','vector.inc')))
        
        # add the DiscreteSampler information
        files.cp(pjoin(MG5DIR,'vendor', 'DiscreteSampler', 'DiscreteSampler.f'), 
                 pjoin(self.dir_path, 'Source'))
        files.cp(pjoin(MG5DIR,'vendor', 'DiscreteSampler', 'StringCast.f'), 
                 pjoin(self.dir_path, 'Source'))
        
        # We need to create the correct open_data for the pdf
        self.write_pdf_opendata()
        
        
    #===========================================================================
    # Call MadAnalysis5 to generate the default cards for this process
    #=========================================================================== 
    def create_default_madanalysis5_cards(self, history, proc_defs, processes,
                            ma5_path, output_dir, levels = ['parton','hadron']):
        """ Call MA5 so that it writes default cards for both parton and
        post-shower levels, tailored for this particular process."""
        
        if len(levels)==0:
            return
        start = time.time()
        logger.info('Generating MadAnalysis5 default cards tailored to this process')
        try:
            MA5_interpreter = common_run_interface.CommonRunCmd.\
                          get_MadAnalysis5_interpreter(MG5DIR,ma5_path,loglevel=100)
        except (Exception, SystemExit) as e:
            logger.warning('Fail to create a MadAnalysis5 instance. Therefore the default analysis with MadAnalysis5 will be empty')
            return
        if MA5_interpreter is None:
            return

        # expand merged-flavor beam codes (81/82/...) so MA5 recognises the legs
        proc_defs = self.expand_merged_particle_legs(proc_defs)

        MA5_main = MA5_interpreter.main
        for lvl in ['parton','hadron']:
            if lvl in levels:
                card_to_generate = pjoin(output_dir,'madanalysis5_%s_card_default.dat'%lvl)
                try:
                    text = MA5_main.madgraph.generate_card(history, proc_defs, processes,lvl)
                except (Exception, SystemExit) as e:
                    # keep the default card (skip only)
                    logger.warning('MadAnalysis5 failed to write a %s-level'%lvl+
                                                  ' default analysis card for this process.')
                    logger.warning('Therefore, %s-level default analysis with MadAnalysis5 will be empty.'%lvl)
                    error=StringIO()
                    traceback.print_exc(file=error)
                    logger.debug('MadAnalysis5 error was:')
                    logger.debug('-'*60)
                    logger.debug(error.getvalue()[:-1])
                    logger.debug('-'*60)
                else:
                    open(card_to_generate,'w').write(text)
        stop = time.time()
        if stop-start >1:
            logger.info('Cards created in %.2fs' % (stop-start))

    #===========================================================================
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===========================================================================
    def write_procdef_mg5(self, file_pos, modelname, process_str):
        """ write an equivalent of the MG4 proc_card in order that all the Madevent
        Perl script of MadEvent4 are still working properly for pure MG5 run."""

        proc_card_template = template_files.mg4_proc_card.mg4_template
        process_template = template_files.mg4_proc_card.process_template
        process_text = ''
        coupling = ''
        new_process_content = []


        # First find the coupling and suppress the coupling from process_str
        #But first ensure that coupling are define whithout spaces:
        process_str = process_str.replace(' =', '=')
        process_str = process_str.replace('= ', '=')
        process_str = process_str.replace(',',' , ')
        #now loop on the element and treat all the coupling
        for info in process_str.split():
            if '=' in info:
                coupling += info + '\n'
            else:
                new_process_content.append(info)
        # Recombine the process_str (which is the input process_str without coupling
        #info)
        process_str = ' '.join(new_process_content)

        #format the SubProcess
        replace_dict = {'process': process_str,
                        'coupling': coupling}
        process_text += process_template.substitute(replace_dict)
        
        replace_dict = {'process': process_text,
                                            'model': modelname,
                                            'multiparticle':''}
        text = proc_card_template.substitute(replace_dict)
        
        if file_pos:
            ff = open(file_pos, 'w')
            ff.write(text)
            ff.close()
        else:
            return replace_dict


    def pass_information_from_cmd(self, cmd):
        """Pass information for MA5"""
        
        self.proc_defs = cmd._curr_proc_defs

    #===========================================================================
    # Create jpeg diagrams, html pages,proc_card_mg5.dat and madevent.tar.gz
    #===========================================================================
    def finalize(self, matrix_elements, history='', mg5options={}, flaglist=[], second_exporter=None):
        """Function to finalize v4 directory, for inheritance.""" 

        filename = pjoin(self.dir_path,'Source','makefile')
        if not second_exporter:
            self.write_source_makefile(writers.FileWriter(filename), self.model)
        else:
           replace_dict = self.write_source_makefile(None, model=self.model)
           second_exporter.write_source_makefile(writers.FileWriter(filename), model=self.model, default=replace_dict)  

        if second_exporter:
            self.has_second_exporter = second_exporter

        if self.has_second_exporter and hasattr(self.has_second_exporter, 'run_card_class'):
            with misc.TMP_variable(self, 'run_card_class', self.has_second_exporter.run_card_class):
                self.create_run_card(matrix_elements, history)
        else:
            self.create_run_card(matrix_elements, history)
        self.create_MA5_cards(matrix_elements, history)
    
    def create_MA5_cards(self,matrix_elements,history):
        """ A wrapper around the creation of the MA5 cards so that it can be 
        bypassed by daughter classes (i.e. in standalone)."""
        if 'madanalysis5_path' in self.opt and not \
                self.opt['madanalysis5_path'] is None and not self.proc_defs is None:
            processes = None
            if isinstance(matrix_elements, group_subprocs.SubProcessGroupList):            
                processes = [me.get('processes')  for megroup in matrix_elements 
                                        for me in megroup['matrix_elements']]
            elif matrix_elements:
                processes = [me.get('processes') 
                                 for me in matrix_elements['matrix_elements']]
            
            self.create_default_madanalysis5_cards(
                history, self.proc_defs, processes,
                self.opt['madanalysis5_path'], pjoin(self.dir_path,'Cards'),
                levels = ['hadron','parton'])
            
            for level in ['hadron','parton']:
                # Copying these cards turn on the use of MadAnalysis5 by default.
                if os.path.isfile(pjoin(self.dir_path,'Cards','madanalysis5_%s_card_default.dat'%level)):
                    shutil.copy(pjoin(self.dir_path,'Cards','madanalysis5_%s_card_default.dat'%level),
                                pjoin(self.dir_path,'Cards','madanalysis5_%s_card.dat'%level))

    #===========================================================================
    # Create the proc_characteristic file passing information to the run_interface
    #===========================================================================
    def create_proc_charac(self, matrix_elements=None, history="", **opts):
        
        self.proc_characteristic.write(pjoin(self.dir_path, 'SubProcesses', 'proc_characteristics'))

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self):
        """Function to write a matrix.f file, for inheritance.
        """
        pass

    def _check_crossing_support(self):
        """Note that this output cannot read folded crossings.

        `--use_crossing` (on by default) tells the generation not to write out
        the crossed subprocesses separately, because the matrix element is
        expected to reach them through an extended FLAV_IDX instead. Only the
        folding-capable standalone backends implement that decoding.

        This used to refuse the export and ask the user to regenerate with
        --use_crossing=False. It no longer does: the crossed subprocesses are
        recorded as metadata at generation, so an output that cannot read them
        gets them expanded back into explicit subprocesses automatically (see
        MadGraphCmd._expand_recorded_crossings, applied on both the grouped and
        the ungrouped path). Erroring out here would additionally be wrong for
        the many processes that fold NO crossing at all -- nothing would be
        missing from their output -- and it fired on the flag rather than on the
        data. --use_crossing=False stays available, but is no longer needed just
        to reach a non-folding output.
        """

        if self.supports_crossing:
            return
        if not self.opt.get('use_crossing', False):
            return
        logger.debug("The '%s' output does not read folded crossings; any "
                     "recorded crossed subprocess will be expanded back into "
                     "an explicit subprocess.",
                     self.opt.get('export_format', 'unknown'))

    def _configure_flavor_mask_from_cmd_options(self):
        """Honor `--mask=True|False` from the output command line."""

        if 'mask' not in self.cmd_options:
            return
        val = self.cmd_options['mask']
        if isinstance(val, bool):
            self.use_flavor_mask = val
        elif isinstance(val, str):
            token = val.strip().lower()
            if token in ('false', '0', 'no', 'off'):
                self.use_flavor_mask = False
            elif token in ('true', '1', 'yes', 'on'):
                self.use_flavor_mask = True

    def _build_flavor_group_lookup(self, model):
        """Return (pdg_or_group_id -> group_position, max_group_size)."""

        merged_particles = (model.get('merged_particles') or {}) if model else {}
        pdg_to_group_pos = {}
        max_group_size = 0

        for merged_id, members in merged_particles.items():
            members = list(members)
            if members:
                max_group_size = max(max_group_size, len(members))
                # If a merged pseudo-id appears in a flavor tuple, map it to a
                # deterministic valid partner index.
                pdg_to_group_pos[int(merged_id)] = 1
                pdg_to_group_pos[-int(merged_id)] = 1
            for pos, pdg in enumerate(members, 1):
                pdg = int(pdg)
                pdg_to_group_pos[pdg] = pos
                pdg_to_group_pos[-pdg] = pos

        return pdg_to_group_pos, max_group_size

    def _map_flavor_to_group_pos(self, flavor, pdg_to_group_pos, max_group_size=0):
        """Map a raw flavor token to a valid FLV_COUPLING partner index."""

        f = int(flavor)
        if f in pdg_to_group_pos:
            return pdg_to_group_pos[f]
        af = abs(f)
        if af in pdg_to_group_pos:
            return pdg_to_group_pos[af]
        if max_group_size and 1 <= af <= max_group_size:
            return af
        # Keep non-merged particles as their original PDG so downstream
        # flavor-dependent logic (e.g. reweighting/broken-sym bookkeeping)
        # still sees the physical flavor assignment.
        return f

    def _compress_mask_list_to_flavor_groups(self, matrix_element, allowed_flavors,
                                             object_masks):
        """Project per-flavor masks onto coupling-equivalent flavor groups."""

        grouped_flavors = [list(group)
                           for group in matrix_element.get_external_flavors_with_iden()]
        if not grouped_flavors:
            return allowed_flavors, object_masks

        if len(grouped_flavors) == len(allowed_flavors) and all(
                len(group) == 1 and tuple(group[0]) == tuple(flavor)
                for group, flavor in zip(grouped_flavors, allowed_flavors)):
            return allowed_flavors, object_masks

        flavor_to_idx = {tuple(flavor): idx
                         for idx, flavor in enumerate(allowed_flavors)}
        runtime_flavors = []
        group_bitsets = []
        for group in grouped_flavors:
            runtime_flavors.append(tuple(group[0]))
            bits = 0
            for flavor in group:
                bits |= (1 << flavor_to_idx[tuple(flavor)])
            group_bitsets.append(bits)

        grouped_masks = []
        for mask in object_masks:
            grouped_mask = 0
            for group_idx, bits in enumerate(group_bitsets):
                if mask & bits:
                    grouped_mask |= (1 << group_idx)
            grouped_masks.append(grouped_mask)

        return runtime_flavors, grouped_masks

    def _build_flavor_index_masks(self, object_masks, n_flavors, nwords):
        """Transpose a per-object flavor bitset list into per-word, per-flavor
        index bitsets, plus the OR-combined active mask.

        object_masks[i] is the flavor bitset of wavefunction/amplitude i. In
        the result index_masks[word][flav] bit p is set iff object
        64*word + p contributes for flavor flav, and active[word] is the OR
        over all flavors of that word.
        """

        index_masks = [[0] * n_flavors for _ in range(nwords)]
        for obj_idx, mask in enumerate(object_masks):
            word = obj_idx // 64
            bit = obj_idx % 64
            for flav_idx in range(n_flavors):
                if (mask >> flav_idx) & 1:
                    index_masks[word][flav_idx] |= (1 << bit)
        active = [0] * nwords
        for word in range(nwords):
            for flav_idx in range(n_flavors):
                active[word] |= index_masks[word][flav_idx]
        return index_masks, active

    def _format_flavor_mask_decl(self, n_flavors, n_wfs, n_amps,
                                 nwords_wf, nwords_amp,
                                 wf_index_masks, amp_index_masks,
                                 active_wf_index_masks, active_amp_index_masks,
                                 flav_table_flat, thread_flav_idx=False):
        """Format the Fortran declaration / DATA block for the flavor-mask
        machinery. Shared verbatim by the standalone and madevent matrix
        element exporters so the runtime lookup layout cannot drift apart.

        When *thread_flav_idx* is True the resolved flavor index is passed into
        GET_AMP as the FLAV_IDX argument, so the per-call FLAV_TABLE scan is
        gone: only the loop counters needed to rebuild FLAVOR and copy the
        per-flavor masks remain (MASK_J, MASK_K). The default keeps the
        self-contained lookup locals used by the madevent / matchbox backends.
        """

        def _fmt_int8_2d(name, matrix):
            items = []
            for row in matrix:
                for v in row:
                    # INTEGER*8 is signed; convert unsigned 64-bit values that
                    # have the high bit set to their two's-complement equivalent
                    # so gfortran does not reject them with "integer too big".
                    if v >= (1 << 63):
                        v -= (1 << 64)
                    items.append('%d_8' % v)
            return '      DATA %s / %s /' % (name, ', '.join(items))

        def _fmt_int_array(name, values):
            items = [str(v) for v in values]
            return '      DATA %s / %s /' % (name, ', '.join(items))

        # WF_INDEX_MASK / AMP_INDEX_MASK are declared (NWORDS, NMASK_FLAV).
        # Fortran DATA fills column-major (first index fastest), so the value
        # stream must be word-fastest. wf_index_masks/amp_index_masks are
        # indexed [word][flavor]; transpose to [flavor][word] before emitting.
        def _transpose(matrix):
            return [list(col) for col in zip(*matrix)] if matrix else matrix

        decl_lines = [
            'C     Flavor-mask machinery (compute_flavor_masks).',
            '      INTEGER NMASK_FLAV, NMASK_WF, NMASK_AMP',
            '      INTEGER NWORDS_WF, NWORDS_AMP',
            '      PARAMETER (NMASK_FLAV=%d)' % n_flavors,
            '      PARAMETER (NMASK_WF=%d, NMASK_AMP=%d)' % (n_wfs, n_amps),
            '      PARAMETER (NWORDS_WF=%d, NWORDS_AMP=%d)' % (nwords_wf, nwords_amp),
            '      INTEGER*8 WF_INDEX_MASK(NWORDS_WF, NMASK_FLAV)',
            '      INTEGER*8 AMP_INDEX_MASK(NWORDS_AMP, NMASK_FLAV)',
            '      INTEGER*8 CURRENT_WF_MASK(NWORDS_WF)',
            '      INTEGER*8 CURRENT_AMP_MASK(NWORDS_AMP)',
            '      INTEGER*8 ACTIVE_WF_MASK(NWORDS_WF)',
            '      INTEGER*8 ACTIVE_AMP_MASK(NWORDS_AMP)',
            ('      INTEGER MASK_J, MASK_K' if thread_flav_idx else
             '      INTEGER FLAV_IDX_LOOKUP, MASK_I, MASK_J, MASK_K\n'
             '      LOGICAL FLAV_MATCH'),
            '      INTEGER FLAV_TABLE(NEXTERNAL, NMASK_FLAV)',
            _fmt_int8_2d('WF_INDEX_MASK', _transpose(wf_index_masks)),
            _fmt_int8_2d('AMP_INDEX_MASK', _transpose(amp_index_masks)),
            _fmt_int8_2d('ACTIVE_WF_MASK', [active_wf_index_masks]),
            _fmt_int8_2d('ACTIVE_AMP_MASK', [active_amp_index_masks]),
            _fmt_int_array('FLAV_TABLE', flav_table_flat),
        ]
        return '\n'.join(decl_lines)

    def _format_flavor_mask_setup(self, leading_comment=None,
                                  append_amp_init=False,
                                  thread_flav_idx=False):
        """Format the Fortran runtime flavor-lookup block for the flavor-mask
        machinery. Shared by the standalone and madevent matrix element
        exporters; leading_comment and append_amp_init cover the only
        backend-specific wrapping around the common lookup loop.

        When *thread_flav_idx* is True the caller has already resolved the
        flavor index once (in SMATRIX) and passed it down as FLAV_IDX, so the
        per-call FLAV_TABLE scan is replaced by a direct rebuild of FLAVOR and a
        direct copy of the FLAV_IDX-th mask column. FLAV_IDX <= 0 (an unresolved
        flavor) falls back to FLAV_TABLE column 1 with the all-flavors-active
        masks, matching the miss behaviour of the lookup variant.
        """

        setup_lines = []
        if leading_comment:
            setup_lines.append(leading_comment)
        if thread_flav_idx:
            setup_lines += [
                '      IF (FLAV_IDX .GE. 1 .AND. FLAV_IDX .LE. NMASK_FLAV) THEN',
                '        DO MASK_J = 1, NEXTERNAL',
                '          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, FLAV_IDX)',
                '        ENDDO',
                '        DO MASK_K = 1, NWORDS_WF',
                '          CURRENT_WF_MASK(MASK_K) = WF_INDEX_MASK(MASK_K, FLAV_IDX)',
                '        ENDDO',
                '        DO MASK_K = 1, NWORDS_AMP',
                '          CURRENT_AMP_MASK(MASK_K) = AMP_INDEX_MASK(MASK_K, FLAV_IDX)',
                '        ENDDO',
                '      ELSE',
                'C       Unresolved flavor: rebuild from column 1 and keep all',
                'C       active calls enabled (HELAS still checks compatibility).',
                '        DO MASK_J = 1, NEXTERNAL',
                '          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, 1)',
                '        ENDDO',
                '        DO MASK_K = 1, NWORDS_WF',
                '          CURRENT_WF_MASK(MASK_K) = ACTIVE_WF_MASK(MASK_K)',
                '        ENDDO',
                '        DO MASK_K = 1, NWORDS_AMP',
                '          CURRENT_AMP_MASK(MASK_K) = ACTIVE_AMP_MASK(MASK_K)',
                '        ENDDO',
                '      ENDIF',
            ]
            if append_amp_init:
                setup_lines += [
                    'C     Zero-initialise AMP so that JAMP reads 0 from any slot whose',
                    'C     CALL is skipped by the IAND guard below. Without this, AMP',
                    'C     would carry uninitialised stack data into JAMP.',
                    '      AMP(:) = (0D0, 0D0)',
                ]
            return '\n'.join(setup_lines)
        setup_lines += [
            '      FLAV_IDX_LOOKUP = 0',
            '      DO MASK_I = 1, NMASK_FLAV',
            '        FLAV_MATCH = .TRUE.',
            '        DO MASK_J = 1, NEXTERNAL',
            '          IF (FLAVOR(MASK_J) .NE. FLAV_TABLE(MASK_J, MASK_I)) THEN',
            '            FLAV_MATCH = .FALSE.',
            '            EXIT',
            '          ENDIF',
            '        ENDDO',
            '        IF (FLAV_MATCH) THEN',
            '          FLAV_IDX_LOOKUP = MASK_I',
            '          EXIT',
            '        ENDIF',
            '      ENDDO',
            'C     If the lookup misses, keep all active calls enabled. HELAS',
            'C     still checks flavor compatibility internally, so this is a',
            'C     safe fallback for grouped flavor tables and MadSpin probes.',
            '      IF (FLAV_IDX_LOOKUP .EQ. 0) THEN',
            '        DO MASK_K = 1, NWORDS_WF',
            '          CURRENT_WF_MASK(MASK_K) = ACTIVE_WF_MASK(MASK_K)',
            '        ENDDO',
            '        DO MASK_K = 1, NWORDS_AMP',
            '          CURRENT_AMP_MASK(MASK_K) = ACTIVE_AMP_MASK(MASK_K)',
            '        ENDDO',
            '      ELSE',
            '        DO MASK_K = 1, NWORDS_WF',
            '          CURRENT_WF_MASK(MASK_K) = WF_INDEX_MASK(MASK_K, FLAV_IDX_LOOKUP)',
            '        ENDDO',
            '        DO MASK_K = 1, NWORDS_AMP',
            '          CURRENT_AMP_MASK(MASK_K) = AMP_INDEX_MASK(MASK_K, FLAV_IDX_LOOKUP)',
            '        ENDDO',
            '      ENDIF',
        ]
        if append_amp_init:
            setup_lines += [
                'C     Zero-initialise AMP so that JAMP reads 0 from any slot whose',
                'C     CALL is skipped by the IAND guard below. Without this, AMP',
                'C     would carry uninitialised stack data into JAMP.',
                '      AMP(:) = (0D0, 0D0)',
            ]
        return '\n'.join(setup_lines)

    def _get_flavor_mask_blocks(self, matrix_element):
        """Return declaration/setup blocks for per-call flavor masks.

        This madevent variant projects the per-flavor masks onto
        coupling-equivalent flavor groups; ProcessExporterFortranSA overrides
        the method to keep every flavor instead. Both share the Fortran
        formatting through _build_flavor_index_masks, _format_flavor_mask_decl
        and _format_flavor_mask_setup.
        """

        if not getattr(self, 'use_flavor_mask', False):
            return ('', '', 0, 0)

        allowed_flavors = matrix_element.compute_flavor_masks()
        if not allowed_flavors:
            return ('', '', 0, 0)

        if matrix_element.flavor_mask_is_trivial():
            return ('', '', len(allowed_flavors), (1 << len(allowed_flavors)) - 1)

        all_wfs = matrix_element.get_all_wavefunctions()
        all_amps = matrix_element.get_all_amplitudes()
        wf_numbers = [wf.get('number') for wf in all_wfs
                      if isinstance(wf.get('number'), int) and wf.get('number') > 0]
        amp_numbers = [amp.get('number') for amp in all_amps
                       if isinstance(amp.get('number'), int) and amp.get('number') > 0]
        n_wfs = max(matrix_element.get_number_of_wavefunctions(),
                    len(all_wfs),
                    max(wf_numbers) if wf_numbers else 0)
        n_amps = max(matrix_element.get_number_of_amplitudes(),
                     len(all_amps),
                     max(amp_numbers) if amp_numbers else 0)
        nwords_wf = max(1, (n_wfs + 63) // 64)
        nwords_amp = max(1, (n_amps + 63) // 64)

        wf_masks = [0] * n_wfs
        amp_masks = [0] * n_amps
        for wf in all_wfs:
            idx = wf.get('number')
            if isinstance(idx, int) and 0 < idx <= n_wfs:
                wf_masks[idx - 1] = wf['flavor_mask'] if 'flavor_mask' in wf else 0
        for amp in all_amps:
            idx = amp.get('number')
            if isinstance(idx, int) and 0 < idx <= n_amps:
                amp_masks[idx - 1] = amp['flavor_mask'] if 'flavor_mask' in amp else 0

        # madevent collapses coupling-equivalent flavors into one runtime
        # table entry; the standalone exporter overrides this method to skip
        # the compression and keep every flavor.
        runtime_flavors, wf_masks = self._compress_mask_list_to_flavor_groups(
            matrix_element, allowed_flavors, wf_masks)
        _, amp_masks = self._compress_mask_list_to_flavor_groups(
            matrix_element, allowed_flavors, amp_masks)
        n_flavors = len(runtime_flavors)

        active_flavor_mask = 0
        for amp_mask in amp_masks:
            active_flavor_mask |= amp_mask
        if active_flavor_mask == 0:
            active_flavor_mask = (1 << n_flavors) - 1

        wf_index_masks, active_wf_index_masks = self._build_flavor_index_masks(
            wf_masks, n_flavors, nwords_wf)
        amp_index_masks, active_amp_index_masks = self._build_flavor_index_masks(
            amp_masks, n_flavors, nwords_amp)

        model = matrix_element.get('processes')[0].get('model')
        pdg_to_group_pos, max_group_size = self._build_flavor_group_lookup(model)
        flav_table_flat = []
        for flavor in runtime_flavors:
            for p in flavor:
                flav_table_flat.append(self._map_flavor_to_group_pos(
                    p, pdg_to_group_pos, max_group_size))

        decl_block = self._format_flavor_mask_decl(
            n_flavors, n_wfs, n_amps, nwords_wf, nwords_amp,
            wf_index_masks, amp_index_masks,
            active_wf_index_masks, active_amp_index_masks, flav_table_flat)
        setup_block = self._format_flavor_mask_setup()

        return (decl_block, setup_block, n_flavors, active_flavor_mask)

    #===========================================================================
    # write_pdf_opendata
    #===========================================================================
    def write_pdf_opendata(self):
        """ modify the pdf opendata file, to allow direct access to cluster node
        repository if configure"""
        
        if not self.opt["cluster_local_path"]:
            changer = {"pdf_systemwide": ""}
        else: 
            to_add = """
            tempname='%(path)s'//Tablefile
            open(IU,file=tempname,status='old',ERR=1)
            return
 1          tempname='%(path)s/Pdfdata/'//Tablefile
            open(IU,file=tempname,status='old',ERR=2)
            return
 2          tempname='%(path)s/lhapdf'//Tablefile
            open(IU,file=tempname,status='old',ERR=3)
            return            
 3          tempname='%(path)s/../lhapdf/pdfsets/'//Tablefile
            open(IU,file=tempname,status='old',ERR=4)
            return              
 4          tempname='%(path)s/../lhapdf/pdfsets/6.1/'//Tablefile
            open(IU,file=tempname,status='old',ERR=5)
            return  
            """ % {"path" : self.opt["cluster_local_path"]}
            
            changer = {"pdf_systemwide": to_add}


        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "opendata.f"))        
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_opendata.f"),"r").read()
        ff.writelines(template % changer)

        # Do the same for lhapdf set
        if not self.opt["cluster_local_path"]:
            changer = {"cluster_specific_path": ""}
        else:
            to_add="""
         LHAPath='%(path)s/PDFsets'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return        
         LHAPath='%(path)s/../lhapdf/pdfsets/6.1/'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return
         LHAPath='%(path)s/../lhapdf/pdfsets/'
         Inquire(File=LHAPath, exist=exists)
         if(exists)return  
         LHAPath='./PDFsets'            
         """ % {"path" : self.opt["cluster_local_path"]}
            changer = {"cluster_specific_path": to_add}

        # this is for LHAPDF
        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"))        
        #ff = open(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"),"w")
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_wrap_lhapdf.f"),"r").read()
    
        NLO = isinstance(self, madgraph.iolibs.export_fks.ProcessExporterFortranFKS)
        ff.writelines(template % changer, {'LO': not NLO})

        # this is for eMELA
        ff = writers.FortranWriter(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_emela.f"))        
        #ff = open(pjoin(self.dir_path, "Source", "PDF", "pdfwrap_lhapdf.f"),"w")
        template = open(pjoin(MG5DIR, "madgraph", "iolibs", "template_files", "pdf_wrap_emela.f"),"r").read()
        ff.writelines(template % changer)
        
        
        return



    #===========================================================================
    # write_maxparticles_file
    #===========================================================================
    def write_maxparticles_file(self, writer, matrix_elements):
        """Write the maxparticles.inc file for MadEvent"""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            maxparticles = max([me.get_nexternal_ninitial()[0] for me in \
                              matrix_elements.get('matrix_elements')])
        else:
            maxparticles = max([me.get_nexternal_ninitial()[0] \
                              for me in matrix_elements])

        lines = "integer max_particles\n"
        lines += "parameter(max_particles=%d)" % maxparticles

        # Write the file
        writer.writelines(lines)

        return True

    
    #===========================================================================
    # export the model
    #===========================================================================
    def export_model_files(self, model_path):
        """Configure the files/link of the process according to the model"""

        # Import the model
        for file in os.listdir(model_path):
            if os.path.isfile(pjoin(model_path, file)):
                shutil.copy2(pjoin(model_path, file), \
                                     pjoin(self.dir_path, 'Source', 'MODEL'))

        # add file for EWA 
        template = open(pjoin(MG5DIR,'madgraph/iolibs/template_files/madevent_electroweakFlux.inc')).read()
        fsock = open(pjoin(self.dir_path, 'Source', 'ElectroweakFlux.inc'),'w')
        fsock.write(template % {'MW': 'wmass','MZ':'zmass'})                 
        fsock.close() 
        ln(pjoin(self.dir_path, 'Source', 'ElectroweakFlux.inc'), self.dir_path + '/Source/PDF')


    def make_model_symbolic_link(self):
        """Make the copy/symbolic links"""
        model_path = self.dir_path + '/Source/MODEL/'
        if os.path.exists(pjoin(model_path, 'ident_card.dat')):
            mv(model_path + '/ident_card.dat', self.dir_path + '/Cards')
        if os.path.exists(pjoin(model_path, 'particles.dat')):
            ln(model_path + '/particles.dat', self.dir_path + '/SubProcesses')
            ln(model_path + '/interactions.dat', self.dir_path + '/SubProcesses')
        cp(model_path + '/param_card.dat', self.dir_path + '/Cards')
        mv(model_path + '/param_card.dat', self.dir_path + '/Cards/param_card_default.dat')
        ln(model_path + '/coupl.inc', self.dir_path + '/Source')
        ln(model_path + '/coupl.inc', self.dir_path + '/SubProcesses')
        self.make_source_links()
        
    def make_source_links(self):
        """ Create the links from the files in sources """

        ln(self.dir_path + '/Source/run.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxparticles.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/run_config.inc', self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/lhe_event_infos.inc', self.dir_path + '/SubProcesses', log=False)
        

    #===========================================================================
    # export the helas routine
    #===========================================================================
    def export_helas(self, helas_path):
        """Configure the files/link of the process according to the model"""

        # Import helas routine
        for filename in os.listdir(helas_path):
            filepos = pjoin(helas_path, filename)
            if os.path.isfile(filepos):
                if filepos.endswith('Makefile.template'):
                    cp(filepos, self.dir_path + '/Source/DHELAS/Makefile')
                elif filepos.endswith('Makefile'):
                    pass
                else:
                    cp(filepos, self.dir_path + '/Source/DHELAS')
    # following lines do the same but whithout symbolic link
    # 
    #def export_helas(mgme_dir, dir_path):
    #
    #        # Copy the HELAS directory
    #        helas_dir = pjoin(mgme_dir, 'HELAS')
    #        for filename in os.listdir(helas_dir): 
    #            if os.path.isfile(pjoin(helas_dir, filename)):
    #                shutil.copy2(pjoin(helas_dir, filename),
    #                            pjoin(dir_path, 'Source', 'DHELAS'))
    #        shutil.move(pjoin(dir_path, 'Source', 'DHELAS', 'Makefile.template'),
    #                    pjoin(dir_path, 'Source', 'DHELAS', 'Makefile'))
    #  

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,
                                         me_number, **opt):
        """Routine to generate a subprocess directory (for inheritance)"""

        pass

    #===========================================================================
    # get_source_libraries_list
    #===========================================================================
    def get_source_libraries_list(self):
        """ Returns the list of libraries to be compiling when compiling the
        SOURCE directory. It is different for loop_induced processes and 
        also depends on the value of the 'output_dependencies' option"""
        
        return ['$(LIBDIR)libmodel.$(libext)',
                '$(LIBDIR)libdhelas.$(libext)',
                '$(LIBDIR)libpdf.$(libext)',
                '$(LIBDIR)libgammaUPC.$(libext)',
                '$(LIBDIR)libcernlib.$(libext)',
                '$(LIBDIR)libbias.$(libext)']

    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model=None):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = ' '.join(self.get_source_libraries_list()+['$(LIBRARIES)'])
        if self.opt['model'] == 'mssm' or self.opt['model'].startswith('mssm-'):
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc vector.inc\n\tcd MODEL; make
MODEL/MG5_param.dat: ../Cards/param_card.dat\n\t../bin/madevent treatcards param
param_card.inc: MODEL/MG5_param.dat\n\t../bin/madevent treatcards param\n'''
        else:
            model_line='''$(LIBDIR)libmodel.$(libext): MODEL param_card.inc vector.inc\n\tcd MODEL; make    
param_card.inc: ../Cards/param_card.dat\n\t../bin/madevent treatcards param\n'''
        
        replace_dict= {'libraries': set_of_lib, 
                       'model':model_line,
                       'additional_dsample': '',
                       'additional_dependencies':'',
                       'additional_clean':'',
                       'running': ''} 

        if self.opt['running']:
            replace_dict['running'] ="  $(LIBDIR)librunning.$(libext): RUNNING\n\tcd RUNNING; make"
            replace_dict['libraries'] += " $(LIBDIR)librunning.$(libext) "
        
        if writer:
            text = open(path).read() % replace_dict
            writer.write(text)
            
        return replace_dict

    #===========================================================================
    # write_nexternal_madspin
    #===========================================================================
    def write_nexternal_madspin(self, writer, nexternal, ninitial):
        """Write the nexternal_prod.inc file for madspin"""

        replace_dict = {'flavor_mask_decl':'',
                        'flavor_mask_setup':''}

        replace_dict['nexternal'] = nexternal
        replace_dict['ninitial'] = ninitial

        file = """ \
          integer    nexternal_prod
          parameter (nexternal_prod=%(nexternal)d)
          integer    nincoming_prod
          parameter (nincoming_prod=%(ninitial)d)""" % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict

    #===========================================================================
    # write_helamp_madspin
    #===========================================================================
    def write_helamp_madspin(self, writer, ncomb):
        """Write the helamp.inc file for madspin"""

        replace_dict = {}

        replace_dict['ncomb'] = ncomb

        file = """ \
          integer    ncomb1
          parameter (ncomb1=%(ncomb)d)
          double precision helamp(ncomb1)    
          common /to_helamp/helamp """ % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict



    #===========================================================================
    # write_nexternal_file
    #===========================================================================
    def write_nexternal_file(self, writer, nexternal, ninitial):
        """Write the nexternal.inc file for MG4"""

        replace_dict = {}

        replace_dict['nexternal'] = nexternal
        replace_dict['ninitial'] = ninitial

        file = """ \
          integer    nexternal
          parameter (nexternal=%(nexternal)d)
          integer    nincoming
          parameter (nincoming=%(ninitial)d)""" % replace_dict

        # Write the file
        if writer:
            writer.writelines(file)
            return True
        else:
            return replace_dict
    #===========================================================================
    # write_pmass_file
    #===========================================================================
    def write_pmass_file(self, writer, matrix_element):
        """Write the pmass.inc file for MG4"""

        model = matrix_element.get('processes')[0].get('model')
        
        lines = []
        for wf in matrix_element.get_external_wavefunctions():
            mass = model.get('particle_dict')[wf.get('pdg_code')].get('mass')
            if mass.lower() != "zero":
                mass = "abs(%s)" % mass

            lines.append("pmass(%d)=%s" % \
                         (wf.get('number_external'), mass))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_ngraphs_file
    #===========================================================================
    def write_ngraphs_file(self, writer, nconfigs):
        """Write the ngraphs.inc file for MG4. Needs input from
        write_configs_file."""

        file = "       integer    n_max_cg\n"
        file = file + "parameter (n_max_cg=%d)" % nconfigs

        # Write the file
        writer.writelines(file)

        return True

    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, matrix_element):
        """Write the leshouche.inc file for MG4"""

        # Write the file
        writer.writelines(self.get_leshouche_lines(matrix_element, 0))

        return True

    #===========================================================================
    # get_leshouche_lines
    #===========================================================================
    def get_leshouche_lines(self, matrix_element, numproc, drop_icolup=False):
        """Write the leshouche.inc file for MG4

        With *drop_icolup* the ICOLUP table is omitted for any ME whose colour
        flows have a canonical code: the consumer (addmothers) rebuilds the tags
        from colorflow.inc instead, so the table would be dead weight. It is
        still written for an ME without a usable code, which is what addmothers
        falls back to. Only the madevent exporters set this -- MadWeight ships
        no addmothers.f and keeps reading ICOLUP."""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        if drop_icolup and self._color_code_tables(matrix_element):
            drop_icolup = True
        else:
            drop_icolup = False

        lines = []
        real_iproc = -1
        # Both sources of flavor multiplicity (several processes mapped onto one
        # matrix element, and merged legs within a process) are enumerated by
        # HelasMatrixElement.get_flavor_pdg_combinations, shared with the mg7
        # exporter so the two backends cannot drift apart.
        processes = matrix_element.get('processes')
        for iproc, (pdg_lists, has_merged_particles) in enumerate(
                matrix_element.get_flavor_pdg_combinations(self.model)):
            proc = processes[iproc]
            legs = proc.get_legs_with_decays()
            real_iproc += 1
            if has_merged_particles:
                for ids in pdg_lists:
                    lines.append("DATA (IDUP(i,%d,%d),i=1,%d)/%s/" % \
                         (real_iproc + 1, numproc+1, nexternal,
                          ",".join([str(id) for id in ids])))
                    real_iproc += 1
            else:
                lines.append("DATA (IDUP(i,%d,%d),i=1,%d)/%s/" % \
                         (real_iproc + 1, numproc+1, nexternal,
                          ",".join([str(id) for id in pdg_lists[0]])))

            if iproc == 0 and numproc == 0:
                for i in [1, 2]:
                    lines.append("DATA (MOTHUP(%d,i),i=1,%2r)/%s/" % \
                             (i, nexternal,
                              ",".join([ "%3r" % 0 ] * ninitial + \
                                       [ "%3r" % i ] * (nexternal - ninitial))))

            # Here goes the color connections corresponding to the JAMPs
            # Only one output, for the first subproc!
            if iproc == 0 and not drop_icolup:
                # If no color basis, just output trivial color flow
                if not matrix_element.get('color_basis'):
                    for i in [1, 2]:
                        lines.append("DATA (ICOLUP(%d,i,1,%d),i=1,%2r)/%s/" % \
                                 (i, numproc+1,nexternal,
                                  ",".join([ "%3r" % 0 ] * nexternal)))

                else:
                    # First build a color representation dictionnary
                    repr_dict = {}
                    for l in legs:
                        repr_dict[l.get('number')] = \
                            proc.get('model').get_particle(l.get('id')).get_color()\
                            * (-1)**(1+l.get('state'))
                    # Get the list of color flows
                    color_flow_list = \
                        matrix_element.get('color_basis').get_flow_basis().\
                                  color_flow_decomposition(repr_dict, ninitial)
                    # And output them properly
                    for cf_i, color_flow_dict in enumerate(color_flow_list):
                        for i in [0, 1]:
                            lines.append("DATA (ICOLUP(%d,i,%d,%d),i=1,%2r)/%s/" % \
                                 (i + 1, cf_i + 1, numproc+1, nexternal,
                                  ",".join(["%3r" % color_flow_dict[l.get('number')][i] \
                                            for l in legs])))

        return lines




    #===========================================================================
    # write_maxamps_file
    #===========================================================================
    def write_maxamps_file(self, writer, maxamps, maxflows, max_flav_per_proc,
                           maxproc,maxsproc):
        """Write the maxamps.inc file for MG4."""

        file = "       integer    maxamps, maxflow, maxproc, maxsproc, maxflavperproc\n"
        file = file + "parameter (maxamps=%d, maxflow=%d)\n" % \
               (maxamps, maxflows)
        file = file + "parameter (maxproc=%d, maxsproc=%d)\n" % \
               (maxproc, maxsproc)
        file += "parameter (maxflavperproc=%d)" % max_flav_per_proc

        # Write the file
        writer.writelines(file)

        return True



        raise Exception("This function is deprecated. maxamps.inc is no longer used in MG5_aMC.")
        file = "       integer    maxamps, maxflow, maxproc, maxsproc\n"
        file = file + "parameter (maxamps=%d, maxflow=%d)\n" % \
               (maxamps, maxflows)
        file = file + "parameter (maxproc=%d, maxsproc=%d)" % \
               (maxproc, maxsproc)

        # Write the file
        writer.writelines(file)

        return True


    #===========================================================================
    # Routines to output UFO models in MG4 format
    #===========================================================================

    def convert_model(self, model, wanted_lorentz = [],
                             wanted_couplings = []):
        """ Create a full valid MG4 model from a MG5 model (coming from UFO)"""

        # Make sure aloha is in quadruple precision if needed
        old_aloha_mp=aloha.mp_precision
        aloha.mp_precision=self.opt['mp']
        self.model = model
        # create the MODEL
        write_dir=pjoin(self.dir_path, 'Source', 'MODEL')
        self.opt['exporter'] = self.__class__
        if 'vector_size' in self.opt:
            self.opt['output_options']['vector_size'] = self.opt['vector_size']
        if 'vector_size' not in self.opt['output_options']:
            self.opt['output_options']['vector_size'] = self.default_vector_size

        model_builder = UFO_model_to_mg4(model, write_dir, self.opt + self.proc_characteristic)
        model_builder.build(wanted_couplings)

        # Backup the loop mode, because it can be changed in what follows.
        old_loop_mode = aloha.loop_mode

        # Create the aloha model or use the existing one (for loop exporters
        # this is useful as the aloha model will be used again in the 
        # LoopHelasMatrixElements generated). We do not save the model generated
        # here if it didn't exist already because it would be a waste of
        # memory for tree level applications since aloha is only needed at the
        # time of creating the aloha fortran subroutines.
        if hasattr(self, 'aloha_model'):
            aloha_model = self.aloha_model
        else:
            try:
                with misc.MuteLogger(['madgraph.models'], [60]):
                    aloha_model = create_aloha.AbstractALOHAModel(os.path.basename(model.get('modelpath')))
            except (ImportError, UFOError):
                aloha_model = create_aloha.AbstractALOHAModel(model.get('modelpath'))
        aloha_model.add_Lorentz_object(model.get('lorentz'))

        # Compute the subroutines
        if wanted_lorentz:
            aloha_model.compute_subset(wanted_lorentz)
        else:
            aloha_model.compute_all(save=False)

        # Write them out
        write_dir=pjoin(self.dir_path, 'Source', 'DHELAS')
        options= {}
        options['vector.inc'] = True if self.opt['export_format']=='madevent' else False
        aloha_model.write(write_dir, 'Fortran', options=options)

        # Revert the original aloha loop mode
        aloha.loop_mode = old_loop_mode

        #copy Helas Template
        cp(MG5DIR + '/aloha/template_files/Makefile_F', write_dir+'/makefile')
        if any([any([tag.startswith('L') for tag in d[1]]) for d in wanted_lorentz]):
            cp(MG5DIR + '/aloha/template_files/aloha_functions_loop.f',
                                                 write_dir+'/aloha_functions.f')
            aloha_model.loop_mode = False
        else:
            if aloha.unitary_gauge !=3:
                cp(MG5DIR + '/aloha/template_files/aloha_functions.f',
                                                 write_dir+'/aloha_functions.f')
            else:
                cp(MG5DIR + '/aloha/template_files/aloha_functions_fd.f',
                                                 write_dir+'/aloha_functions.f')

        # For models with tensor (spin-2) or Rarita-Schwinger (spin-3/2)
        # particles, ALOHA generates routines whose tensor parameter is
        # TYPE(ALOHA2D) (W(16)), but the caller's wavefunction array
        # stores all slots as TYPE(ALOHA). Make TYPE(ALOHA) share the
        # TYPE(ALOHA2D) memory layout *only in this model's DHELAS copy*
        # so the tensor routine no longer overruns its slot. Models
        # without tensors keep the compact TYPE(ALOHA) %W(4) layout.
        if self.model and any(p.get('spin') in [4, 5]
                              for p in self.model.get('particles') if p):
            self._widen_aloha_type(pjoin(write_dir, 'aloha_functions.f'))

        create_aloha.write_aloha_file_inc(write_dir, '.f', '.o')

        # Make final link in the Process
        self.make_model_symbolic_link()
    
        # Re-establish original aloha mode
        aloha.mp_precision=old_aloha_mp
    

    @staticmethod
    def _widen_aloha_type(aloha_file):
        """Extend TYPE(ALOHA) %W to 16 complex (matching TYPE(ALOHA2D))
        so a uniform TYPE(ALOHA) wavefunction array can safely hold
        tensor wavefunctions written by TXXXXX/VVT2_*. Only used for
        models that contain spin-2 or spin-3/2 particles.

        Operates in place on the just-copied aloha_functions.f, so the
        rest of the install (and other models) keep the compact %W(4)
        layout for free."""
        import re
        with open(aloha_file) as fh:
            text = fh.read()
        # Patch the TYPE ALOHA block (and its MP_ALOHA mirror) to use
        # the same %W size as the TYPE ALOHA2D block. Touch only the
        # %W declaration; leave %P / %flv_index alone.
        def _bump(match):
            block = match.group(0)
            block = re.sub(r'double complex\s*::\s*W\(\d+\)',
                           'double complex::W(16)', block)
            block = re.sub(r'complex\*32\s*::\s*W\(\d+\)',
                           'complex*32 :: W(16)', block)
            return block
        text = re.sub(
            r'TYPE\s+ALOHA\s*\n.*?END\s+TYPE\s+ALOHA\s*\n',
            _bump, text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(
            r'TYPE\s+MP_ALOHA\s*\n.*?END\s+TYPE\s+MP_ALOHA\s*\n',
            _bump, text, flags=re.IGNORECASE | re.DOTALL)
        with open(aloha_file, 'w') as fh:
            fh.write(text)

    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""

        return False, matrix_element
        
    #===========================================================================
    # Helper functions
    #===========================================================================
    def get_mg5_info_lines(self):
        """Return info lines for MG5, suitable to place at beginning of
        Fortran files"""

        info = misc.get_pkg_info()
        info_lines = ""
        if info and 'version' in info and  'date' in info:
            info_lines = "#  Generated by MadGraph5_aMC@NLO v. %s, %s\n" % \
                         (info['version'], info['date'])
            info_lines = info_lines + \
                         "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"
        else:
            info_lines = "#  Generated by MadGraph5_aMC@NLO\n" + \
                         "#  By the MadGraph5_aMC@NLO Development Team\n" + \
                         "#  Visit launchpad.net/madgraph5 and amcatnlo.web.cern.ch"        

        return info_lines

    def get_process_info_lines(self, matrix_element):
        """Return info lines describing the processes for this matrix element"""

        return"\n".join([ "C " + process.nice_string().replace('\n', '\nC * ') \
                         for process in matrix_element.get('processes')])


    def get_helicity_lines(self, matrix_element,array_name='NHEL', add_nb_comb=False):
        """Return the Helicity matrix definition lines for this matrix element"""

        #misc.sprint(matrix_element.get_external_nhel())

        helicity_line_list = []
        i = 0            
        if add_nb_comb:
            spins = matrix_element.get_spin_state()
            spins.insert(0, len(spins))
            helicity_line_list.append(\
                ("DATA ("+array_name+"(I,0),I=1,%d) /" + \
                 ",".join(['%2r'] * (len(spins)-1)) + "/") % tuple(spins))
            
        for helicities in matrix_element.get_helicity_matrix():
            i = i + 1
            int_list = [i, len(helicities)]
            int_list.extend(helicities)
            helicity_line_list.append(\
                ("DATA ("+array_name+"(I,%4r),I=1,%d) /" + \
                 ",".join(['%2r'] * len(helicities)) + "/") % tuple(int_list))

        return "\n".join(helicity_line_list)

    @staticmethod
    def _fortran_data_stmt(name, values, per_line=10):
        """Emit a fixed-form 'DATA name /v1,v2,.../' statement with
        continuation lines (column-6 '&') so long value lists stay within the
        Fortran line-length limit. name may contain an implied-DO, e.g.
        '(STATES(I,1),I=1,3)'."""
        strs = ["%d" % v for v in values]
        if len(strs) <= per_line:
            return "      DATA %s /%s/" % (name, ",".join(strs))
        lines = ["      DATA %s /" % name]
        for i in range(0, len(strs), per_line):
            seg = strs[i:i + per_line]
            tail = "," if i + per_line < len(strs) else "/"
            lines.append("     & %s%s" % (",".join(seg), tail))
        return "\n".join(lines)

    def _helstate_data(self, matrix_element):
        """Return the Fortran DATA blocks for the canonical helicity
        encoder/decoder that replaces the explicit NHEL config table.

        A helicity configuration is encoded as a single mixed-radix integer
        (the 'canonical code') over the per-leg helicity states, with the last
        external leg as the least-significant digit -- matching the
        itertools.product ordering used by get_helicity_matrix(). For a
        non-polarized process this makes the code of the i-th row exactly i, so
        HELALLOW is simply [1..NCOMB] and nothing is relabelled; a polarization
        restriction ({0}/{L}/...) keeps the *full* per-leg multiplicity as the
        radix (so helicity 0 / longitudinal stays a first-class state) and
        leaves HELALLOW as the selected, non-contiguous subset of codes.

        Returns a dict with keys:
          maxhel          - max per-leg helicity multiplicity (STATES 1st dim)
          nhstate_data    - DATA for NHSTATE(NEXTERNAL)   (states per leg)
          states_data     - DATA for STATES(MAXHEL,NEXTERNAL) (helicity values)
          hel_allow_data  - DATA for HELALLOW(NCOMB)      (allowed codes)
        """
        model = matrix_element.get('processes')[0].get('model')
        pdict = model.get('particle_dict')
        ext = matrix_element.get_external_wavefunctions()
        # Full per-leg helicity states, allow_reverse=True so the value order
        # matches get_helicity_matrix() for non-polarized legs (code==row).
        states = [pdict[wf.get('pdg_code')].get_helicity_states(True)
                  for wf in ext]
        nstate = [len(s) for s in states]
        nexternal = len(ext)
        maxhel = max(nstate) if nstate else 1

        # Allowed canonical codes: encode each enumerated helicity row.
        allowed = []
        for row in matrix_element.get_helicity_matrix():
            code = 0
            for k, val in enumerate(row):
                code = code * nstate[k] + states[k].index(val)
            allowed.append(code + 1)

        states_lines = []
        for k in range(nexternal):
            vals = [states[k][i] if i < nstate[k] else 0
                    for i in range(maxhel)]
            states_lines.append(self._fortran_data_stmt(
                '(STATES(I,%d),I=1,%d)' % (k + 1, maxhel), vals))

        return {'maxhel': maxhel,
                'nhstate_data': self._fortran_data_stmt('NHSTATE', nstate),
                'states_data': "\n".join(states_lines),
                'hel_allow_data': self._fortran_data_stmt('HELALLOW', allowed)}

    def get_ic_line(self, matrix_element):
        """Return the IC definition line coming after helicities, required by
        switchmom in madevent"""

        nexternal = matrix_element.get_nexternal_ninitial()[0]
        int_list = list(range(1, nexternal + 1))

        return "DATA (IC(I,1),I=1,%i) /%s/" % (nexternal,
                                                     ",".join([str(i) for \
                                                               i in int_list]))

    def set_chosen_SO_index(self, process, squared_orders):
        """ From the squared order constraints set by the user, this function
        finds what indices of the squared_orders list the user intends to pick.
        It returns this as a string of comma-separated successive '.true.' or 
        '.false.' for each index."""
        
        user_squared_orders = process.get('squared_orders')
        split_orders = process.get('split_orders')
        
        if len(user_squared_orders)==0:
            return ','.join(['.true.']*len(squared_orders))
        
        res = []
        for sqsos in squared_orders:
            is_a_match = True
            for user_sqso, value in user_squared_orders.items():
                if user_sqso == 'WEIGHTED' :
                    logger.debug('WEIGHTED^2%s%s encoutered. Please check behavior for' + \
                            'https://bazaar.launchpad.net/~maddevelopers/mg5amcnlo/3.0.1/revision/613', \
                            (process.get_squared_order_type(user_sqso), sqsos[split_orders.index(user_sqso)]))
                if user_sqso not in split_orders:
                    is_a_match = False
                elif (process.get_squared_order_type(user_sqso) =='==' and \
                        value!=sqsos[split_orders.index(user_sqso)]) or \
                   (process.get_squared_order_type(user_sqso) in ['<=','='] and \
                                value<sqsos[split_orders.index(user_sqso)]) or \
                   (process.get_squared_order_type(user_sqso) == '>' and \
                                value>=sqsos[split_orders.index(user_sqso)]):
                    is_a_match = False
                    break
            res.append('.true.' if is_a_match else '.false.')
            
        return ','.join(res)

    def get_split_orders_lines(self, orders, array_name, n=5):
        """ Return the split orders definition as defined in the list orders and
        for the name of the array 'array_name'. Split rows in chunks of size n."""
        
        ret_list = []  
        for index, order in enumerate(orders):      
            for k in range(0, len(order), n):
                ret_list.append("DATA (%s(%3r,i),i=%3r,%3r) /%s/" % \
                  (array_name,index + 1, k + 1, min(k + n, len(order)),
                              ','.join(["%5r" % i for i in order[k:k + n]])))
        return ret_list
    
    def format_integer_list(self, list, name, n=5):
        """ Return an initialization of the python list in argument following 
        the fortran syntax using the data keyword assignment, filling an array 
        of name 'name'. It splits rows in chunks of size n."""
        
        ret_list = []
        for k in range(0, len(list), n):
            ret_list.append("DATA (%s(i),i=%3r,%3r) /%s/" % \
                  (name, k + 1, min(k + n, len(list)),
                                  ','.join(["%5r" % i for i in list[k:k + n]])))
        return ret_list

#    def write_namelist_file(self, matrix_element, dirpath):
#
#        fsock = open(pjoin(dirpath, 'namelist.def'), 'w')
#
#        fsock.write(' &NM_CF\n')
#        if not matrix_element.get('color_matrix'):
#            fsock.write('  CF = 1\n')
#        else:
#            cf = []
#            for index, denominator in \
#                enumerate(matrix_element.get('color_matrix').\
#                                                 get_line_denominators()): 
#                num_list = matrix_element.get('color_matrix').\
#                                            get_line_numerators(index, denominator)
#                num_list[index] /= 2
#                cf += [str(int(2*coeff)) for coeff in num_list[index:]]
#            fsock.write('  CF = %s\n' % (','.join(cf))) 
#            fsock.write(' /\n')



    @staticmethod
    def jamp_fold_spanning_tree(permutations, size):
        """Same walk as ColorBasisSymmetry.spanning_tree, over an index set
        given directly as permutations rather than as color basis keys."""

        keep = []
        parent_uf = list(range(size))

        def find(x):
            while parent_uf[x] != x:
                parent_uf[x] = parent_uf[parent_uf[x]]
                x = parent_uf[x]
            return x

        for perm in permutations:
            used = False
            for i, j in enumerate(perm):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent_uf[ri] = rj
                    used = True
            if used:
                keep.append(perm)

        representative = [-1] * size
        parent = [None] * size
        representatives = []
        for start in range(size):
            if representative[start] != -1:
                continue
            representatives.append(start)
            representative[start] = start
            queue = collections.deque([start])
            while queue:
                current = queue.popleft()
                for local, perm in enumerate(keep):
                    image = perm[current]
                    if representative[image] == -1:
                        representative[image] = start
                        parent[image] = (current, local)
                        queue.append(image)
        return representatives, representative, parent, keep

    def get_color_matrix_encoding(self, matrix_element):
        """Describe the color matrix by one line per orbit of the index
        permutations leaving the color basis invariant, plus the permutations
        needed to reach every other line from it (see ColorBasisSymmetry).

        Every line of the matrix is one of those lines with its columns
        permuted, so this replaces the N*(N+1)/2 entries by (nrep+ngen+3)*N
        numbers. That is only a gain once the basis is large enough, and None
        is returned otherwise so that the entries are written out as before."""

        color_matrix = matrix_element.get('color_matrix')
        if not color_matrix:
            return None
        # an asymmetric matrix does not have the line structure exploited here
        if color_matrix._col_basis1 is not color_matrix._col_basis2:
            return None

        keys = color_matrix._sorted_keys1
        nb_color = len(keys)
        symmetry = color_amp.ColorBasisSymmetry(keys)
        if not symmetry.has_symmetry():
            return None

        folding = self.get_jamp_folding(matrix_element)
        if folding and folding['sign'] < 0:
            # The rebuilt form cannot carry the weight a permutation picks up
            # when it sends a line onto its own partner. The sum runs over the
            # folded matrix either way, so there is no falling back to the
            # unfolded encoding here: it has to be written out instead.
            return None
        if folding:
            # reversing commutes with permuting the indices, so a permutation
            # of the lines is also a permutation of the pairs
            slot = folding['slot']
            nb_color = len(folding['representatives'])
            induced = [[slot[perm[line]]
                        for line in folding['representatives']]
                       for perm in symmetry.generators1]
            representatives, representative, parent, gens = \
                    self.jamp_fold_spanning_tree(induced, nb_color)
        else:
            representatives, representative, parent, gens = \
                                                    symmetry.spanning_tree()

        # Writing the entries out is well trodden and the compressed form
        # carries a routine of its own, so only take it when it pays clearly.
        # In practice this leaves everything below about a hundred color
        # structures alone, which is where the matrix is not the bulk of the
        # generated file anyway.
        size = (len(representatives) + len(gens) + 3) * nb_color
        if size * self.color_encoding_margin > nb_color * (nb_color + 1) // 2:
            return None

        place = dict((line, index) for index, line in enumerate(representatives))
        if folding:
            denominator, folded = self.jamp_folded_color_matrix(
                        matrix_element, folding['reverse'], folding['sign'])
            rows = [folded[line] for line in representatives]
        else:
            denominator = max(color_matrix.get_line_denominators())
            rows = []
            for line in representatives:
                num_list = color_matrix.get_line_numerators(line, denominator)
                assert all(int(i) == i for i in num_list)
                rows.append([int(i) for i in num_list])

        return {'denom': denominator,
                'nb_color': nb_color,
                'rows': rows,
                'gens': gens,
                # for each line, the line it comes from and the generator
                # reaching it, or (0,0) when the line is a representative
                'parent': [(0, 0) if p is None else (p[0] + 1, p[1] + 1)
                           for p in parent],
                'slot': [place[representative[i]] + 1
                         for i in range(nb_color)]}

    def get_color_data_lines(self, matrix_element, n=128):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return ["DATA %(proc_prefix)sDenom/1/", "DATA %(proc_prefix)sCF/1/"]

        if self.get_color_matrix_encoding(matrix_element):
            # the entries are rebuilt at run time by INIT_CF, only the overall
            # denominator is still needed here
            denominator = max(matrix_element.get('color_matrix').\
                                                    get_line_denominators())
            return ["DATA %%(proc_prefix)sDenom/%(denom)i/" % \
                                                       {'denom': denominator}]

        folding = self.get_jamp_folding(matrix_element)
        if folding:
            denominator, folded = self.jamp_folded_color_matrix(
                        matrix_element, folding['reverse'], folding['sign'])
            ret_list = ["DATA %%(proc_prefix)sDenom/%(denom)i/" %
                        {'denom': denominator}]
            cf_index = 0
            for index in range(len(folded)):
                row = folded[index]
                for k in range(index, len(row), n):
                    chunk = row[k:k + n]
                    ret_list.append(
                        "DATA (%%(proc_prefix)sCF(i),i=%3r,%3r) /%s/" %
                        (cf_index + 1, cf_index + len(chunk),
                         ','.join("%i" % ((1 if (k == index and pos == 0)
                                           else 2) * int(v))
                                  for pos, v in enumerate(chunk))))
                    cf_index += len(chunk)
            return ret_list

        ret_list = []
        my_cs = color.ColorString()
        denominator = max(matrix_element.get('color_matrix').get_line_denominators())
        ret_list.append("DATA %%(proc_prefix)sDenom/%(denom)i/" % {'denom':denominator})

        cf_index = 0
        col_basis = matrix_element.get('color_matrix')._col_basis1
        is_asym = matrix_element.get('color_matrix')._col_basis1 is not matrix_element.get('color_matrix')._col_basis2
        for index in range(len(col_basis)):
            num_list = matrix_element.get('color_matrix').get_line_numerators(index, denominator)
            assert all(int(i) == i for i in num_list)
            if is_asym:
                min_k = 0
            else:
                min_k = index # only include the upper diagonal
            for k in range(min_k, len(num_list), n):
                chunk = num_list[k:k+n]
                if is_asym:
                    ret_list.append("DATA (%%(proc_prefix)sCF(i,%3r),i=%3r,%3r) /%s/" % \
                                    (index+1, k + 1, k+len(chunk),
                                     ','.join([("%i" % (int(i))) for i in chunk])))  
                else: 
                    ret_list.append("DATA (%%(proc_prefix)sCF(i),i=%3r,%3r) /%s/" % \
                                    (cf_index+1, cf_index + len(chunk),
                                     ','.join([("%i" % ((1 if (k==index and pos==0) else 2)*int(i))) for pos,i in enumerate(chunk)])))
                cf_index += len(chunk)

            my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[index])
            ret_list.append("C %s" % repr(my_cs))

        return ret_list

    @staticmethod
    def get_int_data_lines(name, values, n=128, var='i'):
        """DATA statements filling the one dimensional integer array name."""

        lines = []
        for start in range(0, len(values), n):
            chunk = values[start:start + n]
            lines.append("      DATA (%s(%s),%s=%d,%d) /%s/" % \
                         (name, var, var, start + 1, start + len(chunk),
                          ','.join(str(int(v)) for v in chunk)))
        return lines

    def get_color_init_routine(self, matrix_element, proc_prefix,
                               suffix=''):
        """Fortran source rebuilding the color matrix from its compressed
        description, or an empty routine when the entries are written out."""

        encoding = self.get_color_matrix_encoding(matrix_element)
        nb_color = encoding['nb_color'] if encoding else \
                   (len(matrix_element.get('color_matrix')._sorted_keys1)
                    if matrix_element.get('color_matrix') else 0)
        header = ["      SUBROUTINE %sINIT_CF%s()" % (proc_prefix, suffix)]
        if not encoding:
            return header + ["      RETURN", "      END"]

        nb_rep = len(encoding['rows'])
        nb_gen = len(encoding['gens'])
        body = header + [
            "C     Rebuild the color matrix from one line per",
            "C     orbit of the index permutations leaving the",
            "C     color basis invariant. Every other line is one",
            "C     of those with its columns permuted, which is",
            "C     what following CFPAR back to the representative",
            "C     line gives. Done once, on the first call.",
            "      IMPLICIT NONE",
            "      INTEGER NCOLOR, NCFREP, NCFGEN",
            "      PARAMETER (NCOLOR=%d)" % nb_color,
            "      PARAMETER (NCFREP=%d)" % nb_rep,
            "      PARAMETER (NCFGEN=%d)" % nb_gen,
            "      INTEGER %sCF(NCOLOR*(NCOLOR+1)/2)" % proc_prefix,
            "      INTEGER %sDENOM" % proc_prefix,
            "      COMMON /%scolor_matrix%s/ %sCF,%sDENOM" % \
                            (proc_prefix, suffix, proc_prefix, proc_prefix),
            "      INTEGER CFROW(NCOLOR*NCFREP)",
            "      INTEGER CFGEN(NCOLOR*NCFGEN)",
            "      INTEGER CFPAR(2*NCOLOR)",
            "      INTEGER CFSLOT(NCOLOR)",
            "      INTEGER PERM(NCOLOR)",
            "      INTEGER I,J,NODE,G,CF_INDEX,BASE",
            "      LOGICAL CF_DONE",
            "      DATA CF_DONE/.FALSE./",
            "      SAVE CF_DONE",
        ]
        body += self.get_int_data_lines("CFROW",
                            sum(encoding['rows'], []))
        body += self.get_int_data_lines("CFGEN",
                            sum(([x + 1 for x in g] for g in encoding['gens']),
                                []))
        body += self.get_int_data_lines("CFPAR",
                            sum(([p[0], p[1]] for p in encoding['parent']), []))
        body += self.get_int_data_lines("CFSLOT", encoding['slot'])
        body += [
            "      IF (CF_DONE) RETURN",
            "      CF_DONE = .TRUE.",
            "      CF_INDEX = 0",
            "      DO I = 1, NCOLOR",
            "        DO J = 1, NCOLOR",
            "          PERM(J) = J",
            "        ENDDO",
            "        NODE = I",
            "        DO WHILE (CFPAR(2*NODE-1) .NE. 0)",
            "          G = (CFPAR(2*NODE)-1)*NCOLOR",
            "          DO J = 1, NCOLOR",
            "            PERM(J) = CFGEN(G+PERM(J))",
            "          ENDDO",
            "          NODE = CFPAR(2*NODE-1)",
            "        ENDDO",
            "        BASE = (CFSLOT(NODE)-1)*NCOLOR",
            "        CF_INDEX = CF_INDEX + 1",
            "        %sCF(CF_INDEX) = CFROW(BASE+PERM(I))" % proc_prefix,
            "        DO J = I+1, NCOLOR",
            "          CF_INDEX = CF_INDEX + 1",
            "          %sCF(CF_INDEX) = 2*CFROW(BASE+PERM(J))" % proc_prefix,
            "        ENDDO",
            "      ENDDO",
            "      END",
        ]
        return body

    def get_den_factor_line(self, matrix_element):
        """Return the denominator factor line for this matrix element"""
        return "DATA IDEN/%2r/" % \
               matrix_element.get_denominator_factor()

    def get_flow_jamp_lines(self, projection, JAMP_format, AMP_format):
        """The Kleiss-Kuijf definitions of the trace JAMPs in terms of the DDM
        ones. The common subexpression pass is skipped: the map has only a
        handful of terms per line and its temporaries would collide with the
        ones of the JAMP definitions proper."""

        cmd_options = dict(self.cmd_options)
        self.cmd_options['jamp_optim'] = False
        try:
            lines, nb_temp = self.get_JAMP_lines(projection,
                                                 JAMP_format=JAMP_format,
                                                 AMP_format=AMP_format)
        finally:
            self.cmd_options = cmd_options

        return lines

    def set_color_flow_lines_sa(self, matrix_element, replace_dict, ncolor):
        """Same as set_color_flow_lines, for the standalone template: the JAMPs
        are not split per amplitude order there, and the flow JAMPs get their
        own routine so that they can be timed on their own."""

        prefix = replace_dict['proc_prefix']
        color_basis = matrix_element.get('color_basis')
        flow_basis = color_basis.get_flow_basis() if color_basis else None

        if flow_basis is None or flow_basis is color_basis:
            replace_dict['ncolor_flow'] = ncolor
            replace_dict['jampflow_decl'] = ''
            replace_dict['jampflow_call'] = ''
            replace_dict['jampflow_routine'] = ''
            return ncolor

        ncolor_flow = max(1, len(flow_basis))
        projection = color_basis.get_flow_projection()
        lines = self.get_flow_jamp_lines(projection, JAMP_format="JAMPF(%s)",
                                         AMP_format="JAMP(%s)")

        replace_dict['ncolor_flow'] = ncolor_flow
        replace_dict['jampflow_decl'] = "\n".join([
            "      INTEGER NCOLOR_FLOW",
            "      PARAMETER (NCOLOR_FLOW=%d)" % ncolor_flow,
            "      COMPLEX*16 JAMPF(NCOLOR_FLOW)",
            "      DOUBLE PRECISION %sJAMP2(NCOLOR_FLOW)" % prefix,
            "      COMMON /%sJAMP2_COMMON/ %sJAMP2" % (prefix, prefix)])
        # accumulated exactly like madevent does, so that the work is real
        replace_dict['jampflow_call'] = "\n".join([
            "      CALL %sGET_JAMPF(JAMP,JAMPF)" % prefix,
            "      DO I = 1, NCOLOR_FLOW",
            "        %sJAMP2(I) = %sJAMP2(I)" % (prefix, prefix),
            "     $           + DABS(DBLE(JAMPF(I)*DCONJG(JAMPF(I))))",
            "      ENDDO"])
        replace_dict['jampflow_routine'] = "\n".join([
            "      SUBROUTINE %sGET_JAMPF(JAMP,JAMPF)" % prefix,
            "CF2PY INTENT(OUT) :: JAMPF",
            "CF2PY INTENT(IN) :: JAMP",
            "      IMPLICIT NONE",
            "      INTEGER    NCOLOR, NCOLOR_FLOW",
            "      PARAMETER (NCOLOR=%d)" % ncolor,
            "      PARAMETER (NCOLOR_FLOW=%d)" % ncolor_flow,
            "      COMPLEX*16 IMAG1",
            "      PARAMETER (IMAG1=(0D0,1D0))",
            "      COMPLEX*16 JAMP(NCOLOR), JAMPF(NCOLOR_FLOW)"] + lines +
            ["      END"])

        logger.debug('Color sum on %d DDM structures, color flow on %d trace '
                     'structures (%d Kleiss-Kuijf terms)',
                     ncolor, ncolor_flow,
                     sum(len(row) for row in projection))

        return ncolor_flow

    def set_color_flow_lines(self, matrix_element, replace_dict, ncolor):
        """Fill in replace_dict everything the matrix element template needs to
        know about the color flow basis, and return its size.

        For a fully adjoint (multi-gluon) process the color sum can be done on
        the (n-2)! Del Duca-Dixon-Maltoni basis, but a color flow still has to
        be picked among the (n-1)! trace structures. The trace JAMPs are then
        not built from the amplitudes but obtained from the DDM ones through
        the Kleiss-Kuijf relations, which is (n-1) times cheaper."""

        color_basis = matrix_element.get('color_basis')
        flow_basis = color_basis.get_flow_basis() if color_basis else None

        if flow_basis is None or flow_basis is color_basis:
            replace_dict['ncolor_flow'] = ncolor
            replace_dict['jampflow_decl'] = ''
            replace_dict['jampflow_lines'] = ''
            replace_dict['jamp_flow'] = 'JAMP'
            return ncolor

        ncolor_flow = max(1, len(flow_basis))
        projection = color_basis.get_flow_projection()

        # The Kleiss-Kuijf map only acts on color, so it is the same for every
        # split order
        lines = []
        for iso in range(replace_dict['nAmpSplitOrders']):
            lines.extend(self.get_flow_jamp_lines(projection,
                    JAMP_format="JAMPF(%%s,%d)" % (iso + 1),
                    AMP_format="JAMP(%%s,%d)" % (iso + 1)))

        replace_dict['ncolor_flow'] = ncolor_flow
        replace_dict['jampflow_decl'] = \
                             '    COMPLEX*16 JAMPF(NCOLOR_FLOW,NAMPSO)'
        replace_dict['jampflow_lines'] = '\n'.join(lines)
        replace_dict['jamp_flow'] = 'JAMPF'

        logger.debug('Color sum on %d DDM structures, color flow on %d trace '
                     'structures (%d Kleiss-Kuijf terms)',
                     ncolor, ncolor_flow,
                     sum(len(row) for row in projection))

        return ncolor_flow

    @staticmethod
    def get_crossing_permutation(cross, nexternal):
        """Return (perm, ic, valid) for the crossing code CROSS.

        CROSS decomposes as I*(NEXTERNAL+1)+J, with I and J the crossing
        partners of particle 1 and particle 2 (0 meaning "leave that particle
        alone"). The base is NEXTERNAL+1, not NEXTERNAL, so that I and J range
        over 0..NEXTERNAL and can therefore designate the last particle too.
        perm[slot] is the 0-based index of the original leg sitting in that
        slot, and ic[slot] is -1 for a leg that changed between the initial and
        the final state. This mirrors exactly what APPLY_CROSSING does in the
        generated fortran, so both stay in sync.

        *valid* is False for the overlapping-swap codes, which must not be used.
        CROSS asks for two independent transpositions, (particle1, I) and
        (particle2, J). When BOTH are active and they share a slot they no
        longer compose into an involution but into a 3-cycle, and the two code
        paths that consume this permutation (GET_PDG_FOR_FLAVOR building the
        signature, and APPLY_CROSSING_TABLE evaluating the matrix element) then
        disagree, one applying the permutation and the other its inverse --
        invisible for disjoint swaps (all involutions) but wrong for a cycle.
        Such a code is pure redundancy: every physical process it could reach is
        also reached by a DISJOINT swap, so it is marked invalid and its callers
        refuse it (SPINCOL_CROSS_TABLE gets 0, which SMATRIX and
        GET_PDG_FOR_FLAVOR both map to a null result). The two transpositions
        {1,I} and {2,J} are both active iff I not in {0,1} and J not in {0,2}
        (I==1 / J==2 swap a particle with itself, a no-op like 0), and they
        overlap iff I==2 or J==1 or I==J.
        """
        base = nexternal + 1
        i_part = cross // base
        j_part = cross % base
        perm = list(range(nexternal))
        ic = [1] * nexternal

        valid = not (i_part not in (0, 1) and j_part not in (0, 2)
                     and (i_part == 2 or j_part == 1 or i_part == j_part))

        def swap(slot_a, slot_b):
            perm[slot_a], perm[slot_b] = perm[slot_b], perm[slot_a]
            ic[slot_a] = -ic[slot_a]
            ic[slot_b] = -ic[slot_b]

        # I==1 (resp. J==2) would swap a particle with itself: degenerate, so
        # treated as "no crossing" just like 0.
        if i_part not in (0, 1):
            swap(0, i_part - 1)
        if j_part not in (0, 2):
            swap(1, j_part - 1)
        return perm, ic, valid

    @staticmethod
    def breaks_crossing_symmetry(process):
        """True if `process` constrains a specific s-channel propagator.

        Crossing permutes legs between the initial and the final state, so a
        channel that is s-channel in the generated process is not s-channel in
        its crossings. A constraint naming a specific s-channel therefore does
        not survive the crossing and the crossing machinery must not be emitted:
          - required_s_channels  (the `> A >` syntax)
          - forbidden_s_channels (the `$$` syntax, diagram removed)
        `forbidden_onsh_s_channels` (a single `$`) only forbids the on-shell
        *region* of a kept diagram, so it does not break crossing symmetry and
        is deliberately not listed here.

        Works for both Process and ProcessDefinition (same attributes), and
        recurses into decay chains, whose constraints bind just as much. A decay
        chain itself does NOT break crossing: p p > t t~ j j, t > ... still
        crosses at the production level (force-onshell decays ride along on the
        legs they hang off), so crossing stays enabled -- the crossing tables
        just have to be built over the full decay leaves (see
        compute_crossing_tables) so the identical-particle/denominator factors
        reflect the real final state.
        """
        if process.get('required_s_channels') or \
           process.get('forbidden_s_channels'):
            return True
        # Crossing is a tree-level construction; a perturbative (loop / loop-
        # induced) process must not go through it. Its matrix element has no
        # flavor/PDG crossing tables (compute_crossing_pdg_entries would index
        # past the end), so treat it as crossing-breaking to keep every
        # crossing gate -- and the crossed-group detection -- clear of it.
        if process.get('perturbation_couplings'):
            return True
        return any(ProcessExporterFortran.breaks_crossing_symmetry(decay)
                   for decay in process.get('decay_chains'))

    def fill_crossing_replace_dict(self, matrix_element, replace_dict,
                                   use_crossing):
        """Fill the crossing-machinery holes of matrix_standalone_v4.inc.

        The extended FLAV_IDX (a flavor *and* a crossing) and everything
        decoding it are only written out when the process was generated with
        --use_crossing=True (the default) *and* the process definition pins no
        specific s-channel (see breaks_crossing_symmetry). Otherwise the
        crossed subprocesses are generated as separate matrix elements instead,
        so the crossing machinery would be dead code: the tables, the
        APPLY_CROSSING/GET_CROSS_PERM/GET_SPINCOL_CROSS/GET_IDENT_CROSS routines
        are not emitted at all and each hole below gets the plain code path
        (FLAV_IDX is then a bare flavor index in [1,NFLAV]).

        Requires proc_prefix, nflav and den_factor_line to be set already.
        """
        prefix = replace_dict['proc_prefix']

        if not use_crossing:
            replace_dict.update({
                'crossing_routines': '',
                'iden_cross_lines': '',
                'smatrix_cross_decl':
                    'C     Generated without crossing symmetry: FLAV_IDX is a plain'
                    '\nC     flavor index, there is no crossing to decode.',
                'smatrix_cross_decode': '',
                'smatrix_cross_apply': '',
                'smatrix_goodhel_gate':
                    '                IF (GOODHEL(IHEL,FLAV_USE) .OR. NTRY(FLAV_USE)'
                    ' .LT. 20.OR.USERHEL.NE.-1) THEN',
                'smatrix_goodhel_train':
                    '                    IF (T .NE. 0D0 .AND. .NOT.    '
                    'GOODHEL(IHEL,FLAV_USE)) THEN\n'
                    '                        GOODHEL(IHEL,FLAV_USE)=.TRUE.\n'
                    '                    ENDIF',
                'smatrix_matrix_call':
                    '                    T=%sMATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_USE)'
                    % prefix,
                'smatrix_iden_line':
                    'C     IDEN carries the identical-particle factor of the'
                    ' representative\nC     flavor; BROKEN_SYM corrects it for'
                    ' the actual one.'
                    '\n      ANS=ANS/DBLE(IDEN)*%sBROKEN_SYM(FLAVOR)' % prefix,
                'inter_rescale_decl': '',
                'inter_rescale_body':
                    'C     The static IDEN GET_INTER divides by carries the'
                    ' identical-particle\nC     factor of the representative'
                    ' flavor, so BROKEN_SYM must correct it for\nC     the actual'
                    ' one, exactly as SMATRIX does with ANS/IDEN*BROKEN_SYM.'
                    '\n      RESCALE = DBLE(%sBROKEN_SYM(FLAVOR))' % prefix,
                'density_cross_apply': self.CROSS_PASSTHROUGH % {
                    'nhel_copy': 'NHELUSE(:,:) = NHEL(:,:)'},
                'allinter_cross_apply': '      IC(:)=1\n' + self.CROSS_PASSTHROUGH % {
                    'nhel_copy': 'NHELUSE(:) = NHEL(:)'},
                'pdg_cross_snippets': self.PDG_CROSS_SNIPPETS_OFF,
                'nhel_idx_decl':
                    'C     Generated without crossing symmetry: FLAV_IDX_IN is a'
                    ' plain\nC     flavor index, so only BROKEN_SYM can move the'
                    ' denominator.',
                'nhel_idx_body':
                    'C     Mirrors SMATRIX exactly: ANS=ANS/IDEN*BROKEN_SYM means'
                    ' the effective\nC     denominator is IDEN/BROKEN_SYM. The'
                    ' division is exact -- BROKEN_SYM is\nC     the ratio of the'
                    ' representative to the actual identical-particle\nC     count,'
                    ' and IDEN carries the representative one as a factor.'
                    '\n      IDEN_STAR = IDEN_STAR / %sBROKEN_SYM(FLAVOR)' % prefix,
            })
            return

        replace_dict['iden_cross_lines'] = \
            self.get_iden_cross_lines(matrix_element)
        replace_dict['ident_resonance'] = \
            self.compute_crossing_tables(matrix_element)['ident_resonance']
        replace_dict.update(dict(
            (key, value % {'proc_prefix': prefix,
                           'den_factor_line': replace_dict['den_factor_line']})
            for key, value in self.CROSSING_SNIPPETS.items()))
        # CROSS_GHIDX (in the crossing routines below) recomputes the crossed
        # -> identity helicity row map at runtime; it needs only the small
        # per-crossing GHFILT flag plus the STATES/NHSTATE the encoder uses (in
        # get_helicity_matrix()'s default allow_reverse=True order, so the map is
        # built in the same order the NHEL table is emitted).
        hel_data = self._helstate_data(matrix_element)
        replace_dict['maxhel'] = hel_data['maxhel']
        replace_dict['nhstate_data'] = hel_data['nhstate_data']
        replace_dict['states_data'] = hel_data['states_data']
        replace_dict['ghfilt_data'] = self.format_integer_data_lines(
            'GHFILT', self.compute_ghfilt(matrix_element, allow_reverse=True))
        replace_dict['pdg_cross_snippets'] = tuple(
            snippet % {'proc_prefix': prefix}
            for snippet in self.PDG_CROSS_SNIPPETS_ON)
        replace_dict['nhel_idx_decl'] = (
            '      INTEGER %(prefix)sGET_SPINCOL_CROSS\n'
            '      INTEGER %(prefix)sGET_IDENT_CROSS' % {'prefix': prefix})
        replace_dict['nhel_idx_body'] = (
            'C     Mirrors SMATRIX branch for branch: IDEN/BROKEN_SYM uncrossed,\n'
            'C     GET_SPINCOL_CROSS*GET_IDENT_CROSS crossed. Keeping the CROSS=0\n'
            'C     branch on the old path (rather than letting the crossed formula\n'
            'C     cover it) is what guarantees no change for existing callers.\n'
            '      IF (NHI_CROSS .EQ. 0) THEN\n'
            '        IDEN_STAR = IDEN_STAR / %(prefix)sBROKEN_SYM(FLAVOR)\n'
            '      ELSE\n'
            '        IDEN_STAR = %(prefix)sGET_SPINCOL_CROSS(NHI_CROSS)\n'
            '     &   * %(prefix)sGET_IDENT_CROSS(NHI_CROSS, FLAVOR)\n'
            '      ENDIF' % {'prefix': prefix})
        crossing_template = pjoin(_file_path, 'iolibs', 'template_files',
                                  'matrix_standalone_crossing_v4.inc')
        replace_dict['crossing_routines'] = \
            open(crossing_template).read() % replace_dict

    def fill_crossing_replace_dict_me(self, matrix_element, replace_dict,
                                      use_crossing, proc_id, xgrow_map=None):
        """Fill the crossing holes of matrix_madevent_group_v4.inc.

        The madevent group SMATRIX differs structurally from the standalone one
        (runtime IFLAV, GOODHEL/NTRY carry a flavor dimension, IVEC, and the NSF
        flags are baked into the helas calls rather than read from an IC array),
        so it gets its own holes and OFF fills. With crossing off every hole
        reproduces the historical madevent code, so a non-crossing output is
        unchanged; the extended-FLAV_IDX decode / APPLY_CROSSING path is only
        written out when use_crossing is True (added in the ON slice).

        ``xgrow_map`` (Track-A bases only) is ``{cross: (dep_proc_id, cmap)}``
        for every within-group router flavor routed here: which subprocess the
        crossed call is FOR, and that subprocess's diagram -> this module's
        diagram map under the crossing. It drives the multi-channel row; see
        the ``me_confsub_j`` fill below.
        """
        pid = str(proc_id)
        if not use_crossing:
            replace_dict.update({
                'smatrix_me_cross_decl':
                    'C     Generated without crossing symmetry: IFLAV is a plain'
                    '\nC     flavor index, there is no crossing to decode.',
                'smatrix_me_cross_decode': '',
                'me_flav_key': 'IFLAV',
                'me_goodhel_idx': 'I',
                'me_goodhel_train_guard': '',
                'smatrix_me_goodhel_or': '',
                'me_matrix_args': 'P ,NHEL(1,I),IFLAV,I,AMP2, JAMP2, IVEC',
                'smatrix_me_iden_line':
                    '    ANS=ANS/DBLE(IDEN)*BROKEN_SYM%s(FLAVOR_FOR_SYM)' % pid,
                'crossing_routines_me': '',
                'me_matrix_ic_param': '',
                'me_matrix_ic_decl': '',
                # Multi-channel row: without crossing this matrix element is
                # only ever called for its own subprocess, so its own CONFSUB
                # row is the right one and AMP2 is already in its numbering.
                'me_confsub_j': 'CONFSUB(%s, I)' % pid,
                # helicity-recycling template variant (matrix<i>_hel):
                'smatrix_hel_cross_decl':
                    'C     Generated without crossing symmetry: IFLAV is a plain'
                    '\nC     flavor index, there is no crossing to decode.',
                'smatrix_hel_cross_decode': '',
                'hel_matrix_call_args': 'P ,IFLAV, TS, AMP2, JAMP2, IVEC',
                'hel_matrix_ic_param': '',
                # No crossing: every call is the uncrossed base process, so the
                # C-parity de-duplication is always applicable.
                'me_csym_cross_ok': '.TRUE.',
                'hel_csym_cross_ok': '.TRUE.',
            })
            return

        # ON path. The crossing routines must not collide across the matrix<i>.f
        # files linked into one group executable, so they are named with a
        # per-proc_id qualifier (GET_CROSS_PERM stays prefix-less in standalone).
        #
        # NFLAV must be the count that the madevent GET_FLAVOR table is sized by,
        # i.e. get_external_flavors_with_iden() (== replace_dict 'max_flavor',
        # what MAXFLAVPERPROC/FLAVOR(NEXTERNAL,max_flavor) use), NOT the standalone
        # _build_flav_table_flat() (compute_flavor_masks): for a merged group ME
        # the two differ (e.g. Q Q~ > g g: iden 1 vs masks 4), and the extended
        # FLAV_IDX decode CROSS=(IFLAV-1)/NFLAV, FLAV=mod(IFLAV-1,NFLAV)+1 must
        # land FLAV in [1, max_flavor]. This also matches compute_crossing_pdg_
        # entries (used by partition_crossing_classes), so the routed FLAV_IDX
        # decodes the same way here.
        nflav = len(matrix_element.get_external_flavors_with_iden())
        cp = 'CR%s_' % pid
        crossing_template = pjoin(_file_path, 'iolibs', 'template_files',
                                  'matrix_standalone_crossing_v4.inc')
        hel_data = self._helstate_data(matrix_element)
        crossing_routines = open(crossing_template).read() % {
            'proc_prefix': cp,
            'nflav': nflav,
            'iden_cross_lines': self.get_iden_cross_lines(matrix_element),
            'ident_resonance':
                self.compute_crossing_tables(matrix_element)['ident_resonance'],
            'maxhel': hel_data['maxhel'],
            'nhstate_data': hel_data['nhstate_data'],
            'states_data': hel_data['states_data'],
            'ghfilt_data': self.format_integer_data_lines(
                'GHFILT', self.compute_ghfilt(matrix_element,
                                              allow_reverse=True))}
        # ---- multi-channel row for calls routed here by a within-group router.
        # CHANNEL and AMP2 are both in THIS module's diagram numbering (the
        # router already translated CHANNEL through the crossing), but the loop
        # that builds XTOT must enumerate the configs of the subprocess the call
        # is FOR: GET_CHANNEL_CUT(P, I) is evaluated on the DEPENDENT's momenta,
        # so I has to be a config of the dependent's row, and the AMP2 slot
        # paired with it is the dependent's diagram sent through the crossing.
        # Walking our own row instead pairs each amplitude with a different
        # config's cut -- a bijective relabel, so the weights still sum to 1 and
        # the cross section is unchanged, but the importance sampling is
        # mis-paired. Both lookups are resolved from CROSSUSE through baked
        # tables rather than a common block set by the router: madevent runs
        # vectorised (IVEC/warps) and mutable shared state would race.
        #
        # Safe to key on the crossing code alone: a base that serves a
        # within-group router is never also a cross-group (Track B) base --
        # compute_crossgroup_routing skips any group that has within-group
        # routing -- so no foreign crossing can reach these tables.
        ngraphs_me = len(matrix_element.get('diagrams'))
        nxc = (matrix_element.get_nexternal_ninitial()[0] + 1) ** 2 - 1
        xg_rows, xg_cols = {}, {}
        xg_cfg = [list(range(0, ngraphs_me + 1))]   # column 1 = identity
        for cross in sorted(xgrow_map or {}):
            dep_pid, cmap = xgrow_map[cross]
            # Only a clean permutation of our own diagrams is usable: anything
            # else (a fallback map, or a dependent with a different diagram
            # count) keeps our own row, i.e. the historical behaviour.
            if not 1 <= cross <= nxc or \
                    sorted(cmap) != list(range(1, ngraphs_me + 1)):
                continue
            col = [0] + list(cmap)
            if col not in xg_cfg:
                xg_cfg.append(col)
            xg_rows[cross] = dep_pid
            xg_cols[cross] = xg_cfg.index(col) + 1
        if xg_rows:
            def _data2d(name, icol, values, per_line=10):
                out = []
                for s in range(0, len(values), per_line):
                    chunk = values[s:s + per_line]
                    out.append('      DATA (%s(I,%d),I=%d,%d) /%s/'
                               % (name, icol, s, s + len(chunk) - 1,
                                  ','.join(str(v) for v in chunk)))
                return out
            xg_lines = ['      INTEGER IXROW, IXR',
                        '      INTEGER XGROWT(0:%d), XGCOLT(0:%d)' % (nxc, nxc),
                        self.format_integer_data_lines(
                            'XGROWT', [xg_rows.get(c, int(pid))
                                       for c in range(nxc + 1)]),
                        self.format_integer_data_lines(
                            'XGCOLT', [xg_cols.get(c, 1)
                                       for c in range(nxc + 1)]),
                        '      INTEGER XGCFG(0:%d,%d)'
                        % (ngraphs_me, len(xg_cfg))]
            for icol, col in enumerate(xg_cfg):
                xg_lines += _data2d('XGCFG', icol + 1, col)
            xg_decl = '\n' + '\n'.join(xg_lines)
            xg_decode = ('\n      IXROW = XGROWT(CROSSUSE)'
                         '\n      IXR = XGCOLT(CROSSUSE)')
            # Slot 0 of every XGCFG column is 0, so a config this subprocess has
            # no diagram for still reads back as 0 and is skipped as before.
            confsub_j = 'XGCFG(CONFSUB(IXROW, I), IXR)'
        elif id(matrix_element) in getattr(self, '_crossgroup_base_mes', ()):
            # Cross-group (Track B) base: same defect, but this object is
            # SYMLINKED into the dependent P directories (write_crossgroup_mk),
            # so one binary serves them all and the row cannot be baked here --
            # a dependent's configs live in ITS directory's config_subproc_map,
            # and GET_CHANNEL_CUT already resolves to the dependent's genps.o.
            # Take the row from XGROW<pid>, which every directory defines for
            # itself in its own auto_dsig.f (see write_xgrow_routines): the
            # identity (our own CONFSUB row) where we are generated, the routed
            # subprocess's row composed with the crossing map in a dependent's.
            # LMAXCONFIGS is a single global maximum (Source/maxconfigs.inc,
            # symlinked), so the loop bound is the same in every directory.
            xg_decl = '\n      INTEGER XGJROW(LMAXCONFIGS)'
            xg_decode = '\n      CALL XGROW%s(CROSSUSE, XGJROW)' % pid
            confsub_j = 'XGJROW(I)'
        else:
            xg_decl, xg_decode = '', ''
            confsub_j = 'CONFSUB(%s, I)' % pid
        replace_dict.update({
            'me_confsub_j': confsub_j,
            'smatrix_me_cross_decl': (
                '      INTEGER NFLAV\n'
                '      PARAMETER (NFLAV=%(nflav)d)\n'
                '      INTEGER FLAV_USE, CROSSUSE, IDENUSE, XKCR\n'
                '      INTEGER IC(NEXTERNAL), IC0(NEXTERNAL)\n'
                '      REAL*8 PUSE(0:3,NEXTERNAL)\n'
                '      INTEGER NHELUSE(NEXTERNAL,NCOMB)\n'
                '      INTEGER %(cp)sGET_SPINCOL_CROSS\n'
                '      INTEGER %(cp)sGET_IDENT_CROSS\n'
                # runtime good-helicity remap: GHIDXA(I) is the identity row that
                # gates crossed row I (0 = not filterable), precomputed once per
                # SMATRIX call from the crossing permutation XGPERM/XGSGN.
                '      INTEGER GHIDXA(NCOMB), XGPERM(NEXTERNAL)\n'
                '      INTEGER XGSGN(NEXTERNAL), XGDUM, XGH'
                '%(xg_decl)s'
                ) % {'nflav': nflav, 'cp': cp, 'xg_decl': xg_decl},
            # Decode the crossing and build the crossed P/NHEL/IC once, before the
            # helicity loop. An unusable crossing (spin*color = 0) has a zero ME.
            'smatrix_me_cross_decode': (
                '      CROSSUSE = (IFLAV-1) / NFLAV\n'
                '      IDENUSE = %(cp)sGET_SPINCOL_CROSS(CROSSUSE)\n'
                '      IF (IDENUSE.EQ.0) THEN\n'
                '        ANS = 0D0\n'
                '        IHEL = 1\n'
                '        ICOL = 1\n'
                '        RETURN\n'
                '      ENDIF\n'
                '      DO XKCR=1,NEXTERNAL\n'
                '        IC0(XKCR) = 1\n'
                '      ENDDO\n'
                '      CALL %(cp)sAPPLY_CROSSING_TABLE(IFLAV, NCOMB, P, NHEL,\n'
                '     &   IC0, PUSE, NHELUSE, IC, FLAV_USE)\n'
                # Precompute the crossed->identity helicity-row map once (the
                # crossing permutation does not depend on the row), so the shared
                # GOODHEL filter (keyed by the reduced FLAV_USE) can gate crossed
                # rows through it just like the standalone. CROSS=0 gives
                # GHIDXA(I)=I, i.e. the historical unfiltered-flavor behaviour.
                '      CALL %(cp)sGET_CROSS_PERM(IFLAV, XGPERM, XGSGN, XGDUM)\n'
                '      DO XGH=1,NCOMB\n'
                '        CALL %(cp)sCROSS_GHIDX(CROSSUSE, XGPERM, XGSGN,\n'
                '     &   NHEL(1,XGH), GHIDXA(XGH))\n'
                '      ENDDO'
                '%(xg_decode)s'
                ) % {'cp': cp, 'xg_decode': xg_decode},
            'me_flav_key': 'FLAV_USE',
            # The shared GOODHEL filter (keyed by the reduced flavor) is gated
            # and trained through the runtime remap GHIDXA: crossed row I is good
            # iff identity row GHIDXA(I) is. GHIDXA(I)=0 (non-filterable crossing)
            # forces the row to be computed (.OR. GHIDXA(I).EQ.0) and never
            # trained (GHIDXA(I).NE.0 guard). The index is clamped with MAX(...,1)
            # because the gate reads GOODHEL before the .EQ.0 guard and fortran
            # does not short-circuit .OR.; the clamped value is only ever read
            # when GHIDXA(I).EQ.0 already forces the branch true, so it is inert.
            'me_goodhel_idx': 'MAX(GHIDXA(I),1)',
            'me_goodhel_train_guard': 'GHIDXA(I).NE.0 .AND. ',
            'smatrix_me_goodhel_or': ' .OR. GHIDXA(I).EQ.0',
            'me_matrix_args':
                'PUSE ,NHELUSE(1,I),IC,FLAV_USE,I,AMP2, JAMP2, IVEC',
            # Uncrossed keeps IDEN/BROKEN_SYM; crossed rebuilds the denominator
            # as initial spin*color (per crossing) times the identical-final
            # factor of the actual flavors (per flavor).
            'smatrix_me_iden_line': (
                '      IF (CROSSUSE.EQ.0) THEN\n'
                '        ANS=ANS/DBLE(IDEN)*BROKEN_SYM%(pid)s(FLAVOR_FOR_SYM)\n'
                '      ELSE\n'
                '        ANS=ANS/DBLE(IDENUSE*%(cp)sGET_IDENT_CROSS(CROSSUSE,\n'
                '     &   FLAVOR_FOR_SYM))\n'
                '      ENDIF'
                ) % {'pid': pid, 'cp': cp},
            'crossing_routines_me': crossing_routines,
            'me_matrix_ic_param': 'IC,',
            'me_matrix_ic_decl': '    INTEGER IC(NEXTERNAL)',
            # Helicity-recycling variant (matrix<i>_hel -> matrix<i>_optim). The
            # recycled MATRIX bakes its helicity set; feeding it the crossed
            # momenta PUSE and IC evaluates that set at the crossed kinematics,
            # which is exactly the crossed ME -- no NHEL table (nor a helicity
            # remap) is needed here. What the set must BE is the catch: IC carries
            # the crossing's sign flips but nothing carries its slot permutation,
            # so the set has to cover tau(G_base) as well (see
            # write_crossgroup_helunion / _crossgroup_base_helsignmap).
            'smatrix_hel_cross_decl': (
                '      INTEGER NFLAV\n'
                '      PARAMETER (NFLAV=%(nflav)d)\n'
                '      INTEGER FLAV_USE, CROSSUSE, IDENUSE, XKCR\n'
                '      INTEGER PERM(NEXTERNAL), SGN(NEXTERNAL), IC(NEXTERNAL)\n'
                '      REAL*8 PUSE(0:3,NEXTERNAL)\n'
                '      INTEGER %(cp)sGET_SPINCOL_CROSS\n'
                '      INTEGER %(cp)sGET_IDENT_CROSS'
                '%(xg_decl)s'
                ) % {'nflav': nflav, 'cp': cp, 'xg_decl': xg_decl},
            'smatrix_hel_cross_decode': (
                '      CROSSUSE = (IFLAV-1) / NFLAV\n'
                '      IDENUSE = %(cp)sGET_SPINCOL_CROSS(CROSSUSE)\n'
                '      IF (IDENUSE.EQ.0) THEN\n'
                '        ANS = 0D0\n'
                '        IHEL = 1\n'
                '        ICOL = 1\n'
                '        RETURN\n'
                '      ENDIF\n'
                '      CALL %(cp)sGET_CROSS_PERM(IFLAV, PERM, SGN, FLAV_USE)\n'
                '      DO XKCR=1,NEXTERNAL\n'
                '        PUSE(0,XKCR) = P(0,PERM(XKCR))\n'
                '        PUSE(1,XKCR) = P(1,PERM(XKCR))\n'
                '        PUSE(2,XKCR) = P(2,PERM(XKCR))\n'
                '        PUSE(3,XKCR) = P(3,PERM(XKCR))\n'
                '        IC(XKCR) = SGN(XKCR)\n'
                '      ENDDO'
                '%(xg_decode)s'
                ) % {'cp': cp, 'xg_decode': xg_decode},
            'hel_matrix_call_args': 'PUSE ,IC, FLAV_USE, TS, AMP2, JAMP2, IVEC',
            'hel_matrix_ic_param': 'IC,',
            # C-parity de-duplication only for the uncrossed base process
            # (CROSSUSE 0): a crossing permutes/sign-flips the helicities, so a
            # base-row FLIP is not the crossed C-parity partner. Crossed
            # dependents keep the full helicity sum.
            'me_csym_cross_ok': 'CROSSUSE.EQ.0',
            'hel_csym_cross_ok': 'CROSSUSE.EQ.0',
        })

    # (decl, decode, apply) for GET_PDG_FOR_FLAVOR without crossing: FLAV_IDX_IN
    # is a bare flavor index, so there is nothing to permute or conjugate.
    PDG_CROSS_SNIPPETS_OFF = (
        'C     Generated without crossing symmetry: FLAV_IDX_IN is a plain\n'
        'C     flavor index, so the PDGs are read straight off the table.',
        '      FP_FLAV = FLAV_IDX_IN',
        """      DO FP_I = 1, NEXTERNAL
        PDGS(FP_I) = FP_PDG_TABLE(FP_I, FP_FLAV)
      ENDDO""")

    # The same three holes with crossing on. GET_CROSS_PERM is reused rather
    # than re-deriving I/J here, so the PDGs reported can never disagree with
    # the legs the matrix element actually evaluates: PERM(K) is the input slot
    # landing in crossed slot K and SGN(K)=-1 marks exactly the legs that
    # swapped between the initial and the final state, which are the ones the
    # crossed process sees as their own antiparticle.
    PDG_CROSS_SNIPPETS_ON = (
        """      INTEGER FP_PERM(NEXTERNAL), FP_SGN(NEXTERNAL)
      INTEGER FP_CROSS
      INTEGER %(proc_prefix)sGET_SPINCOL_CROSS""",
        '      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX_IN, FP_PERM, FP_SGN,\n'
        '     & FP_FLAV)',
        """C     A crossing with a null spin*color entry is one SMATRIX itself maps
C     to a zero matrix element (out of range, or not applicable). Report no
C     PDGs for it rather than a signature that cannot be evaluated.
      FP_CROSS = (FLAV_IDX_IN-1) / NFLAV
      IF (%(proc_prefix)sGET_SPINCOL_CROSS(FP_CROSS) .EQ. 0) THEN
        RETURN
      ENDIF
      DO FP_I = 1, NEXTERNAL
        IF (FP_SGN(FP_I) .EQ. 1) THEN
          PDGS(FP_I) = FP_PDG_TABLE(FP_PERM(FP_I), FP_FLAV)
        ELSE
          PDGS(FP_I) = FP_ANTI_TABLE(FP_PERM(FP_I), FP_FLAV)
        ENDIF
      ENDDO""")

    # Copy the arguments through unchanged: same shape as the crossing block it
    # replaces, so its (single) caller does not have to know which is which.
    CROSS_PASSTHROUGH = """C     No crossing to apply: the arguments go through unchanged.
      PUSE(:,:) = P(:,:)
      %(nhel_copy)s
      ICUSE(:) = IC(:)
      DO IPART=1,N_CHANGING
        CPOS(IPART) = POS(IPART)
      ENDDO"""

    # The crossing-aware variants of the same holes. Kept here rather than in
    # the template because the template can only hold one variant per hole.
    CROSSING_SNIPPETS = {
        'smatrix_cross_decl': """C     CROSSUSE is the crossing carried by FLAV_IDX and IDENUSE the initial
C     state spin*color average of the process it crosses into.
      INTEGER IDENUSE, CROSSUSE
      INTEGER %(proc_prefix)sGET_SPINCOL_CROSS
      INTEGER %(proc_prefix)sGET_IDENT_CROSS
C     Crossed copies of the arguments, built ONCE per SMATRIX call (see the
C     BEGIN CODE section). They are only touched when a crossing is actually
C     requested, so the uncrossed path pays nothing for them.
      REAL*8 PUSE(0:3,NEXTERNAL)
      INTEGER NHELUSE(NEXTERNAL,NCOMB)
      INTEGER ICUSE(NEXTERNAL)
      INTEGER DUMFLAV
C     GHIDX is the identity row whose shared GOODHEL bit gates the current
C     crossed row, recomputed at runtime by CROSS_GHIDX (which owns the small
C     per-crossing GHFILT flag table); XGPERM/XGSGN are the crossing's slot
C     permutation and NSF signs, fetched once per call (see smatrix_cross_apply).
      INTEGER GHIDX
      INTEGER XGPERM(NEXTERNAL), XGSGN(NEXTERNAL), XGDUM""",

        'smatrix_cross_decode': """C     CROSS = (FLAV_IDX-1)/NFLAV is the crossing to apply. IDENUSE is 0 for a
C     crossing that cannot be applied, whose matrix element is identically zero.
      CROSSUSE = (FLAV_IDX-1) / NFLAV
      IDENUSE = %(proc_prefix)sGET_SPINCOL_CROSS(CROSSUSE)
      IF (IDENUSE.EQ.0) THEN
        ANS = 0D0
        RETURN
      ENDIF""",

        'smatrix_cross_apply': """C     Fetch the crossing's slot permutation / NSF signs once (the good-helicity
C     gate below reuses them per helicity via CROSS_GHIDX). Cheap, and the
C     identity crossing returns the identity permutation.
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX, XGPERM, XGSGN, XGDUM)
C     Apply the crossing ONCE, here, rather than once per helicity: the whole
C     NHEL table is permuted in one go (the crossing is a fixed slot
C     permutation, identical for every row) together with the momenta and the
C     NSF/NSV flags. When CROSSUSE is 0 nothing is copied at all and the loop
C     below passes the original arrays straight through, exactly as it did
C     before crossings existed.
      IF (CROSSUSE.NE.0) THEN
        CALL %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX, NCOMB, P, NHEL,
     &   JC, PUSE, NHELUSE, ICUSE, DUMFLAV)
      ENDIF""",

        'smatrix_goodhel_gate': """C     The good-helicity filter (GOODHEL) is shared by every crossing of a
C     flavor, but a crossing permutes and flips helicities, so a crossed row
C     and its identity counterpart are different rows. CROSS_GHIDX sends crossed
C     row IHEL to the identity row that gates it (sigma^-1, recomputed from the
C     config); GHIDX=0 means the crossing is not filterable (an initial-initial
C     swap, or a crossing that cannot be applied) so its every helicity is
C     computed. For CROSSUSE=0 it returns IHEL, exactly the historical gate.
                CALL %(proc_prefix)sCROSS_GHIDX(CROSSUSE, XGPERM, XGSGN,
     &           NHEL(1,IHEL), GHIDX)
                IF (GHIDX.EQ.0 .OR. GOODHEL(GHIDX,FLAV_USE) .OR. NTRY(FLAV_USE).LT.20 .OR. USERHEL.NE.-1) THEN""",

        'smatrix_goodhel_train': """C     Train the SHARED filter through the same map: mark the IDENTITY row
C     GHIDX good, so GOODHEL always stores the identity pattern whatever
C     crossing is being evaluated. GHIDX=0 (non-filterable crossing) never
C     trains. For CROSSUSE=0 GHIDX=IHEL, so this is the historical training.
                    IF (T .NE. 0D0 .AND. GHIDX.NE.0 .AND. .NOT.GOODHEL(GHIDX,FLAV_USE)) THEN
                        GOODHEL(GHIDX,FLAV_USE)=.TRUE.
                    ENDIF""",

        'smatrix_matrix_call': """                    IF (CROSSUSE.EQ.0) THEN
                      T=%(proc_prefix)sMATRIX(P ,NHEL(1,IHEL),JC(1),FLAV_USE)
                    ELSE
                      T=%(proc_prefix)sMATRIX(PUSE,NHELUSE(1,IHEL),ICUSE(1)
     &                 ,FLAV_USE)
                    ENDIF""",

        'smatrix_iden_line': """C     Uncrossed: keep the historical path untouched (IDEN carries the
C     representative's identical factor and BROKEN_SYM corrects it per flavor).
C     Crossed: BROKEN_SYM's tables describe the uncrossed final state and
C     cannot express the crossed one, so rebuild the denominator instead as
C     initial state spin*color (per crossing) times the identical final state
C     factor of the actual crossed flavors (per flavor).
      IF (CROSSUSE.EQ.0) THEN
        ANS=ANS/DBLE(IDEN)*%(proc_prefix)sBROKEN_SYM(FLAVOR)
      ELSE
        ANS=ANS/DBLE(IDENUSE*%(proc_prefix)sGET_IDENT_CROSS(CROSSUSE,
     &   FLAVOR))
      ENDIF""",

        'inter_rescale_decl': """      INTEGER CROSS, DCROSS, IDEN
      INTEGER %(proc_prefix)sGET_SPINCOL_CROSS
      INTEGER %(proc_prefix)sGET_IDENT_CROSS
      %(den_factor_line)s""",

        'inter_rescale_body': """      CROSS = (FLAV_IDX-1)/NFLAV
      IF (CROSS.EQ.0) THEN
C     Uncrossed: the static IDEN carries the identical-particle factor of the
C     representative flavor, so BROKEN_SYM must correct it for the actual one,
C     exactly as SMATRIX does with ANS/IDEN*BROKEN_SYM.
        RESCALE = DBLE(%(proc_prefix)sBROKEN_SYM(FLAVOR))
      ELSE
C     Crossed: BROKEN_SYM's tables describe the uncrossed final state and are
C     useless here; rebuild the whole denominator instead (see SMATRIX) and
C     undo the IDEN that GET_INTER divided by.
        DCROSS = %(proc_prefix)sGET_SPINCOL_CROSS(CROSS)
     &   * %(proc_prefix)sGET_IDENT_CROSS(CROSS, FLAVOR)
        IF (DCROSS.EQ.0) THEN
          RESCALE = 0D0
        ELSE
          RESCALE = DBLE(IDEN)/DBLE(DCROSS)
        ENDIF
      ENDIF""",

        'density_cross_apply': """      CALL %(proc_prefix)sAPPLY_CROSSING_TABLE(FLAV_IDX, NB_NHEL, P, NHEL,
     & IC, PUSE, NHELUSE, ICUSE, DUMFLAV)
C     POS is given in uncrossed slots; PERM(K) is the uncrossed slot sitting in
C     crossed slot K, so invert it to move POS into the crossed numbering.
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX, PERM, SGN, DUMFLAV)
      DO IPART=1,N_CHANGING
        DO I=1,NEXTERNAL
          IF (PERM(I).EQ.POS(IPART)) CPOS(IPART) = I
        ENDDO
      ENDDO""",

        'allinter_cross_apply': """C     IC starts at +1 everywhere; APPLY_CROSSING flips it for the legs that the
C     crossing carried by FLAV_IDX moves across.
      IC(:)=1
      CALL %(proc_prefix)sAPPLY_CROSSING(FLAV_IDX, P, NHEL, IC, PUSE,
     & NHELUSE, ICUSE, DUMFLAV)
      CALL %(proc_prefix)sGET_CROSS_PERM(FLAV_IDX, PERM, SGN, DUMFLAV)
      DO IPART = 1, N_CHANGING
        DO I = 1, NEXTERNAL
          IF (PERM(I).EQ.POS(IPART)) CPOS(IPART) = I
        ENDDO
      ENDDO""",
    }

    def get_iden_cross_lines(self, matrix_element):
        """Return the DATA lines backing the crossing-dependent denominator.

        SMATRIX must divide by the averaging/symmetry factor of the *crossed*
        process. That factor splits in two, and the two halves must be handled
        differently:

        - the initial state spin*color average changes with the crossing (a
          gluon pulled into the initial state takes the color average from 3 to
          8) but NOT with the flavor, since every particle of a flavor group
          shares its spin and color. It is emitted as SPINCOL_CROSS_TABLE,
          indexed by CROSS.
        - the identical final state factor changes with the FLAVOR: e.g.
          d d~ > g u u~ crossed gives d g > d u u~ (nothing identical) while
          d d~ > g d d~ crossed gives d g > d d d~ (two identical d). It cannot
          be tabulated on CROSS alone, and the existing BROKEN_SYM cannot help:
          its tables describe the *uncrossed* final state, so for this process
          it emits COMP_OLD=1 and returns 1 whatever flavor array it is given.
          It is therefore computed at runtime by GET_IDENT_CROSS, from the two
          per-particle tables below.

        The per-slot representative PDG (BASEPID) and FLAVOR source slot (SRC)
        GET_IDENT_CROSS needs are not tabulated per crossing: they follow from
        the crossing's own PERM/IC (the same GET_SPINCOL_CROSS decodes) applied
        to two NEXTERNAL-long base tables. IDS_BASE is the base process PDG of
        each leg; ANTIPID_BASE is its charge conjugate (used for a leg that
        swapped between the initial and the final state). Slot k of crossing
        CROSS then reads leg PERM(k), conjugated when IC(k) flipped, and looks
        up FLAVOR(PERM(k)); two crossed final legs are identical iff they share
        both. This drops the two NCROSS*NEXTERNAL-long tables.

        A crossing that cannot be applied gets a 0 spin*color entry, which
        SMATRIX maps to a null matrix element.
        """
        tables = self.compute_crossing_tables(matrix_element)

        return '\n'.join([
            self.format_integer_data_lines('SPINCOL_PART', tables['spincol_part']),
            self.format_integer_data_lines('IDS_BASE', tables['ids_base']),
            self.format_integer_data_lines('ANTIPID_BASE', tables['antipid_base']),
            self.format_integer_data_lines('COUNTABLE', tables['countable'])])

    @staticmethod
    def _leaf_block_sizes(process):
        """Per core leg, the number of decay leaves it expands to.

        A decay chain's matrix element runs over the decay *leaves*, but a
        crossing acts at the *production* level: it may only permute whole
        production legs, and a decaying production leg carries its whole decay
        block (all its leaves) as one unit. This returns a list parallel to
        ``process.get('legs')`` giving each core leg's leaf count -- 1 for a
        non-decaying leg (a single leaf that a crossing may move), >1 for a
        decaying resonance (a block a crossing must never split or pull into the
        initial state). Non-decay processes get all 1s, so every downstream use
        is a no-op for them. Mirrors base_objects.get_legs_with_decays exactly:
        decays are matched to final legs in leg order, first id-match wins.
        """
        decays = list(process.get('decay_chains'))
        sizes = []
        for leg in process.get('legs'):
            if not leg.get('state') or not decays:
                sizes.append(1)
                continue
            ids = [d.get('legs')[0].get('id') for d in decays]
            if leg.get('id') in ids:
                decay = decays.pop(ids.index(leg.get('id')))
                sizes.append(len(decay.get_legs_with_decays()) - 1)
            else:
                sizes.append(1)
        return sizes

    def compute_crossing_tables(self, matrix_element):
        """Build the crossing tables as plain python int lists (model-agnostic).

        Returns a dict with, for every crossing code CROSS in
        0..(NEXTERNAL+1)**2-1:
          'spincol' : SPINCOL_CROSS_TABLE[CROSS], the initial-state spin*color
                      average of the crossed process (0 = crossing that must not
                      be applied: out of range, impossible, or an overlapping
                      swap, see get_crossing_permutation);
          'basepid' : flattened CROSS*NEXTERNAL+slot -> representative signed PDG
                      of the particle landing in that crossed slot (conjugated
                      when the leg swapped between the initial and the final
                      state);
          'source'  : flattened CROSS*NEXTERNAL+slot -> 0-based index of the
                      original leg that moved into that slot (FLAVOR is NOT
                      permuted, so this says which FLAVOR entry a slot reads);
          'perm'    : flattened CROSS*NEXTERNAL+slot -> 0-based perm[slot];
          'ic'      : flattened CROSS*NEXTERNAL+slot -> +-1 NSF sign of that slot;
          'nexternal', 'ninitial'.

        Both the fortran (get_iden_cross_lines) and the C++ standalone exporter
        consume this, so the two backends can never disagree about a crossing.
        """
        process = matrix_element.get('processes')[0]
        model = process.get('model')
        # For a decay chain the crossing acts at the production level but the
        # matrix element (and its NEXTERNAL) is over the decay *leaves*, so the
        # crossing tables must span the leaves too: the two z of e+ e- > z z
        # look like an identical pair on the core legs, yet z > mu+ mu- and
        # z > e+ e- make the real final state non-identical (denominator 4, not
        # 8). get_legs_with_decays() is the plain legs for a non-decay process.
        legs = process.get_legs_with_decays() \
            if hasattr(process, 'get_legs_with_decays') else process.get('legs')
        nexternal = len(legs)
        leg_ids = [leg.get('id') for leg in legs]
        # polarization restricts the number of helicity states of a leg; it is
        # attached to the leg, and a crossing moves legs around, so carry it.
        polarizations = [leg.get('polarization') for leg in legs]

        # Per LEAF: the size of the production block it belongs to, and whether
        # it is 'countable' for the identical-final factor. A crossing permutes
        # production legs, so a decaying leg's whole block (its >1 leaves) moves
        # as a unit; the CROSS codes can only transpose single leaves, so any
        # crossing that would carry a block leaf into the initial state (splitting
        # the block, or making a decaying resonance an initial particle) is
        # rejected below. block_size is 1 for every leaf of a non-decay process,
        # so decay chains are the only ones this constrains.
        block_size = []
        # Referenced through the class, not self: the C++/mg7 exporters call
        # compute_crossing_tables unbound with a non-Fortran self (see the
        # get_iden_cross_lines docstring), which has no _leaf_block_sizes.
        for size in ProcessExporterFortran._leaf_block_sizes(process):
            block_size.extend([size] * size)
        assert len(block_size) == nexternal, \
            'leaf block sizes %s do not span NEXTERNAL %d' % (block_size,
                                                              nexternal)
        # A block leaf (size > 1) is a decay product locked inside a resonance:
        # it never counts toward the identical-final factor at the leaf level
        # (that factor is resonance-level, see ident_resonance below). A single
        # leaf (size 1) is a genuine external and does count.
        countable = [1 if size == 1 else 0 for size in block_size]

        def particle(pdg):
            return model.get('particle_dict')[pdg]

        ninitial = len([leg for leg in legs if not leg.get('state')])

        spincol = []
        basepid = []
        source = []
        perm_flat = []
        ic_flat = []
        # CROSS = I*(NEXTERNAL+1)+J with I and J both in 0..NEXTERNAL.
        for cross in range((nexternal + 1) * (nexternal + 1)):
            perm, ic, valid = ProcessExporterFortran.get_crossing_permutation(
                cross, nexternal)
            if not valid:
                # Overlapping-swap code: pure redundancy, and inconsistent
                # between GET_PDG_FOR_FLAVOR and APPLY_CROSSING (see
                # get_crossing_permutation). A 0 spin*color marks it as a
                # crossing that must not be applied, exactly as for one that
                # genuinely cannot be; both SMATRIX and GET_PDG_FOR_FLAVOR then
                # refuse it via GET_SPINCOL_CROSS==0.
                spincol.append(0)
                slot_ids = list(leg_ids)
            else:
                try:
                    # A leg that swapped between the initial and the final state
                    # is seen as its own antiparticle by the crossed process.
                    slot_ids = [leg_ids[perm[slot]] if ic[slot] == 1
                                else particle(leg_ids[perm[slot]]).get_anti_pdg_code()
                                for slot in range(nexternal)]

                    # Two codes that name no crossing, rejected exactly like an
                    # impossible one: a 0 spin*color makes SMATRIX and
                    # GET_PDG_FOR_FLAVOR both return a null result. slot_ids is
                    # still the permuted signature so the IDS_BASE/BASEPID
                    # rebuild sanity below stays consistent. GET_CROSS_PERM
                    # applies the same two rules at runtime.
                    #
                    # 1. A leg conjugated without changing side. The two legs of
                    #    a same-side transposition are both conjugated while
                    #    neither moves across, which is no crossing at all: for
                    #    2 -> N that is the beam swap (XI==2 / XJ==1), giving
                    #    e.g. u~ g > e+ ve d, not even charge conserving; for
                    #    1 -> N it is every XJ swap.
                    # 2. A decay-block leaf carried across the initial/final
                    #    line: it would split the block (pull one decay product
                    #    into the initial state) or make a decaying resonance an
                    #    initial particle. For a non-decay process every
                    #    block_size is 1, so this one never fires.
                    if any(ic[slot] == -1 and
                           ((slot < ninitial) == (perm[slot] < ninitial)
                            or block_size[perm[slot]] > 1)
                           for slot in range(nexternal)):
                        spincol.append(0)
                    else:
                        # The crossing always keeps slots 1..ninitial initial.
                        factor = 1
                        for slot in range(ninitial):
                            pol = polarizations[perm[slot]]
                            factor *= len(pol) if pol else \
                                len(particle(slot_ids[slot]).get_helicity_states())
                            # get('color') is signed for antiparticles; only the
                            # size of the representation matters for the average.
                            factor *= abs(particle(slot_ids[slot]).get('color'))
                        spincol.append(factor)
                except (KeyError, IndexError):
                    spincol.append(0)
                    slot_ids = list(leg_ids)

            basepid.extend(slot_ids)
            source.extend(perm[slot] for slot in range(nexternal))
            perm_flat.extend(perm)
            ic_flat.extend(ic)

        # Per-particle spin*color (states * |color repr|), for every base leg.
        # It is conjugation-invariant (a particle and its antiparticle share
        # both), so a crossing's initial-state spin*color is just the product of
        # these over the legs that land in the initial slots -- which is how
        # GET_SPINCOL_CROSS recomputes SPINCOL_CROSS_TABLE at runtime from the
        # NEXTERNAL-long SPINCOL_PART instead of the NCROSS-long table.
        spincol_part = []
        for slot in range(nexternal):
            pol = polarizations[slot]
            nspin = len(pol) if pol else \
                len(particle(leg_ids[slot]).get_helicity_states())
            spincol_part.append(nspin * abs(particle(leg_ids[slot]).get('color')))

        # Per-particle base PDG and its charge conjugate, one entry per base
        # leg. GET_IDENT_CROSS rebuilds BASEPID_CROSS_TABLE / SRC_CROSS_TABLE at
        # runtime from these two NEXTERNAL-long tables plus the crossing PERM/IC,
        # instead of storing the two NCROSS*NEXTERNAL-long tables.
        ids_base = list(leg_ids)
        antipid_base = [particle(pid).get_anti_pdg_code() for pid in leg_ids]

        # ident_resonance: the part of the identical-final factor a crossing
        # leaves untouched. A crossing only ever permutes the single-leaf
        # (countable) legs -- decay blocks stay put -- so the crossed identical
        # factor is (n! over the crossed countable final legs) times this
        # constant. It collects everything a leaf-level count over the crossed
        # legs cannot see: identical resonances decaying identically, and the
        # identical particles locked inside each decay block. base_non_chain is
        # the identical factor of the base's own countable final legs, so
        # dividing it out of the resonance-level identical_particle_factor leaves
        # exactly that constant. For a non-decay process every final leg is
        # countable and there are no resonances, so base_non_chain equals the
        # whole identical factor and ident_resonance is 1 -- GET_IDENT_CROSS then
        # reduces to the historical plain leaf count.
        final_countable = collections.defaultdict(int)
        for slot in range(ninitial, nexternal):
            if countable[slot]:
                final_countable[(leg_ids[slot],
                                 tuple(polarizations[slot] or []))] += 1
        base_non_chain = 1
        for count in final_countable.values():
            base_non_chain *= math.factorial(count)
        identical = matrix_element.get('identical_particle_factor')
        assert identical % base_non_chain == 0, \
            'Countable identical factor %d does not divide the identical-' \
            'particle factor %d' % (base_non_chain, identical)
        ident_resonance = identical // base_non_chain

        # Sanity: for the identity crossing, spin*color times the identical
        # factor must rebuild the static IDEN, else this and
        # get_denominator_factor have drifted apart. A decay chain's identical
        # factor is resonance-level (two z decaying the same way count once,
        # differently not at all), so it is checked through
        # identical_particle_factor rather than a leaf count; the initial
        # spin*color (which may carry a sign from an antiparticle beam in
        # get_denominator_factor but not in the abs-based spincol) is only
        # required to divide IDEN.
        if process.get('decay_chains'):
            assert matrix_element.get_denominator_factor() % spincol[0] == 0, \
                'Crossing initial spin*color does not divide IDEN: ' \
                '%s vs %s' % (spincol[0],
                              matrix_element.get_denominator_factor())
        else:
            assert spincol[0] * identical == \
                matrix_element.get_denominator_factor(), \
                'Crossing denominator disagrees with get_denominator_factor: ' \
                '%s*%s vs %s' % (spincol[0], identical,
                                 matrix_element.get_denominator_factor())
        # Sanity: the small per-particle tables reproduce the per-crossing
        # tables the runtime routines used to read. SPINCOL_PART -> the
        # initial-state spin*color; IDS_BASE/ANTIPID_BASE plus the crossing
        # PERM/IC -> BASEPID_CROSS_TABLE / SRC_CROSS_TABLE (checked for the
        # applicable crossings, the only ones GET_IDENT_CROSS is ever asked).
        for cross in range((nexternal + 1) * (nexternal + 1)):
            perm, ic, valid = \
                ProcessExporterFortran.get_crossing_permutation(cross, nexternal)
            expect = 0 if not valid else 1
            if valid:
                for slot in range(ninitial):
                    expect *= spincol_part[perm[slot]]
            assert expect == spincol[cross] or spincol[cross] == 0, \
                'SPINCOL_PART product %s != SPINCOL_CROSS_TABLE %s at CROSS %d' \
                % (expect, spincol[cross], cross)
            if not valid:
                continue
            for slot in range(nexternal):
                bp = ids_base[perm[slot]] if ic[slot] == 1 \
                    else antipid_base[perm[slot]]
                assert bp == basepid[cross * nexternal + slot] and \
                    perm[slot] == source[cross * nexternal + slot], \
                    'IDS_BASE/ANTIPID_BASE rebuild != BASEPID/SRC at CROSS ' \
                    '%d slot %d' % (cross, slot)

        return {'spincol': spincol, 'spincol_part': spincol_part,
                'ids_base': ids_base, 'antipid_base': antipid_base,
                'basepid': basepid, 'source': source,
                'perm': perm_flat, 'ic': ic_flat,
                'countable': countable, 'ident_resonance': ident_resonance,
                'nexternal': nexternal, 'ninitial': ninitial}

    def _flavor_rep_rows(self, matrix_element):
        """PDG-table row representing each madevent / C++ / mg7 flavor index.

        The two tables involved are indexed differently and only look alike:

        * ``_build_flav_pdg_tables`` is indexed by ``compute_flavor_masks()`` --
          ONE ROW PER PHYSICAL FLAVOR COMBINATION (15 rows for ``Q Q~ > t t~
          Q Q~`` with three quark flavors).
        * those backends' flavor index counts the COUPLING-EQUIVALENCE CLASSES
          of ``get_external_flavors_with_iden()`` (3 for the same matrix
          element), and the FLAVOR table they read is built from each class's
          representative ``flav[0]`` -- see the ``get_flavor_matrix`` fills.

        Row ``f`` of the first table is the representative of class ``f`` only
        while the leading masks rows happen to BE the representatives, which
        stops holding from three merged flavors on: for ``Q Q~ > t t~ Q Q~``
        class 2 (``q q~' > t t~ q q~'``, the mixed t-channel one) is masks row 3,
        while row 2 is ``q q~ > t t~ q'' q~''``, a member of class 1. Taking the
        ordinal therefore names a process the flavor index does not select, and
        the consumers (partition_crossing_classes' routing, the recorded-crossing
        intersection behind crossed_flavors.dat, the C++ demo_pdg table) match on
        exactly that signature.

        So look the representative up instead of assuming it. Returns one
        0-based row per flavor class. The ordinal is kept as a fall-back for a
        representative that cannot be located -- not expected, decay chains span
        the leaves on both sides and do line up, but a wrong row is a better
        outcome than a traceback in a table this deep in the exporter.
        """
        masks = matrix_element.compute_flavor_masks()
        classes = list(matrix_element.get_external_flavors_with_iden())
        rowof = {tuple(mask): row for row, mask in enumerate(masks)}
        rows = []
        for flav0, members in enumerate(classes):
            row = rowof.get(tuple(members[0])) if members else None
            if row is None:
                logger.debug(
                    'Crossing: flavor class %d of %s has no row in the flavor '
                    'mask table; falling back to the ordinal.'
                    % (flav0, matrix_element.get('processes')[0].shell_string()))
                row = flav0 if flav0 < len(masks) else 0
            rows.append(row)
        return rows

    def compute_crossing_pdg_entries(self, matrix_element, zero_based=True):
        """Enumerate the reachable extended flavor indices and their crossed PDG.

        Returns a list of ``(index, cross, flav0, pdg_tuple)`` for every crossing
        code CROSS that can actually be applied (SPINCOL_CROSS_TABLE[CROSS] != 0,
        i.e. skipping the out-of-range / impossible / overlapping-swap codes) and
        every flavor ``flav0`` in ``0..NFLAV-1``:

        * ``index`` -- the extended flavor index that selects (CROSS, flav0),
          decoded 0-based as ``cross*NFLAV + flav0`` (``zero_based=False`` gives
          the 1-based fortran form). **NFLAV here is the madevent / C++ / mg7
          one**, ``get_external_flavors_with_iden()`` -- the count those backends
          size their flavor table by, deliberately not the STANDALONE fortran
          NFLAV, which comes from _build_flav_table_flat (compute_flavor_masks)
          and is a different, usually larger number: 1 vs 2 for
          ``p p > w+ j, w+ > e+ ve``, 2 vs 4 for ``p p > z j``, 1/1/9 vs 1/4/12
          for ``p p > j j``. See the NFLAV comment in get_crossing_routines.
          So ``index`` is meaningful to partition_crossing_classes (madevent
          routing) and to the C++ demo_pdg table, and NOT to the standalone
          fortran PY_<prefix>GET_PDG_FOR_FLAVOR: a caller holding a standalone
          module must take NFLAV from PY_<prefix>GET_FLAVOR_LAYOUT and build the
          index itself (reweight_interface.build_cross_resolve does). ``cross``
          and ``pdg_tuple`` carry no such convention and are good everywhere.
        * ``cross``  -- the crossing code (0 == identity).
        * ``flav0``  -- the 0-based reduced flavor.
        * ``pdg_tuple`` -- the *signed physical* PDG of each leg, in the leg order
          the momenta must be supplied in for that index (legs permuted and
          conjugated where they swapped between the initial and the final state).

        This is the python twin of the fortran runtime GET_PDG_FOR_FLAVOR *for
        the signature*: the C++ and mg7 standalones have no runtime PDG
        accessor, so their crossed PDG signatures are computed here instead (the
        same logic that fills the check_sa demo table). The backends agree on
        which PDG tuple a (CROSS, flavor) names; they do NOT share one index
        convention, see ``index`` above. Both helpers are referenced through the
        class so a non-Fortran ``self`` (the C++/mg7 exporter, or a throwaway)
        can reuse them unbound.
        """
        tables = ProcessExporterFortran.compute_crossing_tables(
            self, matrix_element)
        spincol = tables['spincol']
        perm = tables['perm']
        ic = tables['ic']
        nx = tables['nexternal']
        ncross = len(spincol)
        n_flav = len(matrix_element.get_external_flavors_with_iden())
        _, pdg_flat, antipdg_flat = \
            ProcessExporterFortran._build_flav_pdg_tables(self, matrix_element)
        # The pdg tables are indexed by physical flavor combination, not by
        # flavor index; _flavor_rep_rows bridges the two.
        rep_rows = ProcessExporterFortran._flavor_rep_rows(
            self, matrix_element)

        entries = []
        for cross in range(ncross):
            if spincol[cross] == 0:
                continue
            for flav0 in range(n_flav):
                row = rep_rows[flav0]
                pdg = []
                for k in range(nx):
                    src = perm[cross * nx + k]
                    if ic[cross * nx + k] == 1:
                        pdg.append(pdg_flat[row * nx + src])
                    else:
                        pdg.append(antipdg_flat[row * nx + src])
                index = cross * n_flav + flav0
                if not zero_based:
                    index += 1
                entries.append((index, cross, flav0, tuple(pdg)))
        return entries

    def find_reorder_candidates(self, matrix_elements):
        """Modules that keep their own matrix<i>.f ONLY because one flavor class
        is listed with its final legs the other way round.

        Pure analysis -- it changes no routing and no output. It names the work a
        split would have to do, and it is the check that says whether a split is
        worth attempting for a given process at all.

        A module drops its matrix<i>.f only when EVERY flavor routes
        (partition_crossing_classes), so one stubborn class keeps a whole 14-
        diagram matrix element alive. For ``Q Q~ > t t~ Q Q~`` off
        ``Q Q > t t~ Q Q`` that class is the flavor-changing annihilation
        ``q q~ > t t~ q' q~'``: the crossing (I=0, J=5) delivers it as
        ``(q~', q')`` while the module lists ``(q', q~')``. The module cannot fix
        that by relabelling itself -- its leg pattern is shared by all its rows,
        the FLAVOR table carrying unsigned group POSITIONS -- and no single
        ordering suits all three of its classes anyway: flipping it repairs the
        annihilation class and breaks the mixed t-channel one.

        Peeling the class out into its own subprocess, GENERATED in the order the
        crossing reaches, removes the conflict: written that way the process
        keeps its diagrams (7 either way) and its signature matches the crossing
        exactly, so it routes with no permutation applied anywhere at run time.
        That is the point of doing it at generation rather than at the call site:
        diagrams, configs, colour basis, helicity table, leshouche and flavor
        table are then all built together in one order, and none of the
        base->dependent maps needs composing with anything.

        Returns ``{me_index: [(flav0, sigma, base_index, iflav), ...]}`` naming,
        per module, the classes that need peeling; ``sigma`` is the final-leg
        permutation their signature needs (0-based, indexed by the base's crossed
        slot). Modules absent from the dict are already fine -- either they route
        as they are, or a reorder would not save them either.
        """
        n = len(matrix_elements)
        if not n:
            return {}
        nini = matrix_elements[0].get_nexternal_ninitial()[1]

        def canon(pdg):
            return (tuple(pdg[:nini]), tuple(sorted(pdg[nini:])))

        def reorder(crossed, sig):
            if tuple(crossed[:nini]) != tuple(sig[:nini]):
                return None
            nx = len(sig)
            sigma = list(range(nx))
            free = [k for k in range(nini, nx) if crossed[k] != sig[k]]
            taken = set(range(nini)) | set(k for k in range(nini, nx)
                                           if k not in free)
            for k in free:
                for j in range(nini, nx):
                    if j not in taken and sig[j] == crossed[k]:
                        sigma[k] = j
                        taken.add(j)
                        break
                else:
                    return None
            return tuple(sigma)

        sig_by_flav, exact, loose = [], [], []
        for me in matrix_elements:
            sbf, cm_e, cm_l = {}, {}, {}
            for idx, cross, flav0, pdg in \
                    self.compute_crossing_pdg_entries(me, zero_based=False):
                if cross == 0:
                    sbf[flav0] = pdg
                cm_e.setdefault(pdg, (cross, idx, pdg))
                cm_l.setdefault(canon(pdg), (cross, idx, pdg))
            nflav = (max(sbf) + 1) if sbf else 0
            sig_by_flav.append([sbf[f] for f in range(nflav)])
            exact.append(cm_e)
            loose.append(cm_l)

        # Replay the real (exact-match) partition so the answer reflects the
        # bases routing actually picks.
        bases, blocked = [], {}
        for i in range(n):
            hits, ok = [], bool(bases)
            for flav0, sig in enumerate(sig_by_flav[i]):
                hit = None
                for b in bases:
                    cx = exact[b].get(sig)
                    if cx is not None and cx[0] != 0:
                        hit = True
                        break
                if hit is None:
                    ok = False
                    blocked.setdefault(i, []).append(flav0)
            if not ok:
                bases.append(i)

        out = {}
        for i, blocked_flavs in blocked.items():
            if i not in bases:
                continue                      # already routes; nothing to peel
            peel, savable = [], True
            for flav0 in blocked_flavs:
                sig = sig_by_flav[i][flav0]
                found = None
                for b in bases:
                    if b >= i:
                        continue              # only earlier modules are bases
                    cx = loose[b].get(canon(sig))
                    if cx is None or cx[0] == 0:
                        continue
                    sigma = reorder(cx[2], sig)
                    if sigma is not None:
                        found = (flav0, sigma, b, cx[1])
                        break
                if found is None:
                    savable = False           # a reorder would not save it
                    break
                peel.append(found)
            if savable and peel:
                out[i] = peel
        return out

    def partition_crossing_classes(self, matrix_elements):
        """Route each subprocess *flavor* to a base matrix element via crossing.

        The crossing relates whole flavor combinations, not whole modules: a
        flavor-merged matrix element bundles flavors that cross to *different*
        bases (e.g. within a group ``u u~ > u u~`` is a crossing of ``u u > u u``
        while its module-mate ``d d~ > u u~`` is not). So the sharing that lets
        one matrix<i>.f serve several subprocesses is decided per flavor: a
        module can drop its own matrix<i>.f only when EVERY one of its flavors is
        a genuine crossing (cross != 0) of some *base* module's flavor; otherwise
        it stays a base and keeps its own matrix<i>.f.

        Bases are chosen greedily in order. Returns ``(bases, routing)``:

        * ``bases``   -- the matrix_element indices that keep their own
          matrix<i>.f (their SMATRIX, driven by an extended FLAV_IDX, also serves
          the flavors routed to them).
        * ``routing`` -- a list parallel to ``matrix_elements``; ``routing[i]``
          has one ``(base_index, iflav)`` per flavor of member ``i`` (in flavor
          order), naming the base module whose ``SMATRIX`` evaluates that flavor
          and the 1-based extended ``FLAV_IDX`` to call it with. A base routes
          each of its own flavors to itself with the plain (cross 0) index.

        Signatures are the crossed physical PDG tuples of compute_crossing_pdg_
        entries, the same key check_crossing matches on, so the momentum order a
        member supplies already matches what the base SMATRIX expects for that
        index.
        """
        n = len(matrix_elements)
        # Per ME: identity signature of each flavor (flavor order) and the map
        # from any crossed signature it can reach to (cross, 1-based FLAV_IDX).
        sig_by_flav = []
        crossmap = []
        for me in matrix_elements:
            sbf = {}
            cm = {}
            for idx, cross, flav0, pdg in \
                    self.compute_crossing_pdg_entries(me, zero_based=False):
                if cross == 0:
                    sbf[flav0] = pdg
                cm.setdefault(pdg, (cross, idx))
            nflav = (max(sbf) + 1) if sbf else 0
            sig_by_flav.append([sbf[f] for f in range(nflav)])
            crossmap.append(cm)

        bases = []
        routing = [None] * n
        for i in range(n):
            cover = []
            coverable = bool(bases)   # nothing to route to before the first base
            for sig in sig_by_flav[i]:
                hit = None
                for b in bases:
                    cx = crossmap[b].get(sig)
                    if cx is not None and cx[0] != 0:  # a genuine crossing of b
                        hit = (b, cx[1])
                        break
                if hit is None:
                    coverable = False
                    break
                cover.append(hit)
            if coverable:
                routing[i] = cover            # drop i's matrix.f; route each flavor
            else:
                bases.append(i)               # i keeps its own matrix.f (a base)
                routing[i] = [(i, crossmap[i][sig][1]) for sig in sig_by_flav[i]]
        return bases, routing

    def compute_crossgroup_routing(self, subproc_groups):
        """Cross-group crossing (Track B): find whole subprocess GROUPS whose
        matrix element is a crossing of another group's, so the dependent group
        can REUSE (symlink) the base group's compiled matrix element instead of
        generating and compiling its own. Used for e.g. lepton/photon beams where
        each initial state lands in its own single-process P directory and the
        crossings relate different P directories (partition_crossing_classes is
        group-agnostic -- it clusters by crossed-PDG signature -- so it is fed the
        flat list of every group's matrix elements).

        Returns a dict keyed by ``(group_enum_idx, me_idx)`` for the DEPENDENT
        members only; each value carries the base group's directory, the base
        SMATRIX's proc_id, the base matrix_element (for the COLMAP/CONFIGMAP
        remaps) and the crossed 1-based FLAV_IDX per flavor. Bases are absent
        (they keep their own matrix element). Only a dependent whose EVERY flavor
        crosses to a SINGLE base matrix element is routed; anything else keeps its
        own matrix element (so the sharing is always a clean whole-ME reuse).
        """
        if not self.opt.get('use_crossing', False):
            return {}
        # Consider only groups whose every member is a within-group BASE (no
        # router). A group that ALREADY has within-group crossing routing (the
        # hadronic p p groups where several crossings co-locate under a `j`
        # multiparticle) is left to Track A -- mixing its base(s) with those
        # routers is fragile, so it is excluded here. The lepton/photon single-
        # process groups are all bases; a p p run additionally exposes the cross-
        # P-directory crossings that within-group routing cannot reach (e.g.
        # g g > q q~ vs q q~ > g g, in their own P directories).
        flat = []                       # (group_enum_idx, me_idx, matrix_element)
        for gi, group in enumerate(subproc_groups):
            mes_g = group.get('matrix_elements')
            # A group that breaks crossing (pinned s-channel, or a perturbative
            # / loop-induced matrix element) has no crossing tables -- skip it
            # before partition_crossing_classes, which would index past the end.
            if any(self.breaks_crossing_symmetry(proc)
                   for me in mes_g for proc in me.get('processes')):
                continue
            g_bases, _ = self.partition_crossing_classes(mes_g)
            if len(g_bases) < len(mes_g):
                continue                # within-group routing -> leave to Track A
            for mi, me in enumerate(mes_g):
                flat.append((gi, mi, me))
        if not flat:
            return {}
        # A pinned s-channel does not survive crossing (see breaks_crossing_
        # symmetry): fall back to independent matrix elements.
        if any(self.breaks_crossing_symmetry(proc)
               for (_, _, me) in flat for proc in me.get('processes')):
            return {}
        mes = [me for (_, _, me) in flat]
        bases, routing = self.partition_crossing_classes(mes)
        result = {}
        for flat_i, (gi, mi, me) in enumerate(flat):
            if flat_i in bases:
                continue
            route = routing[flat_i]                 # per flavor: (base_flat, iflav)
            base_flats = set(bflat for (bflat, _) in route)
            if len(base_flats) != 1:
                # flavors cross to different bases (a merged group): no single ME
                # to symlink, keep this member's own matrix element.
                continue
            base_gi, base_mi, base_me = flat[base_flats.pop()]
            base_group = subproc_groups[base_gi]
            result[(gi, mi)] = {
                'base_dir': 'P%d_%s' % (base_group.get('number'),
                                        base_group.get('name')),
                'base_proc_id': base_mi + 1,
                'base_me': base_me,
                'flav_idx': [iflav for (_, iflav) in route],
            }
        return result

    def compute_ghremap(self, matrix_element, allow_reverse=True):
        """Build the good-helicity remap table for the crossing filter.

        The good-helicity filter (GOODHEL) is shared by all crossings of a
        flavor, but a crossing permutes and flips helicities, so identity and
        crossed have different good-helicity SETS. The crossed set is the
        identity set transformed by the crossing's own helicity-row permutation
        sigma, where sigma sends identity row h to the row whose config is
        (ic[k]*nhel[perm[k], h])_k -- permute the legs and flip the helicity of
        the swapped ones, with (perm, ic) from get_crossing_permutation. A
        crossed row H is therefore good iff the identity row sigma^-1(H) is
        good, so the filter can stay shared as long as it is consulted (and
        trained) through sigma^-1. See standalone-cross-symmetry memory.

        Returns a flat list of length NCROSS*NCOMB indexed CROSS*NCOMB + H (H
        the 0-based helicity row), each entry being the 0-based identity row
        sigma^-1(H) that gates crossed row H, or None when the crossing must
        not be filtered (compute every helicity, never train):
          - CROSS==0 -> the identity (entry == H): the uncrossed path is
            completely unchanged;
          - a genuine crossing whose active partners are all final particles ->
            sigma^-1(H);
          - an initial-initial swap, or an invalid / inapplicable crossing ->
            None. The sigma relation only holds when the active partners are
            final; an initial-initial swap breaks it (it overcounts at 2->3),
            so those disable the filter and keep the full-computation result.

        allow_reverse must match the order the NHEL table is emitted in for the
        backend consuming the result (True for the fortran get_helicity_lines,
        False for the C++ get_helicity_matrix).
        """
        # Reference the class explicitly (not self) so the C++ standalone
        # exporter can reuse this via ProcessExporterFortran.compute_ghremap
        # with a non-Fortran self, exactly like compute_crossing_tables.
        tables = ProcessExporterFortran.compute_crossing_tables(
            self, matrix_element)
        spincol = tables['spincol']
        nexternal = tables['nexternal']
        ninitial = tables['ninitial']
        base = nexternal + 1
        ncross = base * base
        hel_matrix = [tuple(row) for row in
                      matrix_element.get_helicity_matrix(allow_reverse)]
        ncomb = len(hel_matrix)
        row_index = {row: h for h, row in enumerate(hel_matrix)}

        remap = []
        for cross in range(ncross):
            perm, ic, valid = \
                ProcessExporterFortran.get_crossing_permutation(cross, nexternal)
            i_part, j_part = cross // base, cross % base
            final_only = ((i_part in (0, 1) or i_part > ninitial) and
                          (j_part in (0, 2) or j_part > ninitial))
            derivable = (valid and spincol[cross] != 0 and
                         (cross == 0 or final_only))
            block = [None] * ncomb
            if derivable:
                for h in range(ncomb):
                    config = tuple(ic[k] * hel_matrix[h][perm[k]]
                                   for k in range(nexternal))
                    big_h = row_index.get(config)
                    if big_h is None:
                        # The permuted config is not a table row: the crossing
                        # is not a bijection on the rows, so it cannot be
                        # derived. Disable the filter for it (safe fallback).
                        block = [None] * ncomb
                        break
                    block[big_h] = h
            remap.extend(block)
        return remap

    def compute_ghfilt(self, matrix_element, allow_reverse=True):
        """Per-crossing filterability flags for the runtime good-helicity remap.

        Returns a list of length NCROSS: 1 if crossing CROSS is filterable (its
        helicity-row permutation sigma is a clean bijection -- see
        compute_ghremap), 0 otherwise (initial-initial swap, inapplicable, or a
        non-bijection). This is the small flag table that replaces the full
        GHREMAP(NCROSS*NCOMB) row table: at runtime the row map itself is
        recomputed by permuting+sign-flipping the config and re-encoding it (see
        the CROSS_GHIDX routine), so only the per-crossing yes/no survives as
        DATA. A whole compute_ghremap block is either fully derivable or fully
        None, so this loses nothing."""
        # Reference the class explicitly (not self) so a non-Fortran self (the
        # C++ standalone exporter) can reuse this via
        # ProcessExporterFortran.compute_ghfilt, exactly like compute_ghremap.
        remap = ProcessExporterFortran.compute_ghremap(
            self, matrix_element, allow_reverse)
        nexternal = matrix_element.get_nexternal_ninitial()[0]
        ncross = (nexternal + 1) * (nexternal + 1)
        ncomb = len(remap) // ncross
        return [0 if all(x is None for x in remap[c * ncomb:(c + 1) * ncomb])
                else 1 for c in range(ncross)]

    @staticmethod
    def format_integer_data_lines(name, values, per_line=10):
        """Emit 'DATA (name(I),I=a,b) /.../' lines for a 0-based table."""
        lines = []
        for start in range(0, len(values), per_line):
            chunk = values[start:start + per_line]
            lines.append('      DATA (%s(I),I=%d,%d) /%s/' %
                         (name, start, start + len(chunk) - 1,
                          ','.join(str(value) for value in chunk)))
        return '\n'.join(lines)

    def get_icolamp_lines(self, mapconfigs, matrix_element, num_matrix_element):
        """Return the ICOLAMP matrix, showing which JAMPs contribute to
        which configs (diagrams)."""

        ret_list = []

        booldict = {False: ".false.", True: ".true."}

        if not matrix_element.get('color_basis'):
            # No color, so only one color factor. Simply write a ".true." 
            # for each config (i.e., each diagram with only 3 particle
            # vertices
            configs = len(mapconfigs)
            ret_list.append("DATA(icolamp(1,i,%d),i=1,%d)/%s/" % \
                            (num_matrix_element, configs,
                             ','.join([".true." for i in range(configs)])))
            return ret_list


        # There is a color basis - create a list showing which JAMPs have
        # contributions to which configs

        # Only want to include leading color flows, so find max_Nc. This is
        # about color flows, so always the trace basis
        color_basis = matrix_element.get('color_basis').get_flow_basis()

        # We don't want to include the power of Nc's which come from the potential
        # loop color trace (i.e. in the case of a closed fermion loop for example)
        # so we subtract it here when computing max_Nc
        max_Nc = max(sum([[(v[4]-v[5]) for v in val] for val in
                                                      color_basis.values()],[]))

        # Crate dictionary between diagram number and JAMP number
        diag_jamp = {}
        for ijamp, col_basis_elem in \
                enumerate(sorted(color_basis.keys())):
            for diag_tuple in color_basis[col_basis_elem]:
                # Only use color flows with Nc == max_Nc. However, notice that
                # we don't want to include the Nc power coming from the loop
                # in this counting.
                if (diag_tuple[4]-diag_tuple[5]) == max_Nc:
                    diag_num = diag_tuple[0] + 1
                    # Add this JAMP number to this diag_num
                    diag_jamp[diag_num] = diag_jamp.setdefault(diag_num, []) + \
                                          [ijamp+1]
                else:
                    self.proc_characteristic['single_color'] = False

        colamps = ijamp + 1
        for iconfig, num_diag in enumerate(mapconfigs):        
            if num_diag == 0:
                continue

            # List of True or False 
            bool_list = [(i + 1 in diag_jamp[num_diag]) for i in range(colamps)]
            # Add line
            ret_list.append("DATA(icolamp(i,%d,%d),i=1,%d)/%s/" % \
                                (iconfig+1, num_matrix_element, colamps,
                                 ','.join(["%s" % booldict[b] for b in \
                                           bool_list])))

        return ret_list

    @staticmethod
    def get_multi_channel_dictionary(diagrams, config_map): 
        """diagrams should be from matrix_element.get('diagrams')"""


        config_to_diag_dict = {}
        if config_map:
            # In this case, we need to sum up all amplitudes that have
            # identical topologies, as given by the config_map (which
            # gives the topology/config for each of the diagrams
            # Combine the diagrams with identical topologies
            for idiag, diag in enumerate(diagrams):
                if config_map[idiag] == 0:
                    continue
                try:
                    config_to_diag_dict[config_map[idiag]].append(idiag)
                except KeyError:
                    config_to_diag_dict[config_map[idiag]] = [idiag]
        else:
            # Get minimum legs in a vertex
            vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
                diagrams if diag.get_vertex_leg_numbers()!=[]]
            minvert = min(vert_list) if vert_list!=[] else 0

            for idiag, diag in enumerate(diagrams):
                # Ignore any diagrams with 4-particle vertices.
                if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                    continue
                config_to_diag_dict[config_map[idiag]] = [idiag]

        return  config_to_diag_dict


    def get_amp2_lines(self, matrix_element, config_map = [], replace_dict=None):
        """Return the amp2(i) = sum(amp for diag(i))^2 lines"""

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        # Get minimum legs in a vertex
        vert_list = [max(diag.get_vertex_leg_numbers()) for diag in \
                     matrix_element.get('diagrams') if diag.get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        ret_lines = []
        if config_map:
            # In this case, we need to sum up all amplitudes that have
            # identical topologies, as given by the config_map (which
            # gives the topology/config for each of the diagrams
            diagrams = matrix_element.get('diagrams')
            config_to_diag_dict = self.get_multi_channel_dictionary(diagrams, config_map)
            # Write out the AMP2s summing squares of amplitudes belonging
            # to eiher the same diagram or different diagrams with
            # identical propagator properties.  Note that we need to use
            # AMP2 number corresponding to the first diagram number used
            # for that AMP2.
            
            for config in sorted(config_to_diag_dict.keys()):

                line = "AMP2(%(num)d)=AMP2(%(num)d)+" % \
                       {"num": (config_to_diag_dict[config][0] + 1)}

                amp = "+".join(["AMP(%(num)d)" % {"num": a.get('number')} for a in \
                                  sum([diagrams[idiag].get('amplitudes') for \
                                       idiag in config_to_diag_dict[config]], [])])
                
                # Not using \sum |M|^2 anymore since this creates troubles
                # when ckm is not diagonal due to the JIM mechanism.
                if '+' in amp:
                    amp = "(%s)*dconjg(%s)" % (amp, amp)
                else:
                    amp = "%s*dconjg(%s)" % (amp, amp)
                
                line =  line + "%s" % (amp)
                #line += " * get_channel_cut(p, %s) " % (config)
                ret_lines.append(line)
        else:
            for idiag, diag in enumerate(matrix_element.get('diagrams')):
                # Ignore any diagrams with 4-particle vertices.
                if diag.get_vertex_leg_numbers()!=[] and max(diag.get_vertex_leg_numbers()) > minvert:
                    continue
                # Now write out the expression for AMP2, meaning the sum of
                # squared amplitudes belonging to the same diagram
                line = "AMP2(%(num)d)=AMP2(%(num)d)+" % {"num": (idiag + 1)}
                line += "+".join(["AMP(%(num)d)*dconjg(AMP(%(num)d))" % \
                                  {"num": a.get('number')} for a in \
                                  diag.get('amplitudes')])
                ret_lines.append(line)

        return ret_lines

    #===========================================================================
    # Returns the data statements initializing the coeffictients for the JAMP
    # decomposition. It is used when the JAMP initialization is decided to be 
    # done through big arrays containing the projection coefficients.
    #===========================================================================    
    def get_JAMP_coefs(self, color_amplitudes, color_basis=None, tag_letter="",\
                       n=50, Nc_value=3):
        """This functions return the lines defining the DATA statement setting
        the coefficients building the JAMPS out of the AMPS. Split rows in
        bunches of size n.
        One can specify the color_basis from which the color amplitudes originates
        so that there are commentaries telling what color structure each JAMP
        corresponds to."""
        
        if(not isinstance(color_amplitudes,list) or 
           not (color_amplitudes and isinstance(color_amplitudes[0],list))):
                raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_coefs")

        res_list = []
        my_cs = color.ColorString()
        for index, coeff_list in enumerate(color_amplitudes):
            # Create the list of the complete numerical coefficient.
            coefs_list=[coefficient[0][0]*coefficient[0][1]*\
                        (fractions.Fraction(Nc_value)**coefficient[0][3]) for \
                        coefficient in coeff_list]
            # Create the list of the numbers of the contributing amplitudes.
            # Mutliply by -1 for those which have an imaginary coefficient.
            ampnumbers_list=[coefficient[1]*(-1 if coefficient[0][2] else 1) \
                              for coefficient in coeff_list]
            # Find the common denominator.  
            commondenom=abs(reduce(math.gcd, coefs_list).denominator)
            num_list=[(coefficient*commondenom).numerator \
                      for coefficient in coefs_list]
            res_list.append("DATA NCONTRIBAMPS%s(%i)/%i/"%(tag_letter,\
                                                         index+1,len(num_list)))
            res_list.append("DATA DENOMCCOEF%s(%i)/%i/"%(tag_letter,\
                                                         index+1,commondenom))
            if color_basis:
                my_cs.from_immutable(sorted(color_basis.keys())[index])
                res_list.append("C %s" % repr(my_cs))
            for k in range(0, len(num_list), n):
                res_list.append("DATA (NUMCCOEF%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in num_list[k:k + n]])))
                res_list.append("DATA (AMPNUMBERS%s(%3r,i),i=%6r,%6r) /%s/" % \
                    (tag_letter,index + 1, k + 1, min(k + n, len(num_list)),
                                 ','.join(["%6r" % i for i in ampnumbers_list[k:k + n]])))
                pass
        return res_list


    def get_JAMP_lines_split_order(self, col_amps, split_order_amps, 
          split_order_names=None, JAMP_format="JAMP(%s,{0})", AMP_format="AMP(%s)",
          orbit=False, proc_prefix=''):
        """Return the JAMP = sum(fermionfactor * AMP(i)) lines from col_amps 
        defined as a matrix element or directly as a color_amplitudes dictionary.
        The split_order_amps specifies the group of amplitudes sharing the same
        amplitude orders which should be put in together in a given set of JAMPS.
        The split_order_amps is supposed to have the format of the second output 
        of the function get_split_orders_mapping function in helas_objects.py.
        The split_order_names is optional (it should correspond to the process
        'split_orders' attribute) and only present to provide comments in the
        JAMP definitions in the code."""

        # Let the user call get_JAMP_lines_split_order directly from a 
        error_msg="Malformed '%s' argument passed to the "+\
                 "get_JAMP_lines_split_order function: %s"%str(split_order_amps)
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            color_amplitudes=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error(error_msg%'col_amps')
        else:
            raise MadGraph5Error(error_msg%'col_amps')
        
        # Verify the sanity of the split_order_amps and split_order_names args
        if isinstance(split_order_amps,list):
            for elem in split_order_amps:
                if len(elem)!=2:
                    raise MadGraph5Error(error_msg%'split_order_amps')
                # Check the first element of the two lists to make sure they are
                # integers, although in principle they should all be integers.
                if not isinstance(elem[0],tuple) or \
                   not isinstance(elem[1],tuple) or \
                   not isinstance(elem[0][0],int) or \
                   not isinstance(elem[1][0],int):
                    raise MadGraph5Error(error_msg%'split_order_amps')
        else:
            raise MadGraph5Error(error_msg%'split_order_amps')
        
        if not split_order_names is None:
            if isinstance(split_order_names,list):
                # Should specify the same number of names as there are elements
                # in the key of the split_order_amps.
                if len(split_order_names)!=len(split_order_amps[0][0]):
                    raise MadGraph5Error(error_msg%'split_order_names')
                # Check the first element of the list to be a string
                if not isinstance(split_order_names[0],str):
                    raise MadGraph5Error(error_msg%'split_order_names')                    
            else:
                raise MadGraph5Error(error_msg%'split_order_names')                
        
        # Now scan all contributing orders to be individually computed and 
        # construct the list of color_amplitudes for JAMP to be constructed
        # accordingly.
        res_list=[]
        max_tmp = 0
        for i, amp_order in enumerate(split_order_amps):
            col_amps_order = []
            for jamp in color_amplitudes:
                col_amps_order.append([col_amp for col_amp in jamp if col_amp[1] in amp_order[1]])
            if split_order_names:
                res_list.append('C JAMPs contributing to orders '+' '.join(
                              ['%s=%i'%order for order in zip(split_order_names,
                                                                amp_order[0])]))
            if self.opt['export_format'] in ['madloop_matchbox']:
                res_list.extend(self.get_JAMP_lines(col_amps_order,
                                   JAMP_format=JAMP_format.format(str(i+1)),
                                   JAMP_formatLC="LN"+JAMP_format.format(str(i+1)))[0])
            else:
                # Only one set of definitions fits in the arrays the
                # template declares, so the orbit version is only used when
                # there is a single order to compute.
                toadd, nb_tmp = self.get_JAMP_lines(col_amps_order,
                                   JAMP_format=JAMP_format.format(str(i+1)),
                                   orbit=orbit and len(split_order_amps) == 1,
                                   proc_prefix=proc_prefix,
                                   symmetry_source=col_amps if isinstance(
                                       col_amps,
                                       helas_objects.HelasMatrixElement) else None)
                res_list.extend(toadd)
                max_tmp = max(max_tmp, nb_tmp)         

        return res_list, max_tmp


    def get_JAMP_lines(self, col_amps, JAMP_format="JAMP(%s)", AMP_format="AMP(%s)",
                       split=-1, orbit=False, proc_prefix='',
                       symmetry_source=None):
        """Return the JAMP = sum(fermionfactor * AMP(i)) lines from col_amps
        defined as a matrix element or directly as a color_amplitudes dictionary,
        Jamp_formatLC should be define to allow to add LeadingColor computation
        (usefull for MatchBox)
        The split argument defines how the JAMP lines should be split in order
        not to be too long.
        With orbit on, the common sub-expressions are looked for in a way which
        respects the permutations leaving the color basis invariant, so that
        they can be written as one recipe per orbit (see optimise_jamp)."""

        # Let the user call get_JAMP_lines directly from a MatrixElement or from
        # the color amplitudes lists.
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            color_amplitudes=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                color_amplitudes=col_amps
            else:
                raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_lines")
        else:
            raise MadGraph5Error("Incorrect col_amps argument passed to get_JAMP_lines")

        # the coefficient matrix the optimisation below works on, built once
        # from the same color amplitudes the expanded lines are written from
        all_element = self.jamp_matrix(color_amplitudes)
        res_list = []
        for i, coeff_list in enumerate(color_amplitudes):
            # It might happen that coeff_list is empty if this function was
            # called from get_JAMP_lines_split_order (i.e. if some color flow
            # does not contribute at all for a given order).
            # In this case we simply set it to 0.
            if coeff_list==[]:
                res_list.append(((JAMP_format+"=0D0") % str(i + 1)))
                continue
            # Break the JAMP definition into 'n=split' pieces to avoid having
            # arbitrarly long lines.
            first=True
            n = (len(coeff_list)+1 if split<=0 else split) 
            while coeff_list!=[]:
                coefs=coeff_list[:n]
                coeff_list=coeff_list[n:]
                res = ((JAMP_format+"=") % str(i + 1)) + \
                      ((JAMP_format % str(i + 1)) if not first and split>0 else '')

                first=False
                # Optimization: if all contributions to that color basis element have
                # the same coefficient (up to a sign), put it in front
                list_fracs = [abs(coefficient[0][1]) for coefficient in coefs]
                common_factor = False
                diff_fracs = misc.make_unique(list_fracs)
                if len(diff_fracs) == 1 and abs(diff_fracs[0]) != 1:
                    common_factor = True
                    global_factor = diff_fracs[0]
                    res = res + '%s(' % self.coeff(1, global_factor, False, 0)
                
                # loop for JAMP
                for (coefficient, amp_number) in coefs:
                    if not coefficient:
                        continue
                    if common_factor:
                        res = (res + "%s" + AMP_format) % \
                                                   (self.coeff(coefficient[0],
                                                   coefficient[1] / abs(coefficient[1]),
                                                   coefficient[2],
                                                   coefficient[3]),
                                                   str(amp_number))
                    else:
                        res = (res + "%s" + AMP_format) % (self.coeff(coefficient[0],
                                                   coefficient[1],
                                                   coefficient[2],
                                                   coefficient[3]),
                                                   str(amp_number))
    
                if common_factor:
                    res = res + ')'
                res_list.append(res)
        
        if not self.jamp_optim_enabled():
            return res_list, 0
        else:
            saved = list(res_list)

        if len(all_element) > 1000:
            logger.info("Computing Color-Flow optimization [%s term]", len(all_element))
            start_time = time.time()
        else:
            start_time = 0

        res_list = []

        self.myjamp_count = 0
        # The optimisation itself is language neutral (see jamp_optimiser); it
        # is run one step at a time here rather than through
        # optimise_jamp_matrix because the color basis symmetry has to be read
        # off the matrix once the phase has been taken out of it.
        phase = self.jamp_walk_integers(all_element)
        self.jamp_orbits = None
        # the color basis is read from the matrix element, which is not always
        # what is passed here: the split order version hands over one list of
        # color amplitudes per order and says where they came from
        symmetry = self.get_jamp_symmetry(
                        col_amps if symmetry_source is None else symmetry_source,
                        all_element) if orbit and self.jamp_orbit else None
        new_mat, defs = self.optimise_jamp(all_element, symmetry=symmetry)
        self.jamp_apply_phase(new_mat, phase)
        if start_time:
            logger.info("Color-Flow passed to %s term in %ss. Introduce %i contraction", len(new_mat), int(time.time()-start_time), len(defs))

        
        #misc.sprint("number of iteration", self.myjamp_count)
        def format(frac):
            if isinstance(frac, fractions.Fraction):
                if frac.denominator == 1:
                    return str(frac.numerator)
                else:
                    return "%id0/%id0" % (frac.numerator, frac.denominator)
            elif frac.real == frac:
                #misc.sprint(frac.real, frac)
                return ('%.15e' % (frac.real + 0.0)).replace('e','d')
                #str(float(frac.real)).replace('e','d')
            else:
                return ('(%.15e,%.15e)' % (frac.real + 0.0, frac.imag + 0.0)).replace('e','d')
                #str(frac).replace('e','d').replace('j','*imag1')
                
        
        
        # One recipe per orbit rather than one line per definition, when the
        # symmetry allows it and there are enough definitions for the routine
        # rebuilding them to be worth its own code.
        recipes = None
        if symmetry and len(defs) >= self.jamp_orbit_min_def \
                    and self.jamp_tables_allowed():
            nb_amp = (col_amps if symmetry_source is None
                      else symmetry_source).get_number_of_amplitudes()
            if self.jamp_emit == 'tables':
                recipes = self.jamp_orbit_tables(defs, nb_amp)
            else:
                recipes = self.jamp_orbit_recipes(defs, nb_amp)
        self.jamp_recipes = recipes

        if recipes:
            buffer = self.jamp_buffer()
            tmp_name = lambda k: "%s(NGRAPHS+%d)" % (buffer, k)
            defs = recipes['defs']
            res_list.append("C     The definitions below come in orbits of the")
            res_list.append("C     permutations leaving the color basis")
            res_list.append("C     invariant: all of an orbit are the same")
            res_list.append("C     recipe with the amplitudes permuted, so")
            res_list.append("C     only their operands differ. INIT_JAMP works")
            res_list.append("C     those out once, from one recipe per orbit.")
            if recipes.get('recipes'):
                res_list.append(" CALL %sINIT_JAMP()" % proc_prefix)
                res_list.append(" DO ITMP = 1, NB_TMP_JAMP")
                res_list.append("   AMP(NGRAPHS+ITMP) = AMP(TMP_JAMP_A(ITMP))"
                                " + TMP_JAMP_F(ITMP)*AMP(TMP_JAMP_B(ITMP))")
                res_list.append(" ENDDO")
            else:
                if self.jamp_gather:
                    res_list.append("C     the amplitudes of this helicity are")
                    res_list.append("C     read into one array first, so that")
                    res_list.append("C     the definitions below take both")
                    res_list.append("C     their operands from the same place")
                    res_list.append(" DO ITMP = 1, NGRAPHS")
                    res_list.append("   %s(ITMP) = AMP(ITMP)" % buffer)
                    res_list.append(" ENDDO")
                res_list.append("C     the definitions of one level use none")
                res_list.append("C     of each other, so they are sorted by")
                res_list.append("C     the factor in front of the second")
                res_list.append("C     operand and only the last group of")
                res_list.append("C     each level has to multiply")
                res_list.append(" DO ILEV = 1, NB_LEVEL")
                res_list.append("   DO ITMP = TMP_JAMP_L(5*ILEV-4),"
                                " TMP_JAMP_L(5*ILEV-3)")
                res_list.append("     %s(NGRAPHS+ITMP) = %s(TMP_JAMP_A(ITMP))"
                                " + %s(TMP_JAMP_B(ITMP))"
                                % (buffer, buffer, buffer))
                res_list.append("   ENDDO")
                res_list.append("   DO ITMP = TMP_JAMP_L(5*ILEV-3)+1,"
                                " TMP_JAMP_L(5*ILEV-2)")
                res_list.append("     %s(NGRAPHS+ITMP) = %s(TMP_JAMP_A(ITMP))"
                                " - %s(TMP_JAMP_B(ITMP))"
                                % (buffer, buffer, buffer))
                res_list.append("   ENDDO")
                res_list.append("   DO ITMP = TMP_JAMP_L(5*ILEV-2)+1,"
                                " TMP_JAMP_L(5*ILEV-1)")
                res_list.append("     %s(NGRAPHS+ITMP) = %s(TMP_JAMP_A(ITMP))"
                                " + TMP_JAMP_F(TMP_JAMP_L(5*ILEV)+ITMP"
                                "-TMP_JAMP_L(5*ILEV-2))"
                                "*%s(TMP_JAMP_B(ITMP))"
                                % (buffer, buffer, buffer))
                res_list.append("   ENDDO")
                res_list.append(" ENDDO")
        else:
            tmp_name = lambda k: "TMP_JAMP(%d)" % k
            for i, amp1, amp2, frac, nb in defs:
                if amp1 > 0:
                    amp1 = AMP_format % amp1
                else:
                    amp1 = tmp_name(-amp1)
                if amp2 > 0:
                    amp2 = AMP_format % amp2
                else:
                    amp2 = tmp_name(-amp2)

                if frac not in  [1., -1]:
                    res_list.append(' TMP_JAMP(%d) = %s + (%s) * %s ! used %d times' % (i,amp1, format(frac), amp2, nb))
                elif frac == 1.:
                    res_list.append(' TMP_JAMP(%d) = %s +  %s ! used %d times' % (i,amp1, amp2, nb))
                else:
                    res_list.append(' TMP_JAMP(%d) = %s - %s ! used %d times' % (i,amp1, amp2, nb))

        jamp_res = collections.defaultdict(list)
        max_jamp=0
        for (jamp, var), factor in new_mat.items():
            if var > 0:
                name = ("%s(%%s)" % self.jamp_buffer()) % var \
                       if (recipes and self.jamp_gather) else AMP_format % var
            else:
                if recipes and recipes.get('factor_of'):
                    # the definitions were renumbered, and one of them can be
                    # the opposite of the one the optimisation had
                    where, scale = recipes['factor_of'][-var]
                    factor, var = factor * scale, -where
                name = tmp_name(-var)
            if factor not in [1.]:
                jamp_res[jamp].append("(%s)*%s" % (format(factor), name))
            elif factor ==1:
                jamp_res[jamp].append("%s" % (name))
            max_jamp = max(max_jamp, jamp)
        
        
        for i in range(1,max_jamp+1):
            name = JAMP_format % i
            if not jamp_res[i]:
                res_list.append(" %s = 0d0" %(name))
            else:
                res_list.append(" %s = %s" %(name, '+'.join(jamp_res[i])))

        return res_list, len(defs)

    def optimise_jamp_best(self, all_element, symmetry):
        """Taking whole orbits only pays once there is enough of them to share:
        on a small matrix it can end up asking for more additions than the plain
        scan, which is free to take whatever it likes. g g > t t~ g is such a
        case, 46 additions against 39.

        Small matrices are cheap to optimise, so rather than guess where the
        turn is, do both and keep the shorter. Above that size only the orbit
        version is run: it wins by a wide margin on everything that big, and
        the plain scan is the slow one there."""

        orbit_element, orbit_defs = self.optimise_jamp_equivariant(
                                            dict(all_element), symmetry)
        if len(all_element) > self.jamp_compare_max_size:
            return orbit_element, orbit_defs

        orbits = self.jamp_orbits
        plain_element, plain_defs = self.optimise_jamp(dict(all_element))
        if self.jamp_operation_count(plain_element, plain_defs) < \
                    self.jamp_operation_count(orbit_element, orbit_defs):
            self.jamp_orbits = None
            return plain_element, plain_defs
        self.jamp_orbits = orbits
        return orbit_element, orbit_defs

    #===========================================================================
    # Orbit equivariant version of the JAMP optimisation
    #===========================================================================
    # A permutation of the external color indices which maps the color basis
    # onto itself (see color_amp.ColorBasisSymmetry) also permutes the columns
    # of the JAMP matrix, up to a sign. The whole matrix is then invariant, so
    # the sub-expressions the optimisation looks for come in orbits: every one
    # of them is worth exactly as much as the others. Introducing a whole orbit
    # at a time, rather than one sub-expression at a time as the plain scan
    # does, leaves the matrix invariant at every step, and the definitions can
    # be written as one recipe per orbit.

    _blas_available = None

    @classmethod
    def blas_is_available(cls):
        """Whether a BLAS carrying DSYMM can be linked, asked once."""

        if cls._blas_available is None:
            import subprocess, tempfile, shutil
            probe = ("      PROGRAM P\n"
                     "      DOUBLE PRECISION A(1,1),B(1,1),C(1,1)\n"
                     "      A=1D0\n      B=1D0\n      C=0D0\n"
                     "      CALL DSYMM('L','U',1,1,1D0,A,1,B,1,0D0,C,1)\n"
                     "      END\n")
            work = tempfile.mkdtemp()
            cls._blas_available = False
            cls._blas_flags = ''
            try:
                src = os.path.join(work, 'p.f')
                open(src, 'w').write(probe)
                for flags in ('-framework Accelerate', '-lblas'):
                    try:
                        out = subprocess.call(
                            ['gfortran', src, '-o', os.path.join(work, 'p')]
                            + flags.split(),
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
                    except OSError:
                        break
                    if out == 0:
                        cls._blas_available = True
                        cls._blas_flags = flags
                        break
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return cls._blas_available

    @classmethod
    def blas_available_flags(cls):
        """What a BLAS carrying DSYMM needs on the link line, empty when there
        is none. Unlike blas_link_flags this does not ask whether BLAS was
        wanted, only whether it is there, which is what a backend deciding for
        itself (the C++ color sum) needs."""

        if not cls.blas_is_available():
            return ''
        return cls._blas_flags

    def blas_link_flags(self):
        """What to link the color sum against, empty when BLAS is not taken."""

        if self.blas is False or not self.blas_is_available():
            return ''
        return self._blas_flags

    def blas_wanted(self, nfold):
        """Take BLAS when asked for it, or when it is there and the color
        matrix is big enough that the call is worth setting up."""

        if self.blas is False:
            return False
        if not self.blas_is_available():
            return False
        if self.blas is True:
            return True
        return nfold >= self.blas_min_ncolor

    @staticmethod
    def get_blas_routine(prefix, nfold, ncomb):
        """The color sum for every helicity at once. DSYMM is real, so the
        two parts of JAMP go through separately; the color matrix is real and
        symmetric so that is all it takes."""

        return """
      SUBROUTINE {p}GET_MATRIX_BATCH(JR,JI,NB,ANS)
      IMPLICIT NONE
      INTEGER NFOLD, NCOMB
      PARAMETER (NFOLD={n})
      PARAMETER (NCOMB={c})
      DOUBLE PRECISION JR(NFOLD,NCOMB), JI(NFOLD,NCOMB)
      INTEGER NB
      DOUBLE PRECISION ANS
      INTEGER I,J,K,CFI
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: CFULL(:,:)
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: TR(:,:), TI(:,:)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST
      INTEGER {p}CF(NFOLD*(NFOLD+1)/2)
      INTEGER {p}DENOM
      common /{p}color_matrix/ {p}CF,{p}DENOM
      IF (FIRST) THEN
        CALL {p}INIT_CF()
        ALLOCATE(CFULL(NFOLD,NFOLD))
        ALLOCATE(TR(NFOLD,NCOMB))
        ALLOCATE(TI(NFOLD,NCOMB))
C       What is written out is the upper triangle with its off diagonal
C       doubled, since the scalar sum walks it once. BLAS wants the whole
C       matrix with each entry counted once.
        CFI = 0
        DO I = 1, NFOLD
          DO J = I, NFOLD
            CFI = CFI + 1
            IF (I.EQ.J) THEN
              CFULL(I,J) = DBLE({p}CF(CFI))
            ELSE
              CFULL(I,J) = DBLE({p}CF(CFI))/2D0
              CFULL(J,I) = CFULL(I,J)
            ENDIF
          ENDDO
        ENDDO
        FIRST = .FALSE.
      ENDIF
      CALL DSYMM('L','U',NFOLD,NB,1D0,CFULL,NFOLD,JR,NFOLD,0D0,TR,NFOLD)
      CALL DSYMM('L','U',NFOLD,NB,1D0,CFULL,NFOLD,JI,NFOLD,0D0,TI,NFOLD)
      ANS = 0D0
      DO K = 1, NB
        DO I = 1, NFOLD
          ANS = ANS + TR(I,K)*JR(I,K) + TI(I,K)*JI(I,K)
        ENDDO
      ENDDO
      ANS = ANS / DBLE({p}DENOM)
      END
""".format(p=prefix, n=nfold, c=ncomb)

    def get_color_fold_ampso(self, folding, ncolor):
        """Template replacements for a color sum over one line per reversal
        pair, where JAMP carries a second index for the split orders. Without a
        folding the sum is left on JAMP itself, so nothing is copied."""

        if not folding:
            return {'ncolorfold': ncolor,
                    'color_fold_decl': '',
                    'color_fold_index': '',
                    'color_fold_gather': '',
                    'color_fold_array': 'JAMP'}
        lines = [line + 1 for line in folding['representatives']]
        return {
            'ncolorfold': len(lines),
            'color_fold_decl': (
                "    COMPLEX*16 JFOLD(NCOLORFOLD,NAMPSO)\n"
                "    INTEGER COLREP(NCOLORFOLD)\n"
                "    INTEGER ICF"),
            'color_fold_index': "\n".join(
                self.get_int_data_lines("COLREP", lines, var='ICF')),
            'color_fold_gather': "    JFOLD(:,:) = JAMP(COLREP(:),:)",
            'color_fold_array': 'JFOLD'}

    @staticmethod
    def jamp_column_form(column):
        """Canonical form of one column of the JAMP matrix up to a global sign,
        together with the sign which was taken out."""

        entries = sorted(column.items())
        first = entries[0][1]
        sign = -1 if (first.real, first.imag) < (0., 0.) else 1
        return tuple((i, sign * value) for i, value in entries), sign

    @classmethod
    def jamp_amp_permutation(cls, columns, induced):
        """Permutation of the amplitudes induced by the permutation induced of
        the color basis: return {amp: (amp, sign)} such that

            M[induced[i], sigma(j)] = sign(j) * M[i, j]

        or None if the columns are not mapped onto each other.

        Several amplitudes often have the very same column, so the columns are
        gathered by their canonical form and one target is taken out of each
        group at a time: looking the image up would not give a bijection."""

        groups = collections.defaultdict(collections.deque)
        for j in sorted(columns):
            form, sign = cls.jamp_column_form(columns[j])
            groups[form].append((j, sign))

        action = {}
        for j in sorted(columns):
            image = dict((induced[i - 1] + 1, value)
                         for i, value in columns[j].items())
            form, sign = cls.jamp_column_form(image)
            group = groups.get(form)
            if not group:
                return None
            target, target_sign = group.popleft()
            factor = sign * target_sign
            other = columns[target]
            if len(other) != len(image) or \
                 any(other.get(i) != factor * value
                     for i, value in image.items()):
                return None
            action[j] = (target, factor)
        return action

    def get_jamp_symmetry(self, matrix_element, all_element):
        """Permutations leaving the JAMP matrix invariant: for each of them the
        permutation of the color basis lines, and the permutation of the
        amplitude columns with the sign that goes with it. None when there is
        none, or when the matrix element does not carry a color basis."""

        if not isinstance(matrix_element, helas_objects.HelasMatrixElement):
            return None
        color_basis = matrix_element.get('color_basis')
        if not color_basis or len(color_basis) < 2:
            return None
        symmetry = color_amp.ColorBasisSymmetry(sorted(color_basis.keys()))
        if not symmetry.generators1:
            return None

        columns = collections.defaultdict(dict)
        for (i, j), value in all_element.items():
            if value:
                columns[j][i] = value
        if not columns:
            return None

        nb_line = len(symmetry.keys1)
        rowperms, actions = [], []
        for induced in symmetry.generators1:
            action = self.jamp_amp_permutation(columns, induced)
            if action is None:
                continue
            rowperms.append([0] + [induced[i] + 1 for i in range(nb_line)])
            actions.append(action)
        if not actions:
            return None

        # one line per orbit is enough to see every sub-expression: any other
        # line is the image of one of them, and so are the sub-expressions it
        # holds. This is what keeps the scan below from being quadratic in the
        # number of terms of the whole matrix.
        parent = list(range(nb_line + 1))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for rowperm in rowperms:
            for i in range(1, nb_line + 1):
                ri, rj = find(i), find(rowperm[i])
                if ri != rj:
                    parent[ri] = rj
        line_reps = [i for i in range(1, nb_line + 1) if find(i) == i]

        return {'rowperms': rowperms, 'actions': actions,
                'nb_line': nb_line, 'line_reps': line_reps}

    @staticmethod
    def jamp_operation_image(action, operation):
        """Image of the sub-expression operation=(j1,j2,R) under one
        permutation, and the factor relating the column the image defines to
        the image of the column operation defines."""

        j1, j2, ratio = operation
        first, sign1 = action[j1]
        second, sign2 = action[j2]
        if first < second:
            return (first, second, ratio * sign2 / sign1), sign1
        return (second, first, sign1 / (sign2 * ratio)), sign2 * ratio

    def optimise_jamp_equivariant(self, all_element, symmetry):
        """Same optimisation as optimise_jamp, but introducing whole orbits of
        sub-expressions at a time so that the result is closed under the
        symmetry. Fills self.jamp_orbits with, for every definition, the orbit
        it belongs to and the definition and permutation it comes from."""

        actions = [dict(action) for action in symmetry['actions']]
        line_reps = symmetry['line_reps']
        added = 0
        defs = []
        # (orbit, parent definition, permutation) for every definition
        tree = []
        # the definitions introduced together: none of them uses another, so
        # they can be reordered freely
        levels = []
        nb_orbit = 0

        while True:
            columns = collections.defaultdict(list)
            lines = collections.defaultdict(list)
            for (i, j), value in all_element.items():
                if value:
                    columns[j].append(i)
                    lines[i].append(j)
            for line in lines.values():
                line.sort()

            # every sub-expression is the image of one living on a
            # representative line, so only those have to be looked at
            candidates = set()
            for i in line_reps:
                line = lines.get(i, [])
                for pos, j1 in enumerate(line):
                    value = all_element[(i, j1)]
                    for j2 in line[pos + 1:]:
                        candidates.add((j1, j2, all_element[(i, j2)] / value))

            max_count = 0
            best = []
            for operation in candidates:
                count = len(self.jamp_operation_lines(all_element, columns,
                                                      operation))
                if count > max_count:
                    max_count, best = count, [operation]
                elif count == max_count:
                    best.append(operation)
            if max_count <= 1:
                break

            orbits = self.jamp_operation_orbits(actions, best)
            first_of_level = added + 1
            for orbit, parent in orbits:
                rows = dict((operation,
                             self.jamp_operation_lines(all_element, columns,
                                                       operation))
                            for operation in orbit)
                if not self.jamp_orbit_usable(rows):
                    continue
                index = {}
                for operation in orbit:
                    added += 1
                    index[operation] = added
                    origin, permutation = parent[operation]
                    tree.append((nb_orbit, index[origin] if origin else 0,
                                 permutation))
                    defs.append((added, operation[0], operation[1],
                                 operation[2], len(rows[operation])))
                nb_orbit += 1
                for operation, new in index.items():
                    j1, j2 = operation[0], operation[1]
                    for i in rows[operation]:
                        all_element[(i, -new)] = all_element[(i, j1)]
                        del all_element[(i, j1)]
                        del all_element[(i, j2)]
                for action in actions:
                    for operation, new in index.items():
                        image, factor = self.jamp_operation_image(action,
                                                                  operation)
                        action[-new] = (-index[image], factor)
            if added < first_of_level:
                # nothing could be introduced as a whole orbit
                break
            levels.append((first_of_level, added))
            logger.log(5, "Define %d new shortcut reused %d times",
                       added - first_of_level + 1, max_count)

        self.jamp_orbits = {'tree': tree, 'nb_orbit': nb_orbit,
                            'levels': levels, 'actions': actions,
                            'symmetry': symmetry}

        if self.jamp_emit == 'tables' and self.jamp_greedy_tail:
            # The orbit rounds stop while the JAMP lines still hold a good many
            # terms, since an orbit can only be taken as a whole. The plain
            # scan has no such scruple and can still shorten those lines. Its
            # sub-expressions are not orbits of anything, which rules them out
            # of the recipes, but the table emission does not care: there a
            # definition costs three numbers of DATA and one indirect add,
            # against a term of a line and a direct add.
            all_element, tail = self.optimise_jamp(all_element, added=added)
            defs.extend(tail)

        return all_element, defs

    @staticmethod
    def jamp_operation_lines(all_element, columns, operation):
        """Lines where both columns of the sub-expression are still there with
        its ratio. The values are read from the matrix as it is now, so lines
        already taken by an orbit introduced before are simply gone."""

        j1, j2, ratio = operation
        res = []
        for i in columns.get(j1, ()):
            value = all_element.get((i, j1), 0)
            if not value:
                continue
            other = all_element.get((i, j2), 0)
            if other and other / value == ratio:
                res.append(i)
        return res

    def jamp_operation_orbits(self, actions, operations):
        """Orbits of the sub-expressions, walked breadth first, with the
        (sub-expression, permutation) each of them is reached from."""

        seen = set()
        orbits = []
        for start in sorted(operations, key=lambda op: (op[0], op[1],
                                                        op[2].real,
                                                        op[2].imag)):
            if start in seen:
                continue
            orbit, parent = [start], {start: (None, 0)}
            seen.add(start)
            queue = collections.deque([start])
            while queue:
                current = queue.popleft()
                for position, action in enumerate(actions):
                    image = self.jamp_operation_image(action, current)[0]
                    if image in seen:
                        continue
                    seen.add(image)
                    parent[image] = (current, position + 1)
                    orbit.append(image)
                    queue.append(image)
            orbits.append((orbit, parent))
        return orbits

    @staticmethod
    def jamp_i_power(factor):
        """The exponent of i this factor is, or None when it is not one of the
        four powers of i. The factors the optimisation produces are products of
        signs and of the i the color coefficients carry, so this is what they
        all are in practice."""

        value = complex(factor)
        for exponent, power in enumerate((1, 1j, -1, -1j)):
            if value == power:
                return exponent
        return None

    def jamp_orbit_recipes(self, defs, nb_amp):
        """Describe the definitions by one recipe per orbit: the amplitude
        permutations, the first definition of every orbit, and the definitions
        renumbered so that walking each orbit breadth first from its recipe,
        with those permutations in that order, hands them out in that very
        order. The generated code walks them the same way, so it only needs
        the recipes.

        Returns None when the definitions cannot be described this way, and
        the caller then writes them out one by one as before."""

        orbits = self.jamp_orbits
        if not orbits or not defs:
            return None
        actions = orbits['actions']
        nb_orbit = orbits['nb_orbit']

        # Every factor has to be a power of i. They then form a group of four
        # elements, so the walk can carry them as an exponent modulo four and
        # stays integer arithmetic whatever the process.
        for one_def in defs:
            if self.jamp_i_power(one_def[3]) is None:
                return None
        for action in actions:
            for column, (_image, factor) in action.items():
                if column < 0 and self.jamp_i_power(factor) is None:
                    return None

        # first definition of every orbit
        first = [0] * nb_orbit
        for (orbit, parent, _permutation), one_def in zip(orbits['tree'], defs):
            if not parent:
                first[orbit] = one_def[0]

        # the permutations which are really needed to reach every definition
        # of every orbit: each of them costs one table of amplitude indices
        chosen = []
        rest = list(range(len(actions)))
        while self.jamp_orbit_reach(actions, chosen, first) < len(defs):
            best, best_gain = None, -1
            for position in rest:
                gain = self.jamp_orbit_reach(actions, chosen + [position],
                                             first)
                if gain > best_gain:
                    best, best_gain = position, gain
            if best is None:
                return None
            chosen.append(best)
            rest.remove(best)

        replay = self.jamp_orbit_replay(defs, first, chosen)
        while replay is None and rest:
            # the permutations kept do not reach every definition after all
            chosen.append(rest.pop(0))
            replay = self.jamp_orbit_replay(defs, first, chosen)
        if replay is None:
            return None
        new_defs, recipes, factor_of = replay

        permutations = []
        for permutation in chosen:
            action = orbits['symmetry']['actions'][permutation]
            row = [0] * nb_amp
            for amp, (image, sign) in action.items():
                row[amp - 1] = image if sign > 0 else -image
            if any(value == 0 for value in row):
                return None
            permutations.append(row)

        # an odd power of i anywhere means the factor in front of the second
        # operand is not real, and the array holding it has to be complex
        complex_factor = any(recipe[2] % 2 for recipe in recipes) or \
                         any(self.jamp_i_power(one[3]) % 2 for one in new_defs)
        return {'permutations': permutations, 'recipes': recipes,
                'defs': new_defs, 'nb_amp': nb_amp, 'factor_of': factor_of,
                'complex_factor': complex_factor}

    @staticmethod
    def jamp_hash_size(nb_def):
        """A prime comfortably larger than twice the number of definitions:
        the routine which rebuilds them looks the operand pairs up in a table
        of that size with linear probing."""

        candidate = 2 * nb_def + 101
        while True:
            candidate += 1
            for divisor in range(2, int(candidate ** 0.5) + 1):
                if candidate % divisor == 0:
                    break
            else:
                return candidate

    def jamp_orbit_tables(self, defs, nb_amp):
        """Describe the definitions by the plain list of their operands, to be
        written out as DATA. This is the same loop as the recipes drive, only
        with the table read from the source instead of rebuilt, so no factor is
        out of reach.

        The definitions introduced together carry no dependency, so inside each
        of those groups they are sorted by the factor in front of the second
        operand: the ones adding it, then the ones subtracting it, then the
        rest. The loop then runs over each group with the factor built in and
        only the last one has to multiply."""

        if not defs:
            return None
        by_index = dict((one[0], one) for one in defs)
        levels = self.jamp_definition_levels(defs)
        order, bounds, nb_general = [], [], 0
        for level in levels:
            group = [[], [], []]
            for index in level:
                ratio = complex(by_index[index][3])
                group[0 if ratio == 1 else 1 if ratio == -1 else 2]\
                                                             .append(index)
            start = len(order)
            order += group[0] + group[1] + group[2]
            # first, last of the adding group, last of the subtracting group,
            # last of the rest, and where the factors of that rest start
            bounds.append((start + 1, start + len(group[0]),
                           start + len(group[0]) + len(group[1]), len(order),
                           nb_general))
            nb_general += len(group[2])
        renumber = dict((old, new + 1) for new, old in enumerate(order))

        new_defs = []
        for old in order:
            _k, left, right, ratio, count = by_index[old]
            left = -renumber[-left] if left < 0 else left
            right = -renumber[-right] if right < 0 else right
            new_defs.append((renumber[old], left, right, ratio, count))

        # only the definitions of the third group ever read the factor array
        general = [one for level in bounds
                   for one in range(level[2] + 1, level[3] + 1)]
        return {'defs': new_defs, 'nb_amp': nb_amp, 'recipes': [],
                'bounds': bounds, 'general': general,
                'factor_of': dict((old, (new, 1))
                                  for old, new in renumber.items()),
                'complex_factor': any(complex(new_defs[one - 1][3]).imag
                                      for one in general)}

    @staticmethod
    def jamp_definition_levels(defs):
        """Group the definitions by how deep they sit in their own operands:
        one which uses no other is at the first level, and any other one comes
        after both of the ones it uses. Nothing inside a level uses anything
        else of that level, so they can be reordered freely.

        Read off the operands rather than off the rounds of the optimisation,
        so that whatever the plain scan adds at the end lands where it belongs.
        The operands of a definition always come before it, so one pass is
        enough."""

        depth = {}
        levels = collections.defaultdict(list)
        for index, left, right, _ratio, _count in defs:
            here = 0
            if left < 0:
                here = max(here, depth[-left])
            if right < 0:
                here = max(here, depth[-right])
            depth[index] = here + 1
            levels[here + 1].append(index)
        return [levels[key] for key in sorted(levels)]

    @staticmethod
    def jamp_number_data_lines(name, values, per_line, var='IJMP'):
        """DATA statements filling one array with the given constants."""

        lines = []
        for start in range(0, len(values), per_line):
            chunk = values[start:start + per_line]
            lines.append("      DATA (%s(%s),%s=%d,%d) /%s/" %
                         (name, var, var, start + 1, start + len(chunk),
                          ','.join(chunk)))
        return lines

    def get_jamp_table_lines(self, recipes, proc_prefix):
        """Declarations and DATA for the operand tables."""

        nb_def = len(recipes['defs'])
        nb_amp = recipes['nb_amp']

        def where(column):
            return column if column > 0 else nb_amp - column

        left = [where(one[1]) for one in recipes['defs']]
        right = [where(one[2]) for one in recipes['defs']]
        general = recipes['general']
        if recipes['complex_factor']:
            factor = ['(%s,%s)' %
                      (self.jamp_number(complex(recipes['defs'][one-1][3]).real),
                       self.jamp_number(complex(recipes['defs'][one-1][3]).imag))
                      for one in general]
        else:
            factor = [self.jamp_number(complex(recipes['defs'][one-1][3]).real)
                      for one in general]

        bounds = recipes['bounds']
        lines = [
            "      INTEGER ITMP, ILEV",
            "C     IJMP is the loop variable of the DATA statements below",
            "      INTEGER IJMP",
            "      INTEGER NB_TMP_JAMP, NB_LEVEL, NB_GENERAL",
            "      PARAMETER (NB_TMP_JAMP=%d)" % nb_def,
            "      PARAMETER (NB_LEVEL=%d)" % len(bounds),
            "      PARAMETER (NB_GENERAL=%d)" % max(1, len(general)),
            "      INTEGER TMP_JAMP_A(NB_TMP_JAMP), TMP_JAMP_B(NB_TMP_JAMP)",
            "      INTEGER TMP_JAMP_L(5*NB_LEVEL)",
            "      %s TMP_JAMP_F(NB_GENERAL)" % self.jamp_factor_type(recipes),
        ]
        if self.jamp_gather:
            lines.append("      COMPLEX*16 AMPBUF(%d+NB_TMP_JAMP)" % nb_amp)
        lines += self.get_int_data_lines("TMP_JAMP_A", left,
                                         var="IJMP")
        lines += self.get_int_data_lines("TMP_JAMP_B", right,
                                         var="IJMP")
        lines += self.get_int_data_lines("TMP_JAMP_L",
                                         sum((list(b) for b in bounds), []),
                                         var="IJMP")
        assert len(bounds[0]) == 5
        if general:
            lines += self.jamp_number_data_lines("TMP_JAMP_F", factor,
                                             32 if recipes['complex_factor']
                                             else 64)
        else:
            lines.append("      DATA TMP_JAMP_F/%s/" %
                         ("(0D0,0D0)" if recipes['complex_factor'] else "0D0"))
        return lines

    @staticmethod
    def jamp_number(value):
        """Shortest exact way of writing one of the factors."""

        if value == int(value) and abs(value) < 1e15:
            return "%dD0" % int(value)
        return ("%.15e" % value).replace('e', 'd')

    def jamp_buffer(self):
        """Array the definitions and their operands are read from."""

        return 'AMPBUF' if self.jamp_gather else 'AMP'

    @staticmethod
    def jamp_power_data(recipes):
        """DATA statement for the four powers of i, real when none of them is
        actually needed."""

        if recipes['complex_factor']:
            return "      DATA IPOW/(1D0,0D0),(0D0,1D0),(-1D0,0D0)," \
                   "(0D0,-1D0)/"
        return "      DATA IPOW/1D0,0D0,-1D0,0D0/"

    @staticmethod
    def jamp_factor_type(recipes):
        """The factor in front of the second operand is a power of i, so it is
        only complex when one of those powers is odd."""

        return "COMPLEX*16" if recipes['complex_factor'] \
                            else "DOUBLE PRECISION"

    def get_jamp_decl_lines(self, recipes, proc_prefix):
        """The declarations GET_JAMP needs to run the definitions."""

        if not recipes:
            return []
        if not recipes.get('recipes'):
            return self.get_jamp_table_lines(recipes, proc_prefix)
        return [
            "      INTEGER ITMP",
            "      INTEGER NB_TMP_JAMP",
            "      PARAMETER (NB_TMP_JAMP=%d)" % len(recipes['defs']),
            "      INTEGER TMP_JAMP_A(NB_TMP_JAMP), TMP_JAMP_B(NB_TMP_JAMP)",
            "      %s TMP_JAMP_F(NB_TMP_JAMP)" % self.jamp_factor_type(recipes),
            "      COMMON /%sjamp_recipe/ TMP_JAMP_A,TMP_JAMP_B,TMP_JAMP_F" % \
                                                                  proc_prefix,
        ]

    def get_jamp_init_routine(self, recipes, proc_prefix):
        """Fortran source rebuilding the operands of every color flow
        definition from one recipe per orbit, or nothing when the definitions
        are written out."""

        if not recipes or not recipes.get('recipes'):
            return []
        nb_def = len(recipes['defs'])
        nb_amp = recipes['nb_amp']
        nb_orbit = len(recipes['recipes'])
        nb_perm = len(recipes['permutations'])
        nb_hash = self.jamp_hash_size(nb_def)

        common = [
            "      INTEGER NGRAPHS, NB_TMP_JAMP, NB_HASH",
            "      PARAMETER (NGRAPHS=%d)" % nb_amp,
            "      PARAMETER (NB_TMP_JAMP=%d)" % nb_def,
            "      PARAMETER (NB_HASH=%d)" % nb_hash,
            "      INTEGER TMP_JAMP_A(NB_TMP_JAMP), TMP_JAMP_B(NB_TMP_JAMP)",
            "      %s TMP_JAMP_F(NB_TMP_JAMP)" % self.jamp_factor_type(recipes),
            "      COMMON /%sjamp_recipe/ TMP_JAMP_A,TMP_JAMP_B,TMP_JAMP_F" % \
                                                                  proc_prefix,
            "      INTEGER NUSED",
            "      INTEGER TMP_JAMP_E(NB_TMP_JAMP)",
            "      INTEGER HVAL(NB_HASH)",
            "      INTEGER*8 HKEY(NB_HASH)",
            "      COMMON /%sjamp_build/ NUSED,TMP_JAMP_E,HVAL,HKEY" % \
                                                                  proc_prefix,
        ]

        add = [
            "      SUBROUTINE %sJAMP_ADD(A,B,F,M,SWAP)" % proc_prefix,
            "C     The definition A + i**F*B, added if it is not there yet.",
            "C     Its two operands the other way round, with the inverse",
            "C     factor, give the very same column times i**F, so that one",
            "C     is looked for too; SWAP is the exponent relating what was",
            "C     asked for to what was found.",
            "      IMPLICIT NONE",
            "      INTEGER A,B,F,M,SWAP",
        ] + common + [
            "      INTEGER H, FREE, SHIFT",
            "      INTEGER*8 KEY, OTHER, BASE",
            "      SHIFT = NB_TMP_JAMP + 1",
            "      BASE = NGRAPHS + NB_TMP_JAMP + 2",
            "      KEY = ((A+SHIFT)*BASE+(B+SHIFT))*4+F+1",
            "      OTHER = ((B+SHIFT)*BASE+(A+SHIFT))*4+MOD(4-F,4)+1",
            "      SWAP = 0",
            "      H = INT(MOD(KEY,INT(NB_HASH,8)))+1",
            "      DO WHILE (HKEY(H) .NE. 0)",
            "        IF (HKEY(H) .EQ. KEY) THEN",
            "          M = HVAL(H)",
            "          RETURN",
            "        ENDIF",
            "        H = H+1",
            "        IF (H .GT. NB_HASH) H = 1",
            "      ENDDO",
            "      FREE = H",
            "      H = INT(MOD(OTHER,INT(NB_HASH,8)))+1",
            "      DO WHILE (HKEY(H) .NE. 0)",
            "        IF (HKEY(H) .EQ. OTHER) THEN",
            "          M = HVAL(H)",
            "          SWAP = F",
            "          RETURN",
            "        ENDIF",
            "        H = H+1",
            "        IF (H .GT. NB_HASH) H = 1",
            "      ENDDO",
            "      NUSED = NUSED+1",
            "      M = NUSED",
            "      TMP_JAMP_A(M) = A",
            "      TMP_JAMP_B(M) = B",
            "      TMP_JAMP_E(M) = F",
            "      HKEY(FREE) = KEY",
            "      HVAL(FREE) = M",
            "      END",
            "",
        ]

        body = [
            "      SUBROUTINE %sINIT_JAMP()" % proc_prefix,
            "C     Work out the operands of every color flow definition,",
            "C     starting from one recipe per orbit of the permutations",
            "C     leaving the color basis invariant and walking each orbit",
            "C     with those permutations. Done once, on the first call.",
            "      IMPLICIT NONE",
            "      INTEGER NB_ORBIT, NB_PERM",
            "      PARAMETER (NB_ORBIT=%d)" % nb_orbit,
            "      PARAMETER (NB_PERM=%d)" % nb_perm,
        ] + common + [
            "      INTEGER JPERM(NGRAPHS*NB_PERM)",
            "      INTEGER JREC(3*NB_ORBIT)",
            "      INTEGER JIMG(NB_TMP_JAMP*NB_PERM)",
            "      INTEGER JIMGE(NB_TMP_JAMP*NB_PERM)",
            "      INTEGER I,J,P,A,B,T,EA,EB,M,SWAP,BEGIN",
            "      %s IPOW(0:3)" % self.jamp_factor_type(recipes),
            self.jamp_power_data(recipes),
            "      LOGICAL JAMP_DONE",
            "      DATA JAMP_DONE/.FALSE./",
            "      SAVE JAMP_DONE, JIMG, JIMGE",
        ]
        body += self.get_int_data_lines("JPERM",
                                        sum(recipes['permutations'], []))
        body += self.get_int_data_lines("JREC",
                                        sum((list(one)
                                             for one in recipes['recipes']),
                                            []))
        body += [
            "      IF (JAMP_DONE) RETURN",
            "      JAMP_DONE = .TRUE.",
            "      DO I = 1, NB_HASH",
            "        HKEY(I) = 0",
            "      ENDDO",
            "      NUSED = 0",
            "      DO I = 1, NB_ORBIT",
            "        BEGIN = NUSED",
            "        CALL %sJAMP_ADD(JREC(3*I-2),JREC(3*I-1),JREC(3*I),M,SWAP)"
                                                              % proc_prefix,
            "        J = BEGIN",
            "        DO WHILE (J .LT. NUSED)",
            "          J = J+1",
            "          DO P = 1, NB_PERM",
            "C           an amplitude is permuted with a sign, which is the",
            "C           exponent 0 or 2; a definition brings its own exponent",
            "            A = TMP_JAMP_A(J)",
            "            IF (A .GT. 0) THEN",
            "              T = JPERM((P-1)*NGRAPHS+A)",
            "              A = ABS(T)",
            "              EA = (1-ISIGN(1,T))",
            "            ELSE",
            "              A = -JIMG((-A-1)*NB_PERM+P)",
            "              EA = JIMGE((-TMP_JAMP_A(J)-1)*NB_PERM+P)",
            "            ENDIF",
            "            B = TMP_JAMP_B(J)",
            "            IF (B .GT. 0) THEN",
            "              T = JPERM((P-1)*NGRAPHS+B)",
            "              B = ABS(T)",
            "              EB = (1-ISIGN(1,T))",
            "            ELSE",
            "              B = -JIMG((-B-1)*NB_PERM+P)",
            "              EB = JIMGE((-TMP_JAMP_B(J)-1)*NB_PERM+P)",
            "            ENDIF",
            "C           the image is A' + i**(e+eb-ea)*B', and the column it",
            "C           defines is i**ea times the image of this one",
            "            T = MOD(TMP_JAMP_E(J)+EB-EA+8,4)",
            "            CALL %sJAMP_ADD(A,B,T,M,SWAP)" % proc_prefix,
            "            JIMG((J-1)*NB_PERM+P) = M",
            "            JIMGE((J-1)*NB_PERM+P) = MOD(EA+SWAP,4)",
            "          ENDDO",
            "        ENDDO",
            "      ENDDO",
            "      IF (NUSED .NE. NB_TMP_JAMP) THEN",
            "        WRITE(*,*) 'ERROR: color flow recipes gave',NUSED,",
            "     $             ' definitions instead of',NB_TMP_JAMP",
            "        STOP 1",
            "      ENDIF",
            "C     the operands are read from one array holding the",
            "C     amplitudes first and the definitions after them",
            "      DO I = 1, NB_TMP_JAMP",
            "        IF (TMP_JAMP_A(I) .LT. 0) TMP_JAMP_A(I) = NGRAPHS"
                                                     "-TMP_JAMP_A(I)",
            "        IF (TMP_JAMP_B(I) .LT. 0) TMP_JAMP_B(I) = NGRAPHS"
                                                     "-TMP_JAMP_B(I)",
            "        TMP_JAMP_F(I) = IPOW(TMP_JAMP_E(I))",
            "      ENDDO",
            "      END",
        ]
        return add + body

    def jamp_tables_allowed(self):
        """Whether the definitions may be read from a table rather than written
        out. Both the standalone and the madevent templates declare what that
        needs; madevent reads the amplitudes of the current helicity into a
        buffer first, see jamp_gather."""

        return True

    def jamp_orbit_allowed(self, matrix_element):
        """Whether the orbit equivariant optimisation is used here."""

        if not self.jamp_orbit:
            return False

        if isinstance(self, ProcessExporterFortranME):
            return self.matrix_file in ('matrix_madevent_v4.inc',
                                        'matrix_madevent_group_v4.inc')

        # matchbox and the loop exporters derive from the standalone one but
        # write their own templates
        if type(self) is not ProcessExporterFortranSA:
            return False
        if self.matrix_template != 'matrix_standalone_v4.inc':
            return False
        if self.opt.get('export_format') in ('standalone_msP',
                                             'standalone_msF', 'matchbox',
                                             'madloop_matchbox'):
            return False
        return not matrix_element.get('processes')[0].get('split_orders')

    def jamp_orbit_replay(self, defs, first, chosen):
        """Walk every orbit from its first definition with the given
        permutations, exactly as the generated routine does, and hand out the
        definition numbers in that order. Returns the definitions in the new
        numbering, the recipe of every orbit, and for each old definition the
        new one it became with the factor between the two. None if the walk
        does not reach every definition."""

        amp_action = [self.jamp_orbits['symmetry']['actions'][position]
                      for position in chosen]
        nb_perm = len(chosen)
        by_index = dict((one_def[0], one_def) for one_def in defs)
        # old definition -> (new definition, factor between the two columns)
        factor_of = {}
        left_of, right_of, ratio_of = [], [], []
        image_of, power_of = [], []
        known = {}
        recipes = []

        def store(left, right, ratio):
            """The definition with those operands, added if it is new. The two
            operands can also be the other way round, and the column is then
            the same one up to the ratio, hence the second look up."""

            found = known.get((left, right, ratio))
            if found is not None:
                return found, 1
            # the two operands the other way round with the inverse ratio give
            # the same column times the ratio
            found = known.get((right, left, 1 / ratio))
            if found is not None:
                return found, ratio
            left_of.append(left)
            right_of.append(right)
            ratio_of.append(ratio)
            image_of.extend([0] * nb_perm)
            power_of.extend([0] * nb_perm)
            known[(left, right, ratio)] = len(left_of)
            return len(left_of), 1

        def act(place, column):
            """image of a column and the factor that goes with it"""

            if column > 0:
                return amp_action[place][column]
            where = (-column - 1) * nb_perm + place
            return (-image_of[where],
                    (1, 1j, -1, -1j)[power_of[where]])

        # the same walk is followed on the definitions of the optimisation, so
        # that each of them is matched with the one the generated code builds
        actions = self.jamp_orbits['actions']
        origin_of = []

        for start in first:
            _k, left, right, ratio = by_index[start][:4]
            scale = 1
            if left < 0:
                if -left not in factor_of:
                    return None
                new, factor = factor_of[-left]
                left, scale = -new, factor
            if right < 0:
                if -right not in factor_of:
                    return None
                new, factor = factor_of[-right]
                right, ratio = -new, ratio * factor
            ratio = ratio / scale
            if self.jamp_i_power(ratio) is None:
                return None
            begin = len(left_of)
            new, factor = store(left, right, ratio)
            if new != begin + 1:
                # the first definition of an orbit has to be a new one
                return None
            origin_of.append(start)
            factor_of[start] = (new, scale * factor)
            recipes.append((left, right, self.jamp_i_power(ratio)))

            current = begin
            while current < len(left_of):
                current += 1
                previous = origin_of[current - 1]
                for place in range(nb_perm):
                    image_left, sign_left = act(place, left_of[current - 1])
                    image_right, sign_right = act(place, right_of[current - 1])
                    image_ratio = ratio_of[current - 1] * sign_right / sign_left
                    if self.jamp_i_power(image_ratio) is None:
                        return None
                    where, swap = store(image_left, image_right, image_ratio)
                    sign = sign_left * swap
                    image_of[(current - 1) * nb_perm + place] = where
                    power_of[(current - 1) * nb_perm + place] = \
                        self.jamp_i_power(sign)
                    if where > len(origin_of):
                        origin_of.append(None)
                    # follow the same step on the definitions of the
                    # optimisation to know which one this is
                    image, factor = actions[chosen[place]][-previous]
                    image = -image
                    if origin_of[where - 1] is None:
                        origin_of[where - 1] = image
                    if image not in factor_of:
                        factor_of[image] = (where,
                                            factor_of[previous][1] * sign
                                            / factor)

        if len(factor_of) != len(defs):
            return None

        new_defs = [(i + 1, left_of[i], right_of[i], ratio_of[i], 0)
                    for i in range(len(left_of))]
        return new_defs, recipes, factor_of

    @staticmethod
    def jamp_orbit_reach(actions, chosen, first):
        """How many definitions the given permutations reach from the first
        definition of every orbit."""

        seen = set(first)
        queue = collections.deque(first)
        while queue:
            current = queue.popleft()
            for permutation in chosen:
                image = -actions[permutation][-current][0]
                if image not in seen:
                    seen.add(image)
                    queue.append(image)
        return len(seen)

    @staticmethod
    def jamp_orbit_usable(rows):
        """Restrict an orbit to the entries only one of its sub-expressions
        wants, and say whether what is left can be introduced as a whole. Which
        of two sub-expressions of the same orbit gets a shared entry cannot be
        decided in a way that commutes with the symmetry, so those entries are
        left in the matrix and get another chance in a later round."""

        sizes = set(len(use) for use in rows.values())
        if len(sizes) != 1 or sizes == set([0]):
            return False
        entry = collections.Counter()
        for operation, use in rows.items():
            for i in use:
                entry[(i, operation[0])] += 1
                entry[(i, operation[1])] += 1
        if max(entry.values()) == 1:
            return True
        for operation in list(rows):
            rows[operation] = [i for i in rows[operation]
                               if entry[(i, operation[0])] == 1
                               and entry[(i, operation[1])] == 1]
        sizes = set(len(use) for use in rows.values())
        return len(sizes) == 1 and sizes != set([0])



    def get_pdf_lines(self, matrix_element, ninitial, subproc_group = False, vector=False):
        """Generate the PDF lines for the auto_dsig.f file"""

        processes = matrix_element.get('processes')
        model = processes[0].get('model')

        pdf_definition_lines = ""
        ee_pdf_definition_lines = ""
        pdf_data_lines = ""
        pdf_lines = ""

        if vector:
            pdf_definition_lines_vec = ""
            pdf_data_lines_vec = ""
            pdf_lines = """ NB_WARP_USED = VECSIZE_USED / WARP_SIZE
        IF( NB_WARP_USED * WARP_SIZE .NE. VECSIZE_USED ) THEN
        WRITE(*,*) 'ERROR: NB_WARP_USED * WARP_SIZE .NE. VECSIZE_USED',
     &    NB_WARP_USED, WARP_SIZE, VECSIZE_USED
        STOP
        ENDIF

        DO CURR_WARP=1, NB_WARP_USED
        IF(IMIRROR_VEC(CURR_WARP).EQ.1)THEN
          IB(1) = 1
          IB(2) = 2
        ELSE
          IB(1) = 2
          IB(2) = 1
        ENDIF
        DO IWARP=1, warp_SIZE
          IVEC = (CURR_WARP-1)*WARP_SIZE+IWARP
          """



        if ninitial == 1:
            all_flv = list(matrix_element.get_external_flavors_with_iden())
            if vector:
                # Close the vector loops opened above (no PDFs to fetch for decays)
                pdf_lines += "ENDDO ! IWARP LOOP\n"
                pdf_lines += "ENDDO ! CURRWARP LOOP\n"
                # Set ALL_PD for ALL flavor combinations across all IVEC.
                # No iflav conditioning here (matches ninitial==2 vector pattern):
                # IPSEL will randomly select the flavor, then GET_FLAVOR maps it.
                pdf_lines += "ALL_PD(0,:) = 0d0\nIPROC = 0\n"
                for i, proc in enumerate(processes):
                    process_line = proc.base_string()
                    for grp in all_flv:
                        for one_flv in grp:
                            pdf_lines += "IPROC=IPROC+1 ! " + process_line
                            pdf_lines += "\nDO IVEC=1, VECSIZE_USED\n"
                            pdf_lines += "ALL_PD(IPROC,IVEC)=1d0\n"
                            pdf_lines += "ALL_PD(0,IVEC)=ALL_PD(0,IVEC)+DABS(ALL_PD(IPROC,IVEC))\n"
                            pdf_lines += "ENDDO\n"
            else:
                pdf_lines = "PD(0) = 0d0\nIPROC = 0\n"
                for i, proc in enumerate(processes):
                    process_line = proc.base_string()
                    for nb_flavor in range(len(all_flv)):
                        pdf_lines += 'if(iflav.eq.%d) then\n' % (nb_flavor + 1)
                        for one_flv in all_flv[nb_flavor]:
                            pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                            pdf_lines = pdf_lines + "\nPD(IPROC)=1d0\n"
                            pdf_lines = pdf_lines + "\nPD(0)=PD(0)+PD(IPROC)\n"
                        pdf_lines += ' endif\n'
        else:
            # Pick out all initial state particles for the two beams
            initial_states = [sorted(list(set([p.get_initial_pdg(1) for \
                                               p in processes]))),
                              sorted(list(set([p.get_initial_pdg(2) for \
                                               p in processes])))]

            for one_initial_state in initial_states:
                for i,pdg in enumerate(list(one_initial_state)):
                    if hasattr(self.model, 'merged_particles'):
                        if pdg in self.model['merged_particles']:
                            one_initial_state.remove(pdg)
                            one_initial_state += self.model['merged_particles'][pdg]
                        elif -pdg in self.model['merged_particles']:
                            one_initial_state.remove(pdg)
                            one_initial_state += [-i for i in self.model['merged_particles'][-pdg]]


            if tuple(initial_states) in [([-11],[11]), ([11],[-11]), ([-13],[13]),([13],[-13])]:
                dressed_lep = True
            else:
                dressed_lep = False
            ee_pdf_definition_lines += "DOUBLE PRECISION dummy_components(n_ee)\n"

   
            # Prepare all variable names
            pdf_codes = dict([(p, model.get_particle(p).get_name()) for p in \
                              sum(initial_states,[])])
            for key,val in pdf_codes.items():
                pdf_codes[key] = val.replace('~','x').replace('+','p').replace('-','m')

            # Set conversion from PDG code to number used in PDF calls
            pdgtopdf = {21: 0, 22: 7}

            # Fill in missing entries of pdgtopdf
            for pdg in sum(initial_states,[]):
                if not pdg in pdgtopdf and not pdg in list(pdgtopdf.values()):
                    pdgtopdf[pdg] = pdg
                elif pdg not in pdgtopdf and pdg in list(pdgtopdf.values()):
                    # If any particle has pdg code 7, we need to use something else
                    pdgtopdf[pdg] = 6000000 + pdg
                    
            # Get PDF variable declarations for all initial states
            if vector:
                vector_ext1 = '(VECSIZE_MEMMAX)' # pass to an array from a double
                vector_ext2 = ', VECSIZE_MEMMAX' # add a dimenion
            else:
                vector_ext1, vector_ext2 = '',''

            for i in [0,1]:
                pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d%s" % (pdf_codes[pdg],i+1, vector_ext1) \
                                                 for pdg in \
                                                 initial_states[i]]) + \
                                                 "\n"
                
                ee_pdf_definition_lines += "DOUBLE PRECISION " + \
                                       ",".join(["%s%d_components(n_ee %s)" % (pdf_codes[pdg],i+1, vector_ext2) \
                                                 for pdg in \
                                                 initial_states[i] if abs(pdg) in [11,13]]) + \
                                                 "\n"
                

            # Get PDF data lines for all initial states
            for i in [0,1]:
                pdf_data_lines += "DATA " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in initial_states[i]]) + \
                                                 "/%d*1D0/" % len(initial_states[i]) + \
                                                 "\n"
                if vector:
                    pdf_data_lines_vec += "DATA " + \
                                       ",".join(["%s%d" % (pdf_codes[pdg],i+1) \
                                                 for pdg in initial_states[i]]) + \
                                                 "/%s/" % ','.join(['VECSIZE_MEMMAX*1D0']* len(initial_states[i])) + \
                                                 "\n"


            # Get PDF lines for UPC (non-factorized PDF)
            if 22 in initial_states[0] and 22 in initial_states[1]:
                if subproc_group:
                    pdf_lines = pdf_lines + \
                        "IF (ABS(LPP(IB(1))).EQ.2.AND.ABS(LPP(IB(2))).EQ.2.AND.(PDLABEL(1:4).EQ.'edff'.OR.PDLABEL(1:4).EQ.'chff'))THEN\n"
                    pdf_lines = pdf_lines + \
                        ("%s%d=PHOTONPDFSQUARE(XBK(IB(1)),XBK(IB(2)))\n%s%d=DSQRT(%s%d)\n%s%d=%s%d\n") % \
                        (pdf_codes[22],1,pdf_codes[22],2,pdf_codes[22],1,pdf_codes[22],1,pdf_codes[22],2)
                else:
                    pdf_lines = pdf_lines + \
                        "IF (ABS(LPP(1)).EQ.2.AND.ABS(LPP(2)).EQ.2.AND.(PDLABEL(1:4).EQ.'edff'.OR.PDLABEL(1:4).EQ.'chff'))THEN\n"
                    pdf_lines = pdf_lines + \
                        ("%s%d=PHOTONPDFSQUARE(XBK(1),XBK(2))\n%s%d=DSQRT(%s%d)\n%s%d=%s%d\n") % \
                        (pdf_codes[22],1,pdf_codes[22],2,pdf_codes[22],1,pdf_codes[22],1,pdf_codes[22],2)
                pdf_lines = pdf_lines + "ELSE\n"

            # Get PDF lines for all different initial states
            for i, init_states in enumerate(initial_states):
                if subproc_group:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(IB(%d))).GE.1) THEN\n!LP=SIGN(1,LPP(IB(%d)))\n" \
                                 % (i + 1, i + 1)
                    if not vector:
                        if i == 0:
                            pdf_lines = pdf_lines + \
                                "if (DSQRT(Q2FACT(IB(1))).eq.0d0) then\n" +\
                                "  qscale=0d0\n"+\
                                "    do i=3,nexternal\n"+\
                                "      Qscale=Qscale+dsqrt(max(0d0,(PP(0,i)+PP(3,i))*(PP(0,i)-PP(3,i))))\n"+\
                                "    enddo\n"+\
                                "   qscale=qscale/2d0\n"+\
                                "else\n"+\
                                "   qscale=DSQRT(Q2FACT(1))\n"+\
                                "endif\n"
                        else:
                            pdf_lines = pdf_lines + \
                                "if (DSQRT(Q2FACT(IB(2))).ne.0d0) then\n" +\
                                "   qscale=DSQRT(Q2FACT(2))\n" +\
                                "endif\n"
                else:
                    pdf_lines = pdf_lines + \
                           "IF (ABS(LPP(%d)) .GE. 1) THEN\n!LP=SIGN(1,LPP(%d))\n" \
                                 % (i + 1, i + 1)
                    
                for nbi,initial_state in enumerate(init_states):
                    if initial_state in list(pdf_codes.keys()):

                        data = {'part':pdf_codes[initial_state],
                                'beam' : i+1,
                                'pdg': pdgtopdf[initial_state],
                                'vecid': ''
                            }
                        if vector:
                            data['vecid'] = ', IVEC'

                        if vector and subproc_group:
                            template  = "%(part)s%(beam)d(IVEC)=PDG2PDF(LPP(IB(%(beam)d)),%(pdg)d, IB(%(beam)d)," + \
                                         "ALL_XBK(IB(%(beam)d),IVEC),DSQRT(ALL_Q2FACT(%(beam)d, IVEC)))\n"
                            #if dressed_lep and self.opt['vector_size']:
                            #    logger.warning("vector code for lepton pdf not implemented. We removed the option to run dressed lepton")
                            #    self.proc_characteristic['limitations'].append('dressed_ee')
                            #    dressed_lep = False
                        elif subproc_group:
                            template = "%(part)s%(beam)d=PDG2PDF(LPP(IB(%(beam)d)),%(pdg)d, IB(%(beam)d)," + \
                                         "XBK(IB(%(beam)d)), QSCALE)\n"
                        elif vector:
                            template = "%(part)s%(beam)d(IVEC)=PDG2PDF(LPP(%(beam)d),%(pdg)d, %(beam)d," + \
                                         "ALL_XBK(%(beam)d,IVEC),DSQRT(ALL_Q2FACT(%(beam)d,IVEC)))\n"
                            #if dressed_lep:
                            #    raise Exception("vector code for lepton pdf not implemented")
                        else:
                            template = "%(part)s%(beam)d=PDG2PDF(LPP(%(beam)d),%(pdg)d, %(beam)d," + \
                                         "XBK(%(beam)d),DSQRT(Q2FACT(%(beam)d)))\n"
                        if dressed_lep:
                            template += "IF (PDLABEL.EQ.'dressed') %(part)s%(beam)d_components(1:4 %(vecid)s) = ee_components(1:4)\n"

                        pdf_lines = pdf_lines + template % data

                pdf_lines = pdf_lines + "ENDIF\n"

            if 22 in initial_states[0] and 22 in initial_states[1]:
                pdf_lines = pdf_lines + "ENDIF\n"

            if not vector:
                # Add up PDFs for the different initial state particles

                pdf_lines = pdf_lines + "PD(0) = 0d0\nIPROC = 0\n"
                for proc in processes:
                    all_flv = list(matrix_element.get_external_flavors_with_iden())
                    for nb_flavor in range(len(all_flv)):
                        process_line = proc.base_string()
                        pdf_lines += 'if(iflav.eq.%d) then\n' % (nb_flavor+1)
                        for one_flv in all_flv[nb_flavor]:
                            pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                            pdf_lines = pdf_lines + "\nPD(IPROC)="
                            comp_list = []
                            for ibeam in [1, 2]:
                                initial_state = proc.get_initial_pdg(ibeam)
                                if abs(initial_state) in model.get('merged_particles'):
                                    flv = proc.get_initial_flavor(ibeam)
                                    if len(flv) == 0:
                                        sign = 1 if initial_state > 0 else -1
                                        initial_state = sign * one_flv[ibeam-1]
                                    elif len(flv) ==1:
                                        initial_state = flv[0]
                                    else:
                                        # Grouped process: multiple specific quarks are
                                        # possible; use the one specified by this flavor
                                        # combination.
                                        sign = 1 if initial_state > 0 else -1
                                        initial_state = sign * one_flv[ibeam-1]
                                
                                if initial_state in list(pdf_codes.keys()):
                                    pdf_lines = pdf_lines + "%s%d*" % \
                                                (pdf_codes[initial_state], ibeam)
                                    comp_list.append("%s%d" % (pdf_codes[initial_state], ibeam))
                                else:
                                    pdf_lines = pdf_lines + "1d0*"
                                    comp_list.append("DUMMY")
                            
                            # Remove last "*" from pdf_lines
                            pdf_lines = pdf_lines[:-1] + "\n"
                            pdf_lines += 'PD(0)=PD(0)+DABS(PD(IPROC))\n'
                        pdf_lines += ' endif\n'
                        # this is for the lepton collisions with electron luminosity 
                        # put here "%s%d_components(i_ee)*%s%d_components(i_ee)"
                        if dressed_lep:
                            pdf_lines += "if (pdlabel.eq.'dressed')" + \
                                "PD(IPROC)=ee_comp_prod(%s_components,%s_components)\n" % \
                                tuple(comp_list)
                            pdf_lines = pdf_lines + "PD(0)=PD(0)+DABS(PD(IPROC))\n"
                        
                        if not dressed_lep:
                            ee_pdf_definition_lines = ""
            else:
                # Add up PDFs for the different initial state particles
                pdf_lines += "ENDDO ! IWARP LOOP\n"
                pdf_lines += "ENDDO ! CURRWARP LOOP\n"
                pdf_lines = pdf_lines + "ALL_PD(0,:) = 0d0\nIPROC = 0\n"
                for proc in processes:
                    for nb_flavor in range(matrix_element.get_nb_flavors()):
                        comp_list = []
                        process_line = proc.base_string()
                        pdf_lines = pdf_lines + "IPROC=IPROC+1 ! " + process_line
                        pdf_lines += '\n   DO IVEC=1, VECSIZE_USED'
                        pdf_lines = pdf_lines + "\nALL_PD(IPROC,IVEC)="
                        for ibeam in [1, 2]:
                            initial_state = proc.get_initial_pdg(ibeam)
                            if abs(initial_state) in model.get('merged_particles'):
                                flv = proc.get_initial_flavor(ibeam)
                                if len(flv) == 0:
                                    sign = 1 if initial_state > 0 else -1
                                    initial_state = sign * matrix_element.get_external_flavors()[nb_flavor][ibeam-1]
                                elif len(flv) ==1:
                                    initial_state = flv[0]
                                else:
                                    # Grouped process: multiple specific quarks are
                                    # possible; use the one specified by this flavor
                                    # combination.
                                    sign = 1 if initial_state > 0 else -1
                                    initial_state = sign * matrix_element.get_external_flavors()[nb_flavor][ibeam-1]

                            if initial_state in list(pdf_codes.keys()):
                                pdf_lines = pdf_lines + "%s%d(IVEC)*" % \
                                            (pdf_codes[initial_state], ibeam)
                                comp_list.append("%s%d" % (pdf_codes[initial_state], ibeam))
                            else:
                                pdf_lines = pdf_lines + "1d0*"
                                comp_list.append("DUMMY")
                        # Remove last "*" from pdf_lines
                        pdf_lines = pdf_lines[:-1] + "\n"
                        # this is for the lepton collisions with electron luminosity 
                        # put here "%s%d_components(i_ee)*%s%d_components(i_ee)"
                        if dressed_lep:
                            pdf_lines += "if (pdlabel.eq.'dressed')" + \
                                "ALL_PD(IPROC,IVEC)=ee_comp_prod(%s_components(1,IVEC),%s_components(1,IVEC))\n" % \
                                tuple(comp_list)
                        pdf_lines = pdf_lines + "ALL_PD(0,IVEC)=ALL_PD(0,IVEC)+DABS(ALL_PD(IPROC,IVEC))\n"
                        pdf_lines += '\n    ENDDO\n'
                        if not dressed_lep:
                            ee_pdf_definition_lines = ""

        # Remove last line break from the return variables                
        if vector:
            return pdf_definition_lines[:-1], pdf_data_lines_vec[:-1], pdf_lines[:-1], ee_pdf_definition_lines
        else:
            return pdf_definition_lines[:-1], pdf_data_lines[:-1], pdf_lines[:-1], ee_pdf_definition_lines

    #===========================================================================
    # write_props_file
    #===========================================================================
    def write_props_file(self, writer, matrix_element, s_and_t_channels):
        """Write the props.inc file for MadEvent. Needs input from
        write_configs_file."""

        lines = []

        particle_dict = matrix_element.get('processes')[0].get('model').\
                        get('particle_dict')

        for iconf, configs in enumerate(s_and_t_channels):
            for vertex in configs[0] + configs[1][:-1]:
                leg = vertex.get('legs')[-1]
                if leg.get('id') not in particle_dict:
                    # Fake propagator used in multiparticle vertices
                    mass = 'zero'
                    width = 'zero'
                    pow_part = 0
                else:
                    particle = particle_dict[leg.get('id')]
                    # Get mass
                    if particle.get('mass').lower() == 'zero':
                        mass = particle.get('mass')
                    else:
                        mass = "abs(%s)" % particle.get('mass')
                    # Get width
                    if particle.get('width').lower() == 'zero':
                        width = particle.get('width')
                    else:
                        width = "abs(%s)" % particle.get('width')

                    pow_part = 1 + int(particle.is_boson())

                lines.append("prmass(%d,%d)  = %s" % \
                             (leg.get('number'), iconf + 1, mass))
                lines.append("prwidth(%d,%d) = %s" % \
                             (leg.get('number'), iconf + 1, width))
                lines.append("pow(%d,%d) = %d" % \
                             (leg.get('number'), iconf + 1, pow_part))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        configs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        model = matrix_element.get('processes')[0].get('model')
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,
                                                            model)

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""

        lines = []

        s_and_t_channels = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
            for config in configs if [d for d in config if d][0].\
                                             get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any(vert > minvert for vert in [d for d in helas_diags if d]\
              [0].get_vertex_leg_numbers()) :
                # Only 3-vertices allowed in configs.inc except for vertices
                # which originate from a shrunk loop.
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial, model, new_pdg))
                else:
                    stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = list(zip(*[s for s,t in stchannels]))
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("data mapconfig(%d)/%d/" % (nconfigs,
                                                     mapconfigs[iconfig]))

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), nconfigs, len(daughters),
                              ",".join([str(d) for d in daughters])))
                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join([str(d) for d in pdgs])))
                    lines.append("data tprid(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels:
                    lines.append("data tprid(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  abs(last_leg.get('id'))))
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%d/" % nconfigs)

        # Write the file
        writer.writelines(lines)

        return s_and_t_channels

    #===========================================================================
    # Global helper methods
    #===========================================================================

    def coeff(self, ff_number, frac, is_imaginary, Nc_power, Nc_value=3):
        """Returns a nicely formatted string for the coefficients in JAMP lines"""

        total_coeff = ff_number * frac * fractions.Fraction(Nc_value) ** Nc_power

        if total_coeff == 1:
            if is_imaginary:
                return '+imag1*'
            else:
                return '+'
        elif total_coeff == -1:
            if is_imaginary:
                return '-imag1*'
            else:
                return '-'

        res_str = '%+iD0' % total_coeff.numerator

        if total_coeff.denominator != 1:
            # Check if total_coeff is an integer
            res_str = res_str + '/%iD0' % total_coeff.denominator

        if is_imaginary:
            res_str = res_str + '*imag1'

        return res_str + '*'


    def set_fortran_compiler(self, default_compiler, force=False):
        """Set compiler based on what's available on the system"""
               
        # Check for compiler
        if default_compiler['fortran'] and misc.which(default_compiler['fortran']):
            f77_compiler = default_compiler['fortran']
        elif misc.which('gfortran'):
            f77_compiler = 'gfortran'
        elif misc.which('g77'):
            f77_compiler = 'g77'
        elif misc.which('f77'):
            f77_compiler = 'f77'
        elif default_compiler['fortran']:
            logger.warning('No Fortran Compiler detected! Please install one')
            f77_compiler = default_compiler['fortran'] # maybe misc fail so try with it
        else:
            raise MadGraph5Error('No Fortran Compiler detected! Please install one')
        logger.info('Use Fortran compiler ' + f77_compiler)
        
        
        # Check for compiler. 1. set default.
        if default_compiler['f2py']:
            f2py_compiler = default_compiler['f2py']
        else:
            f2py_compiler = ''
        # Try to find the correct one.
        if default_compiler['f2py'] and misc.which(default_compiler['f2py']):
            f2py_compiler = default_compiler['f2py']
        elif misc.which('f2py%d.%d' %(sys.version_info.major, sys.version_info.minor)):
            f2py_compiler = 'f2py%d.%d' %(sys.version_info.major, sys.version_info.minor)
        elif misc.which('f2py%d' %(sys.version_info.major)):
            f2py_compiler = 'f2py%d' %(sys.version_info.major)            
        elif misc.which('f2py'):
            f2py_compiler = 'f2py'


        to_replace = {'fortran': f77_compiler, 'f2py': f2py_compiler}
        
        
        self.replace_make_opt_f_compiler(to_replace)
        # Replace also for Template but not for cluster
        if 'MADGRAPH_DATA' not in os.environ and ReadWrite:
            self.replace_make_opt_f_compiler(to_replace, pjoin(MG5DIR, 'Template', 'LO'))
        
        return f77_compiler

    # an alias for backward compatibility
    set_compiler = set_fortran_compiler


    def set_cpp_compiler(self, default_compiler, force=False):
        """Set compiler based on what's available on the system"""
                
        # Check for compiler
        if default_compiler and misc.which(default_compiler):
            compiler = default_compiler
        elif misc.which('g++'):
            #check if clang version
            p = misc.Popen(['g++', '--version'], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE) 
            out, _ = p.communicate()
            out = out.decode(errors='ignore')
            if 'clang' in str(out) and  misc.which('clang'):
                compiler = 'clang'
            else:
                compiler = 'g++'
        elif misc.which('c++'):
            compiler = 'c++'
        elif misc.which('clang'):
            compiler = 'clang'
        elif default_compiler:
            logger.warning('No c++ Compiler detected! Please install one')
            compiler = default_compiler # maybe misc fail so try with it
        else:
            raise MadGraph5Error('No c++ Compiler detected! Please install one')
        logger.info('Use c++ compiler ' + compiler)
        self.replace_make_opt_c_compiler(compiler)
        # Replace also for Template but not for cluster
        if 'MADGRAPH_DATA' not in os.environ and ReadWrite and \
           not __debug__ and not os.path.exists(pjoin(MG5DIR,'bin','create_release.py')):
            self.replace_make_opt_c_compiler(compiler, pjoin(MG5DIR, 'Template', 'LO'))
        
        return compiler


    def replace_make_opt_f_compiler(self, compilers, root_dir = ""):
        """Set FC=compiler in Source/make_opts"""

        assert isinstance(compilers, dict)
        
        mod = False #avoid to rewrite the file if not needed
        if not root_dir:
            root_dir = self.dir_path
            
        compiler= compilers['fortran']
        f2py_compiler = compilers['f2py']
        if not f2py_compiler:
            f2py_compiler = 'f2py'
        for_update= {'DEFAULT_F_COMPILER':compiler,
                     'DEFAULT_F2PY_COMPILER':f2py_compiler}
        make_opts = pjoin(root_dir, 'Source', 'make_opts')

        try:
            common_run_interface.CommonRunCmd.update_make_opts_full(
                            make_opts, for_update)
        except IOError:
            if root_dir == self.dir_path:
                logger.info('Fail to set compiler. Trying to continue anyway.')            

    def replace_make_opt_c_compiler(self, compiler, root_dir = ""):
        """Set CXX=compiler in Source/make_opts.
        The version is also checked, in order to set some extra flags
        if the compiler is clang (on MACOS)"""
       
        is_clang = misc.detect_if_cpp_compiler_is_clang(compiler)
        is_lc    = misc.detect_cpp_std_lib_dependence(compiler) == '-lc++'


        # list of the variable to set in the make_opts file
        for_update= {'DEFAULT_CPP_COMPILER':compiler,
                     'MACFLAG':'-mmacosx-version-min=10.7' if is_clang and is_lc else '',
                     'STDLIB': '-lc++' if is_lc else '-lstdc++',
                     'STDLIB_FLAG': '-stdlib=libc++' if is_lc and is_clang else ''
                     }

        # for MOJAVE remove the MACFLAG:
        if is_clang:
            import platform
            version, _, _ = platform.mac_ver()
            if not version:# not linux 
                majversion = 14 # set version to remove MACFLAG
            else:
                majversion, version = [int(x) for x in version.split('.',3)[:2]]

            if majversion >= 11 or (majversion ==10 and version >= 14):
                for_update['MACFLAG'] = '-mmacosx-version-min=10.8' if is_lc else ''

        if not root_dir:
            root_dir = self.dir_path
        make_opts = pjoin(root_dir, 'Source', 'make_opts')

        try:
            common_run_interface.CommonRunCmd.update_make_opts_full(
                            make_opts, for_update)
        except IOError:
            if root_dir == self.dir_path:
                logger.info('Fail to set compiler. Trying to continue anyway.')  
    
        return

#===============================================================================
# ProcessExporterFortranSA
#===============================================================================
class ProcessExporterFortranSA(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 StandAlone format."""

    matrix_template = "matrix_standalone_v4.inc"
    f2py_template = "matrix_standalone_f2py.inc"
    f2py_wrapper_all ="f2py_wrapper_all.inc"
    f2py_matrix_splitter = "f2py_splitter.py"
    jamp_optim = True
    jamp_fold = True
    jamp_orbit = True
    # The only exporter implementing the extended FLAV_IDX decoding. The
    # per-matrix-element cases it still cannot cross (msP/msF, matchbox,
    # split orders) are handled by the use_crossing_ic gate in
    # write_matrix_element_v4, which falls back to the uncrossed code.
    supports_crossing = True
    default_vector_size = 0
    # standalone only squares the amplitude, so it can use the DDM basis. It
    # still carries the Kleiss-Kuijf reconstruction of the trace JAMPs, because
    # this is the mode used to time the matrix element and madevent pays it.
    support_ddm_color_basis = True
    ddm_needs_flow_basis = True
    # When True, emit per-call IAND(WF_FLAVOR_MASK/AMP_FLAVOR_MASK,
    # CURRENT_FLAV_BIT) guards in MATRIX so that wavefunctions and amplitudes
    # which contribute zero for the current input flavor are skipped at
    # runtime. Set to False to revert to the unconditional emission.
    use_flavor_mask = True

    def __init__(self, *args,**opts):
        """add the format information compare to standard init"""

        if 'format' in opts:
            self.format = opts['format']
            del opts['format']
        else:
            self.format = 'standalone'

        self.prefix_info = {}
        # proc_prefix -> (list of recorded CROSS codes, complete flag), filled
        # per subprocess directory and written out by write_f2py_splitter; see
        # recorded_crossing_codes.
        self.crossing_records = {}
        ProcessExporterFortran.__init__(self, *args, **opts)

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        self.model = model
        #First copy the full template tree if dir_path doesn't exit
        if os.path.isdir(self.dir_path):
            return
        
        logger.info('initialize a new standalone directory: %s' % \
                        os.path.basename(self.dir_path))
        temp_dir = pjoin(self.mgme_dir, 'Template/LO')
        
        # Create the directory structure
        os.mkdir(self.dir_path)
        os.mkdir(pjoin(self.dir_path, 'Source'))
        os.mkdir(pjoin(self.dir_path, 'Source', 'MODEL'))
        os.mkdir(pjoin(self.dir_path, 'Source', 'DHELAS'))
        os.mkdir(pjoin(self.dir_path, 'SubProcesses'))
        os.mkdir(pjoin(self.dir_path, 'bin'))
        os.mkdir(pjoin(self.dir_path, 'bin', 'internal'))
        os.mkdir(pjoin(self.dir_path, 'lib'))
        os.mkdir(pjoin(self.dir_path, 'Cards'))
        
        # Information at top-level
        #Write version info
        shutil.copy(pjoin(temp_dir, 'TemplateVersion.txt'), self.dir_path)
        try:
            shutil.copy(pjoin(self.mgme_dir, 'MGMEVersion.txt'), self.dir_path)
        except IOError:
            MG5_version = misc.get_pkg_info()
            open(pjoin(self.dir_path, 'MGMEVersion.txt'), 'w').write( \
                "5." + MG5_version['version'])
        
        
       

        if model['running_elements']:
            fsock = open( pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f_sp'), 'r')
            text = fsock.read()
            fsock.close()
            fsock = open(pjoin(self.dir_path, 'SubProcesses', 'makefileP'),'w')
            text = text.replace('BLASLIBS =', 'BLASLIBS = %s' % self.blas_link_flags())
            text = text.replace('LINKLIBS =  -L../../lib/', 'LINKLIBS =  -L../../lib/ -lrunning')
            text = text.replace('LIBS =', 'LIBS = $(LIBDIR)/librunning.$(libext)')
            fsock.write(text)
            fsock.close()
        else:
            # Add file in SubProcesses
            mk = open(pjoin(self.mgme_dir, 'madgraph', 'iolibs',
                            'template_files', 'makefile_sa_f_sp')).read()
            mk = mk.replace('BLASLIBS =', 'BLASLIBS = %s' % self.blas_link_flags())
            open(pjoin(self.dir_path, 'SubProcesses', 'makefileP'), 'w').write(mk)
        

                        
        # Add file in Source
        shutil.copy(pjoin(temp_dir, 'Source', 'make_opts'), 
                    pjoin(self.dir_path, 'Source'))   

        # add the makefile 
        filename = pjoin(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FileWriter(filename),model)          

        # add default vector.inc for SA code
        #filename = pjoin(self.dir_path, 'Source', 'vector.inc')
        #self.write_vector_inc_for_sa(writers.FileWriter(filename), model)

    #===========================================================================
    # handling vector.inc (needed by the model) for SA (assuming no batch)
    #===========================================================================     
    #def write_vector_inc_for_sa(self, writer, model):
    #    """ """
#
#        text="""
#        INTEGER WARP_SIZE
#       PARAMETER (WARP_SIZE=1)
#       INTEGER NB_WARP
#       PARAMETER (NB_WARP=1)
#       INTEGER VECSIZE_MEMMAX
#       PARAMETER (VECSIZE_MEMMAX=1)
#
#"""
#        writer.write(text)
#        return


    #===========================================================================
    # export model files
    #===========================================================================
    def export_model_files(self, model_path):
        """export the model dependent files for V4 model"""

        super(ProcessExporterFortranSA,self).export_model_files(model_path)
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO  
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(os.path.join(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()        
        
        text = open(pjoin(self.dir_path,'SubProcesses','check_sa.f')).read()
        text = text.replace('call setpara(\'param_card.dat\')', 'call setpara(\'param_card.dat\', .true.)')
        fsock = open(pjoin(self.dir_path,'SubProcesses','check_sa.f'), 'w')
        fsock.write(text)
        fsock.close()
        
        self.make_model_symbolic_link()

    #===========================================================================
    # write a procdef_mg5 (an equivalent of the MG4 proc_card.dat)
    #===========================================================================
    def write_procdef_mg5(self, file_pos, modelname, process_str):
        """ write an equivalent of the MG4 proc_card in order that all the Madevent
        Perl script of MadEvent4 are still working properly for pure MG5 run.
        Not needed for StandAlone so just return
        """
        
        return


    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS and MODEL directories, to set up
        everything for running standalone
        """

        source_dir = pjoin(self.dir_path, "Source")
        logger.info("Running make for Source directory")
        try:
            misc.compile(cwd=source_dir, mode='fortran')
        except:
            misc.compile(arg=['../lib/libdhelas.a'], cwd=source_dir, mode='fortran')
            misc.compile(arg=['../lib/libmodel.a'], cwd=source_dir, mode='fortran')

    #===========================================================================
    # Create proc_card_mg5.dat for Standalone directory
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize Standalone MG4 directory by 
           generation proc_card_mg5.dat
           generate a global makefile
           """
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}

        self.compiler_choice(compiler)
        self.make()

        # Write command history as proc_card_mg5
        if history and os.path.isdir(pjoin(self.dir_path, 'Cards')):
            output_file = pjoin(self.dir_path, 'Cards', 'proc_card_mg5.dat')
            history.write(output_file)
        
        ProcessExporterFortran.finalize(self, matrix_elements, 
                                             history, mg5options, flaglist)
        open(pjoin(self.dir_path,'__init__.py'),'w')
        open(pjoin(self.dir_path,'SubProcesses','__init__.py'),'w')

        if False:#'mode' in self.opt and self.opt['mode'] == "reweight":
            #add the module to hande the NLO weight
            files.copytree(pjoin(MG5DIR, 'Template', 'RWGTNLO'),
                          pjoin(self.dir_path, 'Source'))
            files.copytree(pjoin(MG5DIR, 'Template', 'NLO', 'Source', 'PDF'),
                           pjoin(self.dir_path, 'Source', 'PDF'))
            self.write_pdf_opendata()
            
        if self.prefix_info: 
            self.write_f2py_splitter()
            self.write_f2py_makefile(self.model)
            self.write_f2py_check_sa(matrix_elements,
                            pjoin(self.dir_path,'SubProcesses','check_sa.py'))
        else:
            # create a single makefile to compile all the subprocesses
            text = '''\n# For python linking (require f2py part of numpy)\nifeq ($(origin MENUM),undefined)\n  MENUM=2\nendif\n''' 
            deppython = ''
            for Pdir in os.listdir(pjoin(self.dir_path,'SubProcesses')):
                if os.path.isdir(pjoin(self.dir_path, 'SubProcesses', Pdir)):
                    text += '%(0)s/matrix$(MENUM)py.so:\n\tcd %(0)s;make matrix$(MENUM)py.so\n'% {'0': Pdir}
                    deppython += ' %(0)s/matrix$(MENUM)py.so ' % {'0': Pdir}
            text+='all: %s\n\techo \'done\'' % deppython
            
            ff = open(pjoin(self.dir_path, 'SubProcesses', 'makefile'),'a')
            ff.write(text)
            ff.close()


    def write_f2py_splitter(self):
        """write a function to call the correct matrix element"""

        template = open(pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files', self.f2py_matrix_splitter)).read()
        template2 = open(pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files', self.f2py_wrapper_all)).read()

        allids = list(self.prefix_info.keys())
        allprefix = [self.prefix_info[key][0] for key in allids]
        allncomb = [self.prefix_info[key][2] for key in allids]
        alliden = [self.prefix_info[key][3] for key in allids] 
        min_nexternal = min([len(ids[0]) for ids in allids])
        max_nexternal = max([len(ids[0]) for ids in allids])

        info = []
        for (key, pid), (prefix, tag, ncomb, iden) in self.prefix_info.items():
            info.append('#PY %s : %s # %s %s' % (tag, key, prefix, pid))
            
        flavor_text= "  flavor(:) = 1\n"
        flavor_text += " do i =1, npdg\n"
        nb = 0
        for pdg, pids in self.model['merged_particles'].items():
            for group_pos, pid in enumerate(pids, 1):
                if nb !=0:
                    flavor_text += ' else'
                else:
                    flavor_text += ' '
                nb += 1
                # flavor(i) is the 1-based position of the actual PDG within
                # the merged-particle group; pdgs(i) is rewritten to carry
                # the merged-particle ID (with the original sign).
                flavor_text += 'if (abs(pdgs(i)).eq.%i)then\n flavor(i) = %i\n pdgs(i) = Sign(%i, pdgs(i))\n' % (pid, group_pos, pdg)
        if nb>0:
            flavor_text += 'endif\n'
        flavor_text += " enddo\n"

        # `text` dispatches to the raw per-process routines (GET_ALL_INTER /
        # GET_DENSITY), which still take the FLAVOR(NEXTERNAL) array and resolve
        # the flavor index internally. `smtext` is the smatrixhel dispatch: the
        # raw SMATRIXHEL now takes the resolved FLAV_IDX (not the FLAVOR array),
        # so we resolve FLAVOR->FLAV_IDX inline with the per-process
        # GET_FLAVOR_INDEX (which is part of matrix.f, hence linked into this
        # all_matrix module) before calling SMATRIXHEL.
        text = []
        smtext = []
        smatrixhel_prefixes = set()

        for n_ext in range(min_nexternal, max_nexternal+1):
            current_id = [ids[0] for ids in allids if len(ids[0])==n_ext]
            current_pid = [ids[1] for ids in allids if len(ids[0])==n_ext]
            if not current_id:
                continue
            if min_nexternal != max_nexternal:
                if n_ext == min_nexternal:
                    line = '       if (npdg.eq.%i)then' % n_ext
                else:
                    line = '       else if (npdg.eq.%i)then' % n_ext
                text.append(line)
                smtext.append(line)

            for ii,pdgs in enumerate(current_id):
                pid = current_pid[ii]
                condition = '.and.'.join(['%i.eq.pdgs(%i)' %(pdg, i+1) for i, pdg in enumerate(pdgs)])
                if ii==0:
                    line = ' if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition, pid, ii)
                else:
                    line = ' else if(%s.and.(procid.le.0.or.procid.eq.%d)) then ! %i' % (condition,pid,ii)
                text.append(line)
                smtext.append(line)
                prefix = self.prefix_info[(pdgs,pid)][0]
                text.append(' call %s%%(fct_name)s' % prefix)
                smatrixhel_prefixes.add(prefix)
                smtext.append(' call %ssmatrixhel(p, nhel, %sget_flavor_index(flavor), ans)'
                              % (prefix, prefix))
            text.append(' endif')
            smtext.append(' endif')
        #close the function
        if min_nexternal != max_nexternal:
            text.append('endif')
            smtext.append('endif')

        # INTEGER declarations for the per-process GET_FLAVOR_INDEX functions
        # used inline by the smatrixhel dispatch (their name does not start with
        # i-n, so they default to REAL without an explicit declaration).
        flavor_index_decl = '\n'.join('  integer %sget_flavor_index' % prefix
                                      for prefix in sorted(smatrixhel_prefixes))

        # smatrixhel_idx: the same dispatch keyed on the 1-based matrix-element
        # slot (the get_pdg_order / get_prefix index) instead of on the PDG
        # codes, taking the extended FLAV_IDX as given. A FOLDED crossed
        # subprocess has no PDG entry of its own -- that is the whole point of
        # folding -- so the PDG dispatch cannot reach it; a caller that resolved
        # the crossing itself (through GET_PDG_FOR_FLAVOR) holds a slot and an
        # extended index instead. It shares f77_smatrixhel's alphas/scale2
        # handling so that a crossed and an uncrossed evaluation of the same
        # event use the exact same running couplings.
        idxtext = []
        for i, prefix in enumerate(allprefix, 1):
            keyword = 'if' if i == 1 else 'else if'
            idxtext.append(' %s (procindex.eq.%i) then' % (keyword, i))
            idxtext.append(' call %ssmatrixhel(p, nhel, flav_idx, ans)' % prefix)
        if idxtext:
            idxtext.append(' endif')

        all_prefix = set([k[0] for k in self.prefix_info.values()])
        setpara_for_each_matrix = ''
        for prefix in all_prefix:
            setpara_for_each_matrix += " CALL %sINITIALISEMODEL(PATH)\n" % prefix


        params = self.get_model_parameter(self.model)
        parameter_setup =[]
        for key, var in params.items():
            if not key or not var:
                continue
            parameter_setup.append('        CASE ("%s")\n          %s = value' 
                                   % (key, var))

        # part for the resetting of the helicity
        helreset_def = []
        helreset_setup = []
        for prefix in set(allprefix):
            helreset_setup.append(' %shelreset = .true. ' % prefix)
            helreset_def.append(' logical %shelreset \n common /%shelreset/ %shelreset' % (prefix, prefix, prefix))
        
        #nhel
        all_nhel_f2py = ' '
        all_nhel = ''
        all_iden = ''
        nhel_template_f2py = """
        subroutine %(f2py_prefix)s%(prefix)sget_nhel_entry()
        integer %(prefix)snhel(%(next)s,%(ncombs)s)
        common/%(f2py_prefix)s%(prefix)sPROCESS_NHEL/%(prefix)sNHEL
        call %(f2py_prefix)sf77_%(prefix)sget_nhel_entry(%(prefix)sNHEL)

        return
        end 
"""
        nhel_template = """subroutine %(f2py_prefix)sf77_%(prefix)sget_nhel_entry(NHEL)
        integer NHEL(%(next)s,%(ncombs)s)
        integer idendummy
C       Fill NHEL through GET_NHEL rather than reading the PROCESS_NHEL common
C       directly. With the canonical helicity encoder/decoder the table is
C       materialized at runtime (GET_NHEL calls FILL_NHEL), so an early caller
C       -- e.g. reweighting building its per-config helicity map at init, before
C       any matrix-element evaluation -- would otherwise read a table of zeros.
C       Every standalone matrix.f defines GET_NHEL (materializing or DATA-backed),
C       so this also stays correct for split-order processes.
        call %(prefix)sget_nhel(idendummy, NHEL)
        return
        end
"""

        f2py_prefix = ''
        if self.opt['output_options'] and 'prefixf2py' in self.opt['output_options']:
            f2py_prefix = 'f%s_' % self.opt['output_options']['prefixf2py']

        done_prefix = set()
        for prefix, ids, ncomb, iden in zip(allprefix, allids, allncomb, alliden):
            if prefix in done_prefix:
                continue
            done_prefix.add(prefix)
            all_nhel += nhel_template % {'prefix': prefix, 
                                         'next': len(ids[0]), 
                                         'ncombs': ncomb,
                                         'f2py_prefix': f2py_prefix}
            all_nhel_f2py += nhel_template_f2py % {'prefix': prefix, 
                                                   'next': len(ids[0]), 
                                                   'ncombs': ncomb, 
                                                   'f2py_prefix': f2py_prefix}
        # Build IDENS entries ONCE per ME slot (must align 1-to-1 with get_pdg_order / allids).
        all_iden = ''
        for i, iden in enumerate(alliden, start=1):
            all_iden += ' idens(%s) = %s \n' % (i, iden)
        #misc.sprint(all_iden)

        formatting = {'python_information':'\n'.join(info),
                          'smatrixhel': '\n'.join(smtext),
                          'smatrixhel_idx': '\n'.join(idxtext),
                          'flavor_index_decl': flavor_index_decl,
                          'maxpart': max_nexternal,
                          'nb_me': len(allids),
                          'pdgs': ','.join(str(pdg[i]) if i<len(pdg) else '0' 
                                           for i in range(max_nexternal) for (pdg,pid) in allids),
                          'prefix':'\',\''.join(allprefix),
                          'pids': ','.join(str(pid) for (pdg,pid) in allids),
                          'inter_splitter': '\n'.join(text) % {'fct_name': 'GET_ALL_INTER(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, INTER)'},
                          'parameter_setup': '\n'.join(parameter_setup),
                          'helreset_def' : '\n'.join(helreset_def),
                          'helreset_setup' : '\n'.join(helreset_setup),
                          'flavormapping': flavor_text,
                          'setpara_for_each_matrix':setpara_for_each_matrix,
                          'nhel': all_nhel,
                          'f2py_prefix': f2py_prefix,
                          'idens_value': all_iden,
                          'density_splitter': '\n'.join(text) % {'fct_name': 'GET_DENSITY(P, POS, N_CHANGING, ALLOW_HEL, N_COMB, FLAVOR, ALPHAS, SCALE2, INTER)'},
                          
                          }

        formatting['lenprefix'] = len(formatting['prefix'])
        text = template % formatting
        fsock = writers.FortranWriter(pjoin(self.dir_path, 'SubProcesses', 'all_matrix.f'),'w')
        fsock.writelines(text)
        fsock.close()
        formatting['nhel'] = all_nhel_f2py
        text = template2 % formatting
        f2py_wrapper_path = pjoin(self.dir_path, 'SubProcesses', 'f2py_wrapper.f')
        fsock = writers.FortranWriter(f2py_wrapper_path,'w')
        fsock.writelines(text)
        fsock.close()

        # Expose the per-process crossing-aware f2py entry points
        # (PY_<prefix>GET_PDG_FOR_FLAVOR / GET_FLAVOR_LAYOUT / GET_NHEL_IDX /
        # GET_DENSITY_IDX, etc.) in the COMBINED all_matrix module.  They live in
        # each subprocess' self-contained f2py_matrix_wrapper.f and call the
        # M<n>_* routines already linked into liball...me; the combined wrapper is
        # otherwise base-only, so a crossing-aware python caller (MadSpin's
        # density path) could not reach a folded crossed subprocess through it.
        # Concatenate rather than add the files to the f2py command line: f2py's
        # multi-file build leaves the extra wrappers' symbols undefined at
        # dlopen on some platforms, whereas a single scanned source links them.
        wrappers = sorted(glob.glob(pjoin(self.dir_path, 'SubProcesses',
                                          '*', 'f2py_matrix_wrapper.f')))
        if wrappers:
            with open(f2py_wrapper_path, 'a') as fsock:
                for wpath in wrappers:
                    fsock.write('\nC     crossing-aware f2py wrappers from %s\n'
                                % os.path.relpath(wpath,
                                    pjoin(self.dir_path, 'SubProcesses')))
                    fsock.write(open(wpath).read())

        self.write_crossing_records()

    def write_crossing_records(self):
        """List the folded crossed subprocesses for the python consumers of the
        combined f2py module (see recorded_crossing_codes).

        GET_PDG_FOR_FLAVOR tells a caller what process an extended FLAV_IDX
        evaluates, but not whether that crossing is a subprocess the generation
        asked for: its CROSS space is dense and also holds crossings that are
        merely applicable (a Z or a decay product pulled into the initial state).
        Only generation knows the difference, so it is recorded here, one line
        per matrix element,

            <proc_prefix> <complete> <cross code> ...

        with <complete> 0 when a recorded crossed process could not be matched
        to a runtime crossing -- the consumer must then not trust the list to
        cover every folded subprocess. The file is always written (empty lists
        included) so that its absence means "produced before this existed", and
        a consumer can tell that apart from "nothing was folded"."""
        path = pjoin(self.dir_path, 'SubProcesses', 'crossed_flavors.dat')
        with open(path, 'w') as fsock:
            fsock.write('# folded crossed subprocesses, written by MG5aMC\n')
            fsock.write('# <proc_prefix> <complete> <cross code> ...\n')
            for prefix in sorted(self.crossing_records):
                codes, complete = self.crossing_records[prefix]
                fsock.write('%s %d%s\n' % (prefix, 1 if complete else 0,
                                           ''.join(' %d' % c for c in codes)))

    def get_model_parameter(self, model):
        """ returns all the model parameter
        """
        params = {}
        for p in model.get('parameters')[('external',)]:
            name = p.name
            nopref = name[4:] if name.startswith('mdl_') else name
            params[nopref] = name
            
            block = p.lhablock
            lha = '_'.join([str(i) for i in p.lhacode])
            params['%s_%s' % (block.upper(), lha)] = name

        if model['running_elements']:
            add_scale = set()
            for runs in self.model.get('running_elements'):
                for line_run in runs.run_objects:
                    for one_element in line_run:
                        add_scale.add(one_element.lhablock)
            for block in add_scale:
                if block.upper() == "SMINPUTS":
                    continue
                name = block
                params['%s__scale' % (block.upper())] = 'mdl__%s__scale' % (block.upper())
                params['mdl__%s__scale' % (block.upper())] = 'mdl__%s__scale' % (block.upper())

        return params                      
                                        
    def write_f2py_matrix_wrapper(self, writer, replace_dict):
        """ Write the f2py wrapper for matrix element."""

        path =pjoin(_file_path, 'iolibs', 'template_files', self.f2py_template)
        template = open(path).read()
        writer.write(template % replace_dict)

    def write_f2py_check_sa(self, matrix_element, writer):
        """ Write the general check_sa.py in SubProcesses that calls all processes successively."""
        # To be implemented. It is just an example file, i.e. not crucial.
        return
    
    def write_f2py_makefile(self, model):
        """ """
        template = pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f2py')
        destination = pjoin(self.dir_path, 'SubProcesses', 'makefile')

        # Add file in SubProcesses
        if model['running_elements']:
            text = open(template,'r').read()
            text = text.replace('LINKLIBS_ME =  -L../lib/', 'LINKLIBS_ME =  -L../lib/ -lrunning ')
            text = text.replace('LINKLIBS_ALL =  -L../lib/', 'LINKLIBS_ALL =  -L../lib/ -lrunning ')
            open(destination, 'w').write(text)
        else:
             shutil.copy(template, destination)

    def create_MA5_cards(self,*args,**opts):
        """ Overload the function of the mother so as to bypass this in StandAlone."""
        pass

    def compiler_choice(self, compiler):
        """ Different daughter classes might want different compilers.
        So this function is meant to be overloaded if desired."""
        
        self.set_compiler(compiler)

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model, number, **opt):
        """Generate the Pxxxxx directory for a subprocess in MG4 standalone,
        including the necessary matrix.f and nexternal.inc files"""

        # Helper
        def compute_iden_from_pdgs(ids, ninitial, model):
            """
            Helper function to compute denominator factor
            """
            def nhel_from_particle(p):
                spin = int(p.get('spin'))
                # for massless vectors use 2 helicities not 3
                mass = p.get('mass')
                if spin == 3 and (mass == 'ZERO' or str(mass).upper() == 'ZERO'):
                    return 2
                return spin

            def color_dim_from_particle(p):
                # In UFO, color is typically 1, 3, -3, 8, ...
                return abs(int(p.get('color')))

            incoming = ids[:ninitial]
            iden = 1
            for pid in incoming:
                p = model.get_particle(pid)
                iden *= nhel_from_particle(p) * color_dim_from_particle(p)
            return int(iden)
        
        cwd = os.getcwd()
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = pjoin(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

        if self.opt['sa_symmetry']:
            # avoid symmetric output
            for i,proc in enumerate(matrix_element.get('processes')):
                   
                tag = proc.get_tag()     
                legs = proc.get('legs')[:]
                leg0 = proc.get('legs')[0]
                leg1 = proc.get('legs')[1]
                if not leg1.get('state'):
                    proc.get('legs')[0] = leg1
                    proc.get('legs')[1] = leg0
                    flegs = proc.get('legs')[2:]
                    for perm in itertools.permutations(flegs):
                        for i,p in enumerate(perm):
                            proc.get('legs')[i+2] = p
                        dirpath2 =  pjoin(self.dir_path, 'SubProcesses', \
                               "P%s" % proc.shell_string())
                        #restore original order
                        proc.get('legs')[2:] = legs[2:]              
                        if os.path.exists(dirpath2):
                            proc.get('legs')[:] = legs
                            return 0
                proc.get('legs')[:] = legs

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        #try:
        #    os.chdir(dirpath)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % dirpath)
        #    return 0

        logger.info('Creating files in directory %s' % dirpath)




        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Create the matrix.f file and the nexternal.inc file
        if self.opt['export_format']=='standalone_msP':
            filename = pjoin(dirpath, 'matrix_prod.f')
        else:
            filename = pjoin(dirpath, 'matrix.f')
            
        proc_prefix = ''
        if 'prefix' in self.cmd_options:
            if self.cmd_options['prefix'] == 'int':
                proc_prefix = 'M%s_' % number
            elif self.cmd_options['prefix'] == 'proc':
                proc_prefix = matrix_element.get('processes')[0].shell_string().split('_',1)[1]
            else:
                raise Exception('--prefix options supports only \'int\' and \'proc\'')
            ncomb = matrix_element.get_helicity_combinations()
            #iden = matrix_element.get_denominator_factor() 
            for proc in matrix_element.get('processes'):
                ids = [l.get('id') for l in proc.get('legs_with_decays')]
                iden = compute_iden_from_pdgs(ids, ninitial, self.model)
                self.prefix_info[(tuple(ids), proc.get('id'))] = [proc_prefix, proc.get_tag(), ncomb, iden]
            # Which CROSS codes of this matrix element name a crossed subprocess
            # this generation actually requested. Only a python caller holding
            # them can walk the folded crossings without also evaluating the
            # merely-applicable ones; write_f2py_splitter exports them.
            self.crossing_records[proc_prefix] = \
                self.recorded_crossing_codes(matrix_element)

        template = open(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'makefile_sa_f_sp'),'r')
        text = template.read()
        template.close()
        fsock = open(pjoin(self.dir_path, 'SubProcesses', 'makefileP'),'w')
        text = text.replace('BLASLIBS =', 'BLASLIBS = %s' % self.blas_link_flags())
        fsock.write(text)
        fsock.close()

        #important to put that first
        if self.format == 'standalone':
            filename2 = pjoin(dirpath, 'check_sa.f')
            self.write_check_sa(writers.FortranWriter(filename2), matrix_element, proc_prefix)


        replace_dict = self.write_matrix_element_v4(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model,
            proc_prefix=proc_prefix,
            return_replace_dict=True)
        calls = replace_dict.get('return_value', 0)

        self.write_f2py_matrix_wrapper(
            writers.FortranWriter(pjoin(dirpath, 'f2py_matrix_wrapper.f')),
                                  replace_dict=replace_dict)

        # Python convenience wrapper letting callers pass either a FLAVOR array
        # or a single flavor index to the f2py matrix2py module (dispatches to
        # the array or *_idx Fortran entry point). Static helper, copied as-is.
        shutil.copy(pjoin(_file_path, 'iolibs', 'template_files',
                          'f2py_flavor_dispatch.py'),
                    pjoin(dirpath, 'flavor_dispatch.py'))


        if self.opt['export_format'] == 'standalone_msP':
            filename =  pjoin(dirpath,'configs_production.inc')
            mapconfigs, s_and_t_channels = self.write_configs_file(\
                writers.FortranWriter(filename),
                matrix_element)

            filename =  pjoin(dirpath,'props_production.inc')
            self.write_props_file(writers.FortranWriter(filename),
                             matrix_element,
                             s_and_t_channels)

            filename =  pjoin(dirpath,'nexternal_prod.inc')
            self.write_nexternal_madspin(writers.FortranWriter(filename),
                             nexternal, ninitial)

        if self.opt['export_format']=='standalone_msF':
            filename = pjoin(dirpath, 'helamp.inc')
            ncomb=matrix_element.get_helicity_combinations()
            self.write_helamp_madspin(writers.FortranWriter(filename),
                             ncomb)
            
        filename = pjoin(dirpath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(dirpath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(dirpath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))
        


        # Generate diagrams
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
            filename = pjoin(dirpath, "matrix.ps")
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude=True)
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        linkfiles = ['check_sa.f', 'coupl.inc']
        if self.format == 'standalone':
            linkfiles = ['coupl.inc']


        filename = pjoin(dirpath, 'check_sa.f')
        self.write_check_sa(writers.FortranWriter(filename),
                           matrix_element,
                           proc_prefix=proc_prefix)        


        linkfiles = ['coupl.inc']
        for file in linkfiles:
            ln('../%s' % file, cwd=dirpath)
        ln('../makefileP', name='makefile', cwd=dirpath)
        # Return to original PWD
        #os.chdir(cwd)

        if not calls:
            calls = 0
        return calls


    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model):
        """Write the nexternal.inc file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_makefile_source')
        set_of_lib = '$(LIBDIR)libdhelas.$(libext) $(LIBDIR)libmodel.$(libext)'
        model_line='''$(LIBDIR)libmodel.$(libext): MODEL\n\t cd MODEL; make\n'''

        if model['running_elements']:
            running_line = '''$(LIBDIR)librunning.$(libext): RUNNING\n\t cd RUNNING; make\n'''
            set_of_lib += ' $(LIBDIR)librunning.$(libext) '
        else:
            running_line  = '' 

        replace_dict= {'libraries': set_of_lib, 
                       'model':model_line,
                       'additional_dsample': '',
                       'additional_dependencies':'',
                       'additional_clean':'',
                       'running': running_line} 

        text = open(path).read() % replace_dict
        
        if writer:
            writer.write(text)
        
        return replace_dict

    def _format_flavor_rebuild_only(self, n_flavors, flav_table_flat):
        """Decl/setup blocks for the case where FLAVOR must be rebuilt from the
        threaded FLAV_IDX but there is *no* per-call masking (single flavor,
        non-merged, or trivial all-ones mask). Only the FLAV_TABLE and the
        rebuild loop are emitted; the HELAS calls run unguarded so AMP needs no
        zero-init. Returns a 2-tuple (decl_block, setup_block) of strings, each
        a block of Fortran source lines (the declaration block and the
        BEGIN-CODE rebuild block respectively)."""
        items = ', '.join(str(v) for v in flav_table_flat)
        decl = '\n'.join([
            'C     Flavor table for the FLAV_IDX -> FLAVOR rebuild.',
            '      INTEGER NMASK_FLAV',
            '      PARAMETER (NMASK_FLAV=%d)' % n_flavors,
            '      INTEGER MASK_J',
            '      INTEGER FLAV_TABLE(NEXTERNAL, NMASK_FLAV)',
            '      DATA FLAV_TABLE / %s /' % items,
        ])
        setup = '\n'.join([
            'C     Rebuild FLAVOR(NEXTERNAL) from the resolved flavor index.',
            '      IF (FLAV_IDX .GE. 1 .AND. FLAV_IDX .LE. NMASK_FLAV) THEN',
            '        DO MASK_J = 1, NEXTERNAL',
            '          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, FLAV_IDX)',
            '        ENDDO',
            '      ELSE',
            '        DO MASK_J = 1, NEXTERNAL',
            '          FLAVOR(MASK_J) = FLAV_TABLE(MASK_J, 1)',
            '        ENDDO',
            '      ENDIF',
        ])
        return (decl, setup)

    def _get_flavor_mask_blocks(self, matrix_element):
        """Build the Fortran declaration / setup blocks injected into GET_AMP
        (or the monolithic MATRIX) for the always-on flavor machinery.

        The blocks *always* rebuild FLAVOR(NEXTERNAL) from the threaded FLAV_IDX
        via FLAV_TABLE, giving a uniform API (every matrix function takes
        FLAV_IDX). When the ME has merged flavors that select different diagrams
        (non-trivial mask), the per-call IAND mask machinery is emitted on top of
        the rebuild; otherwise only the rebuild is emitted.

        Returns (decl_block, setup_block, n_mask, active_flavor_mask) where
        n_mask is non-zero only for the non-trivial-mask case (it drives
        use_flavor_mask / the HELAS IAND guards). NFLAV for the rebuild itself
        comes from _build_flav_table_flat and is always >= 1.
        """

        n_table, flav_table_flat = self._build_flav_table_flat(matrix_element)

        allowed_flavors = matrix_element.compute_flavor_masks()
        non_trivial = (getattr(self, 'use_flavor_mask', False)
                       and len(allowed_flavors) > 0
                       and not matrix_element.flavor_mask_is_trivial())

        if not non_trivial:
            decl_block, setup_block = self._format_flavor_rebuild_only(
                n_table, flav_table_flat)
            return (decl_block, setup_block, 0, 0)

        n_flavors = len(allowed_flavors)
        all_amps = matrix_element.get_all_amplitudes()
        all_wfs = matrix_element.get_all_wavefunctions()
        n_amps = len(all_amps)
        n_wfs = len(all_wfs)

        # Mask values are indexed by the object's 'number' attribute (1-based,
        # contiguous).
        wf_masks = [0] * n_wfs
        for wf in all_wfs:
            idx = wf.get('number') - 1
            if 0 <= idx < n_wfs:
                wf_masks[idx] = wf['flavor_mask'] if 'flavor_mask' in wf else 0
        amp_masks = [0] * n_amps
        for amp in all_amps:
            idx = amp.get('number') - 1
            if 0 <= idx < n_amps:
                amp_masks[idx] = amp['flavor_mask'] if 'flavor_mask' in amp else 0
        active_flavor_mask = 0
        for amp_mask in amp_masks:
            active_flavor_mask |= amp_mask

        nwords_wf = (n_wfs + 63) // 64
        nwords_amp = (n_amps + 63) // 64

        wf_index_masks, active_wf_index_masks = self._build_flavor_index_masks(
            wf_masks, n_flavors, nwords_wf)
        amp_index_masks, active_amp_index_masks = self._build_flavor_index_masks(
            amp_masks, n_flavors, nwords_amp)

        # FLAV_TABLE holds group positions (built once by _build_flav_table_flat,
        # shared with GET_FLAVOR_INDEX). When the mask is non-trivial the table
        # has one column per allowed flavor, so n_table == n_flavors here.
        decl_block = self._format_flavor_mask_decl(
            n_flavors, n_wfs, n_amps, nwords_wf, nwords_amp,
            wf_index_masks, amp_index_masks,
            active_wf_index_masks, active_amp_index_masks, flav_table_flat,
            thread_flav_idx=True)
        setup_block = self._format_flavor_mask_setup(
            leading_comment='C     Rebuild FLAVOR and select the per-flavor masks.',
            append_amp_init=True, thread_flav_idx=True)

        return (decl_block, setup_block, n_flavors, active_flavor_mask)

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                                write=True, proc_prefix='', return_replace_dict=False):
        """Export a matrix element to a matrix.f file in MG4 standalone format
        if write is on False, just return the replace_dict and not write anything."""


        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if writer:
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter but %s" % type(writer))
            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False


        if 'sa_symmetry' not in self.opt:
            self.opt['sa_symmetry']=False
        # --use_crossing of the generate command (default on); see
        # fill_crossing_replace_dict.
        if 'use_crossing' not in self.opt:
            self.opt['use_crossing']=True

        # ... and gated off per matrix element for processes whose definition
        # pins a specific s-channel, which no crossing of them preserves. This
        # is decided here rather than in the interface so that one constrained
        # `add process` does not disable crossing for the unconstrained ones.
        use_crossing = self.opt['use_crossing'] and \
            not any(self.breaks_crossing_symmetry(proc)
                    for proc in matrix_element.get('processes'))


        # The proc_id is for MadEvent grouping which is never used in SA.
        replace_dict = {'global_variable':'', 'amp2_lines':'',
                                       'proc_prefix':proc_prefix, 'proc_id':'',
                                       'flavor_mask_decl':'',
                                       'flavor_mask_setup':''}

        # Always-on flavor machinery: every matrix function takes FLAV_IDX and
        # rebuilds FLAVOR internally (consistent API). NFLAV is >= 1 (an ME with
        # no merged variants is a single flavor with an all-ones table row).
        n_table, flav_table_flat = self._build_flav_table_flat(matrix_element)
        replace_dict['nflav'] = n_table

        # Build the FLAVOR-rebuild block (always) plus, for merged flavors that
        # select different diagrams, the per-call IAND mask machinery. n_mask is
        # non-zero only in that latter case and drives use_flavor_mask / the
        # HELAS IAND guards. The try/finally ensures we never leak the writer
        # state into the next matrix element.
        mask_decl, mask_setup, n_mask, active_flavor_mask = \
                self._get_flavor_mask_blocks(matrix_element)
        replace_dict['flavor_mask_decl'] = mask_decl
        replace_dict['flavor_mask_setup'] = mask_setup

        fortran_model.use_flavor_mask = (n_mask > 0)
        fortran_model.me_n_flavors = n_mask
        fortran_model.me_active_flavor_mask = active_flavor_mask
        # Only matrix_standalone_v4.inc hands GET_AMP the crossed IC built by
        # APPLY_CROSSING, so it is the only one whose NSF/NSV flags may go
        # through IC. The other variants selected below (msP, msF, matchbox,
        # splitOrders) have no IC to read and must keep the bare flag. Mirror
        # the template choice made further down; split_orders is only fetched
        # again here, which is side-effect free.
        fortran_model.use_crossing_ic = (
            use_crossing
            and self.matrix_template == 'matrix_standalone_v4.inc'
            and self.opt['export_format'] not in ('standalone_msP',
                                                  'standalone_msF',
                                                  'matchbox',
                                                  'madloop_matchbox')
            and not matrix_element.get('processes')[0].get('split_orders'))
        try:
            # Extract helas calls
            helas_calls = fortran_model.get_matrix_element_calls(\
                        matrix_element)
        finally:
            fortran_model.use_flavor_mask = False
            fortran_model.me_n_flavors = 0
            fortran_model.me_active_flavor_mask = None
            fortran_model.use_crossing_ic = False

        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal
        replace_dict['nincoming'] = ninitial

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines. helicity_lines (the explicit NHEL config DATA
        # table) is still consumed by the msP/msF/splitOrders standalone
        # templates; matrix_standalone_v4.inc instead uses the canonical
        # encoder/decoder tables below (NHSTATE/STATES/HELALLOW) and
        # materializes PROCESS_NHEL at runtime via FILL_NHEL.
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines
        hel_data = self._helstate_data(matrix_element)
        replace_dict['maxhel'] = hel_data['maxhel']
        replace_dict['nhstate_data'] = hel_data['nhstate_data']
        replace_dict['states_data'] = hel_data['states_data']
        replace_dict['hel_allow_data'] = hel_data['hel_allow_data']

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        replace_dict['den_factor_line'] = self.get_den_factor_line(matrix_element)

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor
        # |M|^2 is summed over one color flow per reversal pair when the basis
        # allows it, so the color matrix is only over those
        folding = self.get_jamp_folding(matrix_element)
        self.jamp_folding = folding
        nfold = len(folding['representatives']) if folding else ncolor
        replace_dict['ncolorfold'] = nfold
        replace_dict['ncolortriang'] = nfold * (nfold + 1) // 2
        replace_dict['color_fold_index'] = '\n'.join(
            self.get_int_data_lines("COLREP",
                                    [i + 1 for i in folding['representatives']],
                                    var='ICF')) if folding else ''
        replace_dict['color_fold_gather'] = (
            "      DO ICF = 1, NCOLORFOLD\n"
            "        JFOLD(ICF) = JAMP(COLREP(ICF))\n"
            "      ENDDO" if folding else
            "      DO ICF = 1, NCOLOR\n"
            "        JFOLD(ICF) = JAMP(ICF)\n"
            "      ENDDO")
        if not folding:
            replace_dict['color_fold_decl'] = "      INTEGER ICF"
        if folding:
            replace_dict['color_fold_decl'] = \
                "      INTEGER COLREP(NCOLORFOLD)\n      INTEGER ICF"


        replace_dict['hel_avg_factor'] = matrix_element.get_hel_avg_factor()
        replace_dict['beamone_helavgfactor'], replace_dict['beamtwo_helavgfactor'] =\
                                       matrix_element.get_beams_hel_avg_factor()

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines) % {'proc_prefix': replace_dict['proc_prefix']}
        replace_dict['color_init_routine'] = "\n".join(
                self.get_color_init_routine(matrix_element,
                                            replace_dict['proc_prefix']))

        if self.opt['export_format']=='standalone_msP':
        # For MadSpin need to return the AMP2
            amp2_lines = self.get_amp2_lines(matrix_element, [] )
            replace_dict['amp2_lines'] = '\n'.join(amp2_lines)
            replace_dict['global_variable'] = \
         "       Double Precision amp2(NGRAPHS)\n       common/to_amps/  amp2\n"

        # JAMP definition, depends on the number of independent split orders
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        self.jamp_recipes = None

        if len(split_orders)==0:
            replace_dict['nSplitOrders']=''
            # Extract JAMP lines
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines(matrix_element,
                        orbit=self.jamp_orbit_allowed(matrix_element),
                        proc_prefix=replace_dict['proc_prefix'])
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            replace_dict['nSqAmpSplitOrders']=1
            replace_dict['split_order_str_list']=''
            replace_dict['nb_temp_jamp'] = nb_tmp_jamp

        else:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['nAmpSplitOrders']=len(amp_orders)
            replace_dict['nSqAmpSplitOrders']=len(squared_orders)
            replace_dict['nSplitOrders']=len(split_orders)
            replace_dict['split_order_str_list']=str(split_orders)
            amp_so = self.get_split_orders_lines(
                    [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
            sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
            replace_dict['ampsplitorders']='\n'.join(amp_so)
            replace_dict['sqsplitorders']='\n'.join(sqamp_so)           
            jamp_lines, nb_tmp_jamp = self.get_JAMP_lines_split_order(\
                       matrix_element,amp_orders,split_order_names=split_orders)
            replace_dict['nb_temp_jamp'] = nb_tmp_jamp
            # Now setup the array specifying what squared split order is chosen
            replace_dict['chosen_so_configs']=self.set_chosen_SO_index(
                              matrix_element.get('processes')[0],squared_orders)
            
            # For convenience we also write the driver check_sa_splitOrders.f
            # that explicitely writes out the contribution from each squared order.
            # The original driver still works and is compiled with 'make' while
            # the splitOrders one is compiled with 'make check_sa_born_splitOrders'
            check_sa_writer=writers.FortranWriter('check_sa_born_splitOrders.f')
            self.write_check_sa_splitOrders(squared_orders,split_orders,
              nexternal,ninitial,proc_prefix,check_sa_writer)

        if write:
            writers.FortranWriter('nsqso_born.inc').writelines(
                """INTEGER NSQSO_BORN
                   PARAMETER (NSQSO_BORN=%d)"""%replace_dict['nSqAmpSplitOrders'])
            files.cp('nsqso_born.inc', '..')

        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        # The color flow JAMPs, rebuilt from the ones entering the color sum.
        # Standalone does not need a color flow, but it is the mode used to
        # time the matrix element, so it carries the same work as madevent.
        self.set_color_flow_lines_sa(matrix_element, replace_dict, ncolor)

        # The definitions written as one recipe per orbit are held in one
        # array together with the amplitudes, so that the loop running them
        # reads its two operands from the same place.
        recipes = getattr(self, 'jamp_recipes', None)
        replace_dict['jamp_decl'] = '\n'.join(
                self.get_jamp_decl_lines(recipes, replace_dict['proc_prefix']))
        replace_dict['jamp_init_routine'] = '\n'.join(
            self.get_jamp_init_routine(recipes, replace_dict['proc_prefix']))
        if recipes:
            replace_dict['namp_dim'] = 'NGRAPHS+%d' % replace_dict['nb_temp_jamp']
            replace_dict['jamp_tmp_decl'] = ''
        else:
            replace_dict['namp_dim'] = 'NGRAPHS'
            replace_dict['jamp_tmp_decl'] = \
                "      COMPLEX*16 TMP_JAMP(%i)" % replace_dict['nb_temp_jamp']

        # BLAS-3 color sum: every helicity is one column of a single right
        # hand side, so the whole sum is two DSYMM calls instead of one
        # triangular loop per helicity.
        prefix = replace_dict['proc_prefix']
        reps = ([line + 1 for line in folding['representatives']] if folding
                else list(range(1, ncolor + 1)))
        # The batch branch does not go through MATRIX, so it has to carry the
        # color flow JAMPs itself, exactly like the per-helicity path does
        nflow = replace_dict.get('ncolor_flow', ncolor)
        flow_decl, flow_lines = [], []
        if replace_dict.get('jampflow_routine'):
            flow_decl = ["      COMPLEX*16 JAMPFB(%d)" % nflow,
                         "      DOUBLE PRECISION %sJAMP2(%d)" % (prefix, nflow),
                         "      COMMON /%sJAMP2_COMMON/ %sJAMP2" % (prefix,
                                                                    prefix)]
            flow_lines = [
                "            CALL %sGET_JAMPF(JAMPB,JAMPFB)" % prefix,
                "            DO IBH = 1, %d" % nflow,
                "              %sJAMP2(IBH) = %sJAMP2(IBH)" % (prefix, prefix),
                "     $              + DABS(DBLE(JAMPFB(IBH)*DCONJG(JAMPFB(IBH"
                "))))",
                "            ENDDO"]

        if self.blas_wanted(nfold):
            # The batch bypasses MATRIX, so it has to repeat by hand whatever
            # MATRIX's caller does per helicity -- including the crossing.
            # GOODHEL is shared by every crossing of a flavor and indexed by
            # the identity row, so the gate goes through CROSS_GHIDX exactly
            # as %(smatrix_goodhel_gate)s does, and GET_AMP gets the crossed
            # arrays APPLY_CROSSING_TABLE built once before this branch.
            if use_crossing:
                blas_gate = [
                    "          CALL %sCROSS_GHIDX(CROSSUSE, XGPERM, XGSGN,"
                                                                    % prefix,
                    "     $     NHEL(1,IHEL), GHIDX)",
                    "          IF (GHIDX.EQ.0 .OR. GOODHEL(GHIDX,FLAV_USE))"
                    " THEN"]
                blas_amp = [
                    "            IF (CROSSUSE.EQ.0) THEN",
                    "              CALL %sGET_AMP(P,NHEL(1,IHEL),JC(1),"
                    "FLAV_USE,AMPB)" % prefix,
                    "            ELSE",
                    "              CALL %sGET_AMP(PUSE,NHELUSE(1,IHEL),"
                    "ICUSE(1),FLAV_USE,AMPB)" % prefix,
                    "            ENDIF"]
            else:
                blas_gate = ["          IF (GOODHEL(IHEL,FLAV_USE)) THEN"]
                blas_amp = [
                    "            CALL %sGET_AMP(P,NHEL(1,IHEL),JC(1),"
                    "FLAV_USE,AMPB)" % prefix]
            replace_dict['blas_guard_open'] = "("
            replace_dict['blas_guard'] = ") .AND. .NOT.BLASDONE"
            replace_dict['blas_decl'] = "\n".join([
                "      LOGICAL BLASDONE",
                "      INTEGER NBHEL, IBH",
                # NGRAPHS is not in scope here, so size the buffer outright
                "      COMPLEX*16 AMPB(%d), JAMPB(%d)" % (
                    replace_dict['ngraphs'] + (replace_dict['nb_temp_jamp']
                                               if recipes else 0), ncolor),
                "      DOUBLE PRECISION, ALLOCATABLE, SAVE :: JRB(:,:)",
                "      DOUBLE PRECISION, ALLOCATABLE, SAVE :: JIB(:,:)",
                "      INTEGER COLREPB(%d)" % nfold] + flow_decl +
                self.get_int_data_lines("COLREPB", reps, var='IBH'))
            replace_dict['blas_branch'] = "\n".join([
                # NTRY_CSYM>=20 as well: while the C-parity scan is still
                # running the per-helicity loop has to fill TSTORE, and the
                # batch does not. Without crossings NTRY_CSYM tracks NTRY
                # exactly, so this costs nothing.
                "      BLASDONE = .FALSE.",
                "      IF (USERHEL.EQ.-1 .AND. NTRY(FLAV_USE).GE.20",
                "     $    .AND. NTRY_CSYM(FLAV_USE).GE.20",
                "     $    .AND. POLARIZATIONS(0,0).EQ.-1) THEN",
                "        IF (.NOT.ALLOCATED(JRB)) THEN",
                "          ALLOCATE(JRB(%d,NCOMB))" % nfold,
                "          ALLOCATE(JIB(%d,NCOMB))" % nfold,
                "        ENDIF",
                "        NBHEL = 0",
                "        DO IHEL=1,NCOMB",
                # The batch has to honour the C-parity de-duplication the
                # scalar loop below applies, or it evaluates every good
                # helicity where that loop evaluates one per mirror pair --
                # and GET_AMP, not the color sum, is what that costs.
                # De-duplication is all-or-nothing per flavor, so every kept
                # row is doubled by the same factor and one multiply on the
                # total is exact (no per-column scaling, no sqrt(2)).
                "          IF (DEDUP.AND.IHEL.GT.FLIP(IHEL)) CYCLE"] +
                blas_gate + [
                "            NBHEL = NBHEL + 1"] + blas_amp + [
                "            CALL %sGET_JAMP(AMPB,JAMPB)" % prefix] +
                flow_lines + [
                "            DO IBH = 1, %d" % nfold,
                "              JRB(IBH,NBHEL) = DBLE(JAMPB(COLREPB(IBH)))",
                "              JIB(IBH,NBHEL) = DIMAG(JAMPB(COLREPB(IBH)))",
                "            ENDDO",
                "          ENDIF",
                "        ENDDO",
                "        IF (NBHEL.GT.0) THEN",
                "          CALL %sGET_MATRIX_BATCH(JRB,JIB,NBHEL,ANS)" % prefix,
                "          IF (DEDUP) ANS = ANS + ANS",
                "        ENDIF",
                "        BLASDONE = .TRUE.",
                "      ENDIF"])
            replace_dict['blas_routine'] = self.get_blas_routine(
                                                    prefix, nfold, ncomb)
        else:
            # nothing added when BLAS is off, so what is written is exactly
            # what was written before any of this existed
            replace_dict['blas_guard_open'] = ""
            replace_dict['blas_guard'] = ""
            replace_dict['blas_decl'] = ""
            replace_dict['blas_branch'] = ""
            replace_dict['blas_routine'] = ""

        matrix_template = self.matrix_template
        if self.opt['export_format']=='standalone_msP' :
            matrix_template = 'matrix_standalone_msP_v4.inc'
        elif self.opt['export_format']=='standalone_msF':
            matrix_template = 'matrix_standalone_msF_v4.inc'
        elif self.opt['export_format']=='matchbox':
            replace_dict["proc_prefix"] = 'MG5_%i_' % matrix_element.get('processes')[0].get('id')
            replace_dict["color_information"] = self.get_color_string_lines(matrix_element)

        if len(split_orders)>0:
            if self.opt['export_format'] in ['standalone_msP', 'standalone_msF']:
                logger.debug("Warning: The export format %s is not "+\
                  " available for individual ME evaluation of given coupl. orders."+\
                  " Only the total ME will be computed.", self.opt['export_format'])
            elif  self.opt['export_format'] in ['madloop_matchbox', 'matchbox']:
                replace_dict["color_information"] = self.get_color_string_lines(matrix_element)
                matrix_template = "matrix_standalone_matchbox_splitOrders_v4.inc"
            else:
                matrix_template = "matrix_standalone_splitOrders_v4.inc"
        process = matrix_element.get('processes')[0]
        sym_data = self._get_broken_symmetry_data(process, ninitial)
        self._fill_broken_sym_replace_dict(replace_dict, sym_data)
        if matrix_template == 'matrix_standalone_msP_v4.inc':
            bs_func_name = 'BROKEN_SYM_PROD'
            bs_nexternal = replace_dict['nexternal']
        elif matrix_template == 'matrix_standalone_msF_v4.inc':
            bs_func_name = 'BROKEN_SYM'
            bs_nexternal = 'include'
        else:
            bs_func_name = replace_dict['proc_prefix'] + 'BROKEN_SYM'
            bs_nexternal = 'include'
        replace_dict['broken_sym_function'] = \
            self._make_broken_sym_fortran_function(bs_func_name, sym_data, bs_nexternal)

        # GET_FLAVOR_INDEX (FLAVOR->idx) and GET_FLAVOR (idx->FLAVOR) helpers,
        # always emitted. Their names follow the same per-template convention as
        # BROKEN_SYM so msP/msF can be linked in the same MadSpin executable. The
        # templates call the matching names.
        if matrix_template == 'matrix_standalone_msP_v4.inc':
            fi_func_name = 'GET_FLAVOR_INDEX_PROD'
            fa_func_name = 'GET_FLAVOR_PROD'
        elif matrix_template == 'matrix_standalone_msF_v4.inc':
            fi_func_name = 'GET_FLAVOR_INDEX'
            fa_func_name = 'GET_FLAVOR'
        else:
            fi_func_name = replace_dict['proc_prefix'] + 'GET_FLAVOR_INDEX'
            fa_func_name = replace_dict['proc_prefix'] + 'GET_FLAVOR'
        fi_lookup_flat, fi_index_map = self._build_flav_index_lookup(
            matrix_element, n_table, flav_table_flat)
        replace_dict['flavor_index_function'] = \
            self._make_flavor_index_fortran_function(
                fi_func_name, n_table, flav_table_flat,
                nexternal_decl=bs_nexternal,
                lookup_flat=fi_lookup_flat, index_map=fi_index_map)
        replace_dict['flavor_array_function'] = \
            self._make_flavor_array_fortran_function(
                fa_func_name, n_table, flav_table_flat,
                nexternal_decl=bs_nexternal)

        # Per-crossing denominator and the routines decoding an extended
        # FLAV_IDX. Only matrix_standalone_v4.inc has these holes, and they are
        # left empty when the process was generated with --use_crossing=False.
        self.fill_crossing_replace_dict(matrix_element, replace_dict,
                                        use_crossing)

        # GET_PDG_FOR_FLAVOR (extended FLAV_IDX -> per-leg PDG). Must come after
        # fill_crossing_replace_dict, which decides whether it decodes a
        # crossing or just reads the table. Only matrix_standalone_v4.inc has
        # the hole; the key is set unconditionally since an unused replace_dict
        # entry is harmless and the other templates then stay byte-identical.
        n_pdg_flav, pdg_flat, antipdg_flat = \
            self._build_flav_pdg_tables(matrix_element)
        replace_dict['flavor_pdg_function'] = \
            self._make_flavor_pdg_fortran_function(
                replace_dict['proc_prefix'] + 'GET_PDG_FOR_FLAVOR',
                n_pdg_flav, pdg_flat, antipdg_flat,
                replace_dict['pdg_cross_snippets'],
                nexternal_decl=bs_nexternal)

        # f2py entry points taking an extended FLAV_IDX (the only way a python
        # caller can request a crossing, and reach GET_DENSITY_IDX /
        # GET_ALL_INTER_IDX / GET_NHEL_IDX / GET_PDG_FOR_FLAVOR). Those routines
        # only exist in matrix_standalone_v4.inc, so the wrappers are emitted
        # only there; the other standalone templates get an empty hole (the
        # placeholder lives in the shared matrix_standalone_f2py.inc). The
        # snippet is pre-formatted here because the outer '% replace_dict' pass
        # does not re-scan an inserted value for further %(...)s.
        if matrix_template == 'matrix_standalone_v4.inc':
            flav_idx_tmpl = open(pjoin(_file_path, 'iolibs', 'template_files',
                                       'matrix_standalone_f2py_flav_idx.inc')).read()
            nexternal_val = int(replace_dict['nexternal'])
            replace_dict['f2py_flav_idx_wrappers'] = flav_idx_tmpl % {
                'proc_prefix': replace_dict['proc_prefix'],
                'nexternal': nexternal_val,
                'nflav': replace_dict['nflav'],
                'ncomb': replace_dict['ncomb'],
                'ncross': (nexternal_val + 1) ** 2,
            }
        else:
            replace_dict['f2py_flav_idx_wrappers'] = ''

        replace_dict['template_file'] = pjoin(_file_path, 'iolibs', 'template_files', matrix_template)
        replace_dict['template_file2'] = pjoin(_file_path, \
                                   'iolibs/template_files/split_orders_helping_functions.inc')
        if write and writer:
            path = replace_dict['template_file']
            content = open(path).read()
            content = content % replace_dict
            # Write the file
            writer.writelines(content)
            # Add the helper functions.
            if len(split_orders)>0:
                content = '\n' + open(replace_dict['template_file2'])\
                                   .read()%replace_dict
                writer.writelines(content)
            if return_replace_dict:
                replace_dict['return_value'] = len([call for call in helas_calls if call.find('#') != 0])
                return replace_dict
            else:
                return len([call for call in helas_calls if call.find('#') != 0])
        else:
            replace_dict['return_value'] = len([call for call in helas_calls if call.find('#') != 0])
            return replace_dict # for subclass update

    #===========================================================================
    # write_check_sa   
    #===========================================================================
    def _recorded_crossing_matches(self, matrix_element):
        """(matches, complete): the reachable crossing each RECORDED crossed
        subprocess of this matrix element corresponds to.

        Crossing records (merge_crossing='record') say which crossed processes
        are real subprocesses of the generation; the runtime crossing space
        (GET_PDG_FOR_FLAVOR / its python twin compute_crossing_pdg_entries) is a
        dense enumeration of CROSS codes that also contains mathematically
        applicable but unrequested crossings -- e.g. a Z pulled into the initial
        state for p p > z j. Consumers that must not evaluate the latter (the
        check_sa demo, the reweight's folded-crossing lookup) intersect the two
        here.

        `matches` is a list of ``(pdg_signature, cross)`` in the recorded order,
        matched LABEL-AWARE: a recorded process may carry merged multiparticle
        labels (_quark = 81) and so may the reachable signature (a leg that does
        not vary with the flavor index keeps its label), so a label matches any
        member flavor of the same sign, and two labels match when equal. That is
        also why a recorded process is matched as a whole rather than leg by leg:
        the reachable set already encodes the correct flavor pairings, which
        resolving each merged leg on its own would not (it would fabricate e.g. a
        W coupling two same-flavor quarks). Both beam orientations are tried.
        `complete` is False when a recorded process has NO reachable
        instantiation, so a caller can fall back rather than hide a real
        crossing."""
        crossed = matrix_element.get('crossed_processes') \
            if 'crossed_processes' in matrix_element else None
        if not crossed:
            return [], True
        model = matrix_element.get('processes')[0].get('model')
        merged = model.get('merged_particles')

        def leg_matches(leg_id, pdg):
            # Does the reachable PDG instantiate this recorded leg id? Equal ids
            # (two concrete particles, or two identical merged labels) always
            # match; otherwise one of the two may be a merged label covering the
            # other flavor, with the same sign.
            if leg_id == pdg:
                return True
            a, b = abs(leg_id), abs(pdg)
            if (leg_id > 0) != (pdg > 0):
                return False
            return (a in merged and b in merged[a]) or \
                   (b in merged and a in merged[b])

        ninitial = matrix_element.get_nexternal_ninitial()[1]
        # signatures the runtime can actually reach (applicable crossings)
        reachable = [(tuple(pdg), cross) for (_i, cross, _f, pdg) in
                     self.compute_crossing_pdg_entries(matrix_element)]
        # A decay-chain base records its crossings at the PRODUCTION level, but
        # the reachable signatures span the decay leaves (the ME's NEXTERNAL), so
        # the recorded process must be expanded before it can match. The decays
        # never cross (they ride along on their production leg), so re-attaching
        # the base's decay chains and expanding gives the crossed leaf signature.
        base_decays = matrix_element.get('processes')[0].get('decay_chains')

        def crossed_leg_ids(proc):
            if not base_decays:
                return [l.get('id') for l in proc.get('legs')]
            expanded = copy.copy(proc)
            expanded.set('decay_chains', base_decays)
            expanded.set('legs_with_decays', base_objects.LegList())
            return [l.get('id') for l in expanded.get_legs_with_decays()]

        matches, complete = [], True
        for (proc, _bp, _xp) in crossed:
            legs = crossed_leg_ids(proc)
            orients = [legs]
            if ninitial == 2:                 # try the beam-swapped orientation
                orients.append([legs[1], legs[0]] + legs[2:])
            hit = None
            for orient in orients:
                for (r, cross) in reachable:
                    if len(r) == len(orient) and \
                       all(leg_matches(L, P) for L, P in zip(orient, r)):
                        hit = (r, cross)
                        break
                if hit is not None:
                    break
            if hit is None:
                complete = False
                continue
            if hit[1] == 0:
                # The identity: a recorded process that is the base's own beam
                # swap (mirror), not a crossing. Consumers show/reach the base
                # through its own PDG entry, so drop it.
                continue
            matches.append(hit)
        return matches, complete

    def recorded_crossing_codes(self, matrix_element):
        """(cross codes, complete) of the crossed subprocesses folded into this
        matrix element: the CROSS half of every extended FLAV_IDX that names a
        crossing this generation actually requested.

        This is what a python consumer needs to walk the folded crossings
        soundly: it can enumerate GET_PDG_FOR_FLAVOR over
        ``cross*NFLAV + flav`` (getting the exact per-flavor signature, which
        the flavor index and not the code determines) while skipping the codes
        that are merely applicable. See _recorded_crossing_matches."""
        matches, complete = self._recorded_crossing_matches(matrix_element)
        return sorted(set(cross for (_sig, cross) in matches)), complete

    def _crossed_signatures(self, matrix_element):
        """(signatures, complete) for the crossed subprocesses folded into this
        matrix element (merge_crossing='record'), so check_sa can demo exactly
        the crossings that are real subprocesses of the generation -- not every
        mathematically valid crossing of the base.

        Each signature is a representative signed-PDG tuple in the crossed leg
        order, matched at RUNTIME against GET_PDG_FOR_FLAVOR. Matching on the PDG
        rather than the extended index avoids the NFLAV-convention gap between
        the crossing-PDG enumeration and the runtime flavor table. Mirror pairs
        are collapsed (the chosen signature's beam swap is also marked seen).
        See _recorded_crossing_matches for the matching itself."""
        matches, complete = self._recorded_crossing_matches(matrix_element)
        ninitial = matrix_element.get_nexternal_ninitial()[1]
        sigs, seen = [], set()
        for (hit, _cross) in matches:
            mirror = (hit[1], hit[0]) + hit[2:] if ninitial == 2 else hit
            if hit in seen or mirror in seen:
                continue                      # mirror partner already taken
            sigs.append(hit)
            seen.add(hit)
            seen.add(mirror)
        return sigs, complete

    def _get_check_sa_crossing_example(self, matrix_element, proc_prefix):
        """Fortran block for check_sa.f demonstrating the crossed matrix elements.

        Returns '' when crossing is not active for this matrix element (flag
        off, or an s-channel constraint disables it), so the driver is
        unchanged. Otherwise it scans every crossing of the base -- FLIP1 and
        FLIP2 each range over 1..NEXTERNAL, choosing which two legs sit in the
        initial slots -- and, for each, evaluates the crossed matrix element and
        prints the momenta actually used next to their signed PDGs.

        Only the crossings that are REAL subprocesses of the generation (folded
        in via merge_crossing='record') are shown, not every mathematically
        valid crossing: their representative signed-PDG signatures are loaded
        into XCSIG (from _crossed_signatures) and each enumerated crossing is
        kept only if GET_PDG_FOR_FLAVOR matches an XCSIG row. When a folded
        crossing has no reachable signature (e.g. a flavor-changing W), the
        signatures are 'incomplete' and the block falls back to showing every
        applicable crossing (non-zero PDG, minus the FLIP1=1,FLIP2=2 identity).

        The crossing code is CROSS = FLIP1*(NEXTERNAL+1) + FLIP2, matching
        GET_CROSS_PERM's decode (i_part = CROSS/(NEXTERNAL+1),
        j_part = CROSS mod (NEXTERNAL+1)); FLAV_IDX = CROSS*NFLAV + flav, with
        NFLAV emitted as the literal matrix.f value so the encoding matches
        exactly. Degenerate crossings (e.g. FLIP1==FLIP2) decode to all-zero
        PDGs and are skipped by both the match and the fallback.
        """
        use_crossing = self.opt.get('use_crossing', True) and \
            not any(self.breaks_crossing_symmetry(proc)
                    for proc in matrix_element.get('processes'))
        if not use_crossing:
            return ''

        # NFLAV as matrix.f computes it, so CROSS*NFLAV+flav decodes correctly.
        # It is assigned to a local NFLAV here so the loop body reads generically
        # (FLAV_IDX = I*NFLAV+J) instead of a bare literal. The loop is gated
        # behind IF(.FALSE.) unless crossed subprocesses were folded into this ME
        # (merge_crossing='record'): then those partonic contributions have no
        # directory of their own and this driver is the only place they are
        # exercised, so the demo is enabled to actually evaluate them.
        n_table, _ = self._build_flav_table_flat(matrix_element)
        if not matrix_element.get('crossed_processes'):
            # Nothing folded in: keep the dormant example (present but disabled).
            loop_gate = '.false.'
        else:
            loop_gate = '.true.'
        sigs, complete = self._crossed_signatures(matrix_element)

        sep = ('            write (*,*) "-------------------------------------'
               '----------------------------------------"')

        # For the FLAV_IDX already set: print the crossed process -- its per-leg
        # PDG next to the momenta used to evaluate it. Every crossing shown here
        # keeps the massive particles final and only relabels the massless
        # partons, so its mass pattern is P's slot for slot; a standalone
        # (non-crossed) run of that subprocess would draw the very same RAMBO
        # point (identical hard-coded seed, sqrt(s) and per-slot masses). So the
        # base P IS that point, printed row k = P(:,k) with the crossed PDG
        # XPDG(k) -- copy/paste-comparable with the subprocess's own check.
        # XPDG is already set for this FLAV_IDX by the loop body above.
        demo_one = [
            '            CALL %sSMATRIX(P, FLAV_IDX, MATELEM)' % proc_prefix,
            "            write (*,*) 'FLAV_IDX', FLAV_IDX",
            "            write (*,*) '   PDG            E              px"
            "              py              pz'",
            '            DO XCK=1,NEXTERNAL',
            "              write (*,'(1X,I6,4(1X,E15.7))') XPDG(XCK),",
            '     &          P(0,XCK), P(1,XCK), P(2,XCK), P(3,XCK)',
            '            ENDDO',
            '            write (*,*) "Matrix element = ", MATELEM,'
            ' " GeV^",-(2*nexternal-8)',
            sep,
        ]

        lines = [
            '      if(%s) then' % loop_gate,
            '      write (*,*)',
            '      write (*,*) " Crossed processes (folded into this matrix'
            ' element):"',
            '      write (*,*)',
            '      NFLAV = %d' % n_table,
        ]
        if sigs and complete:
            # Load the signed-PDG signatures of the folded crossings, then show
            # only the crossings whose runtime PDG matches one of them (the real
            # subprocesses of this generation, not every valid crossing).
            lines.append('      XCNSIG = %d' % len(sigs))
            for s, sig in enumerate(sigs, 1):
                for k, pid in enumerate(sig, 1):
                    lines.append('      XCSIG(%d,%d) = %d' % (k, s, pid))
            match_cond = 'XCMATCH'
        else:
            # A folded crossing could not be matched to a runtime PDG (e.g. a
            # flavor-changing W subprocess): fall back to every crossing that is
            # applicable here (all-zero PDG = not applicable, skipped), so no real
            # subprocess is hidden.
            lines.append('      XCNSIG = 0')
            match_cond = 'XCVALID'
        lines += [
            'C         FLIP1/FLIP2 pick which legs sit in the two initial slots;',
            'C         1..NEXTERNAL spans every crossing (FLIP1=1,FLIP2=2 = base).',
            '      DO FLIP1=1,NEXTERNAL',
            '        DO FLIP2=1,NEXTERNAL',
            '          DO J=1,NFLAV',
            '            I = FLIP1*(NEXTERNAL+1) + FLIP2',
            '            FLAV_IDX = I*NFLAV+J',
            '            CALL %sGET_PDG_FOR_FLAVOR(FLAV_IDX, XPDG)' % proc_prefix,
        ]
        if sigs and complete:
            lines += [
                'C           Keep this crossing only if its PDG matches a folded',
                'C           subprocess signature.',
                '            XCMATCH = .FALSE.',
                '            DO XCS=1,XCNSIG',
                '              XCVALID = .TRUE.',
                '              DO XCK=1,NEXTERNAL',
                '                IF (XPDG(XCK).NE.XCSIG(XCK,XCS))'
                ' XCVALID = .FALSE.',
                '              ENDDO',
                '              IF (XCVALID) XCMATCH = .TRUE.',
                '            ENDDO',
            ]
        else:
            lines += [
                'C           Applicable here iff its PDG signature is not all-zero,',
                'C           skipping the identity (base process, shown above).',
                '            XCVALID = .FALSE.',
                '            DO XCK=1,NEXTERNAL',
                '              IF (XPDG(XCK).NE.0) XCVALID = .TRUE.',
                '            ENDDO',
                '            IF (FLIP1.EQ.1 .AND. FLIP2.EQ.2) XCVALID = .FALSE.',
            ]
        lines.append('            IF (.NOT.%s) CYCLE' % match_cond)
        lines.extend(demo_one)
        lines += ['          ENDDO', '        ENDDO', '      ENDDO', '      endif']
        return '\n'.join(lines)

    def write_check_sa(self, writer, matrix_element, proc_prefix=''):

        if self.format != 'standalone':
            return

        # Density-mode defaults (overridden if 'density' is in cmd_options).
        # The template uses both %(prefix)s and %(proc_prefix)s; supply both
        # with the same value so the merged flavor-grouping + density paths
        # share a single key set.
        replace_dict = {'prefix': proc_prefix,
                        'proc_prefix': proc_prefix,
                        'use_density': '.false.',
                        'dens_nchanging': 1,
                        'dens_ncomb': 2,
                    'dens_pos': 'if(nincoming.eq.2) then \n       POS(1) = 3 \n        else \n       POS(1) =1 \n        endif',
                        'dens_allow_hel': 'ALLOW_HEL(1) = +1 \n       ALLOW_HEL(2) = -1'}

        if 'density' in self.cmd_options:
            replace_dict['use_density'] = '.true.'
            changing = [int(i) for i in self.cmd_options['density'].split(',')]
            replace_dict['dens_nchanging'] = len(changing)
            replace_dict['dens_pos'] = '\n        '.join(
                   ['POS(%s) = %i' % (i+1, pos) for i,pos in enumerate(changing)])
            get_helicity_per_particle = matrix_element.get_helicity_per_particle()
            changing_hels = [get_helicity_per_particle[pos-1] for pos in changing]
            replace_dict['dens_ncomb'] = math.prod([len(hel) for hel in changing_hels])

            i = 0
            replace_dict['dens_allow_hel'] = ''
            for comb in  itertools.product(*changing_hels):
                for h in comb:
                    i += 1
                    replace_dict['dens_allow_hel'] += ' ALLOW_HEL(%i) = %i\n       ' % (i, h)

        # Flavor-grouping (HEAD path): compute MAXFLAVOR + FLAVOR/PDG_FOR_FLAVOR
        # initialiser code. Required by the merged check_sa.f template even when
        # the model has no merged particles (we emit a trivial single entry).
        all_flavors = matrix_element.get_external_flavors(all_perm=False)
        # Use legs_with_decays so that the PDG list covers all external particles
        # of the combined process (including decay products), matching the length
        # of each flavor tuple returned by get_external_flavors.
        all_pdgs = [l.get('id') for l in matrix_element.get('processes')[0].get('legs_with_decays')]
        map_all_flv = {}
        for flv1 in all_flavors:
            coup = matrix_element.get_coupling_for_flv(flv1, self.model)
            if coup in map_all_flv:
                map_all_flv[coup].append(flv1)
            else:
                map_all_flv[coup] = [flv1]

        pdg_to_flv_index = {}
        for _, opts in self.model.merged_particles.items():
            for j, pdg in enumerate(opts):
                pdg_to_flv_index[pdg] = j + 1

        all_flavors = [flv[0] for flv in map_all_flv.values()]
        maxflavor = max(1, len(all_flavors))
        flavor_text = ['        FLAVOR(:,:) =1']
        for i in range(1, len(all_flavors) + 1):
            for j in range(1, 1 + len(all_flavors[i-1])):
                if all_flavors[i-1][j-1] != 1:
                    pdg = all_flavors[i-1][j-1] * all_pdgs[j-1] // abs(all_pdgs[j-1])
                    flavor_text.append('FLAVOR(%d,%d) = %d ! PDG = %d' % (j, i, pdg_to_flv_index[all_flavors[i-1][j-1]], all_flavors[i-1][j-1]))
                    flavor_text.append('PDG_FOR_FLAVOR(%d,%d) = %d' % (j, i, pdg))
                elif abs(all_pdgs[j-1]) in self.model.get('merged_particles'):
                    pdg = all_flavors[i-1][j-1] * all_pdgs[j-1] // abs(all_pdgs[j-1])
                    flavor_text.append('PDG_FOR_FLAVOR(%d,%d) = %d' % (j, i, pdg))
                else:
                    flavor_text.append('PDG_FOR_FLAVOR(%d,%d) = %d' % (j, i, all_pdgs[j-1]))
        replace_dict['maxflavor'] = maxflavor
        replace_dict['flavor_def'] = '\n        '.join(flavor_text)

        # Crossing-symmetry demonstration: when crossing is active for this
        # matrix element, evaluate one genuinely crossed process (the first
        # valid non-identity crossing) at the same phase-space point and print
        # its per-leg PDG (via GET_PDG_FOR_FLAVOR) and matrix element, so that
        # `make check` visibly exercises the crossing machinery.
        replace_dict['crossing_example'] = \
            self._get_check_sa_crossing_example(matrix_element, proc_prefix)

        fsock =  open(pjoin(self.mgme_dir, 'madgraph', 'iolibs', 'template_files', 'check_sa.f'), 'r')
        text = fsock.read()
        fsock.close()
        text = text % replace_dict
        writer.write(text)
        writer.close()



    #===========================================================================
    # write_check_sa_splitOrders
    #===========================================================================
    def write_check_sa_splitOrders(self,squared_orders, split_orders, nexternal,
                                                nincoming, proc_prefix, writer):
        """ Write out a more advanced version of the check_sa drivers that
        individually returns the matrix element for each contributing squared
        order."""
        
        check_sa_content = open(pjoin(self.mgme_dir, 'madgraph', 'iolibs', \
                             'template_files', 'check_sa_splitOrders.f')).read()
        printout_sq_orders=[]
        for i, squared_order in enumerate(squared_orders):
            sq_orders=[]
            for j, sqo in enumerate(squared_order):
                sq_orders.append('%s=%d'%(split_orders[j],sqo))
            printout_sq_orders.append(\
                    "write(*,*) '%d) Matrix element for (%s) = ',MATELEMS(%d)"\
                                                 %(i+1,' '.join(sq_orders),i+1))
        printout_sq_orders='\n'.join(printout_sq_orders)
        replace_dict = {'printout_sqorders':printout_sq_orders, 
                        'nSplitOrders':len(squared_orders),
                        'nexternal':nexternal,
                        'nincoming':nincoming,
                        'proc_prefix':proc_prefix}
        
        if writer:
            writer.writelines(check_sa_content % replace_dict)
        else:
            return replace_dict

class ProcessExporterFortranMatchBox(ProcessExporterFortranSA):
    """class to take care of exporting a set of matrix element for the Matchbox
    code in the case of Born only routine"""

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'matchbox', 'mp': False,
                        'sa_symmetry': True}

    #specific template of the born
           

    matrix_template = "matrix_standalone_matchbox.inc"
    # matchbox needs the color flow information
    support_ddm_color_basis = False
    

    # Inherits from the standalone exporter but writes its own template, which
    # has no crossing machinery: the capability does not carry over.
    supports_crossing = False

    @staticmethod
    def get_color_string_lines(matrix_element):
        """Return the color matrix definition lines for this matrix element. Split
        rows in chunks of size n."""

        if not matrix_element.get('color_matrix'):
            return "\n".join(["out = 1"])
        
        #start the real work
        color_denominators = matrix_element.get('color_matrix').\
                                                         get_line_denominators()
        matrix_strings = []
        my_cs = color.ColorString()
        for i_color in range(len(color_denominators)):
            # Then write the numerators for the matrix elements
            my_cs.from_immutable(sorted(matrix_element.get('color_basis').keys())[i_color])
            t_str=repr(my_cs)
            t_match=re.compile(r"(\w+)\(([\s\d+\,]*)\)")
            # from '1 T(2,4,1) Tr(4,5,6) Epsilon(5,3,2,1) T(1,2)' returns with findall:
            # [('T', '2,4,1'), ('Tr', '4,5,6'), ('Epsilon', '5,3,2,1'), ('T', '1,2')]
            all_matches = t_match.findall(t_str)
            output = {}
            arg=[]
            for match in all_matches:
                ctype, tmparg = match[0], [m.strip() for m in match[1].split(',')]
                if ctype in ['ColorOne' ]:
                    continue
                if ctype not in ['T', 'Tr' ]:
                    raise MadGraph5Error('Color Structure not handled by Matchbox: %s'  % ctype)
                tmparg += ['0']
                arg +=tmparg
            for j, v in enumerate(arg):
                    output[(i_color,j)] = v

            for key in output:
                if matrix_strings == []:
                    #first entry
                    matrix_strings.append(""" 
                    if (in1.eq.%s.and.in2.eq.%s)then
                    out = %s
                    """  % (key[0], key[1], output[key]))
                else:
                    #not first entry
                    matrix_strings.append(""" 
                    elseif (in1.eq.%s.and.in2.eq.%s)then
                    out = %s
                    """  % (key[0], key[1], output[key]))
        if len(matrix_strings):                
            matrix_strings.append(" else \n out = - 1 \n endif")
        else: 
            return "\n out = - 1 \n "
        return "\n".join(matrix_strings)
    
    def make(self,*args,**opts):
        pass

    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        try:
            misc.compile(cwd=pjoin(self.dir_path,'Source','MODEL'))
        except OSError:
            pass
        return super().finalize(matrix_elements, history, mg5options, flaglist)
    

    def get_JAMP_lines(self, col_amps, JAMP_format="JAMP(%s)", AMP_format="AMP(%s)", split=-1,
                       JAMP_formatLC=None, orbit=False):

        """Adding leading color part of the colorflow. The leading color part
        needs the definitions written out, so the orbit recipes are not used
        here."""
        
        if not JAMP_formatLC:
            JAMP_formatLC= "LN%s" % JAMP_format

        error_msg="Malformed '%s' argument passed to the get_JAMP_lines"
        if(isinstance(col_amps,helas_objects.HelasMatrixElement)):
            col_amps=col_amps.get_color_amplitudes()
        elif(isinstance(col_amps,list)):
            if(col_amps and isinstance(col_amps[0],list)):
                col_amps=col_amps
            else:
                raise MadGraph5Error(error_msg % 'col_amps')
        else:
            raise MadGraph5Error(error_msg % 'col_amps')

        text, nb = super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(col_amps,
                                            JAMP_format=JAMP_format,
                                            AMP_format=AMP_format,
                                            split=-1)
        
        
        # Filter the col_ampls to generate only those without any 1/NC terms
        
        LC_col_amps = []
        for coeff_list in col_amps:
            to_add = []
            for (coefficient, amp_number) in coeff_list:
                if coefficient[3]==0:
                    to_add.append( (coefficient, amp_number) )
            LC_col_amps.append(to_add)
           
        text2, nb2 = super(ProcessExporterFortranMatchBox, self).get_JAMP_lines(LC_col_amps,
                                            JAMP_format=JAMP_formatLC,
                                            AMP_format=AMP_format,
                                            split=-1)
        text += text2 
        
        return text, max(nb,nb2)




#===============================================================================
# ProcessExporterFortranMW
#===============================================================================
class ProcessExporterFortranMW(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadGraph v4 - MadWeight format."""

    matrix_file="matrix_standalone_v4.inc"
    jamp_optim = False

    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranMW, self).copy_template(model)        

        # Add the MW specific file
        misc.copytree(pjoin(MG5DIR,'Template','MadWeight'),
                               pjoin(self.dir_path, 'Source','MadWeight'), True)        
        misc.copytree(pjoin(MG5DIR,'madgraph','madweight'),
                        pjoin(self.dir_path, 'bin','internal','madweight'), True) 
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','setrun.f'),
                                      pjoin(self.dir_path, 'Source','setrun.f'))
        files.mv(pjoin(self.dir_path, 'Source','MadWeight','src','run.inc'),
                                      pjoin(self.dir_path, 'Source','run.inc'))
        # File created from Template (Different in some child class)
        filename = os.path.join(self.dir_path,'Source','run_config.inc')
        self.write_run_config_file(writers.FortranWriter(filename))

        self.handle_cuts_inc()

        try:
            subprocess.call([os.path.join(self.dir_path, 'Source','MadWeight','bin','internal','pass_to_madweight')],
                            stdout = os.open(os.devnull, os.O_RDWR),
                            stderr = os.open(os.devnull, os.O_RDWR),
                            cwd=self.dir_path)
        except OSError:
            # Probably madweight already called
            pass
        
        ln(pjoin(self.dir_path, 'Source','PDF','eepdf.inc'),pjoin(self.dir_path, 'Source'))

        # Copy the different python file in the Template
        self.copy_python_file()
        # create the appropriate cuts.f
        self.get_mw_cuts_version()

        # add the makefile in Source directory 
        filename = os.path.join(self.dir_path,'Source','makefile')
        self.write_source_makefile(writers.FortranWriter(filename), self.model)



    def handle_cuts_inc(self):

        text = open(pjoin(self.dir_path, 'Source', 'cuts.inc'),'r').read()
        text = text.replace('maxjetflavor','dummy_maxjetflavor')

        fsock = open(pjoin(self.dir_path, 'Source', 'cuts.inc'),'w')
        fsock.write(text)

        fsock.write('''            
                logical fixed_extra_scale
                integer maxjetflavor
                double precision mue_over_ref
                double precision mue_ref_fixed
                common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
                ''')
        fsock.close()

        
    #===========================================================================
    # convert_model
    #===========================================================================    
    def convert_model(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
         
        super(ProcessExporterFortranMW,self).convert_model(model, 
                                               wanted_lorentz, wanted_couplings)
         
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
        # This is not safe if there is a '##' or '-' in the path.
        misc.copytree(model_path, 
                               pjoin(self.dir_path,'bin','internal','ufomodel'),
                               ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
        if hasattr(model, 'restrict_card'):
            out_path = pjoin(self.dir_path, 'bin', 'internal','ufomodel',
                                                         'restrict_default.dat')
            if isinstance(model.restrict_card, check_param_card.ParamCard):
                model.restrict_card.write(out_path)
            else:
                files.cp(model.restrict_card, out_path)

    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================        
    def copy_python_file(self):
        """copy the python file require for the Template"""

        # madevent interface
        cp(_file_path+'/interface/madweight_interface.py',
                            self.dir_path+'/bin/internal/madweight_interface.py')
        cp(_file_path+'/interface/extended_cmd.py',
                                  self.dir_path+'/bin/internal/extended_cmd.py')
        cp(_file_path+'/interface/common_run_interface.py',
                            self.dir_path+'/bin/internal/common_run_interface.py')
        cp(_file_path+'/various/misc.py', self.dir_path+'/bin/internal/misc.py')        
        cp(_file_path+'/iolibs/files.py', self.dir_path+'/bin/internal/files.py')
        cp(_file_path+'/iolibs/save_load_object.py', 
                              self.dir_path+'/bin/internal/save_load_object.py') 
        cp(_file_path+'/madevent/gen_crossxhtml.py', 
                              self.dir_path+'/bin/internal/gen_crossxhtml.py')
        cp(_file_path+'/madevent/sum_html.py', 
                              self.dir_path+'/bin/internal/sum_html.py')
        cp(_file_path+'/various/FO_analyse_card.py', 
                              self.dir_path+'/bin/internal/FO_analyse_card.py')                 
        cp(_file_path+'/iolibs/file_writers.py', 
                              self.dir_path+'/bin/internal/file_writers.py')
        #model file                        
        cp(_file_path+'../models/check_param_card.py', 
                              self.dir_path+'/bin/internal/check_param_card.py')   
                
        #madevent file
        cp(_file_path+'/__init__.py', self.dir_path+'/bin/internal/__init__.py')
        cp(_file_path+'/various/lhe_parser.py', 
                                self.dir_path+'/bin/internal/lhe_parser.py')         

        cp(_file_path+'/various/banner.py', 
                                   self.dir_path+'/bin/internal/banner.py')
        cp(_file_path+'/various/shower_card.py', 
                                   self.dir_path+'/bin/internal/shower_card.py')
        cp(_file_path+'/various/cluster.py',
                                       self.dir_path+'/bin/internal/cluster.py')
        # citation tracking (module + bibliography database)
        cp(_file_path+'/various/citation.py',
                                      self.dir_path+'/bin/internal/citation.py')
        cp(_file_path+'/various/citations.bib',
                                     self.dir_path+'/bin/internal/citations.bib')

        # logging configuration
        cp(_file_path+'/interface/.mg5_logging.conf',
                                 self.dir_path+'/bin/internal/me5_logging.conf')
        cp(_file_path+'/interface/coloring_logging.py',
                                 self.dir_path+'/bin/internal/coloring_logging.py')


    #===========================================================================
    # Change the version of cuts.f to the one compatible with MW
    #===========================================================================    
    def get_mw_cuts_version(self, outpath=None):
        """create the appropriate cuts.f
        This is based on the one associated to ME output but:
        1) No clustering (=> remove initcluster/setclscales)
        2) Adding the definition of cut_bw at the file.
        """
        
        template = open(pjoin(MG5DIR,'Template','LO','SubProcesses','cuts.f'))
        
        text = StringIO()
        #1) remove all dependencies in ickkw >1:
        nb_if = 0
        for line in template:
            if 'if(xqcut.gt.0d0' in line:
                nb_if = 1
            if nb_if == 0:
                text.write(line)
                continue
            if re.search(r'if\(.*\)\s*then', line):
                nb_if += 1
            elif 'endif' in line:
                nb_if -= 1
            
        #2) add fake cut_bw (have to put the true one later)
        text.write("""
      logical function cut_bw(p)
      include 'madweight_param.inc'
      double precision p(*)
      if (bw_cut) then
          cut_bw = .true.
      else
          stop 1
      endif
      return
      end
        """)
            
        final = text.getvalue()
        #3) remove the call to initcluster:
        template = final.replace('call initcluster', '! Remove for MW!call initcluster')
        template = template.replace('genps.inc', 'maxparticles.inc')
        #Now we can write it
        if not outpath:
            fsock =  open(pjoin(self.dir_path, 'SubProcesses', 'cuts.f'), 'w')
        elif isinstance(outpath, str):
            fsock = open(outpath, 'w')
        else:
            fsock = outpath
        fsock.write(template)
        
        
        
    #===========================================================================
    # Make the Helas and Model directories for Standalone directory
    #===========================================================================
    def make(self):
        """Run make in the DHELAS, MODEL, PDF and CERNLIB directories, to set up
        everything for running madweight
        """

        source_dir = os.path.join(self.dir_path, "Source")
        logger.info("Running make for Helas")
        misc.compile(arg=['../lib/libdhelas.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for Model")
        misc.compile(arg=['../lib/libmodel.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for PDF")
        misc.compile(arg=['../lib/libpdf.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for gammaUPC")
        misc.compile(arg=['../lib/libgammaUPC.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for CERNLIB")
        misc.compile(arg=['../lib/libcernlib.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for GENERIC")
        misc.compile(arg=['../lib/libgeneric.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for blocks")
        misc.compile(arg=['../lib/libblocks.a'], cwd=source_dir, mode='fortran')
        logger.info("Running make for tools")
        misc.compile(arg=['../lib/libtools.a'], cwd=source_dir, mode='fortran')

    #===========================================================================
    # Create proc_card_mg5.dat for MadWeight directory
    #===========================================================================
    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize Standalone MG4 directory by generation proc_card_mg5.dat"""
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}



        #proc_charac
        if hasattr(self, "nlo_mixed_expansion"):
            self.proc_characteristics['nlo_mixed_expansion'] = mg5options['nlo_mixed_expansion']
        self.create_proc_charac()

        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements)
        ln(pjoin(self.dir_path, 'Source', 'maxparticles.inc'),
           pjoin(self.dir_path, 'Source','MadWeight','blocks'))
        ln(pjoin(self.dir_path, 'Source', 'maxparticles.inc'),
           pjoin(self.dir_path, 'Source','MadWeight','tools'))
        
        self.set_compiler(compiler)
        self.make()
        
        # Write command history as proc_card_mg5
        if os.path.isdir(os.path.join(self.dir_path, 'Cards')):
            output_file = os.path.join(self.dir_path, 'Cards', 'proc_card_mg5.dat')
            history.write(output_file)

        ProcessExporterFortran.finalize(self, matrix_elements,
                                             history, mg5options, flaglist)



    #===========================================================================
    # create the run_card for MW
    #=========================================================================== 
    def create_run_card(self, matrix_elements, history):
        """ """
 
        run_card = banner_mod.RunCard()
    
        # pass to default for MW
        run_card["run_tag"] = "\'not_use\'"
        run_card["fixed_ren_scale"] = "T"
        run_card["fixed_fac_scale"] = "T"
        run_card.remove_all_cut()
                  
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card_default.dat'),
                       template=pjoin(MG5DIR, 'Template', 'MadWeight', 'Cards', 'run_card.dat'),
                       python_template=True)
        run_card.write(pjoin(self.dir_path, 'Cards', 'run_card.dat'),
                       template=pjoin(MG5DIR, 'Template', 'MadWeight', 'Cards', 'run_card.dat'),
                       python_template=True)

    #===========================================================================
    # export model files
    #=========================================================================== 
    def export_model_files(self, model_path):
        """export the model dependent files for V4 model"""
        
        super(ProcessExporterFortranMW,self).export_model_files(model_path)
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO  
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(os.path.join(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()

        # Modify setrun.f
        text = open(os.path.join(self.dir_path,'Source','setrun.f')).read()
        text = text.replace('call setpara(param_card_name)', 'call setpara(param_card_name, .true.)')
        fsock = open(os.path.join(self.dir_path,'Source','setrun.f'), 'w')
        fsock.write(text)
        fsock.close()

        # Modify initialization.f
        text = open(os.path.join(self.dir_path,'SubProcesses','initialization.f')).read()
        text = text.replace('call setpara(param_name)', 'call setpara(param_name, .true.)')
        fsock = open(os.path.join(self.dir_path,'SubProcesses','initialization.f'), 'w')
        fsock.write(text)
        fsock.close()
                
                
        self.make_model_symbolic_link()

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,number, **opt):
        """Generate the Pxxxxx directory for a subprocess in MG4 MadWeight format,
        including the necessary matrix.f and nexternal.inc files"""

        cwd = os.getcwd()
        # Create the directory PN_xx_xxxxx in the specified path
        dirpath = os.path.join(self.dir_path, 'SubProcesses', \
                       "P%s" % matrix_element.get('processes')[0].shell_string())

        try:
            os.mkdir(dirpath)
        except os.error as error:
            logger.warning(error.strerror + " " + dirpath)

        #try:
        #    os.chdir(dirpath)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % dirpath)
        #    return 0


        logger.info('Creating files in directory %s' % dirpath)

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Create the matrix.f file and the nexternal.inc file
        filename = pjoin(dirpath,'matrix.f')
        calls,ncolor = self.write_matrix_element_v4(
            writers.FortranWriter(filename),
            matrix_element,
            fortran_model)

        filename = pjoin(dirpath, 'auto_dsig.f')
        self.write_auto_dsig_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(dirpath, 'configs.inc')
        mapconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            matrix_element)

        filename = pjoin(dirpath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(dirpath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(dirpath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        filename = pjoin(dirpath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(dirpath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(matrix_element.get_all_amplitudes()))

        filename = pjoin(dirpath, 'maxamps.inc')
        self.write_maxamps_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           ncolor,
                           len(matrix_element.get('processes')),
                           1)

        filename = pjoin(dirpath, 'phasespace.inc')
        self.write_phasespace_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           )

        # Generate diagrams
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
            filename = pjoin(dirpath, "matrix.ps")
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude='')
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        #import genps.inc and maxconfigs.inc into Subprocesses
        ln(self.dir_path + '/Source/genps.inc', self.dir_path + '/SubProcesses', log=False)
        #ln(self.dir_path + '/Source/maxconfigs.inc', self.dir_path + '/SubProcesses', log=False)

        linkfiles = ['driver.f', 'cuts.f', 'initialization.f','gen_ps.f', 'makefile', 'coupl.inc','madweight_param.inc', 'run.inc', 'setscales.f', 'genps.inc']

        for file in linkfiles:
            ln('../%s' % file, starting_dir=cwd)
            
        ln('nexternal.inc', '../../Source', log=False, cwd=dirpath)
        ln('leshouche.inc', '../../Source', log=False, cwd=dirpath)
        ln('maxamps.inc', '../../Source', log=False, cwd=dirpath)
        ln('phasespace.inc', '../', log=True, cwd=dirpath)
        ln('../../Source/vector.inc', log=True, cwd=dirpath)
        # Return to original PWD
        #os.chdir(cwd)

        if not calls:
            calls = 0
        return calls

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,proc_id = "", config_map = []):
        """Export a matrix element to a matrix.f file in MG4 MadWeight format"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if writer:
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")

            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor
        replace_dict['proc_prefix'] = '' # Not used in MW

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines) % {'proc_prefix': replace_dict['proc_prefix']}

        mask_decl, mask_setup, n_flavors, active_flavor_mask = \
                self._get_flavor_mask_blocks(matrix_element)
        replace_dict['flavor_mask_decl'] = mask_decl
        replace_dict['flavor_mask_setup'] = mask_setup

        fortran_model.use_flavor_mask = (n_flavors > 0)
        fortran_model.me_n_flavors = n_flavors
        fortran_model.me_active_flavor_mask = active_flavor_mask
        try:
            helas_calls = fortran_model.get_matrix_element_calls(matrix_element)
        finally:
            fortran_model.use_flavor_mask = False
            fortran_model.me_n_flavors = 0
            fortran_model.me_active_flavor_mask = None

        replace_dict['helas_calls'] = "\n".join(helas_calls)

        # Extract JAMP lines
        jamp_lines, nb = self.get_JAMP_lines(matrix_element)
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)

        process = matrix_element.get('processes')[0]
        sym_data = self._get_broken_symmetry_data(process, ninitial)
        self._fill_broken_sym_replace_dict(replace_dict, sym_data)
        if 'group' in self.matrix_file:
            bs_func_name = 'BROKEN_SYM' + str(replace_dict['proc_id'])
        else:
            bs_func_name = replace_dict.get('proc_prefix', '') + 'BROKEN_SYM'
        replace_dict['broken_sym_function'] = \
            self._make_broken_sym_fortran_function(bs_func_name, sym_data)
        
        replace_dict['template_file'] =  os.path.join(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = ''
        
        if writer:
            file = open(replace_dict['template_file']).read()
            file = misc.apply_template(file, replace_dict)
            # Write the file
            writer.writelines(file)
            return len([call for call in helas_calls if call.find('#') != 0]),ncolor
        else:
            replace_dict['return_value'] = (len([call for call in helas_calls if call.find('#') != 0]),ncolor)
            
    #===========================================================================
    # write_source_makefile
    #===========================================================================
    def write_source_makefile(self, writer, model):
        """Write the nexternal.inc file for madweight"""


        path = os.path.join(_file_path,'iolibs','template_files','madweight_makefile_source')
        set_of_lib = '$(LIBRARIES) $(LIBDIR)libdhelas.$(libext) $(LIBDIR)libpdf.$(libext) $(LIBDIR)libgammaUPC.$(libext) $(LIBDIR)libmodel.$(libext) $(LIBDIR)libcernlib.$(libext) $(LIBDIR)libtf.$(libext)'
        text = open(path).read() % {'libraries': set_of_lib}
        writer.write(text)

        return True

    def write_phasespace_file(self, writer, nb_diag):
        """ """
        
        template = """      include 'maxparticles.inc' 
      integer max_branches
      parameter (max_branches=max_particles-1)
      integer max_configs
      parameter (max_configs=%(nb_diag)s)

c     channel position
      integer config_pos,perm_pos
      common /to_config/config_pos,perm_pos
        
        """

        writer.write(template % {'nb_diag': nb_diag})
        

    #===========================================================================
    # write_auto_dsig_file
    #===========================================================================
    def write_auto_dsig_file(self, writer, matrix_element, proc_id = ""):
        """Write the auto_dsig.f file for the differential cross section
        calculation, includes pdf call information (MadWeight format)"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()

        if ninitial not in [1,2]:
            raise writers.FortranWriter.FortranWriterError("""Need ninitial = 1 or 2 to write auto_dsig file""")

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id
        replace_dict['numproc'] = 1

        # Flavor lookup + SMATRIX call default to this subprocess's own matrix
        # element; a cross-group dependent (Track B) overrides them below to route
        # to a base group's symlinked crossing-aware SMATRIX.
        replace_dict['dsig_xg_decl'] = ''
        replace_dict['dsig_xg_decl_vec'] = ''
        replace_dict['dsig_xg_decl_multi'] = ''
        replace_dict['dsig_xg_helper'] = ''
        replace_dict['dsig_getflavor'] = \
            '      CALL GET_FLAVOR%s(IFLAV, FLAVOR)' % proc_id
        replace_dict['dsig_smatrix_call'] = (
            '     CALL SMATRIX%s(P1, IFLAV, RHEL, RCOL,channel,1, DSIGUU,'
            ' selected_hel(1), selected_col(1))' % proc_id)
        # ... and the same for the vectorised (SMATRIX_MULTI) path.
        replace_dict['dsig_getflavor_vec'] = \
            '       CALL GET_FLAVOR%s(IFLAV_VEC(IVEC), FLAVOR)' % proc_id
        replace_dict['dsig_smatrix_vec_name'] = 'SMATRIX%s' % proc_id
        replace_dict['dsig_smatrix_vec_flav'] = 'IFLAV_VEC(IVEC)'
        replace_dict['dsig_smatrix_vec_chan'] = 'channels(IVEC)'
        replace_dict['dsig_smatrix_vec_post'] = ''

        # Set dsig_line
        if ninitial == 1:
            # No conversion, since result of decay should be given in GeV
            dsig_line = "pd(0)*dsiguu"
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(0)*conv*dsiguu"

        replace_dict['dsig_line'] = dsig_line

        # Extract pdf lines
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines
        replace_dict['ee_comp_vars'] = eepdf_vars



        # Lines that differ between subprocess group and regular
        if proc_id:
            replace_dict['numproc'] = int(proc_id)
            replace_dict['passcuts_begin'] = "" 
            replace_dict['passcuts_end'] = "" 
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_subdiag_lines'] = \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
        else:
            replace_dict['passcuts_begin'] = "IF (PASSCUTS(PP)) THEN"
            replace_dict['passcuts_end'] = "ENDIF"
            replace_dict['define_subdiag_lines'] = "" 

        if writer:
            file = open(os.path.join(_file_path, \
                          'iolibs/template_files/auto_dsig_mw.inc')).read()
        
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        configs = [(i+1, d) for i,d in enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        model = matrix_element.get('processes')[0].get('model')
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,matrix_element, model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadWeight"""

        path = os.path.join(_file_path,'iolibs','template_files','madweight_run_config.inc')
        text = open(path).read() % {'chanperjob':'5'}
        writer.write(text)
        return True

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, matrix_element, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""

        lines = []

        particle_dict = matrix_element.get('processes')[0].get('model').\
                        get('particle_dict')

        s_and_t_channels = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
                       for config in configs if [d for d in config if d][0].\
                                                  get_vertex_leg_numbers()!=[]]
        
        minvert = min(vert_list) if vert_list!=[] else 0
        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any([vert > minvert for vert in
                    [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
                # Only 3-vertices allowed in configs.inc
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial,model,new_pdg))
                else:
                    stchannels.append((empty_verts, None))

            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]

            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = list(zip(*[s for s,t in stchannels]))
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            #lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("*     %d       %d " % (nconfigs,
                                                     mapconfigs[iconfig]))

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                line=str(last_leg.get('number'))+" "+str(daughters[0])+"  "+str(daughters[1])
#                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
#                             (last_leg.get('number'), nconfigs, len(daughters),
#                              ",".join([str(d) for d in daughters])))

                if last_leg.get('id') == 21 and 21 not in particle_dict:
                    # Fake propagator used in multiparticle vertices
                    mass = 'zero'
                    width = 'zero'
                    pow_part = 0
                else:
                    if (last_leg.get('id')!=self.model.get_first_non_pdg()):
                      particle = particle_dict[last_leg.get('id')]
                      # Get mass
                      mass = particle.get('mass')
                      # Get width
                      width = particle.get('width')
                    else : # fake propagator used in multiparticle vertices
                      mass= 'zero'
                      width= 'zero'

                line=line+"   "+mass+"  "+width+"   "

                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append(line+" S "+str(last_leg.get('id')))
#                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
#                                 (last_leg.get('number'), nconfigs, nsubprocs,
#                                  ",".join([str(d) for d in pdgs])))
#                    lines.append("data tprid(%d,%d)/0/" % \
#                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels:
                    lines.append(line+" T "+str(last_leg.get('id')))
#		    lines.append("data tprid(%d,%d)/%d/" % \
#                                 (last_leg.get('number'), nconfigs,
#                                  abs(last_leg.get('id'))))
#                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
#                                 (last_leg.get('number'), nconfigs, nsubprocs,
#                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
#        lines.append("# Number of configs")
#        lines.append("data mapconfig(0)/%d/" % nconfigs)
        lines.append(" *    ")  # a line with just a star indicates this is the end of file
        # Write the file
        writer.writelines(lines)

        return s_and_t_channels



#===============================================================================
# ProcessExporterFortranME
#===============================================================================
class ProcessExporterFortranME(ProcessExporterFortran):
    """Class to take care of exporting a set of matrix elements to
    MadEvent format."""

    matrix_file = "matrix_madevent_v4.inc"
    jamp_fold = True
    jamp_orbit = True
    # AMP is indexed by helicity once the matrix element is rewritten for
    # helicity recycling, so the definitions cannot sit at the end of it
    jamp_gather = True
    done_warning_tchannel = False
    # set as soon as one matrix element is written with the batched color
    # sum, so that only then is the library linked in
    blas_used = False

    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{},
                        'hel_recycling': False
                        }
    jamp_optim = True
    default_vector_size = 1
    

    def __new__(cls, *args, **opts):
        """wrapper needed for some plugin"""

        return super(ProcessExporterFortranME, cls).__new__(cls)


    def __init__(self,  dir_path = "", opt=None):
        
        super(ProcessExporterFortranME, self).__init__(dir_path, opt)
        
        # check and format the hel_recycling options as it should if provided 
        if opt and isinstance(opt['output_options'], dict) and \
                                       'hel_recycling' in opt['output_options']:
            self.opt['hel_recycling'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['hel_recycling'], bool, 'hel_recycling')

        if opt and isinstance(opt['output_options'], dict) and \
                                       't_strategy' in opt['output_options']:
            self.opt['t_strategy'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['t_strategy'], int, 't_strategy')

        if opt and isinstance(opt['output_options'], dict) and \
                                       'vector_size' in opt['output_options']:
            self.opt['vector_size'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['vector_size'], int, 'vector_size')
        else:
            self.opt['vector_size'] = 1

        if opt and isinstance(opt['output_options'], dict) and \
                                       'nb_warp' in opt['output_options']:
            self.opt['nb_warp'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['nb_warp'], int, 'nb_warp')
        else:
            self.opt['nb_warp'] = 1

        if opt and isinstance(opt['output_options'], dict) and \
                                       'amp_chunk_size' in opt['output_options']:
            self.opt['amp_chunk_size'] = banner_mod.ConfigFile.format_variable(
                  opt['output_options']['amp_chunk_size'], int, 'amp_chunk_size')
        else:
            self.opt['amp_chunk_size'] = AMP_CHUNK_SIZE_DEFAULT

    def write_amp_chunk_files(self, replace_dict, proc_id):
        """Move the HELAS call sequence of matrix<proc_id>_orig.f out of
        MATRIX<proc_id> and into matrix<proc_id>_origamp<k>.f, one subroutine
        per amp_chunk_size statements, and leave the calls to them behind.

        Returns the number of chunk files written (0 when the sequence is short
        enough to stay inline, which leaves replace_dict untouched and the
        generated file byte-identical to the unchunked output).
        """

        chunk_size = self.opt.get('amp_chunk_size', AMP_CHUNK_SIZE_DEFAULT)
        calls = replace_dict['helas_calls'].split('\n')
        if chunk_size <= 0 or len(calls) <= chunk_size:
            return 0

        chunks = chunk_fortran_statements(calls, chunk_size, fixed_form=False)
        if len(chunks) < 2:
            return 0

        self.set_amp_chunk_replace_keys(replace_dict)
        template = open(pjoin(_file_path,
            'iolibs/template_files/matrix_madevent_ampchunk_v4.inc')).read()
        args = ('P,NHEL,IC,IVEC,FLAVOR,W,AMP%s' %
                replace_dict['amp_chunk_mask_arg'])
        driver = ['C     The HELAS call sequence lives in matrix%s_origamp<k>.f, one'
                  % proc_id,
                  'C     subroutine per %d statements, so that the amplitudes can be'
                  % chunk_size,
                  'C     compiled apart from the JAMP and colour blocks below.']
        for i, chunk in enumerate(chunks):
            chunk_dict = dict(replace_dict)
            chunk_dict['chunk_id'] = str(i + 1)
            chunk_dict['helas_calls'] = '\n'.join(chunk)
            writer = writers.FortranWriter(
                'matrix%s_origamp%d.f' % (proc_id, i + 1))
            writer.writelines(misc.apply_template(template, chunk_dict))
            driver.append('CALL ORIGAMP%s_%d(%s)' % (proc_id, i + 1, args))
        replace_dict['helas_calls'] = '\n'.join(driver)
        return len(chunks)

    def write_amp_chunk_template(self, replace_dict, ime):
        """Write template_matrix<ime>_ampchunk.f, the per-chunk counterpart of
        template_matrix<ime>.f: hel_recycle renders it once per slice of the
        unrolled call sequence into matrix<ime>_optimamp<k>.f. Skipped when the
        chunk size is 0, in which case hel_recycle keeps the sequence inline."""

        if self.opt.get('amp_chunk_size', AMP_CHUNK_SIZE_DEFAULT) <= 0:
            return
        self.set_amp_chunk_replace_keys(replace_dict)
        tfile = open(pjoin(_file_path,
            'iolibs/template_files/matrix_madevent_ampchunk_v4_hel.inc')).read()
        writer = writers.FortranWriter('template_matrix%d_ampchunk.f' % ime)
        writer.uniformcase = False
        writer.writelines(misc.apply_template(tfile, replace_dict))

    def set_amp_chunk_replace_keys(self, replace_dict):
        """Fill the replace_dict holes that only the amplitude-chunk files use:
        the flavor-mask arrays they have to be handed. Their DATA tables stay
        in the matrix element, so the dummies are declared assumed-size.

        Nothing the matrix element itself writes is touched here -- the chunks
        recompute the fake widths from coupl.inc instead of reading them out of
        the matrix element's SAVEd locals -- so a process whose call sequence is
        short enough to stay inline comes out byte-identical."""

        if replace_dict.get('flavor_mask_decl'):
            replace_dict['amp_chunk_mask_arg'] = \
                ',CURRENT_WF_MASK,CURRENT_AMP_MASK'
            replace_dict['amp_chunk_mask_decl'] = (
                'C     Flavor masks of the calling matrix element; the DATA\n'
                'C     tables they are copied from stay there.\n'
                '      INTEGER*8 CURRENT_WF_MASK(*)\n'
                '      INTEGER*8 CURRENT_AMP_MASK(*)')
        else:
            replace_dict['amp_chunk_mask_arg'] = ''
            replace_dict['amp_chunk_mask_decl'] = ''

    # helper function for customise helas writter
    @staticmethod
    def custom_helas_call(call, arg):
        if arg['mass'] == '%(M)s,%(W)s,':
            arg['mass'] = '%(M)s, fk_%(W)s,'
        elif '%(W)s' in arg['mass']:
            raise Exception

        arg['coup'] = re.sub(r'coup(\d+)\)s',r'coup\g<1>)s%(vec\g<1>)s', arg['coup'])

        return call, arg
    
    def copy_template(self, model):
        """Additional actions needed for setup of Template
        """

        super(ProcessExporterFortranME, self).copy_template(model)
        
        # File created from Template (Different in some child class)
        filename = pjoin(self.dir_path,'Source','run_config.inc')
        self.write_run_config_file(writers.FortranWriter(filename))
        
        # The next file are model dependant (due to SLAH convention)
        self.model_name = model.get('name')
        # Add the symmetry.f 
        filename = pjoin(self.dir_path,'SubProcesses','symmetry.f')
        self.write_symmetry(writers.FortranWriter(filename))
        #
        filename = pjoin(self.dir_path,'SubProcesses','addmothers.f')
        self.write_addmothers(writers.FortranWriter(filename))
        # Copy the different python file in the Template
        self.copy_python_file()

        if model["running_elements"]:
            if not os.path.exists(pjoin(MG5DIR, 'Template',"Running")):
                raise Exception("Library for the running have not been installed. To install them please run \"install RunningCoupling\"")
            misc.copytree(pjoin(MG5DIR, 'Template',"Running"), 
                            pjoin(self.dir_path,'Source','RUNNING'))
        
        

    


    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================        
    def copy_python_file(self):
        """copy the python file require for the Template"""

        # madevent interface
        cp(_file_path+'/interface/madevent_interface.py',
                            self.dir_path+'/bin/internal/madevent_interface.py')
        cp(_file_path+'/interface/extended_cmd.py',
                                  self.dir_path+'/bin/internal/extended_cmd.py')
        cp(_file_path+'/interface/common_run_interface.py',
                            self.dir_path+'/bin/internal/common_run_interface.py')
        cp(_file_path+'/various/misc.py', self.dir_path+'/bin/internal/misc.py')        
        cp(_file_path+'/iolibs/files.py', self.dir_path+'/bin/internal/files.py')
        cp(_file_path+'/iolibs/save_load_object.py', 
                              self.dir_path+'/bin/internal/save_load_object.py') 
        cp(_file_path+'/iolibs/file_writers.py', 
                              self.dir_path+'/bin/internal/file_writers.py')
        #model file                        
        cp(_file_path+'../models/check_param_card.py', 
                              self.dir_path+'/bin/internal/check_param_card.py')   
        
        #copy all the file present in madevent directory
        for name in os.listdir(pjoin(_file_path, 'madevent')):
            if name not in ['__init__.py'] and name.endswith('.py'):
                cp(_file_path+'/madevent/'+name, self.dir_path+'/bin/internal/')
        
        #madevent file
        cp(_file_path+'/__init__.py', self.dir_path+'/bin/internal/__init__.py')
        cp(_file_path+'/various/lhe_parser.py', 
                                self.dir_path+'/bin/internal/lhe_parser.py')                        
        cp(_file_path+'/various/banner.py', 
                                   self.dir_path+'/bin/internal/banner.py')
        cp(_file_path+'/various/histograms.py', 
                                   self.dir_path+'/bin/internal/histograms.py')
        cp(_file_path+'/various/plot_djrs.py', 
                                   self.dir_path+'/bin/internal/plot_djrs.py')
        cp(_file_path+'/various/systematics.py', self.dir_path+'/bin/internal/systematics.py')        

        cp(_file_path+'/various/cluster.py',
                                       self.dir_path+'/bin/internal/cluster.py')
        # citation tracking (module + bibliography database)
        cp(_file_path+'/various/citation.py',
                                      self.dir_path+'/bin/internal/citation.py')
        cp(_file_path+'/various/citations.bib',
                                     self.dir_path+'/bin/internal/citations.bib')
        cp(_file_path+'/madevent/combine_runs.py',
                                       self.dir_path+'/bin/internal/combine_runs.py')
        # logging configuration
        cp(_file_path+'/interface/.mg5_logging.conf', 
                                 self.dir_path+'/bin/internal/me5_logging.conf') 
        cp(_file_path+'/interface/coloring_logging.py', 
                                 self.dir_path+'/bin/internal/coloring_logging.py')
        # shower card and FO_analyse_card. 
        #  Although not needed, it is imported by banner.py
        cp(_file_path+'/various/shower_card.py', 
                                 self.dir_path+'/bin/internal/shower_card.py') 
        cp(_file_path+'/various/FO_analyse_card.py', 
                                 self.dir_path+'/bin/internal/FO_analyse_card.py') 
 
 
    def convert_model(self, model, wanted_lorentz = [], 
                                                         wanted_couplings = []):
         
        super(ProcessExporterFortranME,self).convert_model(model, 
                                               wanted_lorentz, wanted_couplings)
         
        IGNORE_PATTERNS = ('*.pyc','*.dat','*.py~')
        try:
            shutil.rmtree(pjoin(self.dir_path,'bin','internal','ufomodel'))
        except OSError as error:
            pass
        model_path = model.get('modelpath')
        # This is not safe if there is a '##' or '-' in the path.
        misc.copytree(model_path, 
                               pjoin(self.dir_path,'bin','internal','ufomodel'),
                               ignore=shutil.ignore_patterns(*IGNORE_PATTERNS))
        if hasattr(model, 'restrict_card'):
            out_path = pjoin(self.dir_path, 'bin', 'internal','ufomodel',
                                                         'restrict_default.dat')
            if isinstance(model.restrict_card, check_param_card.ParamCard):
                model.restrict_card.write(out_path)
            else:
                files.cp(model.restrict_card, out_path)
                
    #===========================================================================
    # export model files
    #=========================================================================== 
    def export_model_files(self, model_path):
        """export the model dependent files"""

        super(ProcessExporterFortranME,self).export_model_files(model_path)
        
        # Add the routine update_as_param in v4 model 
        # This is a function created in the UFO 
        text="""
        subroutine update_as_param()
          call setpara('param_card.dat',.false.)
          return
        end
        """
        ff = open(pjoin(self.dir_path, 'Source', 'MODEL', 'couplings.f'),'a')
        ff.write(text)
        ff.close()
                
        # Add the symmetry.f 
        filename = pjoin(self.dir_path,'SubProcesses','symmetry.f')
        self.write_symmetry(writers.FortranWriter(filename), v5=False)
        
        # Modify setrun.f
        text = open(pjoin(self.dir_path,'Source','setrun.f')).read()
        text = text.replace('call setpara(param_card_name)', 'call setpara(param_card_name, .true.)')
        fsock = open(pjoin(self.dir_path,'Source','setrun.f'), 'w')
        fsock.write(text)
        fsock.close()
        
        self.make_model_symbolic_link()

    #===========================================================================
    # generate_subprocess_directory 
    #===========================================================================
    def generate_subprocess_directory(self, matrix_element,
                                         fortran_model,
                                         me_number, **opt):
        """Generate the Pxxxxx directory for a subprocess in MG4 madevent,
        including the necessary matrix.f and various helper files"""

        cwd = os.getcwd()
        path = pjoin(self.dir_path, 'SubProcesses')


        if not self.model:
            self.model = matrix_element.get('processes')[0].get('model')

        #os.chdir(path)
        # Create the directory PN_xx_xxxxx in the specified path
        subprocdir = "P%s" % matrix_element.get('processes')[0].shell_string()
        try:
            os.mkdir(pjoin(path,subprocdir))
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)

        #try:
        #    os.chdir(subprocdir)
        #except os.error:
        #    logger.error('Could not cd to directory %s' % subprocdir)
        #    return 0

        logger.info('Creating files in directory %s' % subprocdir)
        Ppath = pjoin(path, subprocdir)
        
        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Add the driver.f 
        ncomb = matrix_element.get_helicity_combinations()
        filename = pjoin(Ppath,'driver.f')
        self.write_driver(writers.FortranWriter(filename),ncomb,n_grouped_proc=1,
                          v5=self.opt['v5_model'])


        # Create the matrix.f file, auto_dsig.f file and all inc files
        if 'hel_recycling' in self.opt and self.opt['hel_recycling']:
            filename = pjoin(Ppath, 'matrix_orig.f')
        else:
            filename = pjoin(Ppath, 'matrix.f')
        calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename),
                      matrix_element, fortran_model, subproc_number = me_number)

        filename = pjoin(Ppath, 'auto_dsig.f')
        self.write_auto_dsig_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(Ppath, 'configs.inc')
        mapconfigs, (s_and_t_channels, nqcd_list) = self.write_configs_file(\
            writers.FortranWriter(filename),
            matrix_element)

        filename = pjoin(Ppath, 'config_nqcd.inc')
        self.write_config_nqcd_file(writers.FortranWriter(filename),
                               nqcd_list)

        filename = pjoin(Ppath, 'config_subproc_map.inc')
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           s_and_t_channels)

        filename = pjoin(Ppath, 'coloramps.inc')
        self.write_coloramps_file(writers.FortranWriter(filename),
                             mapconfigs,
                             matrix_element)

        filename = pjoin(Ppath, 'decayBW.inc')
        self.write_decayBW_file(writers.FortranWriter(filename),
                           s_and_t_channels)

        filename = pjoin(Ppath, 'dname.mg')
        self.write_dname_file(writers.FileWriter(filename),
                         "P"+matrix_element.get('processes')[0].shell_string())

        filename = pjoin(Ppath, 'iproc.dat')
        self.write_iproc_file(writers.FortranWriter(filename),
                         me_number)

        filename = pjoin(Ppath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(Ppath, 'colorflow.inc')
        self.write_colorflow_file(writers.FortranWriter(filename),
                             matrix_element)

        filename = pjoin(Ppath, 'maxamps.inc')
        nb_flavor_per_proc = matrix_element.get_nb_flavors()
        # Compute actual MAXPROC: for merged processes each flavor combination
        # generates a separate IDUP row, so MAXPROC must cover all of them.
        nb_idup_rows = 0
        for proc in matrix_element.get('processes'):
            legs = proc.get_legs_with_decays()
            ids = [l.get('id') for l in legs]
            if self.model and 'merged_particles' in self.model and \
                    any(abs(id) in self.model['merged_particles'] for id in ids):
                nb_idup_rows += len(list(sum(
                    matrix_element.get_external_flavors_with_iden(), [])))
            else:
                nb_idup_rows += 1
        self.write_maxamps_file(writers.FortranWriter(filename),
                           len(matrix_element.get('diagrams')),
                           ncolor,
                           nb_flavor_per_proc,
                           max(1, nb_idup_rows),
                           1)

        filename = pjoin(Ppath, 'mg.sym')
        self.write_mg_sym_file(writers.FortranWriter(filename),
                          matrix_element)

        filename = pjoin(Ppath, 'ncombs.inc')
        self.write_ncombs_file(writers.FortranWriter(filename),
                          nexternal)

        filename = pjoin(Ppath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(Ppath, 'ngraphs.inc')
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           len(mapconfigs))


        filename = pjoin(Ppath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(Ppath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        # Find config symmetries and permutations
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(matrix_element)

        filename = pjoin(Ppath, 'symswap.inc')
        self.write_symswap_file(writers.FortranWriter(filename),
                                ident_perms)

        filename = pjoin(Ppath, 'symfact_orig.dat')
        self.write_symfact_file(open(filename, 'w'), symmetry)

        # Generate diagrams
        if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
            filename = pjoin(Ppath, "matrix.ps")
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                 get('diagrams'),
                                              filename,
                                              model=matrix_element.get('processes')[0].\
                                                 get('model'),
                                              amplitude=True)
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        self.link_files_in_SubProcess(Ppath)

        #import nexternal/leshouche in Source
        ln(pjoin(Ppath,'nexternal.inc'), pjoin(self.dir_path,'Source'), log=False)
        ln(pjoin(Ppath,'leshouche.inc'),  pjoin(self.dir_path,'Source'), log=False)
        ln(pjoin(Ppath,'maxamps.inc'),  pjoin(self.dir_path,'Source'), log=False)
        # Return to SubProcesses dir
        #os.chdir(os.path.pardir)

        # Add subprocess to subproc.mg
        filename = pjoin(path, 'subproc.mg')
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)

        # Return to original dir
        #os.chdir(cwd)

        # Generate info page
        gen_infohtml.make_info_html(self.dir_path)


        if not calls:
            calls = 0
        return calls

    link_Sub_files = ['addmothers.f',
                     'cluster.f',
                     'cluster.inc',
                     'coupl.inc',
                     'cuts.f',
                     'cuts.inc',
                     'genps.f',
                     'genps.inc',
                     'idenparts.f',
                     'initcluster.f',
                     'makefile',
                     'message.inc',
                     'myamp.f',
                     'reweight.f',
                     'run.inc',
                     'maxconfigs.inc',
                     'maxparticles.inc',
                     'run_config.inc',
                     'lhe_event_infos.inc',
                     'setcuts.f',
                     'setscales.f',
                     'sudakov.inc',
                     'symmetry.f',
                     'unwgt.f',
                     'dummy_fct.f'
                     ]

    def link_files_in_SubProcess(self, Ppath):
        """ Create the necessary links in the P* directory path Ppath"""
        
        #import genps.inc and maxconfigs.inc into Subprocesses
        ln(self.dir_path + '/Source/genps.inc', 
                                     self.dir_path + '/SubProcesses', log=False)
        ln(self.dir_path + '/Source/maxconfigs.inc',
                                     self.dir_path + '/SubProcesses', log=False)

        linkfiles = self.link_Sub_files

        for file in linkfiles:
            ln('../' + file , cwd=Ppath)    


    def finalize(self, matrix_elements, history, mg5options, flaglist, second_exporter=None):
        """Finalize ME v4 directory by creating jpeg diagrams, html
        pages,proc_card_mg5.dat and madevent.tar.gz."""
        

        if second_exporter:
            self.has_second_exporter = second_exporter

        if 'nojpeg' in flaglist:
            makejpg = False
        else:
            makejpg = True
        if 'online' in flaglist:
            online = True
        else:
            online = False
            
        compiler =  {'fortran': mg5options['fortran_compiler'],
                     'cpp': mg5options['cpp_compiler'],
                     'f2py': mg5options['f2py_compiler']}

        # a matrix element written with the batched color sum needs the
        # library it calls into on the link line
        if self.blas_used:
            makefile = pjoin(self.dir_path, 'SubProcesses', 'makefile')
            if os.path.exists(makefile):
                text = open(makefile).read()
                text = text.replace('BLASLIBS =',
                                    'BLASLIBS = %s' % self.blas_link_flags())
                open(makefile, 'w').write(text)

        # indicate that the output type is not grouped
        if  not isinstance(self, ProcessExporterFortranMEGroup):
            self.proc_characteristic['grouped_matrix'] = False
        self.proc_characteristic['nlo_mixed_expansion'] = mg5options['nlo_mixed_expansion']
        
        self.proc_characteristic['complex_mass_scheme'] = mg5options['complex_mass_scheme']
        self.proc_characteristic['gauge'] = mg5options['gauge']

        # set limitation linked to the model
    
        
        # indicate the PDG of all initial particle
        try:
            pdgs1 = [p.get_initial_pdg(1) for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(1)]
            pdgs2 = [p.get_initial_pdg(2) for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(2)]
        except AttributeError:
            pdgs1 = [p.get_initial_pdg(1) for m in matrix_elements.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(1)]
            pdgs2 = [p.get_initial_pdg(2) for m in matrix_elements.get('matrix_elements') for p in m.get('processes') if p.get_initial_pdg(2)]
        self.proc_characteristic['pdg_initial1'] = pdgs1
        self.proc_characteristic['pdg_initial2'] = pdgs2
        
        
        modelname = self.opt['model']
        if modelname == 'mssm' or modelname.startswith('mssm-'):
            param_card = pjoin(self.dir_path, 'Cards','param_card.dat')
            mg5_param = pjoin(self.dir_path, 'Source', 'MODEL', 'MG5_param.dat')
            check_param_card.convert_to_mg5card(param_card, mg5_param)
            check_param_card.check_valid_param_card(mg5_param)

        # Add the combine_events.f modify param_card path/number of @X
        filename = pjoin(self.dir_path,'Source','combine_events.f')
        try:
            nb_proc =[p.get('id') for me in matrix_elements for m in me.get('matrix_elements') for p in m.get('processes')]
        except AttributeError:
            nb_proc =[p.get('id') for m in matrix_elements.get('matrix_elements') for p in m.get('processes')]
        nb_proc = len(set(nb_proc))
        self.write_combine_events(writers.FortranWriter(filename), nb_proc) # already formatted
        # Write maxconfigs.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxconfigs.inc')
        self.write_maxconfigs_file(writers.FortranWriter(filename),
                                   matrix_elements)
        
        # Write maxparticles.inc based on max of ME's/subprocess groups
        filename = pjoin(self.dir_path,'Source','maxparticles.inc')
        self.write_maxparticles_file(writers.FortranWriter(filename),
                                     matrix_elements)
        
        # Touch "done" file
        os.system('touch %s/done' % pjoin(self.dir_path,'SubProcesses'))

        # Check for compiler
        self.set_compiler(compiler)
        self.set_cpp_compiler(compiler['cpp'])
        

        old_pos = os.getcwd()
        subpath = pjoin(self.dir_path, 'SubProcesses')

        P_dir_list = [proc for proc in os.listdir(subpath) 
                      if os.path.isdir(pjoin(subpath,proc)) and proc[0] == 'P']

        devnull = os.open(os.devnull, os.O_RDWR)
        # Convert the poscript in jpg files (if authorize)
        if makejpg:
            try:
                os.remove(pjoin(self.dir_path,'HTML','card.jpg'))
            except Exception as error:
                pass
            
            if misc.which('gs'):
                logger.info("Generate jpeg diagrams")
                for Pdir in P_dir_list:
                    misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_jpeg-pl')],
                                stdout = devnull, cwd=pjoin(subpath, Pdir))

        logger.info("Generate web pages")
        # Create the WebPage using perl script

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')], \
                                      stdout = devnull,cwd=pjoin(self.dir_path))

        #os.chdir(os.path.pardir)

        obj = gen_infohtml.make_info_html(self.dir_path)
              
        if online:
            nb_channel = obj.rep_rule['nb_gen_diag']
            open(pjoin(self.dir_path, 'Online'),'w').write(str(nb_channel))
        #add the information to proc_charac
        self.proc_characteristic['nb_channel'] = obj.rep_rule['nb_gen_diag']
        
        # Write command history as proc_card_mg5
        if os.path.isdir(pjoin(self.dir_path,'Cards')):
            output_file = pjoin(self.dir_path,'Cards', 'proc_card_mg5.dat')
            history.write(output_file)

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull)

        #crate the proc_characteristic file 
        self.create_proc_charac(matrix_elements, history)

        # create the run_card
        ProcessExporterFortran.finalize(self, matrix_elements, history, mg5options, flaglist)

        # Run "make" to generate madevent.tar.gz file
        if os.path.exists(pjoin(self.dir_path,'SubProcesses', 'subproc.mg')):
            if os.path.exists(pjoin(self.dir_path,'madevent.tar.gz')):
                os.remove(pjoin(self.dir_path,'madevent.tar.gz'))
            misc.call([os.path.join(self.dir_path, 'bin', 'internal', 'make_madevent_tar')],
                        stdout = devnull, cwd=self.dir_path)

        misc.call([pjoin(self.dir_path, 'bin', 'internal', 'gen_cardhtml-pl')],
                        stdout = devnull, cwd=self.dir_path)


        if second_exporter:
            second_exporter.finalize(matrix_elements, history, mg5options, flaglist)



        #return to the initial dir
        #os.chdir(old_pos)               

    #===========================================================================
    # BLAS-3 color sum
    #===========================================================================
    @staticmethod
    def get_blas_routine_me(prefix, proc_id, nfold, nampso, nsqampso,
                            ncomb, cf_dim, chosen_so):
        """The color sum for a whole batch of helicities at once.

        The color matrix is the same for every helicity, so the helicities
        are the columns of a single right hand side and the sum is two
        DSYMM calls. DSYMM is real, so the two parts of JAMP go through
        separately; the color matrix is real and symmetric, so that is all
        it takes, and the two products add up term by term.

        With split orders JAMP carries a second index, and every (M,N) pair
        the squared order mask keeps is one more column pairing. The mask is
        symmetric, since SQSOINDEX adds the two amplitude orders, and that is
        what lets the triangle the scalar sum walks be traded for the whole
        symmetric matrix here."""

        return """
      SUBROUTINE {p}GET_MATRIX_BATCH{i}(JR,JI,NB,ANSB)
      IMPLICIT NONE
      INTEGER NFOLD, NAMPSO, NSQAMPSO, NBMAX
      PARAMETER (NFOLD={n}, NAMPSO={a})
      PARAMETER (NSQAMPSO={q}, NBMAX={c})
      INTEGER NB
      DOUBLE PRECISION JR(NFOLD,NAMPSO,*), JI(NFOLD,NAMPSO,*)
      DOUBLE PRECISION ANSB(*)
      LOGICAL CHOSEN_SO_CONFIGS(NSQAMPSO)
      DATA CHOSEN_SO_CONFIGS/{s}/
      SAVE CHOSEN_SO_CONFIGS
      INTEGER I,J,K,M,N,CFI,NRHS
      DOUBLE PRECISION S
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: CFULL(:,:)
      DOUBLE PRECISION, ALLOCATABLE, SAVE :: TR(:,:), TI(:,:)
      LOGICAL FIRST
      DATA FIRST /.TRUE./
      SAVE FIRST
      INTEGER CF({d})
      INTEGER DENOM
      COMMON /{p}color_matrix{i}/ CF,DENOM
      INTEGER SQSOINDEX{i}
      IF (FIRST) THEN
        CALL {p}INIT_CF{i}()
        ALLOCATE(CFULL(NFOLD,NFOLD))
        ALLOCATE(TR(NFOLD,NAMPSO*NBMAX))
        ALLOCATE(TI(NFOLD,NAMPSO*NBMAX))
C       The triangle written out has its off diagonal doubled, since
C       the scalar sum walks it once. BLAS wants the whole matrix,
C       with every entry counted once.
        CFI = 0
        DO I = 1, NFOLD
          DO J = I, NFOLD
            CFI = CFI + 1
            IF (I.EQ.J) THEN
              CFULL(I,J) = DBLE(CF(CFI))
            ELSE
              CFULL(I,J) = DBLE(CF(CFI))/2D0
              CFULL(J,I) = CFULL(I,J)
            ENDIF
          ENDDO
        ENDDO
        FIRST = .FALSE.
      ENDIF
      NRHS = NB*NAMPSO
      CALL DSYMM('L','U',NFOLD,NRHS,1D0,CFULL,NFOLD,JR,NFOLD,0D0,TR,NFOLD)
      CALL DSYMM('L','U',NFOLD,NRHS,1D0,CFULL,NFOLD,JI,NFOLD,0D0,TI,NFOLD)
      DO K = 1, NB
        S = 0D0
        DO M = 1, NAMPSO
          DO N = 1, NAMPSO
            IF (CHOSEN_SO_CONFIGS(SQSOINDEX{i}(M,N))) THEN
              DO I = 1, NFOLD
                S = S + TR(I,(K-1)*NAMPSO+M)*JR(I,N,K)
                S = S + TI(I,(K-1)*NAMPSO+M)*JI(I,N,K)
              ENDDO
            ENDIF
          ENDDO
        ENDDO
        ANSB(K) = S / DBLE(DENOM)
      ENDDO
      END
""".format(p=prefix, i=proc_id, n=nfold, a=nampso, q=nsqampso, c=ncomb,
           d=cf_dim, s=chosen_so)

    # For every template where MATRIX is one helicity at a time: the call
    # SMATRIX makes in its helicity loop, what selects the helicities worth
    # computing, when the good helicities have settled, the arguments MATRIX
    # takes on top of its own, and the dimension the file declares the color
    # matrix with (the common block is laid out by it, so DENOM only lands
    # where the batched routine looks for it if the two agree).
    blas_me_shape = {
        'matrix_madevent_v4.inc': {
            'call': 'MATRIX%(proc_id)s(P,NHEL(1,I),IFLAV, IVEC)',
            'collect': 'MATRIX%(proc_id)s(P,NHEL(1,IBH),IFLAV,IVEC,'
                       'JRB,JIB,BLASGATE,BLASNB)',
            'select': 'GOODHEL(IBH,IFLAV) .OR. NTRY(IFLAV) .LE. MAXTRIES'
                      '.OR.(ISUM_HEL.NE.0)',
            'settled': 'NTRY(IFLAV).GT.MAXTRIES',
            'cf_dim': 'NFOLD*(NFOLD+1)'},
        # The grouped template is the one crossing symmetry rewrites, so its
        # call/gate are the crossing holes rather than literals; _at_ibh below
        # re-points them at the sweep's own loop variable.
        'matrix_madevent_group_v4.inc': {
            'call': 'MATRIX%(proc_id)s(%(me_matrix_args)s)',
            'collect': 'MATRIX%(proc_id)s(%(me_matrix_args_ibh)s,JRB,JIB,'
                       'BLASGATE,BLASNB)',
            'select': 'GOODHEL(%(me_goodhel_idx_ibh)s,%(me_flav_key)s,'
                      '%(proc_id)s) .OR. '
                      'NTRY(%(me_flav_key)s,%(proc_id)s).LE.MAXTRIES.or.'
                      '(ISUM_HEL.NE.0)%(me_goodhel_or_ibh)s',
            'settled': 'NTRY(%(me_flav_key)s,%(proc_id)s).GT.MAXTRIES',
            'cf_dim': 'NFOLD*(NFOLD+1)/2'},
        }

    # The helicity index inside the crossing holes is always the bare loop
    # variable I; the BLAS pre-sweep runs its own loop over IBH, so the same
    # holes have to be re-pointed at it. Whole-word only, which is exactly
    # right for the strings fill_crossing_replace_dict_me writes: IC, IVEC,
    # IFLAV, IHEL and AMP2/JAMP2 are all left alone.
    _ibh_index = re.compile(r'\bI\b')

    @classmethod
    def _at_ibh(cls, text):
        """Same expression, evaluated at the sweep's IBH instead of at I."""
        return cls._ibh_index.sub('IBH', text)

    def set_blas_replace_dict(self, replace_dict, ncomb, nfold):
        """Template replacements for the BLAS-3 color sum.

        Everything the batched path adds hangs off the end of a line that is
        already there, so with BLAS off the generated file is character for
        character the one written before any of this existed.

        Two shapes are covered. The helicity recycled matrix element already
        walks every helicity inside one call, so there the batch is the loop
        it is already running (the blas_hel_* keys). Everywhere else MATRIX is
        one helicity at a time and SMATRIX is the one holding the loop, so the
        columns are gathered there and the value each helicity ends up with is
        read back out of the batch (the blas_* keys)."""

        keys = ['blas_hel_decl', 'blas_hel_setup', 'blas_hel_gather',
                'blas_hel_gate', 'blas_hel_finish', 'blas_hel_routine',
                'blas_decl', 'blas_arg', 'blas_gather', 'blas_gate',
                'blas_smatrix_decl', 'blas_branch', 'blas_matrix_args',
                'blas_routine']
        shape = self.blas_me_shape.get(self.matrix_file)
        for key in keys:
            replace_dict[key] = ''
        # IBH-indexed twins of the crossing holes, for the pre-sweep loop
        for key in ('me_matrix_args', 'me_goodhel_idx',
                    'smatrix_me_goodhel_or'):
            replace_dict[key.replace('smatrix_me_', 'me_') + '_ibh'] = \
                self._at_ibh(replace_dict.get(key, ''))
        replace_dict['blas_matrix_call'] = \
            (shape['call'] % replace_dict) if shape else ''

        nampso = replace_dict['nAmpSplitOrders']
        if not self.blas_wanted(nfold):
            return
        self.blas_used = True

        prefix = replace_dict['proc_prefix']
        proc_id = replace_dict['proc_id']
        replace_dict['blas_hel_decl'] = "\n".join([
            "",
            "      DOUBLE PRECISION JRB(NCOLORFOLD,NAMPSO,NCOMB)",
            "      DOUBLE PRECISION JIB(NCOLORFOLD,NAMPSO,NCOMB)",
            "      SAVE JRB, JIB",
            "      INTEGER BLASGATE",
            "      LOGICAL BLAS_COLOR_SUM",
            "      COMMON/TO_BLAS_COLOR_SUM/BLAS_COLOR_SUM"])
        replace_dict['blas_hel_setup'] = "\n".join([
            "",
            "      BLASGATE = 1",
            "      IF (BLAS_COLOR_SUM) BLASGATE = 0"])
        replace_dict['blas_hel_gather'] = "\n".join([
            "",
            "        JRB(:,:,K) = DBLE(%s(:,:))"
                                    % replace_dict['color_fold_array'],
            "        JIB(:,:,K) = DIMAG(%s(:,:))"
                                    % replace_dict['color_fold_array']])
        # a zero trip count leaves the scalar sum out without changing a
        # single block, which is what the helicity recycling rewriter walks
        replace_dict['blas_hel_gate'] = "*BLASGATE"
        replace_dict['blas_hel_finish'] = "\n".join([
            "",
            "      IF (BLASGATE.EQ.0) CALL %sGET_MATRIX_BATCH%s(JRB,JIB,"
            "NCOMB,TS)" % (prefix, proc_id)])
        replace_dict['blas_hel_routine'] = self.get_blas_routine_me(
            prefix, proc_id, nfold, nampso,
            replace_dict['nSqAmpSplitOrders'], ncomb,
            'NFOLD*(NFOLD+1)', replace_dict['chosen_so_configs'])

        if not shape or self.opt.get('hel_recycling'):
            # with helicity recycling on, this file is only what the good
            # helicities are found with, and what the rewriter reads: it stays
            # scalar, and the batch lives in the recycled matrix element above
            return

        replace_dict['blas_decl'] = "\n".join([
            "",
            "    DOUBLE PRECISION JRB(NCOLORFOLD,NAMPSO,%d)" % ncomb,
            "    DOUBLE PRECISION JIB(NCOLORFOLD,NAMPSO,%d)" % ncomb,
            "    INTEGER BLASGATE, BLASCOL"])
        replace_dict['blas_arg'] = ",JRB,JIB,BLASGATE,BLASCOL"
        replace_dict['blas_gather'] = "\n".join([
            "",
            "    JRB(:,:,BLASCOL) = DBLE(%s(:,:))"
                                    % replace_dict['color_fold_array'],
            "    JIB(:,:,BLASCOL) = DIMAG(%s(:,:))"
                                    % replace_dict['color_fold_array']])
        # a zero trip count leaves the scalar sum out
        replace_dict['blas_gate'] = "*BLASGATE"
        replace_dict['blas_matrix_args'] = ",JRB,JIB,1,1"
        replace_dict['blas_smatrix_decl'] = "\n".join([
            "",
            "    DOUBLE PRECISION JRB(%d,%d,%d)" % (nfold, nampso, ncomb),
            "    DOUBLE PRECISION JIB(%d,%d,%d)" % (nfold, nampso, ncomb),
            "    SAVE JRB, JIB",
            "    DOUBLE PRECISION BLASB(NCOMB), BLASP(NCOMB)",
            # BLAS_COLOR_SUM itself comes with run.inc, which SMATRIX has
            "    INTEGER BLASIDX(NCOMB), BLASNB, BLASGATE, IBH"])
        # The sweep must skip whatever the loop reading BLASB back will skip,
        # or it evaluates a MATRIX call per C-parity partner for a BLASB entry
        # nothing ever reads. The loop keeps the doubling, so only the wasted
        # work goes away here.
        select = '.NOT.(DEDUP.AND.IBH.GT.FLIP(IBH)) .AND. (%s)' \
            % (shape['select'] % replace_dict)
        # One sweep over the helicities worth computing fills BLASB, either
        # helicity by helicity as before or, once the good helicities have
        # settled, as one batch; the loop below then only reads it back, so
        # nothing is computed twice and AMP2/JAMP2 still add up once.
        replace_dict['blas_branch'] = "\n".join([
            "      BLASGATE = 1",
            "      IF (BLAS_COLOR_SUM .AND. %s) BLASGATE = 0"
                                                    % (shape['settled']
                                                       % replace_dict),
            "      BLASNB = 0",
            "      DO IBH = 1, NCOMB",
            "        IF (%s) THEN" % select,
            "          BLASNB = BLASNB + 1",
            "          BLASIDX(BLASNB) = IBH",
            "          BLASB(IBH) = %s" % (shape['collect'] % replace_dict),
            "        ENDIF",
            "      ENDDO",
            "      IF (BLASGATE.EQ.0 .AND. BLASNB.GT.0) THEN",
            "        CALL %sGET_MATRIX_BATCH%s(JRB,JIB,BLASNB,BLASP)"
                                                    % (prefix, proc_id),
            "        DO IBH = 1, BLASNB",
            "          BLASB(BLASIDX(IBH)) = BLASP(IBH)",
            "        ENDDO",
            "      ENDIF",
            ""])
        replace_dict['blas_matrix_call'] = "BLASB(I)"
        replace_dict['blas_routine'] = self.get_blas_routine_me(
            prefix, proc_id, nfold, nampso,
            replace_dict['nSqAmpSplitOrders'], ncomb,
            shape['cf_dim'], replace_dict['chosen_so_configs'])

    #===========================================================================
    # write_matrix_element_v4
    #===========================================================================
    def write_matrix_element_v4(self, writer, matrix_element, fortran_model,
                           proc_id = "", config_map = [], subproc_number = "",
                           xgrow_map = None):
        """Export a matrix element to a matrix.f file in MG4 madevent format"""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        if writer: 
            if not isinstance(writer, writers.FortranWriter):
                raise writers.FortranWriter.FortranWriterError(\
                "writer not FortranWriter")
            # Set lowercase/uppercase Fortran code
            writers.FortranWriter.downcase = False

        # check if MLM/.../ is supported for this matrix-element and update associate flag
        if self.model and 'MLM' in self.model["limitations"]:
            if 'MLM' not in self.proc_characteristic["limitations"]:
                used_couplings = matrix_element.get_used_couplings(output="set") 
                for vertex in self.model.get('interactions'):
                    particles = [p for p in vertex.get('particles')]
                    if 21 in [p.get('pdg_code') for p in particles]:
                        colors = [par.get('color') for par in particles]
                        if 1 in colors:
                            continue
                        elif 'QCD' not in vertex.get('orders'):
                            for bad_coup in vertex.get('couplings').values():
                                if bad_coup in used_couplings:
                                    self.proc_characteristic["limitations"].append('MLM')
                                    break

        # The proc prefix is not used for MadEvent output so it can safely be set
        # to an empty string.
        replace_dict = {'proc_prefix':'',
                        'set_amp2_line': 'ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT',
                        'flavor_mask_decl':'',
                        'flavor_mask_setup':''}

        # Crossing colour selection: an ME that serves as a base for a crossed
        # dependent publishes its per-flow JAMP2 (in its own flow order) so the
        # dependent can reselect colour natively instead of relabelling the base's
        # own selection -- which was masked with the BASE's ICOLAMP row and can
        # name a flow the dependent's own SELECT_COLOR would never pick. Both
        # crossing paths need it: the cross-group dependent (Track B,
        # _dsig_crossgroup_fills) and the within-group router (Track A,
        # write_matrix_router_file), each calling XG_SELCOL with its OWN IPROC.
        # Emitted only for those bases -- every other madevent ME keeps both holes
        # empty and is byte-identical.
        if (id(matrix_element) in getattr(self, '_crossgroup_base_mes', set())
                or id(matrix_element) in getattr(self, '_router_base_mes', set())):
            replace_dict['xg_jamp2_decl'] = (
                'C     Crossing base: publish this ME\'s per-flow JAMP2 so a'
                '\nC     crossed dependent can reselect colour in its own flow space.'
                '\n      DOUBLE PRECISION XG_JAMP2(0:MAXFLOW,VECSIZE_MEMMAX)'
                '\n      COMMON/TO_XG_JAMP2/XG_JAMP2')
            replace_dict['xg_jamp2_pub'] = (
                '      DO I=0,INT(JAMP2(0))'
                '\n        XG_JAMP2(I,IVEC) = JAMP2(I)'
                '\n      ENDDO')
        else:
            replace_dict['xg_jamp2_decl'] = ''
            replace_dict['xg_jamp2_pub'] = ''

        # Crossing holes of matrix_madevent_group_v4.inc: the group SMATRIX
        # decodes the extended FLAV_IDX and evaluates the crossed process through
        # a runtime IC. Only that template carries the holes (the single-process
        # matrix_madevent_v4.inc does not), and a process whose definition pins a
        # specific s-channel has its crossings generated separately, so it stays
        # on the plain path. When off the fills reproduce the historical code.
        me_use_crossing = (
            self.opt.get('use_crossing', False)
            and self.matrix_file == 'matrix_madevent_group_v4.inc'
            and not any(self.breaks_crossing_symmetry(proc)
                        for proc in matrix_element.get('processes')))
        self.fill_crossing_replace_dict_me(matrix_element, replace_dict,
                                           me_use_crossing, proc_id,
                                           xgrow_map=xgrow_map)

 
        mask_decl, mask_setup, n_flavors, active_flavor_mask = \
                self._get_flavor_mask_blocks(matrix_element)
        replace_dict['flavor_mask_decl'] = mask_decl
        replace_dict['flavor_mask_setup'] = mask_setup

        fortran_model.use_flavor_mask = (n_flavors > 0)
        fortran_model.me_n_flavors = n_flavors
        fortran_model.me_active_flavor_mask = active_flavor_mask
        # With crossing on, the external wavefunction NSF/NSV flag is multiplied
        # by IC(i) so a leg crossed between the initial and final state flips
        # (the crossed P/NHEL/IC are built by APPLY_CROSSING in SMATRIX).
        fortran_model.use_crossing_ic = me_use_crossing
        try:
            helas_calls = fortran_model.get_matrix_element_calls(matrix_element)
        finally:
            fortran_model.use_flavor_mask = False
            fortran_model.me_n_flavors = 0
            fortran_model.me_active_flavor_mask = None
            fortran_model.use_crossing_ic = False
        if fortran_model.width_tchannel_set_tozero and not ProcessExporterFortranME.done_warning_tchannel:
            logger.info("Some T-channel width have been set to zero [new since 2.8.0]\n if you want to keep this width please set \"zerowidth_tchannel\" to False", '$MG:BOLD')
            ProcessExporterFortranME.done_warning_tchannel = True

        replace_dict['helas_calls'] = "\n".join(helas_calls)


        #adding the support for the fake width (forbidding too small width)
        mass_width = matrix_element.get_all_mass_widths()
        mass_width = sorted(list(mass_width))
        width_list = set([e[1] for e in mass_width])
        
        replace_dict['fake_width_declaration'] = \
            ('  double precision fk_%s \n' * len(width_list)) % tuple(width_list)
        replace_dict['fake_width_declaration'] += \
            ('  save fk_%s \n' * len(width_list)) % tuple(width_list)
        fk_w_defs = []
        one_def = ' IF(%(w)s.ne.0d0) then \nfk_%(w)s = SIGN(MAX(ABS(%(w)s), ABS(%(m)s*small_width_treatment)), %(w)s) \n else \n fk_%(w)s = 0d0\n endif\n'     
        for m, w in mass_width:
            if w.lower() == 'zero':
                if ' fk_zero = 0d0' not in fk_w_defs: 
                    fk_w_defs.append(' fk_zero = 0d0')
                continue    
            fk_w_defs.append(one_def %{'m':m, 'w':w})
        replace_dict['fake_width_definitions'] = '\n'.join(fk_w_defs)

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id
        nexternal, ninitial = matrix_element.get_nexternal_ninitial()

        # Extract ncomb
        ncomb = matrix_element.get_helicity_combinations()
        replace_dict['ncomb'] = ncomb

        # Extract helicity lines
        helicity_lines = self.get_helicity_lines(matrix_element)
        replace_dict['helicity_lines'] = helicity_lines

        # Extract IC line
        ic_line = self.get_ic_line(matrix_element)
        replace_dict['ic_line'] = ic_line

        # Extract overall denominator
        # Averaging initial state color, spin, and identical FS particles
        den_factor_line = self.get_den_factor_line(matrix_element)
        replace_dict['den_factor_line'] = den_factor_line

        # Extract ngraphs
        ngraphs = matrix_element.get_number_of_amplitudes()
        replace_dict['ngraphs'] = ngraphs

        # Extract ndiags
        ndiags = len(matrix_element.get('diagrams'))
        replace_dict['ndiags'] = ndiags

        # Set define_iconfigs_lines
        replace_dict['define_iconfigs_lines'] = \
             """INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG
             COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG"""

        # if proc_id:
        #     # Set lines for subprocess group version
        #     # Set define_iconfigs_lines
        #     replace_dict['define_iconfigs_lines'] += \
        #          """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
        #          COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
        #     # Set set_amp2_line
        #     replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(SUBDIAG(%s))/XTOT" % \
        #                                     proc_id
        # else:
        #     # Standard running
        #     # Set set_amp2_line
        #     replace_dict['set_amp2_line'] = "ANS=ANS*AMP2(MAPCONFIG(ICONFIG))/XTOT"

        # Extract nwavefuncs
        nwavefuncs = matrix_element.get_number_of_wavefunctions()
        replace_dict['nwavefuncs'] = nwavefuncs

        # Extract ncolor
        ncolor = max(1, len(matrix_element.get('color_basis')))
        replace_dict['ncolor'] = ncolor
        # |M|^2 is summed over one color flow per reversal pair when the basis
        # allows it. JAMP itself keeps every flow: jamp2 and the color flow
        # selection below read all of them.
        folding = self.get_jamp_folding(matrix_element)
        replace_dict.update(self.get_color_fold_ampso(folding, ncolor))

        # Extract color data lines
        color_data_lines = self.get_color_data_lines(matrix_element)
        replace_dict['color_data_lines'] = "\n".join(color_data_lines) % {'proc_prefix': replace_dict['proc_prefix']}
        # A compressed color matrix is rebuilt at run time, into the common
        # block the matrix element reads it from.
        replace_dict['color_init_routine'] = "\n".join(
                self.get_color_init_routine(matrix_element,
                                            replace_dict['proc_prefix'],
                                            suffix=str(replace_dict['proc_id'])))


        # Set the size of Wavefunction
        if not self.model or any([p.get('spin') in [4,5] for p in self.model.get('particles') if p]):
            replace_dict['wavefunctionsize'] = 18
        else:
            replace_dict['wavefunctionsize'] = 6
            if hasattr(self.model, '_curr_gauge') and self.model._curr_gauge == 'FD':
                replace_dict['wavefunctionsize'] = 7

        # Extract amp2 lines
        amp2_lines = self.get_amp2_lines(matrix_element, config_map, replace_dict)
        replace_dict['amp2_lines'] = '\n'.join(amp2_lines)

        # The JAMP definition depends on the splitting order
        split_orders=matrix_element.get('processes')[0].get('split_orders')
        if len(split_orders)>0:
            squared_orders, amp_orders = matrix_element.get_split_orders_mapping()
            replace_dict['chosen_so_configs']=self.set_chosen_SO_index(
                              matrix_element.get('processes')[0],squared_orders)
            replace_dict['select_configs_if'] = '          IF (CHOSEN_SO_CONFIGS(SQSOINDEX%(proc_id)s(M,N))) THEN' % replace_dict
            replace_dict['select_configs_endif'] = ' endif'
        else:
            # Consider the output of a dummy order 'ALL_ORDERS' for which we
            # set all amplitude order to weight 1 and only one squared order
            # contribution which is of course ALL_ORDERS=2.
            squared_orders = [(2,),]
            amp_orders = [((1,),tuple(range(1,ngraphs+1)))]
            replace_dict['chosen_so_configs'] = '.TRUE.'
            # addtionally set the function to NOT be called
            replace_dict['select_configs_if'] = ''
            replace_dict['select_configs_endif'] = ''
            
        replace_dict['nAmpSplitOrders']=len(amp_orders)
        replace_dict['nSqAmpSplitOrders']=len(squared_orders)
        replace_dict['split_order_str_list']=str(split_orders)
        replace_dict['nSplitOrders']=max(len(split_orders),1)
        amp_so = self.get_split_orders_lines(
                [amp_order[0] for amp_order in amp_orders],'AMPSPLITORDERS')
        sqamp_so = self.get_split_orders_lines(squared_orders,'SQSPLITORDERS')
        replace_dict['ampsplitorders']='\n'.join(amp_so)
        replace_dict['sqsplitorders']='\n'.join(sqamp_so)
        

        # Extract JAMP lines
        # If no split_orders then artificiall add one entry called 'ALL_ORDERS'
        self.jamp_recipes = None
        jamp_lines, nb_temp = self.get_JAMP_lines_split_order(\
                             matrix_element,amp_orders,split_order_names=
                        split_orders if len(split_orders)>0 else ['ALL_ORDERS'],
                        orbit=self.jamp_orbit_allowed(matrix_element))
        replace_dict['jamp_lines'] = '\n'.join(jamp_lines)
        replace_dict['nb_temp_jamp'] = nb_temp
        recipes = getattr(self, 'jamp_recipes', None)
        replace_dict['jamp_decl'] = '\n'.join(
                                self.get_jamp_decl_lines(recipes, ''))
        replace_dict['jamp_tmp_decl'] = '' if recipes else \
                            "    COMPLEX*16 TMP_JAMP(%i)" % nb_temp

        # The color sum can run on the (n-2)! DDM basis while the color flow
        # probabilities keep using the (n-1)! trace one
        ncolor = self.set_color_flow_lines(matrix_element, replace_dict, ncolor)

        # BLAS-3 color sum: the helicities are the columns of a single right
        # hand side, so the whole sum is two DSYMM calls instead of one
        # triangular loop per helicity.
        self.set_blas_replace_dict(replace_dict, ncomb,
                                   int(replace_dict['ncolorfold']))

        if self.beam_polarization == [True, True]:
            replace_dict['beam_polarization'] = """
                         DO JJ=1,nincoming
               IF(POL(JJ).NE.1d0.AND.NHEL(JJ,I).EQ.INT(SIGN(1d0,POL(JJ)))) THEN
                 T=T*ABS(POL(JJ))
               ELSE IF(POL(JJ).NE.1d0)THEN
                 T=T*(2d0-ABS(POL(JJ)))
               ENDIF
             ENDDO
            """
        else:
            replace_dict['beam_polarization'] = ""
            for i in [0,1]:
                if self.beam_polarization[i]:
                    replace_dict['beam_polarization'] = """
                                   ! handling only one beam polarization here. Second beam can be handle via the pdf.
                                   IF(POL(%(bid)i).NE.1d0.AND.NHEL(%(bid)i,I).EQ.INT(SIGN(1d0,POL(%(bid)i)))) THEN
                 T=T*ABS(POL(%(bid)i))
               ELSE IF(POL(%(bid)i).NE.1d0)THEN
                 T=T*(2d0-ABS(POL(%(bid)i)))
               ENDIF """ % {'bid': i+1}




        replace_dict['template_file'] = pjoin(_file_path, \
                          'iolibs/template_files/%s' % self.matrix_file)
        replace_dict['template_file2'] = pjoin(_file_path, \
                          'iolibs/template_files/split_orders_helping_functions.inc')      
        
        s1,s2 = matrix_element.get_spin_state_initial()
        replace_dict['nb_spin_state1'] = s1
        replace_dict['nb_spin_state2'] = s2


        # handling of the flavor:
        all_flav = matrix_element.get_external_flavors_with_iden()
        replace_dict['max_flavor'] = len(all_flav)
        replace_dict['get_flavor_matrix'] = ''

        # The Python flavor tuples store raw PDG codes (needed for check_flavor),
        # but the Fortran FLV_COUPLING % PARTNER array is indexed by 1-based
        # position within the merged particle group.  Build a mapping so that
        # each PDG code is converted to its group position before being written
        # into the DATA statement.
        model = matrix_element.get('processes')[0].get('model')
        pdg_to_group_pos, max_group_size = self._build_flavor_group_lookup(model)

        for i, flav in enumerate(all_flav):
            flav_positions = [str(self._map_flavor_to_group_pos(
                              f, pdg_to_group_pos, max_group_size))
                              for f in flav[0]]
            replace_dict['get_flavor_matrix'] += ' DATA (FLAVOR(i,  %d),i=  1, NEXTERNAL) /%s/\n' % (i+1, ', '.join(flav_positions))

        # In addition to the IFLAV-indexed FLAVOR table above (one row per
        # coupling-equivalence group), emit a second table indexed by the
        # global IPSEL (leshouche row).  BROKEN_SYM needs row-level flavor
        # information to distinguish same-flavor vs different-flavor decay
        # configurations within a single coupling group — without this, the
        # identical-particle factor is never cancelled when it should be.
        all_flav_flat = [flav_tuple for group in all_flav for flav_tuple in group]
        replace_dict['max_flavor_row'] = len(all_flav_flat)
        replace_dict['get_flavor_row_matrix'] = ''
        for i, flav_tuple in enumerate(all_flav_flat):
            flav_positions = [str(self._map_flavor_to_group_pos(
                              f, pdg_to_group_pos, max_group_size))
                              for f in flav_tuple]
            replace_dict['get_flavor_row_matrix'] += ' DATA (FLAVOR_ROW(i,  %d),i=  1, NEXTERNAL) /%s/\n' % (i+1, ', '.join(flav_positions))
        
        # information for computing the correct symmetry factor for each flavor
        process = matrix_element.get('processes')[0]
        sym_data = self._get_broken_symmetry_data(process, ninitial)
        self._fill_broken_sym_replace_dict(replace_dict, sym_data)
        replace_dict['broken_sym_function'] = \
            self._make_broken_sym_fortran_function(
                'BROKEN_SYM' + str(proc_id), sym_data)



        if writer:
            file = open(replace_dict['template_file']).read()
            file = misc.apply_template(file, replace_dict)
            # Add the split orders helper functions.
            file = file + '\n' + misc.apply_template(
                open(replace_dict['template_file2']).read(), replace_dict)
            # Write the file
            writer.writelines(file)
            return len([call for call in helas_calls if call.find('#') != 0]), ncolor
        else:
            replace_dict['return_value'] = (len([call for call in helas_calls if call.find('#') != 0]), ncolor)
            return replace_dict
        
    #===========================================================================
    # _crossgroup_base_files
    #===========================================================================
    def _crossgroup_base_files(self, base_proc_id):
        """Base-group matrix-element source files a cross-group dependent symlinks
        into its own P directory so the makefile compiles the shared crossing-
        aware SMATRIX there too. Correctness-first: the source is reused (symlink)
        but each directory still compiles its own object; sharing the compiled .o
        is a later build step. With helicity recycling the base keeps
        matrix<b>_orig.f plus the template for the run-time optimised copy,
        otherwise a single matrix<b>.f."""
        if self.opt.get('hel_recycling'):
            return ['matrix%d_orig.f' % base_proc_id,
                    'template_matrix%d.f' % base_proc_id]
        return ['matrix%d.f' % base_proc_id]

    def write_crossgroup_mk(self, base_dir, base_proc_id):
        """Write crossgroup.mk in the current (dependent) P directory. Included by
        the shared makefile (`-include crossgroup.mk`), it makes the base group's
        matrix<b> object file be SYMLINKED from the base directory rather than
        recompiled from the symlinked source -- the whole point of the reuse. It is
        built in the base directory first (the specific rule overrides the
        makefile's %.o:%.f pattern; the recursive rule is the standalone ordering
        fallback -- the top-level parallel makefile also orders base before
        dependents).

        With helicity recycling BOTH matrix<b>_orig.o (the full matrix element) and
        matrix<b>_optim.o are shared: gen_ximprove bakes the base optim over
        G_base U tau(G_base) of the crossing class (see crossgroup_helunion.dat),
        so it covers every member. Without recycling the single matrix<b>.o is
        the full, shareable object."""
        objs = ['matrix%d.o' % base_proc_id]
        if self.opt.get('hel_recycling'):
            objs = ['matrix%d_orig.o' % base_proc_id,
                    'matrix%d_optim.o' % base_proc_id]
        lines = ['# Track B cross-group crossing: reuse the base group\'s compiled',
                 '# matrix element (%s) instead of recompiling the symlinked source.'
                 % base_dir]
        for o in objs:
            base_o = pjoin('..', base_dir, o)
            lines.append('%s: %s' % (o, base_o))
            lines.append('\tln -sf %s %s' % (base_o, o))
            lines.append('%s:' % base_o)
            lines.append('\t+$(MAKE) -C %s %s' % (pjoin('..', base_dir), o))
        if self.opt.get('amp_chunk_size', AMP_CHUNK_SIZE_DEFAULT) > 0:
            # ... and, when the base's HELAS call sequence was split out into
            # amplitude files of its own, those objects too. They are globbed
            # in the base directory at make time rather than listed here: the
            # optim ones do not exist until gen_ximprove has recycled the base,
            # which is well after this file is written. The pattern rules below
            # override the shared makefile's %.o: %.f (it is included last), and
            # the extra prerequisites get them built before either binary links.
            base = pjoin('..', base_dir)
            for kind in ('origamp', 'optimamp'):
                var = 'XG_%s' % kind.upper()
                lines.append('%s := $(notdir $(patsubst %%.f,%%.o,'
                             '$(wildcard %s/matrix%d_%s*.f)))'
                             % (var, base, base_proc_id, kind))
                lines.append('matrix%d_%s%%.o:' % (base_proc_id, kind))
                lines.append('\t+$(MAKE) -C %s $@' % base)
                lines.append('\tln -sf %s/$@ $@' % base)
            lines.append('MATRIX += $(XG_ORIGAMP) $(XG_OPTIMAMP)')
            lines.append('MATRIX_HEL += $(XG_ORIGAMP)')
            lines.append('madevent_forhel: $(XG_ORIGAMP)')
            lines.append('madevent: $(XG_ORIGAMP) $(XG_OPTIMAMP)')
        open('crossgroup.mk', 'w').write('\n'.join(lines) + '\n')

    def write_crossgroup_helunion(self, subproc_path):
        """Write crossgroup_helunion.dat in each crossing BASE directory. Each
        line is `<base_proc_id> t1 t2 ... tNCOMB`, the base->base helicity SIGN
        map tau of one dependent crossing (_crossgroup_base_helsignmap): the
        recycled optim's row h contributes to that crossing iff tau[h] is good for
        the base. gen_ximprove reads it and bakes the base optim over the union
        G_base U tau(G_base) over every line, so a single compiled optim serves
        every member of the class. An all-zero row is the sentinel for a crossing
        whose tau is not a clean permutation: keep every config.

        tau and NOT the GHREMAP sigma (_crossed_helicity_configs, permuted=True).
        sigma is the transform of matrix<b>_orig.f, which takes NHEL at run time
        and applies the crossing's slot permutation to it; the recycled
        matrix<b>_optim.f bakes its configs into the HELAS calls and gets only
        (PUSE, IC), so the sign flips survive and the permutation does not.
        Baking the sigma union
        into the optim drops helicity rows the crossed caller needs -- measured
        -28.5% on the q q~ > q q~ cross section, where the routed t-channel
        subprocess got 2 of the 4 rows it needs.

        Both crossing flavours feed this: a Track B cross-group dependent (whose
        base lives in another P directory) and a Track A within-group router
        (whose base is a matrix element of the same directory). Either way the
        recycled optim is entered with crossed momenta, and it bakes its helicity
        configs -- so pruning it to the base's own good-hel biases the crossed
        caller."""
        for base_dir, per_proc in self._crossgroup_helperms.items():
            lines = []
            for base_proc_id, perms in sorted(per_proc.items()):
                for pi in perms:
                    lines.append('%d %s' % (base_proc_id,
                                            ' '.join(str(x) for x in pi)))
            if lines:
                with open(pjoin(subproc_path, base_dir,
                                'crossgroup_helunion.dat'), 'w') as f:
                    f.write('\n'.join(lines) + '\n')

    def write_crossgroup_parallel_makefile(self, subproc_path):
        """Write SubProcesses/makefile_madevent so every P directory builds with a
        single `make -f makefile_madevent -jN` (madevent binaries) or `... forhel`.
        Cross-group dependents (Track B) are ordered AFTER their base directory so
        the base's shared objects exist to be symlinked in; make's dependency graph
        then gives both the ordering and full parallelism. Each target just
        delegates to that directory's own makefile."""
        lines = [
            '# Generated (Track B): build every P directory in one parallel call:',
            '#     make -f makefile_madevent -j           # the madevent binaries',
            '#     make -f makefile_madevent -j forhel    # the madevent_forhel ones',
            '# Cross-group dependents are ordered after their base directory.',
            'PDIRS := $(shell cat subproc.mg 2>/dev/null | tr -d " \\t")',
            'MADEVENT := $(addsuffix /madevent,$(PDIRS))',
            'FORHEL := $(addsuffix /madevent_forhel,$(PDIRS))',
            '',
            '.PHONY: all forhel $(MADEVENT) $(FORHEL)',
            'all: $(MADEVENT)',
            'forhel: $(FORHEL)',
            '',
            '$(MADEVENT) $(FORHEL):',
            '\t+$(MAKE) -C $(@D) $(@F)',
            '',
            '# cross-group ordering (dependent directory waits for its base):',
        ]
        for dep, base in self._crossgroup_dirs:
            lines.append('%s/madevent: %s/madevent' % (dep, base))
            lines.append('%s/madevent_forhel: %s/madevent_forhel' % (dep, base))
        open(pjoin(subproc_path, 'makefile_madevent'), 'w').write(
            '\n'.join(lines) + '\n')

    #===========================================================================
    # _dsig_crossgroup_fills
    #===========================================================================
    def _crossed_helicity_configs(self, base_me, cross, signed=True,
                                  permuted=True):
        """The base helicity rows transformed by the crossing. Three consumers
        need three DIFFERENT transforms, selected by (signed, permuted). Which
        one belongs where is decided by what the code being fed can APPLY at run
        time, and getting it wrong is silent:

        * (True, True) -- the GHREMAP remap sigma[hb][k] = base_row[PERM[k]]*SGN[k],
          the transform the _GOODHEL_PROBE relation validates: a base row is good
          WHEN CROSSED iff sigma^-1 of it is good for the base's own process. This
          is the *loop-index* space of matrix<b>_orig.f, which takes NHEL at run
          time and so realises the full PERM+SGN transform via
          APPLY_CROSSING_TABLE (CROSS_GHIDX is its fortran side). SGN belongs
          here because the crossed physical config
          bh[PERM[k]]*SGN[k]*IC_IN[PERM[k]] reduces to the bare table value
          bh[PERM[k]]*SGN[k] once the common IC_IN[PERM[k]] is stripped.

          CAUTION: G_base U sigma(G_base) is NOT a safe helicity table for the
          recycled matrix<b>_optim.f -- see (True, False) below, which is.

        * (True, False) -- the good-hel-set remap of the RECYCLED optim
          (_crossgroup_base_helsignmap): tau[hb][k] = base_row[k]*SGN[k], a sign
          flip at the crossed legs with NO slot permutation. matrix<b>_optim.f
          bakes its helicity configs into the HELAS calls and takes only
          (PUSE, IC) at run time, so a crossed entry can apply SGN -- through
          IC -- but never PERM. Writing sigma = tau . pi_unsigned (with
          pi_unsigned the (False, True) map, which says which optim row
          reproduces which orig row) gives, for optim row hb, the exact
          statement: hb is non-zero when crossed iff tau[hb] is good for the
          base. So the shared optim's good-hel union is G_base U tau(G_base),
          NOT G_base U sigma(G_base). tau is also always a clean permutation --
          each leg's helicity states are closed under negation -- whereas sigma
          need not be when the crossing swaps legs of different spin.

        * (False, True) -- the event helicity LABEL (the router's digit
          permutation):
          crossed[hb][k] = base_row[PERM[k]], exactly what APPLY_CROSSING_TABLE
          writes into NHEL (it permutes NHEL -- NHEL(XK)=NHEL_IN(PERM(XK)) -- but
          flips only the IC/NSF flags -- IC(XK)=SGN(XK)*IC_IN(PERM(XK))). The LHE
          label is the raw NHEL table value (unwgt.f: jpart(7,i)=nhel(i)), never
          NHEL*IC, and the base MATRIX gives leg k the physical spinor helicity
          NHEL(k)*IC(k)=base_row[PERM[k]]*SGN[k]*IC_IN[PERM[k]]
          =base_row[PERM[k]]*IC_dep[k] (SGN[k]*IC_IN[PERM[k]] is exactly slot k's
          own NSF in the dependent), matching the dependent's native label
          NHEL_dep[k]*IC_dep[k] iff NHEL_dep[k]=base_row[PERM[k]] -- NO extra sign.
          Multiplying SGN here double-counts the flip and mislabels every
          fermion/vector leg that swaps initial<->final.

        Returns (base_rows, crossed_rows) as tuples in the base NHEL order."""
        bh = [tuple(x) for x in base_me.get_helicity_matrix()]
        tables = ProcessExporterFortran.compute_crossing_tables(self, base_me)
        nx = tables['nexternal']
        P = [tables['perm'][cross * nx + k] for k in range(nx)] if permuted \
            else list(range(nx))
        S = [tables['ic'][cross * nx + k] for k in range(nx)] if signed \
            else [1] * nx
        crossed = [tuple(row[P[k]] * S[k] for k in range(nx)) for row in bh]
        return bh, crossed

    def _helicity_row_permutation(self, bh, crossed):
        """1-based row permutation pi[hb] = the index whose base NHEL row equals
        the transformed row of hb, or None if the transform is not a clean
        permutation of the table."""
        bhpos = {cfg: i for i, cfg in enumerate(bh)}
        pi = [bhpos.get(c, -1) for c in crossed]
        if -1 in pi or sorted(pi) != list(range(len(bh))):
            return None
        return [p + 1 for p in pi]

    def _crossgroup_base_helsignmap(self, base_me, cross):
        """1-based base->base helicity permutation tau of a crossing:
        tau[hb] = the base index whose NHEL row equals the row of hb with the
        helicity of every crossed leg negated (SGN, no PERM). This is the
        transform the recycled matrix<b>_optim.f realises when entered with a
        crossing's (PUSE, IC): optim row hb is non-zero for that crossing iff
        tau[hb] is good for the base's own process, so the union good-hel the
        shared optim must be baked over is G_base U tau(G_base). Returns None if
        not a clean permutation (only reachable if the helicity table is not
        closed under negating those legs, e.g. a restricted helicity set)."""
        return self._helicity_row_permutation(
            *self._crossed_helicity_configs(base_me, cross, permuted=False))

    def _diagram_topology_signature(self, me):
        """Per diagram number, the set of its internal propagators as
        (canonical external-leg subset, |PDG|) -- a crossing-covariant topology
        signature. A propagator is identified by the external legs whose momenta
        flow through it (a subset and its complement are the same propagator,
        hence the canonical choice of the two) TOGETHER WITH the particle running
        in it. get_s_and_t_channels numbers the propagators negative,
        external-inward; the final t-channel 'propagator' is a single external
        leg and is dropped (canonical length 1).

        The leg subsets alone are not a fine enough invariant: two diagrams can
        route the same momenta through different particles, and then they share a
        signature, the base lookup loses one of them and _crossgroup_configmap
        degrades to the identity. g g > t t~ u u~ is the standing example -- the
        gluon-exchange diagram and the one carrying the four-gluon vertex through
        its auxiliary field have identical leg subsets and differ only here.

        |PDG| and not PDG: crossing a leg between the initial and the final state
        reverses the momentum flow through every propagator on its path, which
        conjugates them. The magnitude is what is invariant under the relabelling
        -- and staying invariant is the whole point, since this signature is what
        matches a diagram to its counterpart in the crossed process.

        Returns (dict diagram_number -> frozenset of (subset, |PDG|), nexternal).
        """
        nx, nini = me.get_nexternal_ninitial()
        model = me.get('processes')[0].get('model')
        npdg = model.get_first_non_pdg()
        allset = frozenset(range(1, nx + 1))
        canon = lambda s: min(s, allset - s, key=lambda x: (len(x), sorted(x)))
        out = {}
        for diag in me.get('diagrams'):
            sch, tch = diag.get('amplitudes')[0].get_s_and_t_channels(
                nini, model, npdg)
            ext = {i: frozenset([i]) for i in range(1, nx + 1)}
            props = set()
            for vert in list(sch) + list(tch):
                legs = vert.get('legs')
                daughters = [l.get('number') for l in legs[:-1]]
                s = frozenset().union(*[ext.get(d, frozenset([d]))
                                        for d in daughters]) if daughters \
                    else frozenset()
                ext[legs[-1].get('number')] = s
                if 2 <= len(canon(s)):
                    props.add((canon(s), abs(legs[-1].get('id'))))
            out[diag.get('number')] = frozenset(props)
        return out, nx

    def _crossgroup_configmap(self, dep_me, base_me, cross):
        """1-based map from a dependent diagram number to the base diagram number
        of the same topology under the crossing. The dependent's genps samples its
        own config's poles, but the base SMATRIX enhances AMP2(channel), so channel
        must name the matching BASE diagram; otherwise the importance sampling is
        mis-paired (this only affects the variance, never the result -- summing the
        channels gives the full integral for any bijective pairing). Returns the
        identity if the diagrams cannot be cleanly matched -- with a warning,
        because that fallback is otherwise invisible: it is indistinguishable
        from the common and legitimate case of a crossing-covariant numbering,
        every matrix element still agrees to the last digit, and the only symptom
        is a cross section that integrates slowly and unstably behind an error
        estimate that no longer means anything."""
        bsub, nx = self._diagram_topology_signature(base_me)
        dsub, _ = self._diagram_topology_signature(dep_me)
        ngraphs = len(dep_me.get('diagrams'))
        bsig = {v: k for k, v in bsub.items()}
        tables = ProcessExporterFortran.compute_crossing_tables(self, base_me)
        P = [tables['perm'][cross * nx + k] for k in range(nx)]
        d2b = {k + 1: P[k] + 1 for k in range(nx)}   # dep leg -> base leg
        allset = frozenset(range(1, nx + 1))
        canon = lambda s: min(s, allset - s, key=lambda x: (len(x), sorted(x)))

        def bail(why):
            logger.warning(
                'crossing: could not match the diagrams of %s onto %s '
                '(crossing %d): %s. Falling back to the identity config map -- '
                'the cross section stays correct, but the multi-channel '
                'importance sampling of the routed subprocess is mis-paired and '
                'will integrate slowly, with an unreliable error estimate.',
                dep_me.get('processes')[0].shell_string(),
                base_me.get('processes')[0].shell_string(), cross, why)
            return list(range(1, ngraphs + 1))

        if len(bsig) != len(bsub):
            return bail("%d of the base's %d diagrams share a topology "
                        "signature with another"
                        % (len(bsub) - len(bsig), len(bsub)))
        cmap = list(range(1, ngraphs + 1))
        for dd, ds in dsub.items():
            if not 1 <= dd <= ngraphs:
                return bail('diagram number %d is outside 1..%d' % (dd, ngraphs))
            sig = frozenset((canon(frozenset(d2b[l] for l in sub)), pdg)
                            for (sub, pdg) in ds)
            if sig in bsig:
                cmap[dd - 1] = bsig[sig]
            else:
                return bail('diagram %d has no counterpart in the base' % dd)
        if sorted(cmap) != list(range(1, ngraphs + 1)):
            return bail('the matching is not a bijection')
        return cmap

    def _dsig_crossgroup_fills(self, matrix_element, proc_id, crossgroup):
        """Fill the cross-group (Track B) holes of auto_dsig_v4.inc for a
        dependent subprocess that has no matrix element of its own and routes to
        a base group's symlinked crossing-aware SMATRIX.

        * beams -- the dependent cannot define its own GET_FLAVOR (it would clash
          with the symlinked base's), so its FLAVOR table (group-position coded,
          exactly as GET_FLAVOR would return) is inlined as DSIG_XGFLAV and
          indexed by IFLAV for the PDF.
        * SMATRIX -- dispatch to the base SMATRIX with the crossed FLAV_IDX
          (DSIG_XGROUTE(IFLAV)) instead of IFLAV; the base crosses the momenta and
          rebuilds the crossed denominator internally so ANS is this subprocess's
          matrix element. Momenta/PDF/phase space stay this subprocess's own.
        * event helicity/colour -- the base returns selected_hel/selected_col in
          ITS enumeration; the event is written through this subprocess's own
          get_helicities / ICOLUP, so remap the index base -> dependent per flavor
          (DSIG_XGHEL / DSIG_XGCOL). Colour is the identity for colourless.
        * multi-channel -- the base enhances AMP2(channel) in its diagram
          numbering; translate this subprocess's channel to the matching base
          diagram (DSIG_XGCONFIG) so importance sampling stays paired.
        All four maps are emitted only when non-identity.
        """
        base_proc_id = crossgroup['base_proc_id']
        flav_idx = crossgroup['flav_idx']       # per dep flavor -> base FLAV_IDX
        base_me = crossgroup['base_me']
        nflav_base = len(base_me.get_external_flavors_with_iden())
        all_flv = matrix_element.get_external_flavors_with_iden()
        model = self.model or matrix_element.get('processes')[0].get('model')
        pdg_to_group_pos, max_group_size = self._build_flavor_group_lookup(model)

        # Column-major flat DATA (leg fastest, then flavor) -- avoids an implied-
        # do index variable, which need not be declared in every program unit.
        positions = [str(self._map_flavor_to_group_pos(
                         f, pdg_to_group_pos, max_group_size))
                     for flav in all_flv for f in flav[0]]
        decl = ['      INTEGER DSIG_XGFLAV(NEXTERNAL,%d)' % len(all_flv),
                '      DATA DSIG_XGFLAV /%s/' % ','.join(positions),
                '      INTEGER DSIG_XGROUTE(%d)' % len(all_flv),
                '      DATA DSIG_XGROUTE /%s/' % ','.join(str(x) for x in flav_idx)]

        # Per-flavor colour map (base flow -> dependent flow).
        colmap = [self._router_colmap(matrix_element, base_me,
                                      (iflav - 1) // nflav_base)
                  for iflav in flav_idx]
        ncol = len(colmap[0]) if colmap else 0
        # Event helicity: relabel the base's selected helicity code into this
        # (crossed) subprocess's canonical code by permuting the code's
        # mixed-radix digits with the crossing permutation (GET_CROSS_PERM),
        # decoded directly by this subprocess's get_nhel. Replaces the explicit
        # base->dep helicity map. GET_CROSS_PERM takes the extended base index
        # (DSIG_XGROUTE(flav)); cross 0 gives the identity permutation.
        nhstate = [len(s) for s in base_me.get_helicity_per_particle()]
        decl += ['      INTEGER XPERM(NEXTERNAL), XSGN(NEXTERNAL), XDUMF',
                 '      INTEGER XBDIG(NEXTERNAL), XHR, XHK',
                 '      INTEGER XNHS(NEXTERNAL)',
                 '      DATA XNHS /%s/' % ','.join(str(n) for n in nhstate)]
        hel_post = (
            '\n      CALL CR%s_GET_CROSS_PERM(DSIG_XGROUTE({flav}), XPERM,'
            ' XSGN, XDUMF)'
            '\n      IF (selected_hel{idx}.GE.1) THEN'
            '\n        XHR = selected_hel{idx} - 1'
            '\n        DO XHK=NEXTERNAL,1,-1'
            '\n          XBDIG(XHK) = MOD(XHR, XNHS(XHK))'
            '\n          XHR = XHR / XNHS(XHK)'
            '\n        ENDDO'
            '\n        selected_hel{idx} = 0'
            '\n        DO XHK=1,NEXTERNAL'
            '\n          selected_hel{idx} = selected_hel{idx} * XNHS(XPERM(XHK))'
            ' + XBDIG(XPERM(XHK))'
            '\n        ENDDO'
            '\n        selected_hel{idx} = selected_hel{idx} + 1'
            '\n      ENDIF') % base_proc_id
        # Colour: unlike helicity, a base->dep index relabel of selected_col is
        # NOT sufficient. The base SMATRIX picked its flow with select_color,
        # which masks the base-order JAMP2 with THIS (dependent) binary's ICOLAMP
        # + ICONFIG -- mismatched in flow order AND config space -- so the picked
        # flow can be incompatible with the sampled config and addmothers fails
        # to reduce its ICOLUP. Reselect natively instead: permute the base's
        # published per-flow JAMP2 (COMMON/TO_XG_JAMP2) into this subprocess's
        # flow order (DSIG_XGCOL) and run this subprocess's own SELECT_COLOR
        # (its own ICOLAMP + ICONFIG), via the XG_SELCOL helper below. Bit-for-bit
        # a native colour selection. Only needed when colmap is non-identity
        # (identity/colourless: the base's selection is already in this order).
        identity_col = list(range(1, ncol + 1))
        col_active = ncol > 0 and any(cm != identity_col for cm in colmap)
        dsig_xg_helper = ''
        col_scalar_call, col_vec_call = '', ''
        if col_active:
            col_flat = ','.join(str(x) for col in colmap for x in col)
            dsig_xg_helper = self._crossgroup_colsel_helper(
                proc_id, ncol, len(colmap), col_flat)
            col_scalar_call = ('\n      CALL XG_SELCOL%s(RCOL, IFLAV, 1,'
                               ' SELECTED_COL(1))' % proc_id)
            col_vec_call = ('\n        CALL XG_SELCOL%s(COL_RAND(IVEC),'
                            ' IFLAV_VEC(IVEC), IVEC, SELECTED_COL(IVEC))'
                            % proc_id)

        # Multi-channel config remap: the base SMATRIX enhances AMP2(channel) in
        # ITS diagram numbering, but this subprocess's genps samples its own
        # config's poles, so translate the channel to the matching base diagram.
        ngraphs = len(base_me.get('diagrams'))
        configmap = [self._crossgroup_configmap(matrix_element, base_me,
                                                (iflav - 1) // nflav_base)
                     for iflav in flav_idx]
        chan_scalar, chan_vec = 'channel', 'channels(IVEC)'
        if any(cm != list(range(1, ngraphs + 1)) for cm in configmap):
            decl.append('      INTEGER DSIG_XGCONFIG(%d,%d)'
                        % (ngraphs, len(configmap)))
            decl.append('      DATA DSIG_XGCONFIG /%s/'
                        % ','.join(str(x) for col in configmap for x in col))
            chan_scalar = 'DSIG_XGCONFIG(channel, IFLAV)'
            chan_vec = 'DSIG_XGCONFIG(channels(IVEC), IFLAV_VEC(IVEC))'

        # DSIG_XG* are used from three separate program units (DSIG, DSIG_VEC,
        # SMATRIX_MULTI); declare them in each.
        decl_block = '\n'.join(decl) + '\n'

        return {
            'dsig_xg_decl': decl_block,
            'dsig_xg_decl_vec': decl_block,
            'dsig_xg_decl_multi': decl_block,
            'dsig_xg_helper': dsig_xg_helper,
            'dsig_getflavor': '      FLAVOR(:) = DSIG_XGFLAV(:, IFLAV)',
            'dsig_smatrix_call': (
                '     CALL SMATRIX%d(P1, DSIG_XGROUTE(IFLAV), RHEL, RCOL, %s,'
                ' 1, DSIGUU, selected_hel(1), selected_col(1))'
                % (base_proc_id, chan_scalar)
                + hel_post.format(idx='(1)', flav='IFLAV')
                + col_scalar_call),
            # vectorised (SMATRIX_MULTI) path: same routing. The MULTI wrapper
            # itself keeps this subprocess's own name (it is defined in this
            # auto_dsig); only the inner base-SMATRIX call + flavor are routed.
            'dsig_getflavor_vec':
                '       FLAVOR(:) = DSIG_XGFLAV(:, IFLAV_VEC(IVEC))',
            'dsig_smatrix_vec_name': 'SMATRIX%d' % base_proc_id,
            'dsig_smatrix_vec_flav': 'DSIG_XGROUTE(IFLAV_VEC(IVEC))',
            'dsig_smatrix_vec_chan': chan_vec,
            'dsig_smatrix_vec_post': (
                hel_post.format(idx='(IVEC)', flav='IFLAV_VEC(IVEC)')
                + col_vec_call),
        }

    def _crossgroup_colsel_helper(self, proc_id, ncol, nflav, col_flat,
                                  iproc=1):
        """Emit XG_SELCOL<proc_id>, the crossing colour-selection helper for a
        subprocess that gets its matrix element from a crossed base. It permutes
        the base ME's published per-flow JAMP2 (COMMON/TO_XG_JAMP2, base flow
        order) into this subprocess's flow order via DSIG_XGCOL (base flow ->
        dep flow) and runs this subprocess's own SELECT_COLOR (its ICOLAMP row +
        the live ICONFIG), so the returned flow is native to this subprocess --
        consistent with its ICOLUP and its sampled config, unlike a bare
        base->dep index relabel of the base's own (mismatched) selection. The
        DATA is column-major (flow fastest, then flavor); the writer wraps the
        long line.

        ``iproc`` is SELECT_COLOR's matrix-element index, i.e. the ICOLAMP row to
        mask with, and must be THIS subprocess's own. A cross-group dependent
        (Track B) is alone in its P directory and is always 1; a within-group
        router (Track A) shares the directory with its base and passes its own
        proc_id -- the base's row is a different subprocess's and generally
        allows a different set of flows at the same ICONFIG.
        """
        return '\n'.join([
            '      SUBROUTINE XG_SELCOL%s(RCOL, IFLAV, IVEC, ICOL)' % proc_id,
            '      IMPLICIT NONE',
            "      INCLUDE 'genps.inc'",
            "      INCLUDE 'nexternal.inc'",
            "      INCLUDE 'maxconfigs.inc'",
            "      INCLUDE 'maxamps.inc'",
            "      INCLUDE '../../Source/vector.inc'",
            '      DOUBLE PRECISION RCOL',
            '      INTEGER IFLAV, IVEC, ICOL',
            '      INTEGER I',
            '      INTEGER MAPCONFIG(0:LMAXCONFIGS), ICONFIG',
            '      COMMON/TO_MCONFIGS/MAPCONFIG, ICONFIG',
            '      DOUBLE PRECISION XG_JAMP2(0:MAXFLOW,VECSIZE_MEMMAX)',
            '      COMMON/TO_XG_JAMP2/XG_JAMP2',
            '      DOUBLE PRECISION JD(0:MAXFLOW)',
            '      INTEGER DSIG_XGCOL(%d,%d)' % (ncol, nflav),
            '      DATA DSIG_XGCOL /%s/' % col_flat,
            # DSIG_XGCOL is normally a bijection onto 1..ncol, so every slot
            # below JD(0) is written; zero first anyway, so a map that misses a
            # flow degrades to "that flow has no weight" rather than feeding
            # SELECT_COLOR an uninitialised one.
            '      DO I=1,%d' % ncol,
            '        JD(I) = 0D0',
            '      ENDDO',
            '      JD(0) = XG_JAMP2(0,IVEC)',
            '      DO I=1,%d' % ncol,
            '        JD(DSIG_XGCOL(I,IFLAV)) = XG_JAMP2(I,IVEC)',
            '      ENDDO',
            '      CALL SELECT_COLOR(RCOL, JD, ICONFIG, %s, ICOL, IVEC)' % iproc,
            '      END',
        ])

    #===========================================================================
    # write_auto_dsig_file
    #===========================================================================
    def write_auto_dsig_file(self, writer, matrix_element, proc_id = "",
                             crossgroup=None):
        """Write the auto_dsig.f file for the differential cross section
        calculation, includes pdf call information.

        When ``crossgroup`` is given (Track B, cross-group crossing) this
        subprocess has no matrix element of its own: it symlinks a base group's
        crossing-aware SMATRIX and routes to it. The flavor lookup and the
        SMATRIX call are then filled with the routed variants (see
        _dsig_crossgroup_fills); everything else (PDFs, cuts, phase space) stays
        this subprocess's own."""

        if not matrix_element.get('processes') or \
               not matrix_element.get('diagrams'):
            return 0

        nexternal, ninitial = matrix_element.get_nexternal_ninitial()
        self.proc_characteristic['ninitial'] = ninitial
        self.proc_characteristic['nexternal'] = max(self.proc_characteristic['nexternal'], nexternal)

        # Add information relevant for MLM matching:
        # Maximum QCD power in all the contributions
        max_qcd_order = 0
        for diag in matrix_element.get('diagrams'):
            orders = diag.calculate_orders()
            if 'QCD' in orders:
                max_qcd_order = max(max_qcd_order,orders['QCD'])
        max_n_light_final_partons = max(len([1 for id in proc.get_final_ids() 
            if proc.get('model').get_particle(id).get('mass')=='ZERO' and
               proc.get('model').get_particle(id).get('color')>1])
                                    for proc in matrix_element.get('processes'))
        # Maximum number of final state light jets to be matched
        self.proc_characteristic['max_n_matched_jets'] = max(
                               self.proc_characteristic['max_n_matched_jets'],
                                   min(max_qcd_order,max_n_light_final_partons))

        # List of default pdgs to be considered for the CKKWl merging cut
        self.proc_characteristic['colored_pdgs'] = \
          sorted(list(set([abs(p.get('pdg_code')) for p in
            matrix_element.get('processes')[0].get('model').get('particles') if
                                                           p.get('color')>1])))

        if ninitial < 1 or ninitial > 2:
            raise writers.FortranWriter.FortranWriterError("""Need ninitial = 1 or 2 to write auto_dsig file""")

        replace_dict = {}
        replace_dict['additional_header'] = ''
        replace_dict['OMP_LIB'] = " USE OMP_LIB"
        replace_dict['OMP_PREFIX'] = "!$OMP PARALLEL\n!$OMP DO"
        replace_dict['OMP_POSTFIX'] = "!$OMP END DO\n!$OMP END PARALLEL"


        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        # Extract process info lines
        process_lines = self.get_process_info_lines(matrix_element)
        replace_dict['process_lines'] = process_lines

        # Set proc_id
        replace_dict['proc_id'] = proc_id
        replace_dict['numproc'] = 1

        # Flavor lookup + SMATRIX call default to this subprocess's own matrix
        # element; a cross-group dependent (Track B) overrides them below to route
        # to a base group's symlinked crossing-aware SMATRIX.
        replace_dict['dsig_xg_decl'] = ''
        replace_dict['dsig_xg_decl_vec'] = ''
        replace_dict['dsig_xg_decl_multi'] = ''
        replace_dict['dsig_xg_helper'] = ''
        replace_dict['dsig_getflavor'] = \
            '      CALL GET_FLAVOR%s(IFLAV, FLAVOR)' % proc_id
        replace_dict['dsig_smatrix_call'] = (
            '     CALL SMATRIX%s(P1, IFLAV, RHEL, RCOL,channel,1, DSIGUU,'
            ' selected_hel(1), selected_col(1))' % proc_id)
        # ... and the same for the vectorised (SMATRIX_MULTI) path.
        replace_dict['dsig_getflavor_vec'] = \
            '       CALL GET_FLAVOR%s(IFLAV_VEC(IVEC), FLAVOR)' % proc_id
        replace_dict['dsig_smatrix_vec_name'] = 'SMATRIX%s' % proc_id
        replace_dict['dsig_smatrix_vec_flav'] = 'IFLAV_VEC(IVEC)'
        replace_dict['dsig_smatrix_vec_chan'] = 'channels(IVEC)'
        replace_dict['dsig_smatrix_vec_post'] = ''

        # Set dsig_line
        if ninitial == 1:
            # No conversion, since result of decay should be given in GeV
            dsig_line = "pd(0)*dsiguu"
            conv_factor=""
        else:
            # Convert result (in GeV) to pb
            dsig_line = "pd(0)*conv*dsiguu"
            conv_factor="conv*"

        replace_dict['dsig_line'] = dsig_line
        replace_dict['conv'] = conv_factor
        # Extract pdf lines
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                  self.get_pdf_lines(matrix_element, ninitial, proc_id != "")
        replace_dict['pdf_vars'] = pdf_vars
        replace_dict['pdf_data'] = pdf_data
        replace_dict['pdf_lines'] = pdf_lines
        replace_dict['ee_comp_vars'] = eepdf_vars

        # Extract pdf lines vectorised code
        pdf_vars, pdf_data, pdf_lines, eepdf_vars = \
                self.get_pdf_lines(matrix_element, ninitial, proc_id != "", 
                                   vector=max(1,int(self.opt['vector_size'])))
        replace_dict['pdf_vars_vec'] = pdf_vars
        replace_dict['pdf_data_vec'] = pdf_data
        replace_dict['ee_comp_vars_vec'] = eepdf_vars
        replace_dict['pdf_lines_vec'] = pdf_lines


        all_flv = matrix_element.get_external_flavors_with_iden()

        # Lines that differ between subprocess group and regular
        if proc_id:
            replace_dict['numproc'] = int(proc_id)
            replace_dict['passcuts_begin'] = ""
            replace_dict['passcuts_end'] = ""
            # Set lines for subprocess group version
            # Set define_iconfigs_lines
            replace_dict['define_subdiag_lines'] = \
                 """\nINTEGER SUBDIAG(MAXSPROC),IB(2)
                 COMMON/TO_SUB_DIAG/SUBDIAG,IB"""    
            replace_dict['cutsdone'] = ""
            replace_dict['get_channel'] = "SUBDIAG(%s)" % proc_id
            replace_dict['get_channel_vec'] = """
            CHANNELS(IVEC) = CONFSUB(%s,SYMCONF(ICONF_VEC(CURR_WARP)))
            SUBDIAG(%s) = CHANNELS(IVEC) ! only valid if a single process
            channel = SUBDIAG(%s)""" % (proc_id,proc_id, proc_id)
            #SUBDIAG(%s)" % proc_id
            replace_dict['ADDITIONAL_FCT'] = ''
        else:
            replace_dict['passcuts_begin'] = ""#IF (PASSCUTS(PP)) THEN"
            replace_dict['passcuts_end'] = ""#ENDIF"
            replace_dict['define_subdiag_lines'] = "INTEGER IB(2)"
            replace_dict['cutsdone'] = "      cutsdone=.false.\n       cutspassed=.false."
            replace_dict['get_channel'] = "MAPCONFIG(ICONFIG)"
            replace_dict['get_channel_vec'] = " channel  = MAPCONFIG(ICONFIG)"
            # need to extract get_helicities/select color from the group template file
            text = open(pjoin(MG5DIR, 'madgraph', 'iolibs', 'template_files', 'super_auto_dsig_group_v4.inc')).read()
            color_hel_text = writers.FortranWriter.get_routine(text, ['select_color', 'get_helicities'])
            #misc.sprint(color_hel_text)
            get_nhel, get_helicity = [],[]
            get_nhel.append("   integer get_nhel" )
            get_helicity.append("   do i=1,nexternal")
            get_helicity.append(
                    "        nhel(i) = get_nhel(ihel,i)")
            get_helicity.append("enddo")
            replace_dict['call_to_local_get_helicities'] = "\n".join(get_helicity)
            replace_dict['definition_of_local_get_nhel'] = "\n".join(get_nhel)


            #raise Exception
            replace_dict['ADDITIONAL_FCT'] = self.get_dummy_grouping()+ '\n'.join(color_hel_text) % replace_dict
        # extract and replace ncombinations, helicity lines
        ncomb=matrix_element.get_helicity_combinations()
        replace_dict['ncomb']= ncomb
        helicity_lines = self.get_helicity_lines(matrix_element, add_nb_comb=True)
        replace_dict['helicity_lines'] = helicity_lines
        # Canonical helicity decoder tables for GET_NHEL: the per-event helicity
        # label is the mixed-radix code, so GET_NHEL decodes it (per-leg states)
        # rather than indexing an NHEL config table.
        hel_data = self._helstate_data(matrix_element)
        replace_dict['maxhel'] = hel_data['maxhel']
        replace_dict['nhstate_data'] = hel_data['nhstate_data']
        replace_dict['states_data'] = hel_data['states_data']

        context = {'read_write_good_hel':True}
        if not isinstance(self, ProcessExporterFortranMEGroup):            
            replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)
            context['nogrouping'] = True
        else:
            replace_dict['read_write_good_hel'] = ""
            context['nogrouping'] = False


        # extract which flavor to specify
        replace_dict['get_flavor'] = '\nC Not used anymore, just passed the flavor id instead'
        for i,flv in enumerate(all_flv):
            replace_dict['get_flavor'] += 'C %sIF (IFLAV.eq.%d) THEN\n' % ('ELSE' if i != 0 else '',i+1)
            replace_dict['get_flavor'] += 'C    FLAVOR = %s \n' % list(flv[0])
        replace_dict['get_flavor'] += 'C ENDIF\n'
        
        replace_dict['start_ipsel_for_IFLAV'] = '\nC set minimum ipsel for this IFLAV\n'
        ipsel = 0
        for i, flv in enumerate(all_flv):
            replace_dict['start_ipsel_for_IFLAV'] += ' %sIF (IFLAV.eq.%d) THEN\n' % ('ELSE' if i != 0 else '',i+1)
            replace_dict['start_ipsel_for_IFLAV'] += '    ipsel_shift = %d\n' % ipsel
            ipsel += len(flv)
        replace_dict['start_ipsel_for_IFLAV'] += ' ENDIF\n'
        replace_dict['maxflavor'] = len(all_flv)
        replace_dict['get_flavor_matrix'] = ''
        model = self.model or matrix_element.get('processes')[0].get('model')
        pdg_to_group_pos, max_group_size = self._build_flavor_group_lookup(model)
        for i, flav in enumerate(all_flv):
            flav_positions = [str(self._map_flavor_to_group_pos(
                              f, pdg_to_group_pos, max_group_size))
                              for f in flav[0]]
            replace_dict['get_flavor_matrix'] += ' DATA (FLAVOR(i,  %d),i=  1, NEXTERNAL) /%s/\n' % (i+1, ', '.join(flav_positions))


        # Cross-group dependent (Track B): override the flavor lookup + SMATRIX
        # call to route to the symlinked base group's crossing-aware SMATRIX.
        if crossgroup is not None:
            replace_dict.update(
                self._dsig_crossgroup_fills(matrix_element, proc_id, crossgroup))

        if writer:
            file = open(pjoin(_file_path, \
                          'iolibs/template_files/auto_dsig_v4.inc')).read()
            file = file % replace_dict

            # Write the file
            writer.writelines(file, context=context)
        else:
            return replace_dict, context

            
    #===========================================================================
    # get_dummy_grouping
    #===========================================================================
    def get_dummy_grouping(self):
        """ return dummy function for 
        prepare_grouping
        select_grouping
        for situation where they are no grouping
        """

        return """
        
        subroutine PREPARE_GROUPING_CHOICE(PP, WGT, INIT)
        double precision PP(*)
        double precision WGT
        logical INIT
        return
        end

        SUBROUTINE SELECT_GROUPING(IMIRROR, IPROC, ICONF, WGT, IWARP)
        Integer imirror
        integer iproc
        integer iconf
        double precision WGT
        integer iwarp
        return 
        end
        
        
        """


    #===========================================================================
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, mapconfigs, matrix_element):
        """Write the coloramps.inc file for MadEvent"""

        lines = self.get_icolamp_lines(mapconfigs, matrix_element, 1)
        lines.insert(0, "logical icolamp(%d,%d,1)" % \
                        (max(len(list(matrix_element.get('color_basis').keys())), 1),
                         len(mapconfigs)))


        # Write the file
        writer.writelines(lines)

        return True



    #===========================================================================
    # write_colors_file
    #===========================================================================
    def write_colors_file(self, writer, matrix_elements):
        """Write the get_color.f file for MadEvent, which returns color
        for all particles used in the matrix element."""

        if isinstance(matrix_elements, helas_objects.HelasMatrixElement):
            matrix_elements = [matrix_elements]

        model = matrix_elements[0].get('processes')[0].get('model')

        # We need the both particle and antiparticle wf_ids, since the identity
        # depends on the direction of the wf.
        wf_ids = set(sum([sum([sum([[wf.get_pdg_code(),wf.get_anti_pdg_code()] \
                                    for wf in d.get('wavefunctions')],[]) \
                               for d in me.get('diagrams')], []) \
                          for me in matrix_elements], []))

        leg_ids = set(sum([sum([sum([[l.get('id'), 
                          model.get_particle(l.get('id')).get_anti_pdg_code()] \
                                  for l in p.get_legs_with_decays()], []) \
                                for p in me.get('processes')], []) \
                           for me in matrix_elements], []))
        particle_ids = sorted(list(wf_ids.union(leg_ids)))

        lines = """function get_color(ipdg)
        implicit none
        integer get_color, ipdg
        """ 
        for i, part_id in enumerate(particle_ids[:]):
            lines += """%s if(ipdg.eq.%d)then
            get_color=%d
            return
            """ % ('else' if i else '', part_id, model.get_particle(part_id).get_color())
            if abs(part_id) in model['merged_particles']:
                for pdg in model['merged_particles'][abs(part_id)]:
                   lines += """else if(ipdg.eq.%(sign)s%(pdg)d)then
                        get_color=%(sign)s%(color)d
                        return
                        """ % {'sign': '-' if part_id < 0 else '',
                               'pdg': pdg,
                               'color': model.get_particle(part_id).get_color()}   
                  

        # Dummy particle for multiparticle vertices with pdg given by
        # first code not in the model
        lines += """else if(ipdg.eq.%d)then
c           This is dummy particle used in multiparticle vertices
            get_color=2
            return
            """ % model.get_first_non_pdg()
        lines += """else
        write(*,*)'Error: No color given for pdg ',ipdg
        get_color=0        
        return
        endif
        end
        """
        
        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_config_nqcd_file
    #===========================================================================
    def write_config_nqcd_file(self, writer, nqcd_list):
        """Write the config_nqcd.inc with the number of QCD couplings
        for each config"""

        lines = []
        for iconf, n in enumerate(nqcd_list):
            lines.append("data nqcd(%d)/%d/" % (iconf+1, n))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_maxconfigs_file
    #===========================================================================
    def write_maxconfigs_file(self, writer, matrix_elements):
        """Write the maxconfigs.inc file for MadEvent"""

        if isinstance(matrix_elements, helas_objects.HelasMultiProcess):
            maxconfigs = max([me.get_num_configs() for me in \
                              matrix_elements.get('matrix_elements')])
        else:
            maxconfigs = max([me.get_num_configs() for me in matrix_elements])

        lines = "integer lmaxconfigs\n"
        lines += "parameter(lmaxconfigs=%d)" % maxconfigs

        # Write the file
        writer.writelines(lines)

        return True
    
    #===========================================================================
    # read_write_good_hel
    #===========================================================================
    def read_write_good_hel(self, ncomb):
        """return the code to read/write the good_hel common_block"""    

        convert = {'ncomb' : ncomb}
        text = open(pjoin(_file_path, 'iolibs/template_files/matrix_goodhel_helper.inc')).read()
        output = text %   convert
        
        return output
                                
    #===========================================================================
    # write_config_subproc_map_file
    #===========================================================================
    def write_config_subproc_map_file(self, writer, s_and_t_channels):
        """Write a dummy config_subproc.inc file for MadEvent"""

        lines = []

        for iconfig in range(len(s_and_t_channels)):
            lines.append("DATA CONFSUB(1,%d)/1/" % \
                         (iconfig + 1))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, matrix_element):
        """Write the configs.inc file for MadEvent"""

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        model = matrix_element.get('processes')[0].get('model')
        configs = [(i+1, d) for (i, d) in \
                       enumerate(matrix_element.get('diagrams'))]
        mapconfigs = [c[0] for c in configs]
        return mapconfigs, self.write_configs_file_from_diagrams(writer,
                                                            [[c[1]] for c in configs],
                                                            mapconfigs,
                                                            nexternal, ninitial,
                                                            model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_run_config.inc')
        
        if self.proc_characteristic['loop_induced']:
            job_per_chan = 1
        else: 
           job_per_chan = 5
        
        if writer:
            text = open(path).read() % {'chanperjob': job_per_chan} 
            writer.write(text)
            return True
        else:
            return {'chanperjob': job_per_chan} 

    #===========================================================================
    # write_configs_file_from_diagrams
    #===========================================================================
    def write_configs_file_from_diagrams(self, writer, configs, mapconfigs,
                                         nexternal, ninitial, model):
        """Write the actual configs.inc file.
        
        configs is the diagrams corresponding to configs (each
        diagrams is a list of corresponding diagrams for all
        subprocesses, with None if there is no corresponding diagrams
        for a given process).
        mapconfigs gives the diagram number for each config.

        For s-channels, we need to output one PDG for each subprocess in
        the subprocess group, in order to be able to pick the right
        one for multiprocesses."""
        
        lines = []

        s_and_t_channels = []

        nqcd_list = []

        vert_list = [max([d for d in config if d][0].get_vertex_leg_numbers()) \
                       for config in configs if [d for d in config if d][0].\
                                                  get_vertex_leg_numbers()!=[]]
        minvert = min(vert_list) if vert_list!=[] else 0

        # Number of subprocesses
        nsubprocs = len(configs[0])

        nconfigs = 0

        new_pdg = model.get_first_non_pdg()

        for iconfig, helas_diags in enumerate(configs):
            if any([vert > minvert for vert in
                    [d for d in helas_diags if d][0].get_vertex_leg_numbers()]):
                # Only 3-vertices allowed in configs.inc
                continue
            nconfigs += 1

            # Need s- and t-channels for all subprocesses, including
            # those that don't contribute to this config
            empty_verts = []
            stchannels = []
            for h in helas_diags:
                if h:
                    # get_s_and_t_channels gives vertices starting from
                    # final state external particles and working inwards
                    stchannels.append(h.get('amplitudes')[0].\
                                      get_s_and_t_channels(ninitial, model,
                                                           new_pdg))
                else:
                    stchannels.append((empty_verts, None))


            # For t-channels, just need the first non-empty one
            tchannels = [t for s,t in stchannels if t != None][0]
                 
            # pass to ping-pong strategy for t-channel for 3 ore more T-channel
            #  this is directly related to change in genps.f
            tstrat = self.opt.get('t_strategy', 0)
            if isinstance(self, madgraph.loop.loop_exporters.LoopInducedExporterMEGroup):
                tstrat = 2
            tchannels, tchannels_strategy = ProcessExporterFortranME.reorder_tchannels(tchannels, tstrat, self.model)
            
            # For s_and_t_channels (to be used later) use only first config
            s_and_t_channels.append([[s for s,t in stchannels if t != None][0],
                                     tchannels, tchannels_strategy])

            # Make sure empty_verts is same length as real vertices
            if any([s for s,t in stchannels]):
                empty_verts[:] = [None]*max([len(s) for s,t in stchannels])

                # Reorganize s-channel vertices to get a list of all
                # subprocesses for each vertex
                schannels = list(zip(*[s for s,t in stchannels]))
            else:
                schannels = []

            allchannels = schannels
            if len(tchannels) > 1:
                # Write out tchannels only if there are any non-trivial ones
                allchannels = schannels + tchannels

            # Write out propagators for s-channel and t-channel vertices

            lines.append("# Diagram %d" % (mapconfigs[iconfig]))
            # Correspondance between the config and the diagram = amp2
            lines.append("data mapconfig(%d)/%d/" % (nconfigs,
                                                     mapconfigs[iconfig]))
            lines.append("data tstrategy(%d)/%d/" % (nconfigs, tchannels_strategy))
            # Number of QCD couplings in this diagram
            nqcd = 0
            for h in helas_diags:
                if h:
                    try:
                        nqcd = h.calculate_orders()['QCD']
                    except KeyError:
                        pass
                    break
                else:
                    continue

            nqcd_list.append(nqcd)

            for verts in allchannels:
                if verts in schannels:
                    vert = [v for v in verts if v][0]
                else:
                    vert = verts
                daughters = [leg.get('number') for leg in vert.get('legs')[:-1]]
                last_leg = vert.get('legs')[-1]
                lines.append("data (iforest(i,%d,%d),i=1,%d)/%s/" % \
                             (last_leg.get('number'), nconfigs, len(daughters),
                              ",".join([str(d) for d in daughters])))
                if verts in schannels:
                    pdgs = []
                    for v in verts:
                        if v:
                            pdgs.append(v.get('legs')[-1].get('id'))
                        else:
                            pdgs.append(0)
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join([str(d) for d in pdgs])))
                    lines.append("data tprid(%d,%d)/0/" % \
                                 (last_leg.get('number'), nconfigs))
                elif verts in tchannels:
                    lines.append("data tprid(%d,%d)/%d/" % \
                                 (last_leg.get('number'), nconfigs,
                                  abs(last_leg.get('id'))))
                    lines.append("data (sprop(i,%d,%d),i=1,%d)/%s/" % \
                                 (last_leg.get('number'), nconfigs, nsubprocs,
                                  ",".join(['0'] * nsubprocs)))

        # Write out number of configs
        lines.append("# Number of configs")
        lines.append("data mapconfig(0)/%d/" % nconfigs)

        lines.append("#used fake id")
        lines.append("data fake_id/%d/" %new_pdg)

        # Write the file
        writer.writelines(lines)

        return s_and_t_channels, nqcd_list
    


    #===========================================================================
    # reoder t-channels
    #===========================================================================
    
    #ordering = 0    
    @staticmethod
    def reorder_tchannels(tchannels, tstrat, model):
        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        #ProcessExporterFortranME.ordering +=1

        if len(tchannels) < 3 or tstrat == 2 or not model:
            return tchannels, 2
        elif tstrat == 1:
            return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
        elif tstrat == -2:
            return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
        elif tstrat == -1:
            return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels, 1), -1        
        elif len(tchannels) < 4:
            #
            first = tchannels[0]['legs'][1]['number']
            t1 =  tchannels[0]['legs'][-1]['id']
            last = tchannels[-1]['legs'][1]['number']
            t2 = tchannels[-1]['legs'][0]['id']
            m1  = model.get_particle(t1).get('mass') == 'ZERO'
            m2  = model.get_particle(t2).get('mass') == 'ZERO'
            if m2 and not m1:
                return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
            elif m1 and not m2:
                return tchannels, 2
            elif first < last:
                return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
            else:
                return tchannels, 2 
        else:
            first = tchannels[0]['legs'][1]['number']
            t1 =  tchannels[0]['legs'][-1]['id']
            last = tchannels[-1]['legs'][1]['number']
            t2 = tchannels[-1]['legs'][0]['id']
            m1  = model.get_particle(t1).get('mass') == 'ZERO'
            m2  = model.get_particle(t2).get('mass') == 'ZERO'
            
            t12 =  tchannels[1]['legs'][-1]['id']
            m12 = model.get_particle(t12).get('mass') == 'ZERO'
            t22 = tchannels[-2]['legs'][0]['id']
            m22 = model.get_particle(t22).get('mass') == 'ZERO'
            if m2 and not m1:
                if m22:
                    return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
                else:
                    return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
            elif m1 and not m2:
                if m12:
                    return tchannels, 2
                else:
                    return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
            elif m1 and m2 and  len(tchannels) == 4 and not m12: # 3 T propa
                return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2
                # this case seems quite sensitive we tested method 2 specifically and this was not helping in general 
            elif not m1 and not m2 and  len(tchannels) == 4 and m12:
                if first < last:
                    return ProcessExporterFortranME.reorder_tchannels_flipside(tchannels), 1
                return tchannels, 2
            else:
                return ProcessExporterFortranME.reorder_tchannels_pingpong(tchannels), -2


                

    @staticmethod
    def reorder_tchannels_flipside(tchannels):
        """change the tchannel ordering to pass to a ping-pong strategy.
           assume ninitial == 2
        
        We assume that we receive something like this
        
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-1)
                |
                X --------- -1

                X----------  3
                | 
                | (-N+2)
                |                
                X --------- L
                |
                | (-N+1) 
                |                
        -N ----- X ------- P        
        
        coded as 
        (1 -2 > -X) (-X 4 > -X-1) (-X-1 -1 > -X-2) ...
        ((-N+3) 3 > (-N+2)) ((-n+2) L > (-n+1)) ((-n+1) P > -N)
        
        we want to convert this as:
        -N ----- X ------- -2
                |
                | (-N+1) 
                |
                X -------- 4
                | 
                | (-N+2)
                |
                X --------- -1

                X----------  3
                | 
                | (-X-1)
                |                
                X --------- L
                |
                | (-X) 
                |                
        2 ----- X ------- P          
        
        coded as 
        ( 2 P > -X) (-X L > -X-1) (-X-1 3 > -X-2)... (-X-L -2 > -N)
        """
        
        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        if len(tchannels) < 2:
            return tchannels

        out = []
        oldid2new = {}
        
        # initialisation
        # id of the first T-channel (-X)
        propa_id = tchannels[0]['legs'][-1]['number'] 
        #
        # Setup the last vertex to refenence the second id beam
        # -N (need to setup it to 2.
        initialid = tchannels[-1]['legs'][-1]['number']       
        oldid2new[initialid] = 2
        oldid2new[1] = initialid
            
        i = 0 
        while tchannels:
            old_vert = tchannels.pop()
                
            #copy the vertex /leglist to avoid side effects
            new_vert = copy.copy(old_vert)
            new_vert['legs'] = base_objects.LegList([base_objects.Leg(l) for l in old_vert['legs']])
            # vertex taken from the bottom we have 
            # (-N+1 X > -N) we need to flip to pass to 
            # -N X > -N+1 (and then relabel -N and -N+1  
            legs = new_vert['legs'] # shorcut
            id1 = legs[0]['number']
            id2 = legs[1]['number']
            id3 = legs[2]['number']
            # to be secure  we also support (X -N+1 > -N)
            if id3 == id2 -1 and id1 !=1:
                legs[0], legs[1] = legs[1], legs[0]
            #flipping side
            legs[0], legs[2] = legs[2], legs[0]

            # the only new relabelling is the last element of the list
            # always thanks to the above flipping
            old_propa_id = new_vert['legs'][-1]['number'] 
            oldid2new[old_propa_id] = propa_id

            
            #pass to new convention for leg numbering:
            for l in new_vert['legs']:
                if l['number'] in  oldid2new:
                    l['number'] = oldid2new[l['number']]  
                    
            # new_vert is now ready
            out.append(new_vert)
            # prepare next iteration
            propa_id -=1
            i +=1

        return out
    
    @staticmethod
    def reorder_tchannels_pingpong(tchannels, id=2):
        """change the tchannel ordering to pass to a ping-pong strategy.
           assume ninitial == 2
        
        We assume that we receive something like this
        
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-1)
                |
                X --------- -1

                X----------  3
                | 
                | (-N+2)
                |                
                X --------- L
                |
                | (-N+1) 
                |                
        -N ----- X ------- P        
        
        coded as 
        (1 -2 > -X) (-X 4 > -X-1) (-X-1 -1 > -X-2) ...
        ((-N+3) 3 > (-N+2)) ((-n+2) L > (-n+1)) ((-n+1) P > -N)
        
        we want to convert this as:
        1 ----- X ------- -2
                |
                | (-X) 
                |
                X -------- 4
                | 
                | (-X-2)
                |
                X --------- -1

                X----------  3
                | 
                | (-X-3)
                |                
                X --------- L
                |
                | (-X-1) 
                |                
        2 ----- X ------- P          
        
        coded as 
        (1 -2 > -X) (2 P > -X-1) (-X 4 > -X-2) (-X-1 L > -X-3) ...
        """

        # no need to modified anything if 1 or less T-Channel
        #Note that this counts the number of vertex (one more vertex compare to T)
        if len(tchannels) < 2:
            return tchannels

        out = []
        oldid2new = {}
        
        # initialisation
        # id of the first T-channel (-X)
        propa_id = tchannels[0]['legs'][-1]['number'] 
        #
        # Setup the last vertex to refenence the second id beam
        # -N (need to setup it to 2.
        initialid = tchannels[-1]['legs'][-1]['number']       
        oldid2new[initialid] = id


        
        i = 0 
        while tchannels:
            #ping pong by taking first/last element in aternance
            if id ==2:
                if i % 2 == 0:
                    old_vert = tchannels.pop(0)
                else:
                    old_vert = tchannels.pop()
            else:
                if i % 2 != 0:
                    old_vert = tchannels.pop(0)
                else:
                    old_vert = tchannels.pop()
                    
            #copy the vertex /leglist to avoid side effects
            new_vert = base_objects.Vertex(old_vert)
            new_vert['legs'] = base_objects.LegList([base_objects.Leg(l) for l in old_vert['legs']])
            # if vertex taken from the bottom we have 
            # (-N+1 X > -N) we need to flip to pass to 
            # -N X > -N+1 (and then relabel -N and -N+1
            # to be secure  we also support (X -N+1 > -N)
            if (i % 2 ==1 and id ==2) or (i %2 == 0 and id ==1): 
                legs = new_vert['legs'] # shorcut
                id1 = legs[0]['number']
                id2 = legs[1]['number'] 
                if id1 > id2:
                    legs[0], legs[1] = legs[1], legs[0]
                else:
                    legs[0], legs[2] = legs[2], legs[0]
            
            # the only new relabelling is the last element of the list
            # always thanks to the above flipping
            old_propa_id = new_vert['legs'][-1]['number'] 
            oldid2new[old_propa_id] = propa_id

            if i==0 and id==1:
                legs[0]['number'] = 2
            
            #pass to new convention for leg numbering:
            for l in new_vert['legs']:
                if l['number'] in  oldid2new:
                    l['number'] = oldid2new[l['number']]    
            
            # new_vert is now ready
            out.append(new_vert)
            # prepare next iteration
            propa_id -=1
            i +=1

        return out

            
        
        
    
    #===========================================================================
    # write_decayBW_file
    #===========================================================================
    def write_decayBW_file(self, writer, s_and_t_channels):
        """Write the decayBW.inc file for MadEvent"""

        lines = []

        booldict = {None: "0", True: "1", False: "2"}

        for iconf, config in enumerate(s_and_t_channels):
            schannels = config[0]
            for vertex in schannels:
                # For the resulting leg, pick out whether it comes from
                # decay or not, as given by the onshell flag
                leg = vertex.get('legs')[-1]
                lines.append("data gForceBW(%d,%d)/%s/" % \
                             (leg.get('number'), iconf + 1,
                              booldict[leg.get('onshell')]))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_dname_file
    #===========================================================================
    def write_dname_file(self, writer, dir_name):
        """Write the dname.mg file for MG4"""

        line = "DIRNAME=%s" % dir_name

        # Write the file
        writer.write(line + "\n")

        return True

    #===========================================================================
    # write_driver
    #===========================================================================
    def write_driver(self, writer, ncomb, n_grouped_proc, v5=True):
        """Write the SubProcess/driver.f file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_driver.f')
        
        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat'
        # Requiring each helicity configuration to be probed by 10 points for 
        # matrix element before using the resulting grid for MC over helicity
        # sampling.
        # We multiply this by 2 because each grouped subprocess is called at most
        # twice for each IMIRROR.
        replace_dict = {'param_card_name':card, 
                        'ncomb':ncomb,
                        'hel_init_points':n_grouped_proc*10*2}
        if not v5:
            replace_dict['secondparam']=',.true.'
        else:
            replace_dict['secondparam']=''            

        replace_dict['DRIVER_EXTRA_HEADER'] = ""
        replace_dict['DRIVER_EXTRA_INITIALISE'] = ""
        replace_dict['DRIVER_EXTRA_FINALISE'] = ""

        if writer:
            text = open(path).read() % replace_dict
            writer.write(text)
            return True
        else:
            return replace_dict

    def _module_color_flows(self, matrix_element):
        """Return the colour-flow decomposition (leshouche ICOLUP) of an ME as a
        list, one entry per flow, of (colour, anticolour) per leg in leg order.
        None if the ME has no colour basis."""
        if not matrix_element.get('color_basis'):
            return None
        proc = matrix_element.get('processes')[0]
        legs = proc.get_legs_with_decays()
        ninitial = matrix_element.get_nexternal_ninitial()[1]
        repr_dict = {l.get('number'):
                     proc.get('model').get_particle(l.get('id')).get_color()
                     * (-1) ** (1 + l.get('state')) for l in legs}
        # get_flow_basis(): with the DDM color basis the basis elements are
        # products of f's and have no single flow each, so the flows -- and the
        # ICOLUP rows built from them -- come from the trace basis carried
        # alongside, which is also what the JAMP array is indexed by. Without
        # DDM it returns the basis itself.
        flows = matrix_element.get('color_basis').get_flow_basis().\
            color_flow_decomposition(repr_dict, ninitial)
        return [[tuple(cf[l.get('number')]) for l in legs] for cf in flows]

    @staticmethod
    def _color_flow_canon(flow, states):
        """Label-independent canonical form of one colour flow: the set of
        (colour-leg, anticolour-leg) connections, with INITIAL-state legs
        swapping the two roles so that every colour index connects to an
        anticolour index (the LHE convention runs initial-state colour lines
        'through', so without this swap a label can sit in the same slot on two
        legs and the flow is not a bijection). Shared by _router_colmap
        (topology matching) and _color_flow_code."""
        col, anti = {}, {}
        for leg, (c, a) in enumerate(flow):
            if states[leg] is False:
                c, a = a, c
            if c:
                col.setdefault(c, []).append(leg)
            if a:
                anti.setdefault(a, []).append(leg)
        conns = set()
        for lbl in set(list(col) + list(anti)):
            for cc, aa in zip(sorted(col.get(lbl, [])),
                              sorted(anti.get(lbl, []))):
                conns.add((cc, aa))
        return frozenset(conns)

    @staticmethod
    def _color_flow_code(conns):
        """Canonical integer code of a colour flow from its canonical
        connections (see _color_flow_canon).

        Order the colour slots and the anticolour slots by leg -- a gluon holds
        one slot of each kind, a sextet two -- then digit i is the index of the
        anticolour slot that colour slot i connects to, and

            code = sum_i digit_i * N^i          (N = number of anticolour slots)

        This is the colour analogue of the canonical helicity code. It is
        injective over a process's colour basis, and crossing-covariant:
        relabelling the legs with the crossing permutation carries the base
        code onto the crossed process's own code (the initial-state flip is
        what makes the connectivity invariant under a crossing, exactly as the
        conjugate+state flip cancellation does for the helicity). Note the code
        space is N^N while only the basis flows are realised, so -- like the
        helicity allowed-list -- the codes are a sparse subset."""
        ordered = sorted(conns)
        acol = sorted(a for _c, a in conns)
        nslot = len(acol)
        code = 0
        used = set()
        for i, (_c, a) in enumerate(ordered):
            slot = -1
            for j, aa in enumerate(acol):
                if aa == a and j not in used:
                    slot = j
                    break
            if slot < 0:
                return None
            used.add(slot)
            code += slot * (nslot ** i)
        return code

    @staticmethod
    def _color_flow_slots(conns):
        """(colour-slot legs, anticolour-slot legs) of a process, each ordered by
        leg, read off one canonical flow.

        This is FLOW-INDEPENDENT process data -- which legs carry a colour resp.
        anticolour index is fixed by the colour representations (after the
        initial-state flip), not by which flow is picked -- so it is the colour
        analogue of the per-leg helicity-state counts, and it is all a decoder
        needs besides the code itself."""
        return ([c for c, _a in sorted(conns)],
                sorted(a for _c, a in conns))

    @staticmethod
    def _color_flow_decode(code, colslots, acolslots):
        """Inverse of _color_flow_code: rebuild a flow's canonical connections
        from its code and the process's slot structure (see _color_flow_slots).

        digit_i = (code // N^i) %% N is the anticolour slot that colour slot i
        connects to. NOTE: for a leg carrying two slots of the same kind (a
        sextet) encode/decode must agree on the tie-break between its slots;
        that case is untested."""
        nslot = len(acolslots)
        if nslot == 0:
            return frozenset()
        conns = set()
        for i, cleg in enumerate(colslots):
            digit = (code // (nslot ** i)) % nslot
            conns.add((cleg, acolslots[digit]))
        return frozenset(conns)

    def _color_flow_codes(self, matrix_element):
        """Canonical colour-flow codes of an ME, one per colour-basis flow in
        basis order. None if the ME has no colour basis or a flow is not a clean
        colour<->anticolour bijection."""
        flows = self._module_color_flows(matrix_element)
        if not flows:
            return None
        states = [l.get('state') for l in
                  matrix_element.get('processes')[0].get_legs_with_decays()]
        codes = []
        for fl in flows:
            code = self._color_flow_code(self._color_flow_canon(fl, states))
            if code is None:
                return None
            codes.append(code)
        return codes

    def _color_code_tables(self, matrix_element):
        """Per-ME colour tables for the generated fortran, or None if the ME has
        no usable colour code: (codes, colour-slot legs, anticolour-slot legs),
        the two slot lists 1-based so they index the fortran leg arrays.

        This is ALL the colour data an ME needs, and it is per-ME rather than
        per-(base, crossing) pair: the slot structure is flow-independent (see
        _color_flow_slots) and the codes are label-independent, so any crossing
        of this ME reuses the same three arrays."""
        flows = self._module_color_flows(matrix_element)
        if not flows:
            return None
        # A negative tag marks a colour SEXTET (color_flow_decomposition stores
        # it in the opposite slot, so one leg carries two slots of the same
        # kind). The code has no room for that sign, and a decoder rebuilding
        # the tags could not restore it, so leave those to the ICOLUP table.
        if any(c < 0 or a < 0 for fl in flows for c, a in fl):
            return None
        states = [l.get('state') for l in
                  matrix_element.get('processes')[0].get_legs_with_decays()]
        conns = [self._color_flow_canon(fl, states) for fl in flows]
        codes = [self._color_flow_code(c) for c in conns]
        if any(c is None for c in codes) or len(set(codes)) != len(codes):
            return None
        colslots, acolslots = self._color_flow_slots(conns[0])
        if not acolslots:
            return None
        # flow-independence is what lets a single table serve every crossing
        for c in conns[1:]:
            if self._color_flow_slots(c) != (colslots, acolslots):
                return None
        return (codes, [l + 1 for l in colslots], [l + 1 for l in acolslots])

    #===========================================================================
    # get_colorflow_lines / write_colorflow_file
    #===========================================================================
    def get_colorflow_lines(self, matrix_element, numproc):
        """DATA lines of colorflow.inc for one subprocess: the canonical
        colour-flow CODE of each flow plus the slot structure needed to decode
        it (see _color_flow_code / _color_flow_decode).

        addmothers rebuilds the event's colour tags from these instead of
        reading the ICOLUP table, which is why leshouche.inc can drop ICOLUP
        whenever this is emitted. NCOLSLOT is 0 when the ME has no usable code
        (no colour, a sextet, or an epsilon structure); addmothers then falls
        back to ICOLUP, which get_leshouche_lines still writes in that case."""
        tables = self._color_code_tables(matrix_element)
        if not tables:
            return ["DATA NCOLSLOT(%d)/0/" % (numproc + 1)]
        codes, colslots, acolslots = tables
        return [
            "DATA NCOLSLOT(%d)/%d/" % (numproc + 1, len(colslots)),
            "DATA (ICOLCSL(i,%d),i=1,%d)/%s/" % (
                numproc + 1, len(colslots),
                ",".join(str(l) for l in colslots)),
            "DATA (ICOLASL(i,%d),i=1,%d)/%s/" % (
                numproc + 1, len(acolslots),
                ",".join(str(l) for l in acolslots)),
            "DATA (ICOLCODE(i,%d),i=1,%d)/%s/" % (
                numproc + 1, len(codes),
                ",".join(str(c) for c in codes)),
        ]

    def write_colorflow_file(self, writer, matrix_element):
        """Write colorflow.inc for a single (non-grouped) subprocess."""
        writer.writelines(self.get_colorflow_lines(matrix_element, 0))
        return True

    def write_leshouche_file(self, writer, matrix_element):
        """Write leshouche.inc, without the ICOLUP table when the colour code
        can supply the tags (see get_colorflow_lines)."""
        writer.writelines(self.get_leshouche_lines(matrix_element, 0,
                                                   drop_icolup=True))
        return True

    #===========================================================================
    # write_addmothers
    #===========================================================================
    def write_addmothers(self, writer):
        """Write the SubProcess/addmothers.f"""

        path = pjoin(_file_path,'iolibs','template_files','addmothers.f')

        text = open(path).read() % {'iconfig': 'diag_number'}
        writer.write(text)
        
        return True


    #===========================================================================
    # write_combine_events
    #===========================================================================
    def write_combine_events(self, writer, nb_proc=100):
        """Write the SubProcess/driver.f file for MG4"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_combine_events.f')
        
        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat' 
        
        #set maxpup (number of @X in the process card)
            
        text = misc.apply_template(open(path).read(),
                                   {'param_card_name':card, 'maxpup':nb_proc+1})
        #the +1 is just a security. This is not needed but I feel(OM) safer with it.
        writer.write(text)

        return True


    #===========================================================================
    # write_symmetry
    #===========================================================================
    def write_symmetry(self, writer, v5=True):
        """Write the SubProcess/driver.f file for ME"""

        
        path = pjoin(_file_path,'iolibs','template_files','madevent_symmetry.f')

        if self.model_name == 'mssm' or self.model_name.startswith('mssm-'):
            card = 'Source/MODEL/MG5_param.dat'
        else:
            card = 'param_card.dat' 
        
        if v5:
            replace_dict = {'param_card_name':card, 'setparasecondarg':''}      
        else:
            replace_dict= {'param_card_name':card, 'setparasecondarg':',.true.'} 
        
        if writer:
            text = open(path).read() 
            text = text % replace_dict
            writer.write(text)
            return True
        else:
            return replace_dict



    #===========================================================================
    # write_iproc_file
    #===========================================================================
    def write_iproc_file(self, writer, me_number):
        """Write the iproc.dat file for MG4"""
        line = "%d" % (me_number + 1)

        # Write the file
        for line_to_write in writer.write_line(line):
            writer.write(line_to_write)
        return True

    #===========================================================================
    # write_mg_sym_file
    #===========================================================================
    def write_mg_sym_file(self, writer, matrix_element):
        """Write the mg.sym file for MadEvent."""

        lines = []

        # Extract process with all decays included
        final_legs = [leg for leg in matrix_element.get('processes')[0].get_legs_with_decays() if leg.get('state') == True]

        ninitial = len([leg for leg in matrix_element.get('processes')[0].get('legs') if leg.get('state') == False])

        identical_indices = {}

        # Extract identical particle info
        for i, leg in enumerate(final_legs):
            if leg.get('id') in identical_indices:
                identical_indices[leg.get('id')].append(\
                                    i + ninitial + 1)
            else:
                identical_indices[leg.get('id')] = [i + ninitial + 1]

        # Remove keys which have only one particle
        for key in list(identical_indices.keys()):
            if len(identical_indices[key]) < 2:
                del identical_indices[key]

        # Write mg.sym file
        lines.append(str(len(list(identical_indices.keys()))))
        for key in identical_indices.keys():
            lines.append(str(len(identical_indices[key])))
            for number in identical_indices[key]:
                lines.append(str(number))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_mg_sym_file
    #===========================================================================
    def write_default_mg_sym_file(self, writer):
        """Write the mg.sym file for MadEvent."""

        lines = "0"

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_ncombs_file
    #===========================================================================
    def write_ncombs_file(self, writer, nexternal):
        """Write the ncombs.inc file for MadEvent."""

        # ncomb (used for clustering) is 2^nexternal
        file = "       integer    n_max_cl\n"
        file = file + "parameter (n_max_cl=%d)" % (2 ** nexternal)

        # Write the file
        writer.writelines(file)

        return True

    #===========================================================================
    # write_processes_file
    #===========================================================================
    def write_processes_file(self, writer, subproc_group):
        """Write the processes.dat file with info about the subprocesses
        in this group."""

        lines = []

        for ime, me in \
            enumerate(subproc_group.get('matrix_elements')):
            lines.append("%s %s" % (str(ime+1) + " " * (7-len(str(ime+1))),
                                    ",".join(p.base_string() for p in \
                                             me.get('processes'))))
            if me.get('has_mirror_process'):
                mirror_procs = [copy.copy(p) for p in me.get('processes')]
                for proc in mirror_procs:
                    legs = copy.copy(proc.get('legs_with_decays'))
                    legs.insert(0, legs.pop(1))
                    proc.set("legs_with_decays", legs)
                lines.append("mirror  %s" % ",".join(p.base_string() for p in \
                                                     mirror_procs))
            else:
                lines.append("mirror  none")

        # Write the file
        writer.write("\n".join(lines))

        return True

    #===========================================================================
    # write_symswap_file
    #===========================================================================
    def write_symswap_file(self, writer, ident_perms):
        """Write the file symswap.inc for MG4 by comparing diagrams using
        the internal matrix element value functionality."""

        lines = []

        # Write out lines for symswap.inc file (used to permute the
        # external leg momenta
        for iperm, perm in enumerate(ident_perms):
            lines.append("data (isym(i,%d),i=1,nexternal)/%s/" % \
                         (iperm+1, ",".join([str(i+1) for i in perm])))
        lines.append("data nsym/%d/" % len(ident_perms))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_symfact_file
    #===========================================================================
    def write_symfact_file(self, writer, symmetry):
        """Write the files symfact.dat for MG4 by comparing diagrams using
        the internal matrix element value functionality."""

        pos = max(2, int(math.ceil(math.log10(len(symmetry)))))
        form = "%"+str(pos)+"r %"+str(pos+1)+"r"
        # Write out lines for symswap.inc file (used to permute the
        # external leg momenta
        lines = [ form %(i+1, s) for i,s in enumerate(symmetry) if s != 0] 
        # Write the file
        writer.write('\n'.join(lines))
        writer.write('\n')

        return True

    #===========================================================================
    # write_symperms_file
    #===========================================================================
    def write_symperms_file(self, writer, perms):
        """Write the symperms.inc file for subprocess group, used for
        symmetric configurations"""

        lines = []
        for iperm, perm in enumerate(perms):
            lines.append("data (perms(i,%d),i=1,nexternal)/%s/" % \
                         (iperm+1, ",".join([str(i+1) for i in perm])))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_subproc
    #===========================================================================
    def write_subproc(self, writer, subprocdir):
        """Append this subprocess to the subproc.mg file for MG4"""

        # Write line to file
        writer.write(subprocdir + "\n")

        return True

#===============================================================================
# ProcessExporterFortranMEGroup
#===============================================================================
class ProcessExporterFortranMEGroup(ProcessExporterFortranME):
    """Class to take care of exporting a set of matrix elements to
    MadEvent subprocess group format."""


    # the color sum uses the DDM basis, the color flows the trace one
    support_ddm_color_basis = True
    ddm_needs_flow_basis = True

    matrix_file = "matrix_madevent_group_v4.inc"
    grouped_mode = 'madevent'
    # The group SMATRIX decodes an extended FLAV_IDX (M0) and the router lets
    # crossed subprocesses share a base's matrix element, so this exporter can
    # honour --use_crossing (the _check_crossing_support gate lets it through).
    supports_crossing = True
    default_opt = {'clean': False, 'complex_mass':False,
                        'export_format':'madevent', 'mp': False,
                        'v5_model': True,
                        'output_options':{},
                        'hel_recycling': True
                        }


    #===========================================================================
    # write_matrix_router_file
    #===========================================================================
    def _router_colmap(self, router_me, base_me, cross):
        """Map each base colour-flow index to this subprocess's flow index.

        The base picks a colour flow in its own basis and events are written
        through this subprocess's ICOLUP, whose flow ORDER can differ (the
        crossed colour reps decompose the shared colour basis in another order).
        Crossing a base flow (leg j <- base flow leg perm^-1(j), colour <->
        anticolour when that leg swapped initial/final) gives the physical flow;
        it is matched to the local flow of the same topology (label independent).
        Returns a 1-based list indexed by the base flow; identity if unmatchable.
        """
        bflows = self._module_color_flows(base_me)
        rflows = self._module_color_flows(router_me)
        if not bflows or not rflows or len(bflows) != len(rflows):
            return list(range(1, len(rflows or []) + 1))
        nx = router_me.get_nexternal_ninitial()[0]
        rstates = [l.get('state') for l in
                   router_me.get('processes')[0].get_legs_with_decays()]
        perm, ic, _valid = self.get_crossing_permutation(cross, nx)
        inv = [0] * nx
        for s, leg in enumerate(perm):
            inv[leg] = s

        def canon(flow):
            # Topology (label independent), shared with the colour-flow code.
            return self._color_flow_canon(flow, rstates)

        rindex = {}
        for j, fl in enumerate(rflows):
            rindex.setdefault(canon(fl), j + 1)
        colmap = []
        for icol, bf in enumerate(bflows):
            crossed = []
            for j in range(nx):
                c, a = bf[inv[j]]
                if ic[inv[j]] == -1:
                    c, a = a, c
                crossed.append((c, a))
            colmap.append(rindex.get(canon(crossed), icol + 1))
        return colmap

    def write_matrix_router_file(self, writer, matrix_element, fortran_model,
                                 proc_id="", config_map=[], subproc_number="",
                                 routing=None, matrix_elements=None):
        """Write a light matrix<i>.f for a crossed subprocess that shares a base
        subprocess's matrix element. It keeps only GET_FLAVOR<i> (for the PDF)
        and a router SMATRIX<i> that, per flavor, calls the base SMATRIX with the
        crossed FLAV_IDX from partition_crossing_classes; the heavy MATRIX<i> is
        not emitted. get_nhel<i> lives in auto_dsig<i>.f, so it is unaffected.

        Colour is NOT taken from the base's own selection. The base SMATRIX picks
        its flow with SELECT_COLOR masked by the BASE's ICOLAMP row -- a different
        subprocess's row, which at the live ICONFIG generally allows a different
        set of flows -- so relabelling that index into this subprocess's flow
        order (whatever the relabel) can hand the event a topology this
        subprocess's own SELECT_COLOR would never pick, and the crossing-off
        build never produces. Reselect natively instead, exactly as the
        cross-group path does: permute the base's published per-flow JAMP2
        (COMMON/TO_XG_JAMP2, base flow order -- crossing-covariant, so these are
        this subprocess's own per-flow weights) into this subprocess's flow order
        and run SELECT_COLOR with THIS subprocess's proc_id as IPROC
        (_crossgroup_colsel_helper, emitted into this file as XG_SELCOL<i>).

        The base flow -> this subprocess's flow permutation is _router_colmap;
        when it is not a usable bijection the reselect is skipped and the old
        index relabel through the canonical colour-flow CODE is kept (decode the
        base's code, relabel the legs with the crossing permutation, re-encode
        and look it up in this subprocess's own code table, see _color_flow_code),
        with the explicit COLMAP array as the last resort.

        Momenta, PDGs and the helicity index already come out in this
        subprocess's own convention."""
        # Reuse the full builder (writer=None) to get the flavor table and the
        # info/process/nexternal/max_flavor holes; nothing heavy is written.
        replace_dict = self.write_matrix_element_v4(
            None, matrix_element, fortran_model, proc_id=proc_id,
            config_map=config_map, subproc_number=subproc_number)
        dispatch = []
        decl = []
        # Shared temporaries for the runtime helicity encode below. The base
        # returns its selected helicity as ITS canonical code; the event is
        # written through THIS module's get_nhel<i>, which decodes THIS
        # (crossed) module's code -- so relabel by permuting the code's
        # mixed-radix digits with the crossing permutation (GET_CROSS_PERM),
        # exactly the dependent-vs-base relation dep_states[k]==base_states[PERM[k]].
        encode_used = False
        col_used = False
        baked_nhs = {}   # base_index -> baked base-NHSTATE array name
        baked_col = {}   # base_index -> baked base colour table names
        dep_col = self._color_code_tables(matrix_element)
        # Multi-channel config remap. CHANNEL arrives as THIS subprocess's AMP2
        # slot (SUBDIAG = CONFSUB(<this proc>, iconf), a diagram number in this
        # module's numbering), but the base SMATRIX enhances AMP2(CHANNEL) in
        # ITS numbering: AMP2 is filled by the BASE's diagrams evaluated at the
        # CROSSED momenta, so the slot holding |this subprocess's diagram m|^2 is
        # the base diagram carrying m's topology *under the crossing*. Translate
        # the slot with the same map the cross-group path uses for DSIG_XGCONFIG
        # -- and for the same reason; walking the base's own CONFSUB row instead
        # would name the base diagram that shares the topology with legs left in
        # place, which is not the one the crossed momenta filled. Per flavor,
        # since each routes to its own base/crossing. Emitted only when some
        # flavor is non-identity (it usually is not, the diagram numbering being
        # largely crossing-covariant), so most routers are unchanged.
        #
        # The base applies the same map to its multi-channel row (xgrow_map in
        # fill_crossing_replace_dict_me) and accepts it only as a permutation of
        # ITS diagrams, so a base with a different diagram count is left alone on
        # both sides -- crossing partners always have the same count, this only
        # keeps the two ends from disagreeing.
        ngraphs = len(matrix_element.get('diagrams'))
        ident_cfg = list(range(1, ngraphs + 1))
        cfg_cache = {}   # (base_index, cross) -> map; flavors often share one
        configmap = []
        for (b, iflav) in routing:
            key = (b, (iflav - 1) // len(matrix_elements[b]
                                         .get_external_flavors_with_iden()))
            if key not in cfg_cache:
                cmap = self._crossgroup_configmap(
                    matrix_element, matrix_elements[b], key[1])
                if len(matrix_elements[b].get('diagrams')) != ngraphs:
                    cmap = ident_cfg
                cfg_cache[key] = cmap
            configmap.append(cfg_cache[key])
        chan_name = None
        if any(cm != ident_cfg for cm in configmap):
            chan_name = 'XGCONF_%s' % proc_id
            decl.append('      INTEGER XCHAN')
            decl.append('      INTEGER %s(%d,%d)'
                        % (chan_name, ngraphs, len(configmap)))
            decl.append('      DATA %s /%s/' % (
                chan_name, ','.join(str(x) for col in configmap for x in col)))
        # Per-flavor base-flow -> this-subprocess-flow permutation, and whether it
        # supports the native colour reselect (see the docstring). It does when
        # every flavor's map is a bijection of the shared colour basis, which also
        # says the two flow spaces have the same size -- so the base's published
        # JAMP2 fills this subprocess's JD exactly. Anything else (a flow that is
        # not a clean colour<->anticolour bijection, an unmatchable topology, a
        # colourless ME with no flows at all) keeps the historical index relabel.
        ncol_dep = max(1, len(matrix_element.get('color_basis')))
        colmaps = []
        col_native = bool(routing)
        for (base_index, iflav) in routing:
            base_me = matrix_elements[base_index]
            cm = self._router_colmap(
                matrix_element, base_me,
                (iflav - 1) // len(base_me.get_external_flavors_with_iden()))
            colmaps.append(cm)
            if len(cm) != ncol_dep \
                    or ncol_dep != max(1, len(base_me.get('color_basis'))) \
                    or sorted(cm) != list(range(1, ncol_dep + 1)):
                col_native = False
        if col_native:
            # One helper for the whole router; the DATA is column-major (base flow
            # fastest, then flavor), matching _crossgroup_colsel_helper.
            replace_dict['smatrix_router_helper'] = self._crossgroup_colsel_helper(
                proc_id, ncol_dep, len(colmaps),
                ','.join(str(x) for cm in colmaps for x in cm),
                iproc=proc_id)
        for flav0, (base_index, iflav) in enumerate(routing):
            base_me = matrix_elements[base_index]
            nflav_base = len(base_me.get_external_flavors_with_iden())
            cross = (iflav - 1) // nflav_base
            colmap = colmaps[flav0]
            kw = 'IF' if flav0 == 0 else 'ELSE IF'
            dispatch.append('      %s (IFLAV.EQ.%d) THEN' % (kw, flav0 + 1))
            chan = 'channel'
            if chan_name:
                # A config this subprocess has no diagram for gives CHANNEL=0;
                # leave it alone (the base's own multi-channel block already
                # handles what it gets) rather than indexing outside the table.
                chan = 'XCHAN'
                dispatch += [
                    '        XCHAN = channel',
                    '        IF (channel.GE.1.AND.channel.LE.%d) XCHAN ='
                    ' %s(channel,%d)' % (ngraphs, chan_name, flav0 + 1)]
            dispatch.append(
                '        CALL SMATRIX%d(P, %d, RHEL, RCOL, %s, IVEC, ANS,'
                ' IHEL, ICOL)' % (base_index + 1, iflav, chan))
            perm_called = False
            # Encode the crossed helicity code (skip cross 0 = identity).
            if cross != 0:
                encode_used = True
                perm_called = True
                if base_index not in baked_nhs:
                    nsname = 'XNHS%d' % (base_index + 1)
                    nhstate = [len(s) for s in
                               base_me.get_helicity_per_particle()]
                    decl.append('      INTEGER %s(NEXTERNAL)' % nsname)
                    decl.append('      DATA %s /%s/' % (
                        nsname, ','.join(str(n) for n in nhstate)))
                    baked_nhs[base_index] = nsname
                nsname = baked_nhs[base_index]
                dispatch += [
                    '        CALL CR%d_GET_CROSS_PERM(%d, XPERM, XSGN, XDUMF)'
                    % (base_index + 1, iflav),
                    '        XHR = IHEL - 1',
                    '        DO XHK=NEXTERNAL,1,-1',
                    '          XBDIG(XHK) = MOD(XHR, %s(XHK))' % nsname,
                    '          XHR = XHR / %s(XHK)' % nsname,
                    '        ENDDO',
                    '        IHEL = 0',
                    '        DO XHK=1,NEXTERNAL',
                    '          IHEL = IHEL * %s(XPERM(XHK)) + XBDIG(XPERM(XHK))'
                    % nsname,
                    '        ENDDO',
                    '        IHEL = IHEL + 1',
                ]
            if col_native:
                # Discard the base's ICOL entirely and reselect in this
                # subprocess's own flow space, with its own ICOLAMP row -- the
                # base's pick was masked with the base's row and can name a flow
                # this subprocess would never emit. Unconditional: an identity
                # colmap only says the two flow ORDERS agree, it says nothing
                # about the two masks, and it is precisely the identity-colmap
                # routers whose masks were found to disagree.
                dispatch.append('        CALL XG_SELCOL%s(RCOL, %d, IVEC, ICOL)'
                                % (proc_id, flav0 + 1))
                continue
            # Fallback: no usable per-flavor bijection, so keep the historical
            # index relabel of the base's own selection. Skip it when the orders
            # already agree (identity map).
            if not (colmap and colmap != list(range(1, len(colmap) + 1))):
                continue
            base_col = self._color_code_tables(base_me)
            if (dep_col and base_col
                    and len(base_col[1]) == len(dep_col[1])
                    and len(base_col[2]) == len(dep_col[2])):
                # Canonical route: translate through the colour-flow CODE.
                # Decode the base's code into its connections, relabel the legs
                # with the crossing permutation, re-encode in this subprocess's
                # slot order and look the result up in its own code table. The
                # tables are per-ME (shared by every crossing of the same base),
                # where COLMAP was one array per base-flavor pair.
                col_used = True
                if base_index not in baked_col:
                    bcode, bcs, bas = base_col
                    names = ('XCCD%d' % (base_index + 1),
                             'XCCS%d' % (base_index + 1),
                             'XCAS%d' % (base_index + 1))
                    for nm, vals in zip(names, (bcode, bcs, bas)):
                        decl.append('      INTEGER %s(%d)' % (nm, len(vals)))
                        decl.append('      DATA %s /%s/' % (
                            nm, ','.join(str(x) for x in vals)))
                    baked_col[base_index] = names
                cdn, csn, asn = baked_col[base_index]
                ns = len(dep_col[1])
                if not perm_called:
                    dispatch.append(
                        '        CALL CR%d_GET_CROSS_PERM(%d, XPERM, XSGN,'
                        ' XDUMF)' % (base_index + 1, iflav))
                    encode_used = True
                dispatch += [
                    '        IF (ICOL.GE.1.AND.ICOL.LE.%d) THEN' % len(colmap),
                    '          XCBAS = %s(ICOL)' % cdn,
                    '          XCNEW = 0',
                    '          DO XCI=1,%d' % ns,
                    '            XCL = XPERM(XDCS(XCI))',
                    '            XCJ = 1',
                    '            DO XCK=1,%d' % ns,
                    '              IF (%s(XCK).EQ.XCL) XCJ = XCK' % csn,
                    '            ENDDO',
                    '            XCD = MOD(XCBAS / %d**(XCJ-1), %d)' % (ns, ns),
                    '            XCL = XPERM(%s(XCD+1))' % asn,
                    '            DO XCK=1,%d' % ns,
                    '              IF (XDAS(XCK).EQ.XCL) XCD = XCK-1',
                    '            ENDDO',
                    '            XCNEW = XCNEW + XCD * %d**(XCI-1)' % ns,
                    '          ENDDO',
                    '          DO XCK=1,%d' % len(dep_col[0]),
                    '            IF (XDCD(XCK).EQ.XCNEW) ICOL = XCK',
                    '          ENDDO',
                    '        ENDIF',
                ]
            else:
                # No usable code (no colour basis, or a flow that is not a
                # clean colour<->anticolour bijection): keep the explicit map.
                cname = 'COLMAP_%s_%d' % (proc_id, flav0 + 1)
                decl.append('      INTEGER %s(%d)' % (cname, len(colmap)))
                decl.append('      DATA %s /%s/' % (
                    cname, ','.join(str(x) for x in colmap)))
                dispatch.append('        IF (ICOL.GE.1.AND.ICOL.LE.%d)'
                                ' ICOL = %s(ICOL)' % (len(colmap), cname))
        if dispatch:
            dispatch.append('      ENDIF')
        if col_used:
            dcode, dcs, das = dep_col
            for nm, vals in (('XDCD', dcode), ('XDCS', dcs), ('XDAS', das)):
                decl = ['      INTEGER %s(%d)' % (nm, len(vals)),
                        '      DATA %s /%s/' % (
                            nm, ','.join(str(x) for x in vals))] + decl
            decl = ['      INTEGER XCI, XCJ, XCK, XCD, XCL, XCNEW, XCBAS'] \
                + decl
        if encode_used:
            decl = ['      INTEGER XPERM(NEXTERNAL), XSGN(NEXTERNAL), XDUMF',
                    '      INTEGER XBDIG(NEXTERNAL), XHR, XHK'] + decl
        replace_dict['smatrix_router_decl'] = '\n'.join(decl)
        replace_dict['smatrix_router_dispatch'] = '\n'.join(dispatch)
        replace_dict.setdefault('smatrix_router_helper', '')
        tpl = open(pjoin(_file_path, 'iolibs', 'template_files',
                         'matrix_madevent_group_router_v4.inc')).read()
        writer.writelines(misc.apply_template(tpl, replace_dict))
        # Router adds no new matrix-element calls; report the module's own color
        # count so the group's maxflow sizing stays an upper bound.
        calls, ncolor = replace_dict['return_value']
        return 0, ncolor

    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model,
                                         group_number,
                                         second_exporter=None,
                                         second_helas=None):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files."""

        assert isinstance(subproc_group, group_subprocs.SubProcessGroup), \
                                      "subproc_group object not SubProcessGroup"
        


        if not self.model:
            self.model = subproc_group.get('matrix_elements')[0].\
                         get('processes')[0].get('model')


        cwd = os.getcwd()
        path = pjoin(self.dir_path, 'SubProcesses')
        
        os.chdir(path)
        pathdir = os.getcwd()

        # Create the directory PN in the specified path
        subprocdir = "P%d_%s" % (subproc_group.get('number'),
                                 subproc_group.get('name'))
        try:
            os.mkdir(subprocdir)
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)

        try:
            os.chdir(subprocdir)
        except os.error:
            logger.error('Could not cd to directory %s' % subprocdir)
            return 0
        logger.info('Creating files in directory %s' % subprocdir)

        # Create the matrix.f files, auto_dsig.f files and all inc files
        # for all subprocesses in the group

        maxamps = 0
        maxflows = 0
        tot_calls = 0

        matrix_elements = subproc_group.get('matrix_elements')



        # Add the driver.f, all grouped ME's must share the same number of 
        # helicity configuration
        ncomb = matrix_elements[0].get_helicity_combinations()
        for me in matrix_elements[1:]:
            if ncomb!=me.get_helicity_combinations():
                raise MadGraph5Error("All grouped processes must share the "+\
                                       "same number of helicity configurations.")                

        filename = 'driver.f'
        self.write_driver(writers.FortranWriter(filename),ncomb,
                                  n_grouped_proc=len(matrix_elements), v5=self.opt['v5_model'])

        try:
            self.proc_characteristic['hel_recycling'] = self.opt['hel_recycling']
        except KeyError:
            self.proc_characteristic['hel_recycling'] = False
            self.opt['hel_recycling'] = False

        # Crossing merge: partition the group's matrix elements so that a base
        # subprocess keeps its own (crossing-aware) matrix element and the others
        # -- whose every flavor is a crossing of a base flavor -- get only a
        # light router matrix<i>.f that dispatches to the base SMATRIX with the
        # crossed FLAV_IDX (see partition_crossing_classes / the router template).
        group_use_crossing = (
            self.opt.get('use_crossing', False)
            and not any(self.breaks_crossing_symmetry(proc)
                        for me in matrix_elements
                        for proc in me.get('processes')))
        if group_use_crossing:
            crossing_bases, crossing_routing = \
                self.partition_crossing_classes(matrix_elements)
            crossing_bases = set(crossing_bases)
            # A base that actually serves a router must publish its per-flow JAMP2
            # (COMMON/TO_XG_JAMP2), so the router can reselect colour with its OWN
            # ICOLAMP row instead of relabelling the base's masked pick -- see the
            # XG_SELCOL call in write_matrix_router_file. Recorded before the write
            # loop below, since a base can be written before or after its routers.
            # Deliberately NOT merged into _crossgroup_base_mes: that set also
            # drives the Track-B-only XGROW multi-channel row, which a within-group
            # base must not get.
            router_bases = getattr(self, '_router_base_mes', None)
            if router_bases is None:
                router_bases = self._router_base_mes = set()
            for idep, route in enumerate(crossing_routing or []):
                if route is None or idep in crossing_bases:
                    continue
                for (base_index, _iflav) in route:
                    router_bases.add(id(matrix_elements[base_index]))
            # Flag the run interface that this output relies on crossing: a shared
            # matrix element is reused across physically distinct (crossed) initial
            # states. That is fine for the unpolarised proton PDFs, but it is NOT
            # compatible with per-beam polarisation or the EVA luminosity, which
            # depend on the actual beam particle. Tag the limitation only when
            # crossing is materially applied (a router, or a base that evaluates a
            # cross>0 flavor), so ordinary polarised runs are not blocked for
            # nothing. check_card_consistency turns this into a clear error.
            crossing_applied = len(crossing_bases) < len(matrix_elements) or any(
                (iflav - 1) // len(matrix_elements[base_index]
                                   .get_external_flavors_with_iden()) > 0
                for route in crossing_routing if route is not None
                for (base_index, iflav) in route)
            if crossing_applied and \
               'crossing' not in self.proc_characteristic['limitations']:
                self.proc_characteristic['limitations'].append('crossing')
            # Record each router's base->base helicity SIGN map tau, exactly as a
            # cross-group dependent does (crossgroup_helunion.dat). A router sends
            # its call into the base SMATRIX, and with helicity recycling that is
            # the RECYCLED matrix<b>_optim.f, whose helicity configs are baked
            # into the HELAS calls -- it takes no runtime NHEL, so it cannot apply
            # the crossing's slot PERMUTATION the way matrix<b>_orig.f does
            # (CR<b>_APPLY_CROSSING_TABLE permutes NHEL along with the momenta).
            # It can only apply the NSF sign flips, through IC. tau is exactly
            # that residual transform, and optim row hb is non-zero for the
            # crossing iff tau[hb] is good for the base -- so the base's own
            # good-hel SUBSET is not closed under it, and a pruned optim silently
            # drops part of the routed process's helicity sum. gen_ximprove bakes
            # the optim over G_base U tau(G_base) from these lines (and skips the
            # C-parity de-duplication, whose |M|^2 identity is only established
            # for cross 0), which is what the Track B path already does.
            for idep, route in enumerate(crossing_routing or []):
                if route is None or idep in crossing_bases:
                    continue
                for (base_index, iflav) in route:
                    base_me = matrix_elements[base_index]
                    nflav_base = len(base_me.get_external_flavors_with_iden())
                    pi = self._crossgroup_base_helsignmap(
                        base_me, (iflav - 1) // nflav_base)
                    if pi is None:
                        # Not a clean permutation (the crossed legs' helicity
                        # states are not closed under negation).
                        # matrix<b>_orig.f has a run-time escape for that --
                        # GHIDX=0 makes it compute every helicity -- but the
                        # recycled optim is baked and has none, and we cannot say
                        # which configs the router needs. The all-zero row is the
                        # keep-every-config sentinel gen_ximprove understands.
                        pi = [0] * base_me.get_helicity_combinations()
                    # An identity tau (the crossing moves no leg between the
                    # initial and the final state) needs no extra config, but the
                    # line is still written: a non-empty perms list is also what
                    # marks this matrix element as shared by a crossing, which
                    # gen_ximprove needs to keep the C-parity de-duplication off.
                    perms = self._crossgroup_helperms.setdefault(
                        subprocdir, {}).setdefault(base_index + 1, [])
                    if pi not in perms:
                        perms.append(pi)
        else:
            crossing_bases, crossing_routing = None, None
        # Per base: {crossing -> (dependent proc_id, dep-diagram -> base-diagram
        # map)}. The base's multi-channel loop needs both to weight a routed call
        # correctly -- the row says which configs to enumerate (they pair with
        # GET_CHANNEL_CUT on the dependent's momenta), the map turns each of that
        # subprocess's diagrams into the AMP2 slot the crossed evaluation filled.
        # See fill_crossing_replace_dict_me.
        base_xgrow = {}
        if crossing_routing is not None:
            cfg_cache = {}
            for idep, route in enumerate(crossing_routing):
                if route is None or idep in crossing_bases:
                    continue
                for (base_index, iflav) in route:
                    base_me = matrix_elements[base_index]
                    cross = (iflav - 1) // len(
                        base_me.get_external_flavors_with_iden())
                    if not cross:
                        continue
                    key = (idep, base_index, cross)
                    if key not in cfg_cache:
                        cfg_cache[key] = self._crossgroup_configmap(
                            matrix_elements[idep], base_me, cross)
                    base_xgrow.setdefault(base_index, {})[cross] = (
                        idep + 1, cfg_cache[key])

        def _xgrow_kw(ime):
            """Crossing kwargs for this subprocess, or nothing at all.

            Only a Track-A base that actually has a crossed subprocess routed to
            it needs the multi-channel row map. Everything else goes through an
            exporter whose write_matrix_element_v4 does not take the crossing
            kwargs -- notably the loop-induced one, and a loop-induced matrix
            element never crosses anyway (see the perturbative gate in
            generate_matrix_elements) -- so handing it the kwarg is a TypeError.
            """
            xg = base_xgrow.get(ime)
            return {'xgrow_map': xg} if xg else {}

        for ime, matrix_element in \
                enumerate(matrix_elements):
            crossgroup = self._crossgroup.get((group_number, ime))
            if crossgroup is not None:
                # Cross-group dependent (Track B): this subprocess's matrix
                # element is a crossing of a base group's, in another P directory.
                # It generates NO matrix element of its own -- it symlinks the
                # base group's compiled crossing-aware SMATRIX (built once there)
                # and its auto_dsig routes to it with the crossed FLAV_IDX. Only
                # the flavor table (for the PDF) and phase space stay local.
                for fname in self._crossgroup_base_files(crossgroup['base_proc_id']):
                    ln(pjoin('..', crossgroup['base_dir'], fname), log=False)
                # Reuse the base group's COMPILED objects (do not recompile the
                # symlinked source): crossgroup.mk (included by the shared makefile)
                # symlinks matrix<b>_{orig,optim}.o from the base dir, building them
                # there first. Also record the dir pair for the parallel top-level
                # makefile written at finalize.
                self.write_crossgroup_mk(crossgroup['base_dir'],
                                         crossgroup['base_proc_id'])
                self._crossgroup_dirs.append((subprocdir, crossgroup['base_dir']))
                # Record this dependent's base->base helicity SIGN map(s) tau so
                # the base optim can be baked over G_base U tau(G_base) and
                # shared. tau, not the GHREMAP sigma: the recycled optim gets only
                # (PUSE, IC) and so realises the sign flips without the slot
                # permutation -- see _crossgroup_base_helsignmap.
                base_me = crossgroup['base_me']
                nflav_base = len(base_me.get_external_flavors_with_iden())
                perms = self._crossgroup_helperms.setdefault(
                    crossgroup['base_dir'], {}).setdefault(
                    crossgroup['base_proc_id'], [])
                for iflav in crossgroup['flav_idx']:
                    pi = self._crossgroup_base_helsignmap(
                        base_me, (iflav - 1) // nflav_base)
                    if pi is None:
                        # Keep-every-config sentinel, as in the router branch.
                        pi = [0] * base_me.get_helicity_combinations()
                    # An identity tau adds no config, but the line still marks
                    # the base as crossing-shared for gen_ximprove.
                    if pi not in perms:
                        perms.append(pi)
                # ncolor for maxflow sizing: crossing preserves the colour basis,
                # so the dependent's own count is the base's. writer=None writes
                # nothing, it only returns the flavor/colour bookkeeping.
                rd = self.write_matrix_element_v4(
                    None, matrix_element, fortran_model, proc_id=str(ime+1),
                    config_map=subproc_group.get('diagram_maps')[ime],
                    subproc_number=group_number)
                calls, ncolor = 0, rd['return_value'][1]
            elif crossing_routing is not None and ime not in crossing_bases:
                # A router shares a base's matrix element and holds no helicities
                # to recycle. Name it matrix<i>_router.f so the makefile globs it
                # into both build targets while gen_ximprove (which recycles
                # matrix*_orig.f) leaves it alone.
                filename = 'matrix%d_router.f' % (ime+1)
                calls, ncolor = self.write_matrix_router_file(
                    writers.FortranWriter(filename), matrix_element,
                    fortran_model, proc_id=str(ime+1),
                    config_map=subproc_group.get('diagram_maps')[ime],
                    subproc_number=group_number,
                    routing=crossing_routing[ime],
                    matrix_elements=matrix_elements)
            elif self.opt['hel_recycling']:
                filename = 'matrix%d_orig.f' % (ime+1)
                replace_dict = self.write_matrix_element_v4(None,
                                matrix_element,
                                fortran_model,
                                proc_id=str(ime+1),
                                config_map=subproc_group.get('diagram_maps')[ime],
                                subproc_number=group_number,
                                **_xgrow_kw(ime))
                calls,ncolor = replace_dict['return_value']
                # Emit the HELAS call sequence as matrix<i>_origamp<k>.f, one
                # subroutine per amp_chunk_size statements, and leave the calls
                # to them in MATRIX<i>. Short sequences stay inline, so nothing
                # below the high-multiplicity threshold changes at all.
                self.write_amp_chunk_files(replace_dict, str(ime+1))
                tfile = open(replace_dict['template_file']).read()
                file = misc.apply_template(tfile, replace_dict)
                # Add the split orders helper functions.
                file = file + '\n' + misc.apply_template(
                    open(replace_dict['template_file2']).read(), replace_dict)
                # Write the file
                writer = writers.FortranWriter(filename)
                writer.writelines(file)

                #
                # write the dedicated template for helicity recycling
                #
                tfile = open(replace_dict['template_file'].replace('.inc',"_hel.inc")).read()
                file = misc.apply_template(tfile, replace_dict)
                # Add the split orders helper functions.
                file = file + '\n' + misc.apply_template(
                    open(replace_dict['template_file2']).read(), replace_dict)
                # Write the file
                writer = writers.FortranWriter('template_matrix%d.f' % (ime+1))
                writer.uniformcase = False
                writer.writelines(file)

                # ... and the template hel_recycle renders the unrolled call
                # sequence into, one file per chunk, the same way.
                self.write_amp_chunk_template(replace_dict, ime+1)

                
                
                
            else:
                filename = 'matrix%d.f' % (ime+1)
                calls, ncolor = \
                   self.write_matrix_element_v4(writers.FortranWriter(filename),
                                matrix_element,
                                fortran_model,
                                proc_id=str(ime+1),
                                config_map=subproc_group.get('diagram_maps')[ime],
                                subproc_number=group_number,
                                **_xgrow_kw(ime))

            if second_exporter:
                process_exporter_cpp = second_exporter.oneprocessclass(matrix_element,second_helas, prefix=ime)
                dirpath = '.'
                with misc.chdir(dirpath):
                    logger.info('Creating files in directory %s' % dirpath)
                    process_exporter_cpp.path = dirpath
                    # Create the process .h and .cc files
                    process_exporter_cpp.generate_process_files_madevent(proc_id=str(ime+1),
                                        config_map=subproc_group.get('diagram_maps')[ime], 
                                        subproc_number=group_number)
                    for file in second_exporter.to_link_in_P:
                        ln('../%s' % file)    
                # second_exporter.write_matrix_element_madevent(ime,
                #                 matrix_element,
                #                 second_helas,
                #                 proc_id=str(ime+1),
                #                 config_map=subproc_group.get('diagram_maps')[ime],
                #                 subproc_number=group_number
                # )


            filename = 'auto_dsig%d.f' % (ime+1)
            self.write_auto_dsig_file(writers.FortranWriter(filename),
                                 matrix_element,
                                 str(ime+1),
                                 crossgroup=crossgroup)

            # Keep track of needed quantities
            tot_calls += int(calls)
            maxflows = max(maxflows, ncolor)
            maxamps = max(maxamps, len(matrix_element.get('diagrams')))

            # Draw diagrams
            if not 'noeps' in self.opt['output_options'] or self.opt['output_options']['noeps'] != 'True':
                filename = "matrix%d.ps" % (ime+1)
                plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                                        get('diagrams'),
                                                  filename,
                                                  model = \
                                                    matrix_element.get('processes')[0].\
                                                                           get('model'),
                                                  amplitude=True)
                logger.info("Generating Feynman diagrams for " + \
                             matrix_element.get('processes')[0].nice_string())
                plot.draw()

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Generate a list of diagrams corresponding to each configuration
        # [[d1, d2, ...,dn],...] where 1,2,...,n is the subprocess number
        # If a subprocess has no diagrams for this config, the number is 0
        subproc_diagrams_for_config = subproc_group.get('diagrams_for_configs')

        filename = 'auto_dsig.f'
        self.write_super_auto_dsig_file(writers.FortranWriter(filename),
                                   subproc_group, group_number)

        filename = 'coloramps.inc'
        self.write_coloramps_file(writers.FortranWriter(filename),
                                   subproc_diagrams_for_config,
                                   maxflows,
                                   matrix_elements)

        filename = 'config_subproc_map.inc'
        self.write_config_subproc_map_file(writers.FortranWriter(filename),
                                           subproc_diagrams_for_config)

        filename = 'configs.inc'
        nconfigs, (s_and_t_channels, nqcd_list) = self.write_configs_file(\
            writers.FortranWriter(filename),
            subproc_group,
            subproc_diagrams_for_config)

        filename = 'config_nqcd.inc'
        self.write_config_nqcd_file(writers.FortranWriter(filename),
                                    nqcd_list)

        filename = 'decayBW.inc'
        self.write_decayBW_file(writers.FortranWriter(filename),
                           s_and_t_channels)

        filename = 'dname.mg'
        self.write_dname_file(writers.FortranWriter(filename),
                         subprocdir)

        filename = 'iproc.dat'
        self.write_iproc_file(writers.FortranWriter(filename),
                         group_number)

        filename = 'leshouche.inc'
        self.write_leshouche_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = 'colorflow.inc'
        self.write_colorflow_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = 'maxamps.inc'
        # get number of non identical flavor for each matrix element file
        #for me in matrix_elements:
            #misc.sprint(me.get_external_flavors_with_iden())
            #misc.sprint(me.get_nb_flavors())
        #misc.sprint([me.get_nb_flavors() for me in matrix_elements])

        nb_flavor_per_proc = [me.get_nb_flavors() for me in matrix_elements]
        #misc.sprint(os.getcwd(), nb_flavor_per_proc)
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           max(nb_flavor_per_proc),
                           max([me.get_nb_flavors() for me in \
                                matrix_elements]), # THis is max(flavor*process) 
                           len(matrix_elements))

        # Note that mg.sym is not relevant for this case
        filename = 'mg.sym'
        self.write_default_mg_sym_file(writers.FortranWriter(filename))

        filename = 'mirrorprocs.inc'
        self.write_mirrorprocs(writers.FortranWriter(filename),
                          subproc_group)

        filename = 'ncombs.inc'
        self.write_ncombs_file(writers.FortranWriter(filename),
                          nexternal)

        filename = 'nexternal.inc'
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = 'ngraphs.inc'
        self.write_ngraphs_file(writers.FortranWriter(filename),
                           nconfigs)

        filename = 'pmass.inc'
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = 'props.inc'
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

        filename = 'processes.dat'
        files.write_to_file(filename,
                            self.write_processes_file,
                            subproc_group)

        # Find config symmetries and permutations
        symmetry, perms, ident_perms = \
                  diagram_symmetry.find_symmetry(subproc_group)

        filename = 'symswap.inc'
        self.write_symswap_file(writers.FortranWriter(filename),
                                ident_perms)

        filename = 'symfact_orig.dat'
        self.write_symfact_file(open(filename, 'w'), symmetry)
        
        # check consistency
        for i, sym_fact in enumerate(symmetry):
            
            if sym_fact >= 0:
                continue
            if nqcd_list[i] != nqcd_list[abs(sym_fact)-1]:
                misc.sprint(i, sym_fact, nqcd_list[i], nqcd_list[abs(sym_fact)])
                raise Exception("identical diagram with different QCD powwer")
        

        filename = 'symperms.inc'
        self.write_symperms_file(writers.FortranWriter(filename),
                           perms)

        # Generate jpgs -> pass in make_html
        #os.system(pjoin('..', '..', 'bin', 'gen_jpeg-pl'))

        self.link_files_in_SubProcess(pjoin(pathdir,subprocdir))

        #import nexternal/leshouch in Source
        ln('nexternal.inc', '../../Source', log=False)
        ln('leshouche.inc', '../../Source', log=False)
        ln('maxamps.inc', '../../Source', log=False)

        if second_exporter:
            tmp = locals()
            del tmp['self']
            process_exporter_cpp.generate_subprocess_directory_end(**tmp) 

        # Return to SubProcesses dir)
        os.chdir(pathdir)

        # Add subprocess to subproc.mg
        filename = 'subproc.mg'
        files.append_to_file(filename,
                             self.write_subproc,
                             subprocdir)

        # Return to original dir
        os.chdir(cwd)

        if not tot_calls:
            tot_calls = 0
        return tot_calls

    #===========================================================================
    # write_super_auto_dsig_file
    #===========================================================================
    def write_super_auto_dsig_file(self, writer, subproc_group,
                                   group_number=None):
        """Write the auto_dsig.f file selecting between the subprocesses
        in subprocess group mode"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        matrix_elements = subproc_group.get('matrix_elements')

        # Extract process info lines
        process_lines = '\n'.join([self.get_process_info_lines(me) for me in \
                                   matrix_elements])
        replace_dict['process_lines'] = process_lines

        nexternal, ninitial = matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        replace_dict['nsprocs'] = 2*len(matrix_elements)

        # Generate dsig definition line
        dsig_def_line = "DOUBLE PRECISION " + \
                        ",".join(["DSIG%d" % (iproc + 1) for iproc in \
                                  range(len(matrix_elements))])
        replace_dict["dsig_def_line"] = dsig_def_line

        # Generate dsig process lines
        call_dsig_proc_lines = []
        call_dsig_proc_lines_vec = []
        for iproc in range(len(matrix_elements)):
            data = {"num": iproc + 1,
                 "proc": matrix_elements[iproc].get('processes')[0].base_string()}
            call_dsig_proc_lines.append(\
                "IF(IPROC.EQ.%(num)d) DSIGPROC=DSIG%(num)d(P1,IFLAV,WGT,IMODE) ! %(proc)s" % data
                )
            call_dsig_proc_lines_vec.append(\
                "IF(IPROC.EQ.%(num)d) CALL DSIG%(num)d_VEC(ALL_P1,ALL_XBK,ALL_Q2FACT,ALL_CM_RAP,ALL_WGT,IMODE,ALL_OUT,SYMCONF, CONFSUB,ICONF_VEC,IMIRROR_VEC,IFLAV_VEC,VECSIZE_USED) ! %(proc)s" % data
                )

        replace_dict['call_dsig_proc_lines'] = "\n".join(call_dsig_proc_lines)
        replace_dict['call_dsig_proc_lines_vec'] = "\n".join(call_dsig_proc_lines_vec)

        ncomb=matrix_elements[0].get_helicity_combinations()
        replace_dict['read_write_good_hel'] = self.read_write_good_hel(ncomb)

        s1,s2 = matrix_elements[0].get_spin_state_initial()
        replace_dict['nb_spin_state1'] = s1
        replace_dict['nb_spin_state2'] = s2
        
        printzeroamp = []
        for iproc in range(len(matrix_elements)):
            printzeroamp.append(\
                "        call print_zero_amp%i()" % ( iproc + 1))
        replace_dict['print_zero_amp'] = "\n".join(printzeroamp)
        
        
        get_nhel = []
        for iproc in range(len(matrix_elements)):
            get_nhel.append("   integer get_nhel%i   " %(iproc+1) )
            if iproc == 0:
                get_helicity = [' if(iproc.eq.1)then']
            else: 
                get_helicity.append(' elseif(iproc.eq.%s)then' % (iproc+1))
            get_helicity.append("   do i=1,nexternal")
            get_helicity.append(
                "        nhel(i) = get_nhel%i(ihel,i)" % ( iproc + 1))
            get_helicity.append("enddo")
        get_helicity.append(" endif" ) 

        replace_dict['call_to_local_get_helicities'] = "\n".join(get_helicity)
        replace_dict['definition_of_local_get_nhel'] = "\n".join(get_nhel)

        # Generate get_flavor dispatch for the grouped case
        # Each subprocess has its own GET_FLAVOR<N> subroutine (from matrix element template)
        # The wrapper get_flavor(iflav, iproc, flavor) dispatches based on iproc
        get_flavor_decl = []
        get_flavor_call = []
        for iproc in range(len(matrix_elements)):
            get_flavor_decl.append("   external get_flavor%i" % (iproc + 1))
            if iproc == 0:
                get_flavor_call.append(' if(iproc.eq.1)then')
            else:
                get_flavor_call.append(' elseif(iproc.eq.%d)then' % (iproc + 1))
            get_flavor_call.append("   call get_flavor%i(iflav, flavor)" % (iproc + 1))
        get_flavor_call.append(' endif')

        replace_dict['call_to_local_get_flavor'] = "\n".join(get_flavor_call)
        replace_dict['definition_of_local_get_flavor'] = "\n".join(get_flavor_decl)

        if writer:
            file = open(pjoin(_file_path, \
                       'iolibs/template_files/super_auto_dsig_group_v4.inc')).read()
            file = file % replace_dict
            file += self.write_xgrow_routines(subproc_group, group_number)

            # Write the file
            writer.writelines(file)
        else:
            return replace_dict

    def write_xgrow_routines(self, subproc_group, group_number):
        """Per-directory bodies of the XGROW<b> helpers a cross-group (Track B)
        base SMATRIX calls for its multi-channel row (see the me_confsub_j fill).

        The base's compiled matrix<b> object is symlinked into every dependent P
        directory, so it cannot carry the row itself: the row belongs to the
        subprocess the call is FOR, and that subprocess's CONFSUB lives in ITS
        directory. Each directory therefore links its own XGROW<b>, resolved by
        the linker exactly like genps.o (which is why GET_CHANNEL_CUT(P, I) in
        the shared object already means the *dependent's* config I).

        * where the base is generated -- the identity: our own CONFSUB row. Only
          cross 0 ever reaches it (a Track-B base group has no within-group
          router, so its own auto_dsig calls it with a plain FLAV_IDX).
        * in a dependent's directory -- the routed subprocess's own CONFSUB row,
          each of its diagrams mapped to the base AMP2 slot the crossed
          evaluation filled (_crossgroup_configmap, the same map its auto_dsig
          uses for DSIG_XGCONFIG).

        Emitted here because auto_dsig.f is the one file written exactly once per
        P directory, so a base serving several dependents in one directory still
        gets a single definition.
        """
        if group_number is None or not getattr(self, '_crossgroup', None):
            return ''
        mes = subproc_group.get('matrix_elements')
        routines, seen = [], {}
        # Bases generated in this directory: identity row.
        base_ids = getattr(self, '_crossgroup_base_mes', set())
        for ime, me in enumerate(mes):
            if id(me) in base_ids:
                seen[ime + 1] = 'base'
                routines.append(
                    '\n      SUBROUTINE XGROW%(b)d(CROSS, XGJ)\n'
                    'C     Multi-channel row of SMATRIX%(b)d in the directory it\n'
                    'C     is generated in: its own. CROSS is always 0 here.\n'
                    '      IMPLICIT NONE\n'
                    "      INCLUDE 'maxamps.inc'\n"
                    "      INCLUDE 'maxconfigs.inc'\n"
                    '      INTEGER CROSS, XGJ(LMAXCONFIGS), I\n'
                    '      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)\n'
                    "      INCLUDE 'config_subproc_map.inc'\n"
                    '      DO I=1,LMAXCONFIGS\n'
                    '        XGJ(I) = CONFSUB(%(b)d, I)\n'
                    '      ENDDO\n'
                    '      RETURN\n'
                    '      END\n' % {'b': ime + 1})
        # Dependents routed out of this directory: their own row, remapped.
        by_base = {}
        for ime, me in enumerate(mes):
            cg = self._crossgroup.get((group_number, ime))
            if cg is None:
                continue
            base_me = cg['base_me']
            nflav_base = len(base_me.get_external_flavors_with_iden())
            ngraphs_b = len(base_me.get('diagrams'))
            nxc = (base_me.get_nexternal_ninitial()[0] + 1) ** 2 - 1
            for iflav in cg['flav_idx']:
                cross = (iflav - 1) // nflav_base
                if not 1 <= cross <= nxc:
                    continue
                cmap = self._crossgroup_configmap(me, base_me, cross)
                if sorted(cmap) != list(range(1, ngraphs_b + 1)):
                    continue          # unusable map: leave the historical row
                slot = by_base.setdefault(
                    cg['base_proc_id'],
                    {'nxc': nxc, 'ng': ngraphs_b, 'cols': [], 'cross': {}})
                col = (ime + 1, tuple(cmap))
                if col not in slot['cols']:
                    slot['cols'].append(col)
                # Two subprocesses claiming the same crossing would be the same
                # crossed process; keep the first and leave the rest alone.
                slot['cross'].setdefault(cross, slot['cols'].index(col) + 2)
        for b in sorted(by_base):
            if b in seen:
                # This directory both generates SMATRIX<b> and routes to another
                # directory's SMATRIX<b>: one name, two bodies. That collision
                # already exists for SMATRIX<b> itself, so leave it alone.
                logger.warning('Cross-group crossing: SMATRIX%d is both local '
                               'and routed in one directory; keeping the '
                               'historical multi-channel row.' % b)
                continue
            s = by_base[b]
            # Column 1 is the fallback for a crossing this directory does not
            # route (unreachable in practice): the first routed subprocess's own
            # row, unmapped.
            rows = [s['cols'][0][0]] + [c[0] for c in s['cols']]
            cfgs = [list(range(0, s['ng'] + 1))]
            cfgs += [[0] + list(c[1]) for c in s['cols']]
            lines = [
                '\n      SUBROUTINE XGROW%d(CROSS, XGJ)' % b,
                'C     Multi-channel row of the symlinked SMATRIX%d for a call' % b,
                'C     routed out of THIS directory: the routed subprocess own',
                'C     CONFSUB row, each diagram mapped to the AMP2 slot the',
                'C     crossed evaluation filled.',
                '      IMPLICIT NONE',
                "      INCLUDE 'maxamps.inc'",
                "      INCLUDE 'maxconfigs.inc'",
                '      INTEGER CROSS, XGJ(LMAXCONFIGS), I, IXR',
                '      INTEGER CONFSUB(MAXSPROC,LMAXCONFIGS)',
                "      INCLUDE 'config_subproc_map.inc'",
                '      INTEGER XGCOL(0:%d)' % s['nxc'],
                self.format_integer_data_lines(
                    'XGCOL', [s['cross'].get(c, 1)
                              for c in range(s['nxc'] + 1)]),
                '      INTEGER XGROWP(%d)' % len(rows),
                '      DATA XGROWP /%s/' % ','.join(str(x) for x in rows),
                '      INTEGER XGCFG(0:%d,%d)' % (s['ng'], len(cfgs))]
            for icol, col in enumerate(cfgs):
                for st in range(0, len(col), 10):
                    chunk = col[st:st + 10]
                    lines.append('      DATA (XGCFG(I,%d),I=%d,%d) /%s/'
                                 % (icol + 1, st, st + len(chunk) - 1,
                                    ','.join(str(v) for v in chunk)))
            lines += ['      IXR = 1',
                      '      IF (CROSS.GE.0.AND.CROSS.LE.%d) IXR = XGCOL(CROSS)'
                      % s['nxc'],
                      '      DO I=1,LMAXCONFIGS',
                      '        XGJ(I) = XGCFG(CONFSUB(XGROWP(IXR), I), IXR)',
                      '      ENDDO',
                      '      RETURN',
                      '      END']
            routines.append('\n'.join(lines) + '\n')
        return ''.join(routines)
        
    #===========================================================================
    # write_mirrorprocs
    #===========================================================================
    def write_mirrorprocs(self, writer, subproc_group):
        """Write the mirrorprocs.inc file determining which processes have
        IS mirror process in subprocess group mode."""

        lines = []
        bool_dict = {True: '.true.', False: '.false.'}
        matrix_elements = subproc_group.get('matrix_elements')
        for i, me in enumerate(matrix_elements):
            flavors = me.get_external_flavors_with_iden()
            process = me.get('processes')[0]

            # shared with the mg7 exporter (see Process.has_same_initial_multiparticle)
            same_initial_multiparticle = process.has_same_initial_multiparticle()
            if me.get('has_mirror_process'):
                lines.append("DATA (MIRRORPROCS(%i,I),I=1,%d)/%s/" % \
                            (i+1, len(flavors),
                      ",".join(['.true.' for flv in flavors])))
            elif same_initial_multiparticle:
                # If the two initial legs come from the same multiparticle
                # definition, only mixed concrete flavors need mirror calls.
                lines.append("DATA (MIRRORPROCS(%i,I),I=1,%d)/%s/" % \
                            (i+1, len(flavors),
                      ",".join([bool_dict[(flv[0][0] != flv[0][1])] for flv in flavors])))
            else:
                lines.append("DATA (MIRRORPROCS(%i,I),I=1,%d)/%s/" % \
                        (i+1, len(flavors),
                      ",".join(['.false.' for flv in flavors]))) 
                

        lines.append("DATA NB_FLAV /%s/" % (
                      ",".join([str(len(me.get_external_flavors_with_iden())) for \
                                me in matrix_elements])))
        # Write number of individual (identical-coupling) flavors per group per subprocess.
        # N_INDIV_FLAV(K,I) is the number of leshouche rows belonging to coupling group K
        # of matrix element I.  This is used to correctly split the per-group cross section
        # among the individual flavor entries when reporting decay widths.
        max_flav_per_proc = max(len(list(me.get_external_flavors_with_iden()))
                                for me in matrix_elements)
        for i, me in enumerate(matrix_elements):
            groups = list(me.get_external_flavors_with_iden())
            n_per_group = [len(list(g)) for g in groups]
            # Pad to max_flav_per_proc with zeros
            n_per_group += [0] * (max_flav_per_proc - len(n_per_group))
            lines.append("DATA (N_INDIV_FLAV(K,%d),K=1,%d)/%s/" % (
                i + 1, max_flav_per_proc, ",".join(str(n) for n in n_per_group)))
        # Write the file
        writer.writelines(lines)

    #===========================================================================
    # write_maxamps_file
    #===========================================================================
    def write_maxamps_file(self, writer, maxamps, maxflows, max_flav_per_proc,
                           maxproc,maxsproc):
        """Write the maxamps.inc file for MG4."""

        file = "       integer    maxamps, maxflow, maxproc, maxsproc, maxflavperproc\n"
        file = file + "parameter (maxamps=%d, maxflow=%d)\n" % \
               (maxamps, maxflows)
        file = file + "parameter (maxproc=%d, maxsproc=%d)\n" % \
               (maxproc, maxsproc)
        file += "parameter (maxflavperproc=%d)" % max_flav_per_proc

        # Write the file
        writer.writelines(file)

        return True


    #===========================================================================
    # write_addmothers
    #===========================================================================
    def write_addmothers(self, writer):
        """Write the SubProcess/addmothers.f"""

        path = pjoin(_file_path,'iolibs','template_files','addmothers.f')

        text = open(path).read() % {'iconfig': 'lconfig'}
        writer.write(text)
        
        return True


    #===========================================================================
    # write_coloramps_file
    #===========================================================================
    def write_coloramps_file(self, writer, diagrams_for_config, maxflows,
                                   matrix_elements):
        """Write the coloramps.inc file for MadEvent in Subprocess group mode"""

        # Create a map from subprocess (matrix element) to a list of
        # the diagrams corresponding to each config
        subproc_to_confdiag = self.get_confdiag_from_group_mapconfig(diagrams_for_config)
        lines = []

        for subproc in sorted(subproc_to_confdiag.keys()):
            lines.extend(self.get_icolamp_lines(subproc_to_confdiag[subproc],
                                           matrix_elements[subproc],
                                           subproc + 1))

        lines.insert(0, "logical icolamp(%d,%d,%d)" % \
                        (maxflows,
                         len(diagrams_for_config),
                         len(matrix_elements)))

        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # write_config_subproc_map_file
    #===========================================================================
    def write_config_subproc_map_file(self, writer, config_subproc_map):
        """Write the config_subproc_map.inc file for subprocess groups"""

        lines = []
        # Output only configs that have some corresponding diagrams
        iconfig = 0
        for config in config_subproc_map:
            if set(config) == set([0]):
                continue
            lines.append("DATA (CONFSUB(i,%d),i=1,%d)/%s/" % \
                         (iconfig + 1, len(config),
                          ",".join([str(i) for i in config])))
            iconfig += 1
        # Write the file
        writer.writelines(lines)

        return True

    #===========================================================================
    # read_write_good_hel
    #===========================================================================
    def read_write_good_hel(self, ncomb):
        """return the code to read/write the good_hel common_block"""    

        convert = {'ncomb' : ncomb}

        output = """
        subroutine write_good_hel(stream_id)
        implicit none
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC, MAXSPROC)
        INTEGER NTRY(MAXFLAVPERPROC, MAXSPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        write(stream_id,*) GOODHEL
        return
        end
        
        
        subroutine read_good_hel(stream_id)
        implicit none
        include 'genps.inc'
        include 'maxamps.inc'
        integer stream_id
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC, MAXSPROC)
        INTEGER NTRY(MAXFLAVPERPROC, MAXSPROC)
        common/BLOCK_GOODHEL/NTRY,GOODHEL
        read(stream_id,*) GOODHEL
        NTRY(:,:) = MAXTRIES + 1
        return
        end
        
        subroutine init_good_hel()
        implicit none
        include 'maxamps.inc'
        INTEGER                 NCOMB
        PARAMETER (             NCOMB=%(ncomb)d)
        LOGICAL GOODHEL(NCOMB, MAXFLAVPERPROC, MAXSPROC)
        INTEGER NTRY(MAXFLAVPERPROC, MAXSPROC)
        INTEGER I,J

        GOODHEL(:,:,:) = .false.
        NTRY(:,:) = 0
        end
        
        integer function get_maxsproc()
        implicit none
        include 'maxamps.inc'
        
        get_maxsproc = maxsproc
        return 
        end
                
        """ % convert
        
        return output
                           

    #===========================================================================
    # write_configs_file
    #===========================================================================
    @staticmethod
    def get_confdiag_from_group_mapconfig(config_subproc_map, subprocid=None):
            """ This is converting the  mapconfigs generated from the 
                    subproc_group.get('diagrams_for_configs')
                and convert it to a datastructure like expected from the 
                get_icolamp_lines (which does not handle grouping) 
                
                if subproc is None it returns the full output as a dictionary
                with subproc_id as key.
                if provided it returned the associated list for that subproc id.

                Static method since need to be used from cpp case as well.
            """

            subproc_to_confdiag = {}
            for config in config_subproc_map:
                for subproc, diag in enumerate(config):
                    try:
                        subproc_to_confdiag[subproc].append(diag)
                    except KeyError:
                        subproc_to_confdiag[subproc] = [diag]
                        
            if subprocid is None:
                return subproc_to_confdiag
            else:
                return subproc_to_confdiag[subprocid]

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, subproc_group, diagrams_for_config):
        """Write the configs.inc file with topology information for a
        subprocess group. Use the first subprocess with a diagram for each
        configuration."""

        matrix_elements = subproc_group.get('matrix_elements')
        model = matrix_elements[0].get('processes')[0].get('model')

        diagrams = []
        config_numbers = []
        for iconfig, config in enumerate(diagrams_for_config):
            # Check if any diagrams correspond to this config
            if set(config) == set([0]):
                continue
            subproc_diags = []
            for s,d in enumerate(config):
                if d:
                    subproc_diags.append(matrix_elements[s].\
                                         get('diagrams')[d-1])
                else:
                    subproc_diags.append(None)
            diagrams.append(subproc_diags)
            config_numbers.append(iconfig + 1)

        # Extract number of external particles
        (nexternal, ninitial) = subproc_group.get_nexternal_ninitial()

        return len(diagrams), \
               self.write_configs_file_from_diagrams(writer, diagrams,
                                                config_numbers,
                                                nexternal, ninitial,
                                                     model)




    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = pjoin(_file_path,'iolibs','template_files','madevent_run_config.inc')
        if self.proc_characteristic['loop_induced']:
            job_per_chan = 1
        else: 
            job_per_chan = 2
        text = open(path).read() % {'chanperjob':job_per_chan} 
        writer.write(text)
        return True


    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, subproc_group):
        """Write the leshouche.inc file for MG4"""

        all_lines = []

        for iproc, matrix_element in \
            enumerate(subproc_group.get('matrix_elements')):
            all_lines.extend(self.get_leshouche_lines(matrix_element,
                                                 iproc, drop_icolup=True))
        # Write the file
        writer.writelines(all_lines)
        return True

    def write_colorflow_file(self, writer, subproc_group):
        """Write colorflow.inc for a subprocess group (one entry per ME)."""

        all_lines = []
        for iproc, matrix_element in \
            enumerate(subproc_group.get('matrix_elements')):
            all_lines.extend(self.get_colorflow_lines(matrix_element, iproc))
        writer.writelines(all_lines)
        return True


    def finalize(self,*args, second_exporter=None, **opts):

        if second_exporter:
            self.has_second_exporter = second_exporter
        super(ProcessExporterFortranMEGroup, self).finalize(*args, second_exporter=None, **opts)
        #ensure that the grouping information is on the correct value
        self.proc_characteristic['grouped_matrix'] = True

        filename = pjoin(self.dir_path,'Source','makefile')
        if not second_exporter:
            self.write_source_makefile(writers.FileWriter(filename), model=self.model)
        else:
           replace_dict = self.write_source_makefile(None)
           second_exporter.write_source_makefile(writers.FileWriter(filename), model=self.model, default=replace_dict)  

        if second_exporter:
            second_exporter.finalize(*args, **opts)

        
#===============================================================================
# UFO_model_to_mg4
#===============================================================================

python_to_fortran = lambda x: parsers.UFOExpressionParserFortran().parse(x)

class UFO_model_to_mg4(object):
    """ A converter of the UFO-MG5 Model to the MG4 format """

    # The list below shows the only variables the user is allowed to change by
    # himself for each PS point. If he changes any other, then calling 
    # UPDATE_AS_PARAM() (or equivalently MP_UPDATE_AS_PARAM()) will not
    # correctly account for the change.
    PS_dependent_key = ['aS','MU_R']
    mp_complex_format = 'complex*32'
    mp_real_format = 'real*16'
    # Warning, it is crucial none of the couplings/parameters of the model
    # starts with this prefix. I should add a check for this.
    # You can change it as the global variable to check_param_card.ParamCard
    mp_prefix = check_param_card.ParamCard.mp_prefix
    
    def __init__(self, model, output_path, opt=None):
        """ initialization of the objects """

        self.model = model
        self.model_name = model['name']
        self.dir_path = output_path
        
        self.opt = {'complex_mass': False, 'export_format': 'madevent', 'mp':True,
                        'loop_induced': False}
        if opt:
            self.opt.update(opt)
            
        self.coups_dep = []    # (name, expression, type)
        self.coups_indep_noloop = []  # (name, expression, type)
        self.coups_indep_loop = []  # (name, expression, type)
        self.coups_flv_dep = []    # (name, object, [couplings])
        self.coups_flv_indep = []  # (name, object, [couplings])  
        self.params_dep = []   # (name, expression, type)
        self.params_indep = [] # (name, expression, type)
        self.params_ext = []   # external parameter
        self.p_to_f = parsers.UFOExpressionParserFortran(self.model)
        self.mp_p_to_f = parsers.UFOExpressionParserMPFortran(self.model)   
        try:
            vector_size = self.opt['output_options']['vector_size']
            self.vector_size = banner_mod.ConfigFile.format_variable(vector_size, int, 'vector_size')
        except KeyError as error:
            self.vector_size = 0

        try:
            nb_warp = self.opt['output_options']['nb_warp']
        except KeyError:
            nb_warp = 1
        self.nb_warp = max(1, banner_mod.ConfigFile.format_variable(nb_warp, int, 'nb_warp'))
        if self.opt['mp']:
            assert self.vector_size in [0,1]
            self.nb_warp = 1
        self.scales = []
        self.MUE = None # extra parameter loop #2 which is running
        
        if self.model.get('running_elements'):
            all_elements = set()
            add_scale = set()
            for runs in self.model.get('running_elements'):
                for line_run in runs.run_objects:
                    for one_element in line_run:
                        all_elements.add(one_element.name)
                        add_scale.add(one_element.lhablock)
            all_elements.union(set(self.PS_dependent_key))
            self.PS_dependent_key = list(all_elements)
            MUE = [p for p in self.model.get('parameters')[('external',)] if p.lhablock.lower() == 'loop' and tuple(p.lhacode) == (2,)]
            
            if MUE:
                self.MUE = MUE[0]
                self.PS_dependent_key.append(MUE[0].name)
            
            try:
                add_scale.remove('SMINPUTS')
            except Exception:
                pass
            self.scales = add_scale

    
    def pass_parameter_to_case_insensitive(self):
        """modify the parameter if some of them are identical up to the case"""
    
        lower_dict={}
        duplicate = set()
        keys = list(self.model['parameters'].keys())
        keys.sort()
        for key in keys:
            for param in self.model['parameters'][key]:
                lower_name = param.name.lower()
                if not lower_name:
                    continue
                try:
                    lower_dict[lower_name].append(param)
                except KeyError as error:
                    lower_dict[lower_name] = [param]
                else:
                    duplicate.add(lower_name)
                    logger.debug('%s is define both as lower case and upper case.' 
                                 % lower_name)
        if not duplicate:
            return
        
        re_expr = r'''\b(%s)\b'''
        to_change = []
        change={}
        for value in duplicate:
            for i, var in enumerate(lower_dict[value]):
                to_change.append(var.name)
                new_name = '%s%s' % (var.name.lower(), 
                                                  ('__%d'%(i+1) if i>0 else ''))
                change[var.name] = new_name
                var.name = new_name
    
        # Apply the modification to the map_CTcoup_CTparam of the model
        # if it has one (giving for each coupling the CT parameters whcih
        # are necessary and which should be exported to the model.
        if hasattr(self.model,'map_CTcoup_CTparam'):
            for coup, ctparams in self.model.map_CTcoup_CTparam:
                for i, ctparam in enumerate(ctparams):
                    try:
                        self.model.map_CTcoup_CTparam[coup][i] = change[ctparam]
                    except KeyError:
                        pass

        replace = lambda match_pattern: change[match_pattern.groups()[0]]
        rep_pattern = re.compile(re_expr % '|'.join(to_change))
        
        # change parameters
        for key in keys:
            if key == ('external',):
                continue
            for param in self.model['parameters'][key]: 
                param.expr = rep_pattern.sub(replace, param.expr)
            
        # change couplings
        for key in self.model['couplings'].keys():
            for coup in self.model['couplings'][key]:
                coup.expr = rep_pattern.sub(replace, coup.expr)
                
        # change mass/width
        for part in self.model['particles']:
            if str(part.get('mass')) in to_change:
                part.set('mass', rep_pattern.sub(replace, str(part.get('mass'))))
            if str(part.get('width')) in to_change:
                part.set('width', rep_pattern.sub(replace, str(part.get('width'))))                
                
    def refactorize(self, wanted_couplings = []):    
        """modify the couplings to fit with MG4 convention """
            
        # Keep only separation in alphaS + running one        
        keys = list(self.model['parameters'].keys())
        keys.sort(key=len)

        for key in keys:
            to_add = [o for o in self.model['parameters'][key] if o.name]

            if key == ('external',):
                self.params_ext += to_add
            elif any([(k in key) for k in self.PS_dependent_key]):
                self.params_dep += to_add
            else:
                self.params_indep += to_add
                
        # same for couplings + tracking which running happens
        keys = list(self.model['couplings'].keys())
        keys.sort(key=len)
        used_running_key = set()
        for key, coup_list in self.model['couplings'].items():
            if any([(k in key) for k in self.PS_dependent_key]):
                to_add = [c for c in coup_list if
                                   (not wanted_couplings or c.name in \
                                    wanted_couplings)]
                if to_add:
                    self.coups_dep += to_add
                    used_running_key.update(set(key))
            else:
                self.coups_indep_noloop += [c for c in coup_list if
                                     (not wanted_couplings \
                                      or c.name in wanted_couplings \
                                      or f"-{c.name}" in wanted_couplings) and \
                                      not any([tag in c.name.lower() for tag in ['uv', 'r2']])]
                self.coups_indep_loop += [c for c in coup_list if
                                     (not wanted_couplings \
                                      or c.name in wanted_couplings \
                                      or f"-{c.name}" in wanted_couplings) and \
                                      any([tag in c.name.lower() for tag in ['uv', 'r2']])]

        # keep track of all couplings (for backward compatibility and/or tests
        self.coups_indep = self.coups_indep_noloop + self.coups_indep_loop
               
        #store the running parameter that are used
        self.used_running_key = used_running_key     
        # MG4 use G and not aS as it basic object for alphas related computation
        #Pass G in the  independant list
        if 'G' in self.params_dep:
            index = self.params_dep.index('G')
            G = self.params_dep.pop(index)
        #    G.expr = '2*cmath.sqrt(as*pi)'
        #    self.params_indep.insert(0, self.params_dep.pop(index))
        # No need to add it if not defined   

        if 'aS' not in self.params_ext and 'aS' not in self.params_indep:
            logger.critical('aS not define as external parameter adding it!')
            #self.model['parameters']['aS'] = base_objects.ParamCardVariable('aS', 0.138,'DUMMY',(1,))
            self.params_indep.append( base_objects. ModelVariable('aS', '0.138','real'))
            self.params_indep.append( base_objects. ModelVariable('G', '4.1643','real'))

        # Handle flavor couplings
        # strategy picke one of the actual coupling and check if this is a running one or not
        flavor_couplings = [c for c in wanted_couplings if isinstance(c, base_objects.FLV_Coupling)]
        deps = [c.name for c in self.coups_dep]
        for one_flv in flavor_couplings:
            one_coupling = one_flv.get_one_coupling()
            if one_coupling in deps:
                self.coups_flv_dep.append( one_flv)
            else:
                self.coups_flv_indep.append(one_flv)

            
    def build(self, wanted_couplings = [], full=True):
        """modify the couplings to fit with MG4 convention and creates all the 
        different files"""
        
        self.pass_parameter_to_case_insensitive() 
        self.refactorize(wanted_couplings)

        # write the files
        if full:
            if wanted_couplings:
                # extract the wanted ct parameters
                self.extract_needed_CTparam(wanted_couplings=wanted_couplings)
            self.write_all()
            

    def open(self, name, comment='c', format='default'):
        """ Open the file name in the correct directory and with a valid
        header."""
        
        file_path = pjoin(self.dir_path, name)
        
        if format == 'fortran':
            fsock = writers.FortranWriter(file_path, 'w')
            write_class = io.FileIO
            
            write_class.writelines(fsock, comment * 77 + '\n')
            write_class.writelines(fsock, '%(comment)s written by the UFO converter\n' % \
                               {'comment': comment + (6 - len(comment)) *  ' '})
            write_class.writelines(fsock, comment * 77 + '\n\n')
        else:
            fsock = open(file_path, 'w')  
            fsock.writelines(comment * 77 + '\n')
            fsock.writelines('%(comment)s written by the UFO converter\n' % \
                                   {'comment': comment + (6 - len(comment)) *  ' '})
            fsock.writelines(comment * 77 + '\n\n')
        return fsock       

    
    def write_all(self):
        """ write all the files """
        #write the part related to the external parameter
        self.create_ident_card()
        self.create_param_read()
        
        #write the definition of the parameter
        self.create_input()
        self.create_intparam_def(dp=True,mp=False)
        if self.opt['mp']:
            self.create_intparam_def(dp=False,mp=True)
        self.create_ewa()
        
        # definition of the coupling.
        self.create_couplings_flavor_merged()
        self.create_actualize_mp_ext_param_inc()
        self.create_coupl_inc()
        self.create_write_couplings()
        self.create_couplings()
        self.create_printout()
        
        # the makefile
        self.create_makeinc()
        self.create_param_write()

        # The model functions
        self.create_model_functions_inc()
        self.create_model_functions_def()
        
        # The param_card.dat        
        self.create_param_card()
        
        # The get_color/get_spin functions
        self.create_get_color()

        # All the standard files
        self.copy_standard_file()

    ############################################################################
    ##  ROUTINE CREATING THE FILES  ############################################
    ############################################################################

    def copy_standard_file(self):
        """Copy the standard files for the fortran model."""
        
        #copy the library files
        file_to_link = ['formats.inc', \
                        'rw_para.f', 'testprog.f']
    
        for filename in file_to_link:
            cp( MG5DIR + '/models/template_files/fortran/' + filename, \
                                                                self.dir_path)
            
        file = open(os.path.join(MG5DIR,\
                              'models/template_files/fortran/rw_para.f')).read()

        if self.vector_size:
            includes=["include \'../vector.inc\'"]
        else:
            includes = []
        
        includes +=["include \'coupl.inc\'",
                  "include \'input.inc\'",
                  "include \'model_functions.inc\'"]
        if self.opt['mp']:
            includes.extend(["include \'mp_coupl.inc\'","include \'mp_input.inc\'"])
        # In standalone and madloop we do no use the compiled param card but
        # still parse the .dat one so we must load it.
        if self.opt['loop_induced']:
            #loop induced follow MadEvent way to handle the card.
            load_card = ''
            lha_read_filename='lha_read.f' 
            updateloop_default = '.true.'           
        elif self.opt['export_format'] in ['madloop','madloop_optimized', 'madloop_matchbox']:
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read_mp.f'
            updateloop_default = '.true.'
        elif self.opt['export_format'].startswith('standalone') \
            or self.opt['export_format'] in ['madweight', 'plugin']\
            or self.opt['export_format'].startswith('matchbox'):
            load_card = 'call LHA_loadcard(param_name,npara,param,value)'
            lha_read_filename='lha_read.f'
            updateloop_default = '.true.'
        else:
            load_card = ''
            lha_read_filename='lha_read.f'
            updateloop_default = '.false.'
            
        cp( MG5DIR + '/models/template_files/fortran/' + lha_read_filename, \
                                       os.path.join(self.dir_path,'lha_read.f'))
        
        file=file%{'includes':'\n      '.join(includes),
                   'load_card':load_card,
                   'updateloop_default': updateloop_default}
        writer=open(os.path.join(self.dir_path,'rw_para.f'),'w')
        writer.writelines(file)
        writer.close()

        if self.opt['export_format'] in ['madevent', 'FKS5_default', 'FKS5_optimized'] \
            or self.opt['loop_induced']:
            cp( MG5DIR + '/models/template_files/fortran/makefile_madevent', 
                self.dir_path + '/makefile')
            if self.opt['export_format'] in ['FKS5_default', 'FKS5_optimized']:
                path = pjoin(self.dir_path, 'makefile')
                text = open(path).read()
                text = text.replace('madevent','aMCatNLO').replace('../vector.inc', '')
                open(path, 'w').writelines(text)
        elif self.opt['export_format'] in ['standalone', 'standalone_msP','standalone_msF',
                                  'madloop','madloop_optimized', 'standalone_rw', 
                                  'madweight','matchbox','madloop_matchbox', 'plugin']:
            cp( MG5DIR + '/models/template_files/fortran/makefile_standalone', 
                self.dir_path + '/makefile')
        else:
            raise MadGraph5Error('Unknown format')

        if self.opt['export_format'].startswith('standalone'):
            cp( MG5DIR + '/Template/LO/Source/alfas_functions.f', 
                self.dir_path)
            cp( MG5DIR + '/Template/LO/Source/alfas.inc', 
                self.dir_path)

            fsock = open(pjoin(self.dir_path, '..', 'cuts.inc'),'w')
            fsock.write('''            
            logical fixed_extra_scale
            integer maxjetflavor
            double precision mue_over_ref
            double precision mue_ref_fixed
            common/model_setup_running/maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed
            ''')

            if self.model['running_elements']:
                cp( MG5DIR + '/Template/Running',  pjoin(self.dir_path, '..', 'RUNNING'))



    def create_coupl_inc(self):
        """ write coupling.inc """
        
        fsock = self.open('coupl.inc', format='fortran')
        if self.opt['mp']:
            mp_fsock = self.open('mp_coupl.inc', format='fortran')
            mp_fsock_same_name = self.open('mp_coupl_same_name.inc',\
                                            format='fortran')

        # Write header
        header = """C
C NB: VECSIZE_MEMMAX is defined in vector.inc
C NB: vector.inc must be included before coupl.inc
C

                double precision G, all_G%(vec)s
                common/strong/ G, all_G
                 
                double complex gal(2)
                common/weak/ gal
                
                double precision MU_R, all_mu_r%(vec)s
                common/rscale/ MU_R, all_mu_r

                """   % {'vec': ('' if not self.vector_size else '(1)' if self.vector_size<=1 else '(VECSIZE_MEMMAX)')}
                ###   % {'vec': ("(VECSIZE_MEMMAX)" if self.vector_size else '')}
                ###   % {'vec': ("(%i)" % max(1,self.vector_size) if self.vector_size else '')}

        # Nf is the number of light quark flavours
        header = header+"""double precision Nf
                parameter(Nf=%dd0)
                """ % self.model.get_nflav()
        #Nl is the number of massless leptons
        header = header+"""double precision Nl
                parameter(Nl=%dd0)
                """ % self.model.get_nleps()
                
        fsock.writelines(header)
        
        if self.opt['mp']:
            header = """%(real_mp_format)s %(mp_prefix)sG
                    common/MP_strong/ %(mp_prefix)sG
                     
                    %(complex_mp_format)s %(mp_prefix)sgal(2)
                    common/MP_weak/ %(mp_prefix)sgal
                    
                    %(complex_mp_format)s %(mp_prefix)sMU_R
                    common/MP_rscale/ %(mp_prefix)sMU_R

                """




            mp_fsock.writelines(header%{'real_mp_format':self.mp_real_format,
                                  'complex_mp_format':self.mp_complex_format,
                                  'mp_prefix':self.mp_prefix,
                                  'vector_size': '(1)' if self.vector_size else ''})
            mp_fsock_same_name.writelines(header%{'real_mp_format':self.mp_real_format,
                                  'complex_mp_format':self.mp_complex_format,
                                  'mp_prefix':'',
                                  'vector_size': '' if self.vector_size else ''})

        # Write the Mass definition/ common block
        masses = set()
        widths = set()
        if self.opt['complex_mass']:
            complex_mass = set()
            
        for particle in self.model.get('particles'):
            #find masses
            one_mass = particle.get('mass')
            if one_mass.lower() != 'zero':
                masses.add(one_mass)
                
            # find width
            one_width = particle.get('width')
            if one_width.lower() != 'zero':
                widths.add(one_width)
                if self.opt['complex_mass'] and one_mass.lower() != 'zero':
                    complex_mass.add('CMASS_%s' % one_mass)
            
        if masses:
            masses = sorted(list(masses))
            fsock.writelines('double precision '+','.join(masses)+'\n')
            fsock.writelines('common/masses/ '+','.join(masses)+'\n\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_real_format+' '+\
                                                          ','.join(masses)+'\n')
                mp_fsock_same_name.writelines('common/MP_masses/ '+\
                                                        ','.join(masses)+'\n\n')                
                mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                                        self.mp_prefix+m for m in masses])+'\n')
                mp_fsock.writelines('common/MP_masses/ '+\
                            ','.join([self.mp_prefix+m for m in masses])+'\n\n')                

        if widths:
            widths = sorted(list(widths))
            fsock.writelines('double precision '+','.join(widths)+'\n')
            fsock.writelines('common/widths/ '+','.join(widths)+'\n\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_real_format+' '+\
                                                          ','.join(widths)+'\n')
                mp_fsock_same_name.writelines('common/MP_widths/ '+\
                                                        ','.join(widths)+'\n\n')                
                mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                                        self.mp_prefix+w for w in widths])+'\n')
                mp_fsock.writelines('common/MP_widths/ '+\
                            ','.join([self.mp_prefix+w for w in widths])+'\n\n')
        
        # Write the Couplings
        if self.coups_indep:
            c_list = [coupl.name for coupl in self.coups_indep_noloop + self.coups_indep_loop]  
            if c_list:
                fsock.writelines('double complex, target :: '+', '.join(c_list)+'\n') 

        # Write the flavor couplings 
        if self.coups_flv_indep:
            c_list = [coupl.name for coupl in self.coups_flv_indep]
            fsock.writelines('type(flv_coupling) '+', '.join(c_list)+'\n')

        # Write the dependent coupling 
        if self.vector_size and not self.opt['loop_induced']:
            c_list = ['%s(%s)' %(coupl.name, "VECSIZE_MEMMAX") for coupl in self.coups_dep]
        else:
            c_list = [coupl.name for coupl in self.coups_dep] 
        
        if c_list:
            fsock.writelines('double complex, target :: '+', '.join(c_list)+'\n')  

        # Write the flavor dependent couplings
        if self.vector_size and not self.opt['loop_induced']:
            c_list = ['%s(%s)' %(coupl.name, "VECSIZE_MEMMAX") for coupl in self.coups_flv_dep]
        else:
            c_list = [coupl.name for coupl in self.coups_flv_dep] 
        
        if c_list:
            fsock.writelines('type(flv_coupling) '+', '.join(c_list)+'\n')
            if self.opt['loop_induced']:
                raise Exception('Flavor coupling are not supported for loop induced process for the moment')  


        coupling_list = [coupl.name for coupl in self.coups_dep + self.coups_indep_noloop + self.coups_indep_loop + self.coups_flv_dep + self.coups_flv_indep]       

        fsock.writelines('common/couplings/ '+', '.join(coupling_list)+'\n')
        if self.opt['mp']:
            c_list = [coupl.name for coupl in self.coups_indep] 
            if c_list: 
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                   ','.join(c_list)+'\n')
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                 self.mp_prefix+c for c in c_list])+'\n')
            if False: #no vector handling in quadruple for the moment
                c_list = ['%s(%s)' %(coupl.name, "VECSIZE_MEMMAX") for coupl in self.coups_dep]
            else:
                c_list = [coupl.name for coupl in self.coups_dep] 
            if c_list: 
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                   ','.join(c_list)+'\n')
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                 self.mp_prefix+c for c in c_list])+'\n')
            mp_fsock_same_name.writelines('common/MP_couplings/ '+\
                                                 ','.join(coupling_list)+'\n\n')                

            mp_fsock.writelines('common/MP_couplings/ '+\
                     ','.join([self.mp_prefix+c for c in coupling_list])+'\n\n')            
        
        # Write complex mass for complex mass scheme (if activated)
        if self.opt['complex_mass'] and complex_mass:
            fsock.writelines('double complex '+', '.join(complex_mass)+'\n')
            fsock.writelines('common/complex_mass/ '+', '.join(complex_mass)+'\n')
            if self.opt['mp']:
                mp_fsock_same_name.writelines(self.mp_complex_format+' '+\
                                                    ','.join(complex_mass)+'\n')
                mp_fsock_same_name.writelines('common/MP_complex_mass/ '+\
                                                  ','.join(complex_mass)+'\n\n')                
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                                self.mp_prefix+cm for cm in complex_mass])+'\n')
                mp_fsock.writelines('common/MP_complex_mass/ '+\
                    ','.join([self.mp_prefix+cm for cm in complex_mass])+'\n\n')                       
        
    def create_write_couplings(self):
        """ write the file coupl_write.inc """
        
        fsock = self.open('coupl_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' Couplings of %s'  
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""" % self.model_name)
        def format(coupl):
            return 'write(*,2) \'%(name)s = \', %(name)s' % {'name': coupl.name}
        
        # Write the Couplings
        lines = [format(coupl) for coupl in self.coups_dep + self.coups_indep_noloop + self.coups_indep_loop]       
        fsock.writelines('\n'.join(lines))
        
        
    def create_input(self):
        """create input.inc containing the definition of the parameters"""
        
        fsock = self.open('input.inc', format='fortran')
        if self.opt['mp']:
            mp_fsock = self.open('mp_input.inc', format='fortran')
                    
        #find mass/ width since they are already define
        already_def = set()
        for particle in self.model.get('particles'):
            already_def.add(particle.get('mass').lower())
            already_def.add(particle.get('width').lower())
            if self.opt['complex_mass']:
                already_def.add('cmass_%s' % particle.get('mass').lower())
        
        is_valid = lambda name: name.lower() not in ['g', 'mu_r', 'zero'] and \
                                                 name.lower() not in already_def
        
        real_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'real'
                            and is_valid(param.name)]

        real_parameters += [param.name for param in self.params_ext 
                            if param.type == 'real'and 
                               is_valid(param.name)]
        
        # check the parameter is a CT parameter or not
        # if yes, just use the needed ones        
        real_parameters = [param for param in real_parameters \
                                           if self.check_needed_param(param)]

        real_parameters += ['mdl__%s__scale' % s for s in self.scales]
        
        fsock.writelines('double precision '+','.join(real_parameters)+'\n')
        fsock.writelines('common/params_R/ '+','.join(real_parameters)+'\n\n')
        if self.opt['mp']:
            mp_fsock.writelines(self.mp_real_format+' '+','.join([\
                              self.mp_prefix+p for p in real_parameters])+'\n')
            mp_fsock.writelines('common/MP_T_params_R/ '+','.join([\
                            self.mp_prefix+p for p in real_parameters])+'\n\n')        
        
        complex_parameters = [param.name for param in self.params_dep + 
                            self.params_indep if param.type == 'complex' and
                            is_valid(param.name)]

        # check the parameter is a CT parameter or not
        # if yes, just use the needed ones        
        complex_parameters = [param for param in complex_parameters \
                             if self.check_needed_param(param)]

        if complex_parameters:
            fsock.writelines('double complex '+','.join(complex_parameters)+'\n')
            fsock.writelines('common/params_C/ '+','.join(complex_parameters)+'\n\n')
            if self.opt['mp']:
                mp_fsock.writelines(self.mp_complex_format+' '+','.join([\
                            self.mp_prefix+p for p in complex_parameters])+'\n')
                mp_fsock.writelines('common/MP_params_C/ '+','.join([\
                          self.mp_prefix+p for p in complex_parameters])+'\n\n')

    def check_needed_param(self, param):
        """ Returns whether the parameter in argument is needed for this 
        specific computation or not."""
    
        # If this is a leading order model or if there was no CT parameter
        # employed in this NLO model, one can directly return that the 
        # parameter is needed since only CTParameters are filtered.
        if not hasattr(self, 'allCTparameters') or \
               self.allCTparameters is None or self.usedCTparameters is None or \
               len(self.allCTparameters)==0:
            return True
         
        # We must allow the conjugate shorthand for the complex parameter as
        # well so we check wether either the parameter name or its name with
        # 'conjg__' substituted with '' is present in the list.
        # This is acceptable even if some parameter had an original name 
        # including 'conjg__' in it, because at worst we export a parameter 
        # was not needed.
        param = param.lower()
        cjg_param = param.replace('conjg__','',1)
                
        # First make sure it is a CTparameter
        if param not in self.allCTparameters and \
                                          cjg_param not in self.allCTparameters:
            if hasattr(self.model, "notused_ct_params"):
                if param.endswith(('_fin_','_1eps_','_2eps_')):
                    limit = -2
                elif param.endswith(('_1eps','_2eps')):
                    limit =-1
                else:
                    limit = 0
                base = '_'.join(param.split('_')[1:limit])
                if base in self.model.notused_ct_params:
                    return False
            return True
        
        # Now check if it is in the list of CTparameters actually used
        return (param in self.usedCTparameters or \
                                             cjg_param in self.usedCTparameters)
                
    def extract_needed_CTparam(self,wanted_couplings=[]):
        """ Extract what are the needed CT parameters given the wanted_couplings"""
        
        if not hasattr(self.model,'map_CTcoup_CTparam') or not wanted_couplings:
            # Setting these lists to none wil disable the filtering in 
            # check_needed_param
            self.allCTparameters  = None
            self.usedCTparameters = None
            return
        
        # All CTparameters appearin in all CT couplings        
        allCTparameters=list(self.model.map_CTcoup_CTparam.values())
        # Define in this class the list of all CT parameters
        self.allCTparameters=list(\
                            set(itertools.chain.from_iterable(allCTparameters)))

        # All used CT couplings
        w_coupls = [coupl.lower() for coupl in wanted_couplings if isinstance(coupl,str)]
        logger.debug('wanted_couplings: CTparan not supporting merging -> will be problematic for NLO')
        allUsedCTCouplings = [coupl for coupl in 
              self.model.map_CTcoup_CTparam.keys() if coupl.lower() in w_coupls]
        
        # Now define the list of all CT parameters that are actually used
        self.usedCTparameters=list(\
          set(itertools.chain.from_iterable([
            self.model.map_CTcoup_CTparam[coupl] for coupl in allUsedCTCouplings
                                                                            ])))       
        
        # Now at last, make these list case insensitive
        self.allCTparameters = [ct.lower() for ct in self.allCTparameters]
        self.usedCTparameters = [ct.lower() for ct in self.usedCTparameters]
        

    def create_printout(self):
        """create printout.f"""

        replace_dict = {'include_vector': "include '../vector.inc' ! VECSIZE_MEMMAX (needed by coupl.inc)"}

        if not self.vector_size:
            replace_dict['include_vector'] = ''

        fsock = self.open('printout.f', format='fortran')
        text = open(pjoin(MG5DIR , 'models', 'template_files','fortran', 'printout.f')).read()
        text = text % replace_dict
        fsock.write(text)


    def create_ewa(self):
        """create electroweakFlux.inc 
           this file only need the correct name for the mass for the W and Z
        """

        try:
            fsock = self.open(pjoin(self.dir_path,'../PDF/ElectroweakFlux.inc'), format='fortran')
        except:
            logger.debug('No PDF directory do not cfeate ElectroweakFlux.inc')
            return

        masses = {'MZ': '0d0', 'MW': '0d0'}
        count = 0
        for particle in self.model['particles']:
            if particle.get('pdg_code') == 24:
                masses['MW'] = particle.get('mass')
                count += 1
            elif particle.get('pdg_code') == 23:
                masses['MZ'] =  particle.get('mass')
                count += 1
            if count == 2:
                break

        template = open(pjoin(MG5DIR,'madgraph/iolibs/template_files/madevent_electroweakFlux.inc')).read()
        fsock.write(template % masses)                 
        fsock.close()

    def create_intparam_def(self, dp=True, mp=False):
        """ create intparam_definition.inc setting the internal parameters.
        Output the double precision and/or the multiple precision parameters
        depending on the parameters dp and mp. If mp only, then the file names
        get the 'mp_' prefix.
         """

        fsock = self.open('%sintparam_definition.inc'%
                             ('mp_' if mp and not dp else ''), format='fortran')
        
        fsock.write_comments(\
                "Parameters that should not be recomputed event by event.\n")
        fsock.writelines("if(readlha) then\n")
        if dp:        
            fsock.writelines("G = 2 * DSQRT(AS*PI) ! for the first init\n")
        if mp:
            fsock.writelines("MP__G = 2 * SQRT(MP__AS*MP__PI) ! for the first init\n")
            
        for param in self.params_indep:
            if param.name == 'ZERO':
                continue
            # check whether the parameter is a CT parameter
            # if yes,just used the needed ones
            if not self.check_needed_param(param.name):
                continue
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            if mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))    

        fsock.writelines('endif')
        
        fsock.write_comments('\nParameters that should be recomputed at an event by even basis.\n')
        if dp:        
            fsock.writelines("aS = G**2/4/pi\n")
        if mp:
            fsock.writelines("MP__aS = MP__G**2/4/MP__PI\n")

        # these are the parameters needed for the loops
        if hasattr(self, 'allCTparameters') and self.allCTparameters:
            ct_params = [param for param in self.params_dep \
                if self.check_needed_param(param.name) and \
                   param.name.lower() in self.allCTparameters]
        else:
            ct_params = []
        
        for param in self.params_dep:
            # skip the CT parameters, which have already been done before
            if not self.check_needed_param(param.name) or param in ct_params:
                continue
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            elif mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))

        fsock.write_comments('\nParameters that should be updated for the loops.\n')

        # do not skip the evaluation of these parameters in MP
        if not mp and ct_params: fsock.writelines('if (updateloop) then')
        for param in ct_params:
            if dp:
                fsock.writelines("%s = %s\n" % (param.name,
                                            self.p_to_f.parse(param.expr)))
            elif mp:
                fsock.writelines("%s%s = %s\n" % (self.mp_prefix,param.name,
                                            self.mp_p_to_f.parse(param.expr)))

        if not mp and ct_params: fsock.writelines('endif')

        fsock.write_comments("\nDefinition of the EW coupling used in the write out of aqed\n")

        # Let us not necessarily investigate the presence of alpha_EW^-1 of Gf as an external parameter, but also just as a parameter
        if ('aEWM1',) in self.model['parameters'] or \
           any( ('aEWM1'.lower() in [p.name.lower() for p in p_list]) for p_list in self.model['parameters'].values() ):
            if dp:
                fsock.writelines(""" gal(1) = 3.5449077018110318d0 / DSQRT(ABS(aEWM1))
                                 gal(2) = 1d0
                         """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 2 * SQRT(MP__PI/ABS(MP__aEWM1))
                                 %(mp_prefix)sgal(2) = 1d0 
                                 """ %{'mp_prefix':self.mp_prefix})
                pass
        # in Gmu scheme, aEWM1 is not external but Gf is an exteranl variable
        elif ('Gf',) in self.model['parameters']:
            # Make sure to consider complex masses if the complex mass scheme is activated
            if self.opt['complex_mass']:
                mass_prefix = 'CMASS_MDL_'
            else:
                mass_prefix = 'MDL_'

            if dp:
                if self.opt['complex_mass']:
                    fsock.writelines(""" gal(1) = ABS(2.378414230005442133435d0*%(mass_prefix)sMW*SQRT(DCMPLX(1.0D0,0.0d0)-%(mass_prefix)sMW**2/%(mass_prefix)sMZ**2)*DSQRT(MDL_Gf))
                                 gal(2) = 1d0
                         """%{'mass_prefix':mass_prefix})
                else:
                    fsock.writelines(""" gal(1) = 2.378414230005442133435d0*%(mass_prefix)sMW*DSQRT(1D0-%(mass_prefix)sMW**2/%(mass_prefix)sMZ**2)*DSQRT(MDL_Gf)
                                 gal(2) = 1d0
                         """%{'mass_prefix':mass_prefix})
            elif mp:
                if self.opt['complex_mass']:
                    fsock.writelines(""" %(mp_prefix)sgal(1) = ABS(2*%(mp_prefix)s%(mass_prefix)sMW*SQRT(CMPLX(1e0_16,0.0e0_16,KIND=16)-%(mp_prefix)s%(mass_prefix)sMW**2/%(mp_prefix)s%(mass_prefix)sMZ**2)*SQRT(SQRT(2e0_16)*%(mp_prefix)sMDL_Gf))
                                 %(mp_prefix)sgal(2) = 1e0_16
                                 """ %{'mp_prefix':self.mp_prefix,'mass_prefix':mass_prefix})
                else:
                    fsock.writelines(""" %(mp_prefix)sgal(1) = 2*%(mp_prefix)s%(mass_prefix)sMW*SQRT(1e0_16-%(mp_prefix)s%(mass_prefix)sMW**2/%(mp_prefix)s%(mass_prefix)sMZ**2)*SQRT(SQRT(2e0_16)*%(mp_prefix)sMDL_Gf)
                                 %(mp_prefix)sgal(2) = 1e0_16
                                 """ %{'mp_prefix':self.mp_prefix,'mass_prefix':mass_prefix})

                pass
        else:
            if dp:
                logger.warning('$RED aEWM1 and Gf not define in MODEL. AQED will not be written correcty in LHE FILE')
                fsock.writelines(""" gal(1) = 1d0
                                 gal(2) = 1d0
                             """)
            elif mp:
                fsock.writelines(""" %(mp_prefix)sgal(1) = 1e0_16
                                 %(mp_prefix)sgal(2) = 1e0_16
                             """%{'mp_prefix':self.mp_prefix})

    nb_def_by_file = 50
    def create_couplings(self):
        """ create couplings.f and all couplingsX.f """
        
        nb_def_by_file = self.nb_def_by_file
        
        self.create_couplings_main(nb_def_by_file)
        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        
        # For flavor merged couplings, (only dp so far) we need to add the new datastructure
        # and initialise those correctly.
        self.create_couplings_flavor_merged()

        for i in range(nb_coup_indep_noloop):            
            ##### For the independent couplings, we compute the double and multiple
            ##### precision ones together
            # For the EW sudakov approximation, because of the numerical derivatives
            # we need to separate MP vs DP also here
            data = self.coups_indep_noloop[nb_def_by_file * i: 
                             min(len(self.coups_indep_noloop), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1, data, dp=True, mp=False)

            if self.opt['mp']:
                self.create_couplings_part( i + 1, data, dp=False,mp=True)

        for i in range(nb_coup_indep_loop):
            ##### For the independent couplings, we compute the double and multiple
            ##### precision ones together
            # For the EW sudakov approximation, because of the numerical derivatives
            # we need to separate MP vs DP also here
            data = self.coups_indep_loop[nb_def_by_file * i: 
                             min(len(self.coups_indep_loop), nb_def_by_file * (i+1))]
            self.create_couplings_part(i + 1 + nb_coup_indep_noloop, data, dp=True, mp=False)

            if self.opt['mp']:
                self.create_couplings_part( i + 1 + nb_coup_indep_noloop, data, dp=False,mp=True)
            
        for i in range(nb_coup_dep):
            # For the dependent couplings, we compute the double and multiple
            # precision ones in separate subroutines.
            nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
            data = self.coups_dep[nb_def_by_file * i: 
                               min(len(self.coups_dep), nb_def_by_file * (i+1))]
            self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                        dp=True, mp=False, vec=self.vector_size*self.nb_warp)
            if self.opt['mp']:
                self.create_couplings_part( i + 1 + nb_coup_indep , data, 
                                           dp=False, mp=True, vec=self.vector_size*self.nb_warp)
        
    
    def create_couplings_flavor_merged(self):
        """ create the flavor merged couplings """

        template = """
       MODULE MODEL_OBJECT
       type coupptr ! needed to have an array of pointer
           SEQUENCE
           double complex, pointer :: p
       end type coupptr

       TYPE FLV_COUPLING
         SEQUENCE
         INTEGER :: PARTNER(%(max_flavor)i)
         INTEGER :: PARTNER2(%(max_flavor)i)
         TYPE(COUPPTR) :: VAL(%(max_flavor)i)
         END TYPE FLV_COUPLING
         END MODULE MODEL_OBJECT


         subroutine init_flv_couplings()
            use model_object
            implicit none
            %(include_vector)s
            include 'coupl.inc'
            %(loop_decl)s

            %(def_flv)s
        end subroutine init_flv_couplings
            """

        # Single source of truth for the (k1, k2) PARTNER/PARTNER2 indices,
        # shared with the C++/Python backends (see FLV_Coupling docstring).
        _get_k1_k2 = base_objects.FLV_Coupling.get_partner_indices

        def_flv = []
        for coupl in self.coups_flv_indep:
            for key, c in coupl.flavors.items():
                k1, k2 = _get_k1_k2(key)
                def_flv.append(misc.apply_template('%(name)s % PARTNER(%(in)i) = %(out)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                def_flv.append(misc.apply_template('%(name)s % PARTNER2(%(out)i) = %(in)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                def_flv.append(misc.apply_template('%(name)s % VAL(%(in)i) %p  =>  %(coupl)s', {'name': coupl.name, 'in': k1, 'coupl': c}))

        # For alpha_s-dependent flavor couplings the underlying coupling and the
        # FLV_COUPLING itself are both declared as arrays of size VECSIZE_MEMMAX.
        # A scalar pointer cannot be associated to the whole array, so we use a
        # do-loop to point each FLV_COUPLING(j) % VAL(k) % p to its corresponding
        # coupling array element.
        if self.coups_flv_dep:
            if self.vector_size:
                loop_lines = []
                for coupl in self.coups_flv_dep:
                    for key, c in coupl.flavors.items():
                        k1, k2 = _get_k1_k2(key)
                        loop_lines.append(misc.apply_template('%(name)s(j_flv_init) % PARTNER(%(in)i) = %(out)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                        loop_lines.append(misc.apply_template('%(name)s(j_flv_init) % PARTNER2(%(out)i) = %(in)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                        loop_lines.append(misc.apply_template('%(name)s(j_flv_init) % VAL(%(in)i) %p  =>  %(coupl)s(j_flv_init)', {'name': coupl.name, 'in': k1, 'coupl': c}))
                def_flv.append('do j_flv_init = 1, VECSIZE_MEMMAX')
                def_flv.extend(['  ' + l for l in loop_lines])
                def_flv.append('end do')
            else:
                # Non-vectorized dep couplings: same scalar pointer assignment as indep
                for coupl in self.coups_flv_dep:
                    for key, c in coupl.flavors.items():
                        k1, k2 = _get_k1_k2(key)
                        def_flv.append(misc.apply_template('%(name)s % PARTNER(%(in)i) = %(out)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                        def_flv.append(misc.apply_template('%(name)s % PARTNER2(%(out)i) = %(in)i', {'name': coupl.name, 'in': k1, 'out': k2}))
                        def_flv.append(misc.apply_template('%(name)s % VAL(%(in)i) %p  =>  %(coupl)s', {'name': coupl.name, 'in': k1, 'coupl': c}))

        # max size needed for the couplings
        max_flavor = max([len(ids) for ids in self.model['merged_particles'].values()], default=0)

        if self.vector_size:
            include_vector = "include \'../vector.inc\'\n"
            loop_decl = 'integer j_flv_init' if self.coups_flv_dep else ''
        else:
            include_vector = ''
            loop_decl = ''
        replace = {'max_flavor': max_flavor,
                   'include_vector': include_vector,
                   'loop_decl': loop_decl,
                   'def_flv': '\n'.join(def_flv)}
        fsock = self.open('flavor_couplings.f', format='fortran')
        fsock.writelines(template % replace)

        fsock.close()

        # get the list of matrix couplings
        #for interactions in self.model['interactions']:
            # is it too late?




    def create_couplings_main(self, nb_def_by_file=25):
        """ create couplings.f """

        fsock = self.open('couplings.f', format='fortran')
        
        fsock.writelines("""subroutine coup()
                            use model_object
                            implicit none
                            double precision PI, ZERO
                            logical READLHA
                            parameter  (PI=3.141592653589793d0)
                            parameter  (ZERO=0d0)
                            include \'model_functions.inc\'""")
        if self.vector_size:
            fsock.writelines("include \'../vector.inc\'\n")
        if self.opt['mp']:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 
            
        fsock.writelines("""logical updateloop
                            common /to_updateloop/updateloop
                            include \'input.inc\'
                         """)

        fsock.writelines("""    
                            include \'coupl.inc\'
                            READLHA = .true.
                            include \'intparam_definition.inc\'""")
        if self.opt['mp']:
            fsock.writelines("if (updateloop) then\n")
            fsock.writelines("""include \'mp_intparam_definition.inc\'\n""")
            fsock.writelines("endif\n")
        
        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
        


        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1) for i in range(nb_coup_indep_noloop)]))

        fsock.writelines('if (updateloop) then\n')
        fsock.writelines('\n'.join(\
                    ['call coup%s()' %  (i + 1 + nb_coup_indep_noloop) for i in range(nb_coup_indep_loop)]))
        fsock.writelines('\nendif\n')
        
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        fsock.writelines('\n'.join(\
                    ['call coup%(i)s(%(args)s)' %  {'i': nb_coup_indep + i + 1,
                                                    'args':'1' if self.vector_size  else ''} \
                      for i in range(nb_coup_dep)]))

        # the MP-version is there also for those couplings which do not depend 
        #  on the PSP
        if self.opt['mp']:
            fsock.write_comments('\ncouplings in multiple precision\n')

            fsock.writelines('if (updateloop) then\n')

            fsock.writelines('\n'.join(\
                    ['call mp_coup%s()' %  (i + 1) for i in range(nb_coup_indep)]))
        
            fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

            fsock.writelines('\n'.join(\
                    ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                      for i in range(nb_coup_dep)]))

            fsock.writelines('\nendif\n')

        if self.coups_flv_dep or self.coups_flv_indep:
            fsock.writelines('call init_flv_couplings()\n')

        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param(%(args)s)
                            use model_object
                            implicit none
                            %(args_dep)s
                            double precision PI, ZERO
                            logical READLHA, FIRST
                            data first /.true./
                            save first
                            parameter  (PI=3.141592653589793d0)            
                            parameter  (ZERO=0d0)
                            logical updateloop
                            common /to_updateloop/updateloop
                            include \'model_functions.inc\'
                            double precision Gother
                            
                            double precision model_scale
                            common /model_scale/model_scale
                            """ % \
                            {'args': 'vecid' if (self.vector_size) else '',
                            'args_dep': ' integer vecid' if self.vector_size  else ''}
                         )


        if self.opt['export_format'] in ['madevent']:
            fsock.writelines("""
                            include \'../maxparticles.inc\'
                            include \'../cuts.inc\'
                             """)
            if self.vector_size:
                fsock.writelines("""
                            include \'../vector.inc\'
                                 """)
            fsock.writelines("""            
                            include \'../run.inc\'""")        
        elif self.opt['export_format'] in  ['madloop_optimized']:
            if self.vector_size:
                fsock.writelines("""
                            include \'../vector.inc\'
                                 """)
            fsock.writelines("""
                            include \'../maxparticles.inc\'
                            include \'../cuts.inc\'
                            include \'../run.inc\'""")
        else:
            fsock.writelines("""
                            include \'../cuts.inc\'
                            data maxjetflavor,fixed_extra_scale,mue_over_ref,mue_ref_fixed /5,.false.,1d0,91.188/
                            include \'../run.inc\'""")
        fsock.writelines("""
                            double precision alphas 
                            external alphas
                            """)

        fsock.writelines("""include \'input.inc\'
                            include \'coupl.inc\'
                            READLHA = .false.""")
        fsock.writelines("""    
                            include \'intparam_definition.inc\'\n
                            
                         """)
        
        if self.model['running_elements']:
            running_block = self.model.get_running(self.used_running_key) 
            if running_block:
                MUE = [p for p in self.model.get('parameters')[('external',)] if p.lhablock.lower() == 'loop' and tuple(p.lhacode) == (2,)]

                
                
                fsock.write_comments('calculate the running parameter')
                fsock.writelines(' if(fixed_extra_scale.and.first) then')
                if self.MUE:
                    fsock.writelines(' %s = mue_ref_fixed' % self.MUE.name)
                fsock.writelines(' Gother = SQRT(4.0D0*PI*ALPHAS(mue_ref_fixed))') 
                fsock.writelines(' first = .false.') 
                for i in range(len(running_block)):
                    fsock.writelines(" call C_RUNNING_%s(Gother) ! %s \n" % (i+1,list(running_block[i])))   
                fsock.writelines(' elseif(.not.fixed_extra_scale) then')
                fsock.writelines(' Gother = G')
                
                if self.MUE:
                    fsock.writelines(' %s = mue_over_ref*model_scale' % self.MUE.name)
                else:
                    misc.sprint('NO MUE')
                    #raise Exception
                
                fsock.writelines(' if(mue_over_ref.ne.1d0)then')
                fsock.writelines('  Gother = SQRT(4.0D0*PI*ALPHAS(mue_over_ref*model_scale))')
                fsock.writelines(' endif')
                
                for i in range(len(running_block)):
                    fsock.writelines(" call C_RUNNING_%s(Gother) ! %s \n" % (i+1,list(running_block[i])))   
                fsock.writelines('endif')

        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 
                
        fsock.write_comments('\ncouplings needed to be evaluated points by points\n')

        if self.vector_size:
            fsock.writelines("""     ALL_G(VECID) = G   """)

        fsock.writelines('\n'.join(\
                    ['call coup%(i)s(%(args)s)' %  {"i": nb_coup_indep + i + 1, "args": 'vecid' if self.vector_size  else ''} \
                      for i in range(nb_coup_dep)]))
        fsock.writelines('''\n return \n end\n''')

        fsock.writelines("""subroutine update_as_param2(mu_r2,as2 %(args)s)

                            use model_object
                            implicit none
                            
                            double precision PI
                            parameter  (PI=3.141592653589793d0)
                            double precision mu_r2, as2
                            %(args_dep)s
                            include \'model_functions.inc\'"""%
                            {'args': ',vecid' if self.vector_size else '',
                            'args_dep': ' integer vecid' if self.vector_size else ''
                            })
        fsock.writelines("""include \'input.inc\'
                         """)
        if self.vector_size:
            fsock.writelines("       include \'../vector.inc\'\n")
        fsock.writelines("""include \'coupl.inc\'
                            double precision model_scale
                            common /model_scale/model_scale
                            """)
        fsock.writelines("""
                            if (mu_r2.gt.0d0) MU_R = DSQRT(mu_r2)
                            model_scale = DSQRT(mu_r2)
                            G = SQRT(4.0d0*PI*AS2) 
                            AS = as2

                            CALL UPDATE_AS_PARAM(%(args)s)
                         """%
                            {'args': 'vecid' if self.vector_size  else '',
                            'args_dep': ' integer vecid' if self.vector_size else ''
                            }
                            )
                         
                         
        fsock.writelines('''\n return \n end\n''')

        # fsock.writelines("""subroutine update_model_to_scale(scale)
        #                     ! scale in GeV
        #                     implicit none
        #                     double precision scale
        #                     double precision PI
        #                     double precision alphas
        #                     external alphas
        #                     parameter  (PI=3.141592653589793d0)
        #                     double precision mu_r2, as2
        #                     include \'model_functions.inc\'""")
        # fsock.writelines("""include \'input.inc\'
        #                     include \'../vector.inc\'
        #                     include \'coupl.inc\'
        #                     """)
        # fsock.writelines("""
        #                     AS = ALPHAS(scale)
        #                     AS2 = AS*AS
        #                     call update_as_param2(scale**2, AS2)
        #                  """)
        # fsock.writelines('''\n return \n end\n''')




        if self.opt['mp']:
            fsock.writelines("""subroutine mp_update_as_param()
    
                                implicit none
                                logical READLHA
                                include \'model_functions.inc\'""")
            if self.vector_size:
                fsock.writelines("""include \'../vector.inc\'\n""")
            fsock.writelines("""%s MP__PI, MP__ZERO
                                    parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                    parameter (MP__ZERO=0e0_16)
                                    include \'mp_input.inc\'
                                    include \'mp_coupl.inc\'
                            """%self.mp_real_format)
            fsock.writelines("""include \'input.inc\'""")

            fsock.writelines("""include \'coupl.inc\'
                                include \'actualize_mp_ext_params.inc\'
                                READLHA = .false.
                                include \'mp_intparam_definition.inc\'\n
                             """)
            
            nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // nb_def_by_file 
            nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // nb_def_by_file 
            nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
            nb_coup_dep = 1 + len(self.coups_dep) // nb_def_by_file 

            if self.model['running_elements']:
                #running_block = self.model.get_running(self.used_running_key) 
                if running_block:
                    fsock.write_comments('calculate the running parameter')
                    for i in range(len(running_block)):
                        fsock.writelines(" call MP_C_RUNNING_%s(G) ! %s \n" % (i+1,list(running_block[i])))   
            
                    
            fsock.write_comments('\ncouplings needed to be evaluated points by points\n')
    
            fsock.writelines('\n'.join(\
                        ['call mp_coup%s()' %  (nb_coup_indep + i + 1) \
                          for i in range(nb_coup_dep)]))
            fsock.writelines('''\n return \n end\n''')
            
        if self.model['running_elements'] and running_block:
            self.write_running_blocks(fsock, running_block)
    
    def write_running_blocks(self, fsock, running_block):
        
        for block_nb, runparams in enumerate(running_block):
            text = self.write_one_running_block(block_nb, runparams)
            fsock.writelines(text)
            
    
    template_running_gs_gs2 = """
                  SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

      include 'input.inc'
      %(mpinput)s

      include '../cuts.inc'
      include '../vector.inc'
      INCLUDE 'coupl.inc'
      double precision GMU


      double complex mat1(%(size)i,%(size)i), mat2(%(size)i,%(size)i), fullmat(%(size)i,%(size)i), matexp(%(size)i,%(size)i)
      data mat2 /%(mat2)s/
      data mat1 /%(mat1)s/
      double precision C0(%(size)i),Cout(%(size)i)
      data C0 /%(size)i * 0d0/
      logical first
      data first /.true./
      integer i,j,k
      double precision G0,beta0, alphas
      external alphas
      data G0 /0d0/
      double precision r1,r2
      if (first) then
         %(initc0)s
         G0 = SQRT(4.0D0*PI*ALPHAS(mdl__%(scale)s__scale))
         %(check_scale)s
         first = .false.
      endif
      beta0 = 11. - 2./3. * maxjetflavor
      r1 = (1/GMU -1/G0)/ beta0
      r2 = DLOG(G0/GMU)/beta0
      do j=1,%(size)i
         do i=1,%(size)i
            fullmat(j,i) = mat1(j,i) *r1 + mat2(j,i)*r2
         enddo
      enddo
      call c8mat_expm1( %(size)i, fullmat, matexp)
      do j=1,%(size)i
         Cout(j) = 0d0
      enddo

      do i=1,%(size)i
         do j=1,%(size)i
            Cout(j) = Cout(j) + matexp(j,i) * c0(i)
         enddo
      enddo

      %(assignc)s

      return
      end
            """
            
    template_running_gs2 = """
                  SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

      include '../cuts.inc'
      include '../vector.inc'
      INCLUDE 'input.inc'
      %(mpinput)s
      INCLUDE 'coupl.inc'
      double precision GMU

      double complex mat2(%(size)i,%(size)i), fullmat(%(size)i,%(size)i), matexp(%(size)i,%(size)i)
      data mat2 /%(mat2)s/
      double precision C0(%(size)i),Cout(%(size)i)
      data C0 /%(size)i * 0d0/
      logical first
      data first /.true./
      integer i,j,k
      double precision G0,beta0, alphas
      external alphas
      data G0 /0d0/
      double precision r1,r2
      if (first) then
         %(initc0)s
         G0 = SQRT(4.0D0*PI*ALPHAS(mdl__%(scale)s__scale))
         %(check_scale)s
         first = .false.
      endif
      beta0 = 11. - 2./3. * maxjetflavor
      r2 = DLOG(G0/GMU) / beta0 
      do j=1,%(size)i
         do i=1,%(size)i
            fullmat(j,i) = mat2(j,i)*r2
         enddo
      enddo
      call c8mat_expm1( %(size)i, fullmat, matexp)
      do j=1,%(size)i
         Cout(j) = 0d0
      enddo

      do i=1,%(size)i
         do j=1,%(size)i
            Cout(j) = Cout(j) + matexp(j,i) * c0(i)
         enddo
      enddo

      %(assignc)s

      return
      end
            """
            
    template_running_x3 = """
    SUBROUTINE %(mp)sC_RUNNING_%(block_nb)i(GMU)

      IMPLICIT NONE
      DOUBLE PRECISION PI
      PARAMETER  (PI=3.141592653589793D0)

       include '../cuts.inc'
       include '../vector.inc'
      INCLUDE 'input.inc'
      %(mpinput)s
      INCLUDE 'coupl.inc'
      double precision GMU

      double complex mat3
      data mat3 /%(mat3)s/
      double precision C0
      data C0 /0d0/
      logical first
      data first /.true./
      integer i,j,k
      if (first) then
         C0 = %(mp)s%(initc0)s
         first = .false.
         %(check_scale)s
      endif
      
      %(mp)s%(assignc)s =  1/DSQRT( 1/C0/C0 - 2*mat3 *DLOG(MU_R/mdl__%(scale)s__scale))
      
      return
      end
      """
    
    def get_scales(self):

        scales = set()
        
        for elements in self.model["running_elements"]:
            for params in elements.run_objects:
                sparams = [str(p) for p in params]
                if not any(param in runparams for param in sparams):
                    continue
                if 'aS' in sparams or sparams.count('G') == 2:
                    to_update = mat2
                    prefact = 4*math.pi
                    try:
                        sparams.remove('aS')
                    except:
                        sparams.remove('G')
                        sparams.remove('G')
                else:
                    to_update = mat1
                    sparams.remove('G')
                    prefact = 16*math.pi**2
                    
                if len(sparams) == 3:
                    if len(set(sparams)) !=1:
                        raise Exception( "Not supported type of running")
                    mat3 = eval(elements.value)
                    continue
                elif len(sparams) !=2:
                    raise Exception("Not supported type of running")
                id1 = runparams.index(sparams[0])
                id2 = runparams.index(sparams[1])
                assert to_update[id1][id2] == 0
                to_update[id1][id2] = eval(elements.value)*prefact
                for param in params:
                    scales.add(param.lhablock)

        try:
            scales.remove('SMINPUTS')
        except Exception:
            pass

        return scales


    def write_one_running_block(self, block_nb, runparams):
               
        runparams = list(runparams)
        
        size = len(runparams) 
        mat1=[[0]*size for _ in range(size)]
        mat2=[[0]*size for _ in range(size)]
        mat3=0
        scales = set()
        
        for elements in self.model["running_elements"]:
            for params in elements.run_objects:
                sparams = [str(p) for p in params]
                if not any(param in runparams for param in sparams):
                    continue
                if 'aS' in sparams or sparams.count('G') == 2:
                    to_update = mat2
                    prefact = 4*math.pi
                    try:
                        sparams.remove('aS')
                    except:
                        sparams.remove('G')
                        sparams.remove('G')
                else:
                    to_update = mat1
                    sparams.remove('G')
                    prefact = 16*math.pi**2
                    
                if len(sparams) == 3:
                    if len(set(sparams)) !=1:
                        raise Exception( "Not supported type of running")
                    mat3 = eval(elements.value)
                    continue
                elif len(sparams) !=2:
                    raise Exception("Not supported type of running")
                id1 = runparams.index(sparams[0])
                id2 = runparams.index(sparams[1])
                assert to_update[id1][id2] == 0
                try:
                    to_update[id1][id2] = eval(elements.value)*prefact
                except Exception:
                    to_update[id1][id2] = '%s *( %s)' % (prefact, elements.value) 

                for param in params:
                    scales.add(param.lhablock)

        try:
            scales.remove('SMINPUTS')
        except Exception:
            pass
        
        data = {}
        data['block_nb'] = block_nb+1
        data['size'] = size
        data['mp'] = ''
        if mat3:
            template = self.template_running_x3
            data['mat3']
            data['initc0'] = "MDL_%s" % runparams[0]
            data['assignc'] = "MDL_%s" % runparams[0]
            text = template % data
            if self.opt['mp']:
                data['mp'] = 'MP_'
                data['initc0'] = "MP__MDL_%s" % runparams[0]
                data['assignc'] = "MP__MDL_%s" % runparams[0]
                text += template % data 
            return text
        
        data['initc0'] = "\n".join(["c0(%i) = MDL_%s" % (i+1, name)
                                    for i, name in enumerate(runparams)])
        data['assignc'] = "\n".join(["MDL_%s = COUT(%i)" % (name,i+1)
                                    for i, name in enumerate(runparams)])
        data['mp'] = ''
        data['check_scale'] = ''
        
        if len(scales) == 1:
            data['scale'] = scales.pop()
        else:
            one_scale = scales.pop()
            data['scale'] = one_scale
            for scale in scales:
                check_scale = """ if (MDL__%(1)s__SCALE.ne.MDL__%(2)s__SCALE) then
                write(*,*) 'ERROR scale %(1)s and %(2)s need to be equal for the running'
                stop 5
                endif
                """
                data['check_scale'] += check_scale % {'1': one_scale, '2': scale}           

        # need to compute the matrices
        # carefull some component are proportional to aS
        # need to convert those to G^2
        # need to be carefull with prefactor included (none yet)

        
        
        
        
        data['mat1'] = ",".join(["%e" % mat1[j][i] if not isinstance(mat1[j][i], str) else "%e" %0  for i in range(data['size']) for j in range(data['size'])])
        data['mat2'] = ",".join(["%e" % mat2[j][i] if not isinstance(mat2[j][i], str) else "%e" %0 for i in range(data['size']) for j in range(data['size'])])
        
        # add initialization for parameter that have coupling parameter
        for i in range(data['size']):
            for j in range(data['size']):
                if isinstance(mat1[i][j], str):
                    towrite = mat1[i][j].replace('cmath.pi', 'pi')
                    towrite = towrite.replace('cmath.sqrt(', 'SQRT(1d0*')
                    towrite = towrite.replace('math.pi', 'pi')
                    towrite = towrite.replace('math.sqrt(', 'SQRT(1d0*')
                    data['initc0'] += "\n   MAT1(%i,%i) = %s" % (i+1, j+1, towrite)
                if isinstance(mat2[i][j], str):
                    towrite = mat2[i][j].replace('cmath.pi', 'pi')
                    towrite = towrite.replace('cmath.sqrt(', 'SQRT(1d0*')
                    towrite = towrite.replace('math.pi', 'pi')
                    towrite = towrite.replace('math.sqrt(', 'SQRT(1d0*')
                    data['initc0'] += "\n   MAT2(%i,%i) = %s" % (i+1, j+1, towrite)

        data['mpinput'] =''
        if any(mat1[i][j] for i,j in zip(range(size),range(size))):
            template = self.template_running_gs_gs2
        else:
            template = self.template_running_gs2
        
        text = template % data
        if self.opt['mp']:
            data['mp'] = 'MP_'
            data['mpinput']="INCLUDE 'mp_input.inc'"
            data['initc0'] = "\n".join(["c0(%i) = MP__MDL_%s" % (i+1, name)
                                    for i, name in enumerate(runparams)])
            # add initialization for parameter that have coupling parameter
            for i in range(data['size']):
                for j in range(data['size']):
                    if isinstance(mat1[i][j], str):
                        towrite = mat1[i][j].replace('cmath.pi', 'MP__pi')
                        towrite = towrite.replace('cmath.sqrt(', 'SQRT((1_E16*')
                        towrite = towrite.replace('math.pi', 'MP__pi')
                        towrite = towrite.replace('math.sqrt(', 'SQRT(1_E16*')
                        data['initc0'] += "\n   MAT1(%i,%i) = %s" % (i+1, j+1, mat1[i][j].replace('MDL_', 'MP__MDL_'))
                    if isinstance(mat2[i][j], str):
                        towrite = mat2[i][j].replace('cmath.pi', 'MP__pi')
                        towrite = towrite.replace('cmath.sqrt(', 'SQRT((1_E16*')
                        towrite = towrite.replace('math.pi', 'MP__pi')
                        towrite = towrite.replace('math.sqrt(', 'SQRT(1_E16*')
                        data['initc0'] += "\n   MAT2(%i,%i) = %s" % (i+1, j+1, mat2[i][j].replace('MDL_', 'MP__MDL_'))

            data['assignc'] = "\n".join(["MP__MDL_%s = COUT(%i)" % (name,i+1)
                                    for i, name in enumerate(runparams)])
            text += template % data   
            
        return text

    def create_couplings_part(self, nb_file, data, dp=True, mp=False, vec=False):
        """ create couplings[nb_file].f containing information coming from data.
        Outputs the computation of the double precision and/or the multiple
        precision couplings depending on the parameters dp and mp.
        If mp is True and dp is False, then the prefix 'MP_' is appended to the
        filename and subroutine name.
        """

        if self.opt['loop_induced']:
            vec = False
        
        fsock = self.open('%scouplings%s.f' %('mp_' if mp and not dp else '',
                                                     nb_file), format='fortran')
        fsock.writelines("""subroutine %(mp)scoup%(nb_file)s( %(args)s)
          use model_object
          implicit none
          %(def_args)s
          include \'model_functions.inc\'"""% {'mp': 'mp_' if mp and not dp else '',
                                               'nb_file': nb_file,
                                               'args': 'vecid' if (vec and not mp) else '',
                                               'def_args': '  integer vecid' if vec else ''})

        if self.vector_size:
            fsock.writelines("""include '../vector.inc'\n""")

        if dp:
            fsock.writelines("""
              double precision PI, ZERO
              parameter  (PI=3.141592653589793d0)
              parameter  (ZERO=0d0)
              include 'input.inc'""")
            fsock.writelines("""include 'coupl.inc'""")
        if mp:
            fsock.writelines("""%s MP__PI, MP__ZERO
                                parameter (MP__PI=3.1415926535897932384626433832795e0_16)
                                parameter (MP__ZERO=0e0_16)
                                include \'mp_input.inc\'
                                include \'mp_coupl.inc\'
                        """%self.mp_real_format) 

        for coupling in data:
            if dp:  

                fsock.writelines('%(name)s%(index)s = %(expr)s' % {'name': coupling.name,
                                          'index': '(vecid)' if vec else '',
                                          'expr': self.p_to_f.parse(coupling.expr)})
            if mp:
                fsock.writelines('%(mp)s%(name)s%(index)s = %(expr)s' % {'mp': self.mp_prefix,
                                          'name': coupling.name,
                                          'index': '', #no vectorization in quadruple
                                          'expr': self.mp_p_to_f.parse(coupling.expr)})
        fsock.writelines('end')

    def create_model_functions_inc(self):
        """ Create model_functions.inc which contains the various declarations
        of auxiliary functions which might be used in the couplings expressions
        """

        additional_fct = []
        # check for functions define in the UFO model
        ufo_fct = self.model.get('functions')
        if ufo_fct:
            for fct in ufo_fct:
                # already handle by default
                if str(fct.name) not in ["complexconjugate", "re", "im", "sec", 
                       "csc", "asec", "acsc", "theta_function", "cond", 
                       "condif", "reglogp", "reglogm", "reglog", "recms", "arg",
                                    "grreglog","regsqrt","B0F","b0f","sqrt_trajectory",
                                    "log_trajectory"]:
                    additional_fct.append(fct.name)
        # put in lower case and remove duplicate
        additional_fct = list({f.lower():'' for f in additional_fct if f.lower() not in ['condif', 'reglog', 'reglogp', 'reglogm', 'recms', 'arg', 'grreglog', 'regsqrt']}) 
        fsock = self.open('model_functions.inc', format='fortran')
        fsock.writelines("""double complex cond
          double complex condif
          double complex reglog
          double complex reglogp
          double complex reglogm
          double complex regsqrt
          double complex grreglog
          double complex recms
          double complex arg
          double complex B0F
          double complex sqrt_trajectory
          double complex log_trajectory
          %s
          """ % "\n".join(["          double complex %s" % i for i in additional_fct]))

        
        if self.opt['mp']:
            fsock.writelines("""%(complex_mp_format)s mp_cond
          %(complex_mp_format)s mp_condif
          %(complex_mp_format)s mp_reglog
          %(complex_mp_format)s mp_reglogp
          %(complex_mp_format)s mp_reglogm
          %(complex_mp_format)s mp_regsqrt
          %(complex_mp_format)s mp_grreglog
          %(complex_mp_format)s mp_recms
          %(complex_mp_format)s mp_arg
          %(complex_mp_format)s mp_B0F
          %(complex_mp_format)s mp_sqrt_trajectory
          %(complex_mp_format)s mp_log_trajectory
          %(additional)s
          """ %\
          {"additional": "\n".join(["          %s mp_%s" % (self.mp_complex_format, i) for i in additional_fct]),
           'complex_mp_format':self.mp_complex_format
           }) 

    def create_model_functions_def(self):
        """ Create model_functions.f which contains the various definitions
        of auxiliary functions which might be used in the couplings expressions
        Add the functions.f functions for formfactors support
        """

        fsock = self.open('model_functions.f', format='fortran')
        fsock.writelines(r"""double complex function cond(condition,truecase,falsecase)
          implicit none
          double complex condition,truecase,falsecase
          if(condition.eq.(0.0d0,0.0d0)) then
             cond=truecase
          else
             cond=falsecase
          endif
          end
          
          double complex function condif(condition,truecase,falsecase)
          implicit none
          logical condition
          double complex truecase,falsecase
          if(condition) then
             condif=truecase
          else
             condif=falsecase
          endif
          end

          double complex function recms(condition,expr)
          implicit none
          logical condition
          double complex expr
          if(condition)then
             recms=expr
          else
             recms=dcmplx(dble(expr))
          endif
          end

          double complex function reglog(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          if(arg.eq.(0.0d0,0.0d0)) then
             reglog=(0.0d0,0.0d0)
          else
             reglog=log(arg)
          endif
          end

          double complex function reglogp(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          if(arg.eq.(0.0d0,0.0d0))then
             reglogp=(0.0d0,0.0d0)
          else
             if(dble(arg).lt.0.0d0.and.dimag(arg).lt.0.0d0)then
                reglogp=log(arg) + TWOPII
             else
                reglogp=log(arg)
             endif
          endif
          end

          double complex function reglogm(arg_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          if(arg.eq.(0.0d0,0.0d0))then
             reglogm=(0.0d0,0.0d0)
          else
             if(dble(arg).lt.0.0d0.and.dimag(arg).gt.0.0d0)then
                reglogm=log(arg) - TWOPII
             else
                reglogm=log(arg)
             endif
          endif
          end

          double complex function regsqrt(arg_in)
          implicit none
          double complex arg_in
          double complex arg
          arg=arg_in
          if(dabs(dimag(arg)).eq.0.0d0)then
             arg=dcmplx(dble(arg),0.0d0)
          endif
          if(dabs(dble(arg)).eq.0.0d0)then
             arg=dcmplx(0.0d0,dimag(arg))
          endif
          regsqrt=sqrt(arg)
          end

          double complex function grreglog(logsw,expr1_in,expr2_in)
          implicit none
          double complex TWOPII
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex expr1_in,expr2_in
          double complex expr1,expr2
          double precision logsw
          double precision imagexpr
          logical firstsheet
          expr1=expr1_in
          expr2=expr2_in
          if(dabs(dimag(expr1)).eq.0.0d0)then
             expr1=dcmplx(dble(expr1),0.0d0)
          endif
          if(dabs(dble(expr1)).eq.0.0d0)then
             expr1=dcmplx(0.0d0,dimag(expr1))
          endif
          if(dabs(dimag(expr2)).eq.0.0d0)then
             expr2=dcmplx(dble(expr2),0.0d0)
          endif
          if(dabs(dble(expr2)).eq.0.0d0)then
             expr2=dcmplx(0.0d0,dimag(expr2))
          endif
          if(expr1.eq.(0.0d0,0.0d0))then
             grreglog=(0.0d0,0.0d0)
          else
             imagexpr=dimag(expr1)*dimag(expr2)
             firstsheet=imagexpr.ge.0.0d0
             firstsheet=firstsheet.or.dble(expr1).ge.0.0d0
             firstsheet=firstsheet.or.dble(expr2).ge.0.0d0
             if(firstsheet)then
                grreglog=log(expr1)
             else
                if(dimag(expr1).gt.0.0d0)then
                   grreglog=log(expr1) - logsw*TWOPII
                else
                   grreglog=log(expr1) + logsw*TWOPII
                endif
             endif
          endif
          end

          module b0f_caching

          type b0f_node
          double complex p2,m12,m22
          double complex value
          type(b0f_node),pointer::parent
          type(b0f_node),pointer::left
          type(b0f_node),pointer::right
          end type b0f_node

          contains

          subroutine b0f_search(item, head, find)
          implicit none
          type(b0f_node),pointer,intent(inout)::head,item
          logical,intent(out)::find
          type(b0f_node),pointer::item1
          integer::icomp
          find=.false.
          nullify(item%parent)
          nullify(item%left)
          nullify(item%right)
          if(.not.associated(head))then
             head => item
             return
          endif
          item1 => head
          do
             icomp=b0f_node_compare(item,item1)
             if(icomp.lt.0)then
                if(.not.associated(item1%left))then
                   item1%left => item
                   item%parent => item1
                   exit
                else
                   item1 => item1%left
                endif
             elseif(icomp.gt.0)then
                if(.not.associated(item1%right))then
                   item1%right => item
                   item%parent => item1
                   exit
                else
                   item1 => item1%right
                endif
             else
                find=.true.
                item%value=item1%value
                exit
             endif
          enddo
          return
          end

          integer function b0f_node_compare(item1,item2) result(res)
          implicit none
          type(b0f_node),pointer,intent(in)::item1,item2
          res=complex_compare(item1%p2,item2%p2)
          if(res.ne.0)return
          res=complex_compare(item1%m22,item2%m22)
          if(res.ne.0)return
          res=complex_compare(item1%m12,item2%m12)
          return
          end

          integer function real_compare(r1,r2) result(res)
          implicit none
          double precision r1,r2
          double precision maxr,diff
          double precision tiny
          parameter (tiny=-1d-14)
          maxr=max(abs(r1),abs(r2))
          diff=r1-r2
          if(maxr.le.1d-99.or.abs(diff)/max(maxr,1d-99).le.abs(tiny))then
             res=0
             return
          endif
          if(diff.gt.0d0)then
             res=1
             return
          else
             res=-1
             return
          endif
          end

          integer function complex_compare(c1,c2) result(res)
          implicit none
          double complex c1,c2
          double precision r1,r2
          r1=dble(c1)
          r2=dble(c2)
          res=real_compare(r1,r2)
          if(res.ne.0)return
          r1=dimag(c1)
          r2=dimag(c2)
          res=real_compare(r1,r2)
          return
          end

          end module b0f_caching

          double complex function B0F(p2,m12,m22)
          use b0f_caching
          implicit none
          double complex p2,m12,m22
          double complex zero,TWOPII
          parameter (zero=(0.0d0,0.0d0))
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double precision M,M2,Ga,Ga2
          double precision tiny
          parameter (tiny=-1d-14)
          double complex logterms
          double complex log_trajectory
          logical use_caching
          parameter (use_caching=.true.)
          type(b0f_node),pointer::item
          type(b0f_node),pointer,save::b0f_bt
          integer init
          save init
          data init /0/
          logical find
          IF(m12.eq.zero)THEN
c           it is a special case
c           refer to Eq.(5.48) in arXiv:1804.10017
            M=DBLE(p2) ! M^2
            M2=DBLE(m22) ! M2^2
            IF(M.LT.tiny.OR.M2.LT.tiny)THEN
            WRITE(*,*)'ERROR:B0F is not well defined when M^2,M2^2<0'
            STOP
            ENDIF
            M=DSQRT(DABS(M))
            M2=DSQRT(DABS(M2))
            IF(M.EQ.0d0)THEN
               Ga=0d0
            ELSE
               Ga=-DIMAG(p2)/M
            ENDIF
            IF(M2.EQ.0d0)THEN
               Ga2=0d0
            ELSE
               Ga2=-DIMAG(m22)/M2
            ENDIF
            IF(p2.ne.m22.and.p2.ne.zero.and.m22.ne.zero)THEN
               b0f=(m22-p2)/p2*LOG((m22-p2)/m22)
               IF(M.GT.M2.and.Ga*M2.GT.Ga2*M)THEN
                  b0f=b0f-TWOPII
               ENDIF
               RETURN
            ELSE
                WRITE(*,*)'ERROR:B0F is not supported for a simple form'
                STOP
            ENDIF
          ENDIF
c         the general case
c         trajectory method as advocated in arXiv:1804.10017 (Eq.(E.47))
          if(use_caching)then
             if(init.eq.0)then
                nullify(b0f_bt)
                init=1
             endif
             allocate(item)
             item%p2=p2
             item%m12=m12
             item%m22=m22
             find=.false.
             call b0f_search(item,b0f_bt,find)
             if(find)then
                b0f=item%value
                deallocate(item)
                return
             else
                logterms=log_trajectory(100,p2,m12,m22)
                b0f=-LOG(p2/m22)+logterms
                item%value=b0f
                return
             endif
          else
             logterms=log_trajectory(100,p2,m12,m22)
             b0f=-LOG(p2/m22)+logterms
          endif
          RETURN
          end

          double complex function sqrt_trajectory(n_seg,p2,m12,m22)
c         only needed when p2*m12*m22=\=0
          implicit none
          integer n_seg ! number of segments
          double complex p2,m12,m22
          double complex zero,one
          parameter (zero=(0.0d0,0.0d0),one=(1.0d0,0.0d0))
          double complex gamma0,gamma1
          double precision M,Ga,dGa,Ga_start
          double precision Gai,intersection
          double complex argim1,argi,p2i
          double complex gamma0i,gamma1i
          double precision tiny
          parameter (tiny=-1d-24)
          integer i
          double precision prefactor
          IF(ABS(p2*m12*m22).EQ.0d0)THEN
            WRITE(*,*)'ERROR:sqrt_trajectory works when p2*m12*m22/=0'
            STOP
          ENDIF
          M=DBLE(p2) ! M^2
          M=DSQRT(DABS(M))
          IF(M.EQ.0d0)THEN
             Ga=0d0
          ELSE
             Ga=-DIMAG(p2)/M
          ENDIF
c         Eq.(5.37) in arXiv:1804.10017
          gamma0=one+m12/p2-m22/p2
          gamma1=m12/p2-dcmplx(0d0,1d0)*ABS(tiny)/p2
          IF(ABS(Ga).EQ.0d0)THEN
             sqrt_trajectory=SQRT(gamma0**2-4d0*gamma1)
             RETURN
          ENDIF
c         segments from -DABS(tiny*Ga) to Ga
          Ga_start=-DABS(tiny*Ga)
          dGa=(Ga-Ga_start)/n_seg
          prefactor=1d0
          Gai=Ga_start
          p2i=dcmplx(M**2,-Gai*M)
          gamma0i=one+m12/p2i-m22/p2i
          gamma1i=m12/p2i-dcmplx(0d0,1d0)*ABS(tiny)/p2i
          argim1=gamma0i**2-4d0*gamma1i
          DO i=1,n_seg
             Gai=dGa*i+Ga_start
             p2i=dcmplx(M**2,-Gai*M)
             gamma0i=one+m12/p2i-m22/p2i
             gamma1i=m12/p2i-dcmplx(0d0,1d0)*ABS(tiny)/p2i
             argi=gamma0i**2-4d0*gamma1i
             IF(DIMAG(argi)*DIMAG(argim1).LT.0d0)THEN
                intersection=DIMAG(argim1)*(DBLE(argi)-DBLE(argim1))
                intersection=intersection/(DIMAG(argi)-DIMAG(argim1))
                intersection=intersection-DBLE(argim1)
                IF(intersection.GT.0d0)THEN
                   prefactor=-prefactor
                ENDIF
             ENDIF
             argim1=argi
          ENDDO
          sqrt_trajectory=SQRT(gamma0**2-4d0*gamma1)*prefactor
          RETURN
          end

          double complex function log_trajectory(n_seg,p2,m12,m22)
c         sum of log terms appearing in Eq.(5.35) of arXiv:1804.10017
c         only needed when p2*m12*m22=\=0
          implicit none
c         4 possible logarithms appearing in Eq.(5.35) of arXiv:1804.10017
c         log(arg(i)) with arg(i) for i=1 to 4
c         i=1: (ga_{+}-1)
c         i=2: (ga_{-}-1)
c         i=3: (ga_{+}-1)/ga_{+}
c         i=4: (ga_{-}-1)/ga_{-}
          integer n_seg ! number of segments
          double complex p2,m12,m22
          double complex zero,one,half,TWOPII
          parameter (zero=(0.0d0,0.0d0),one=(1.0d0,0.0d0))
          parameter (half=(0.5d0,0.0d0))
          parameter (TWOPII=2.0d0*3.1415926535897932d0*(0.0d0,1.0d0))
          double complex gamma0,gammap,gammam,sqrtterm
          double precision M,Ga,dGa,Ga_start
          double precision Gai,intersection
          double complex argim1(4),argi(4),p2i,sqrttermi
          double complex gamma0i,gammapi,gammami
          double precision tiny
          parameter (tiny=-1d-14)
          integer i,j
          double complex addfactor(4)
          double complex sqrt_trajectory
          IF(ABS(p2*m12*m22).EQ.0d0)THEN
            WRITE(*,*)'ERROR:log_trajectory works when p2*m12*m22/=0'
            STOP
          ENDIF
          M=DBLE(p2) ! M^2
          M=DSQRT(DABS(M))
          IF(M.EQ.0d0)THEN
             Ga=0d0
          ELSE
             Ga=-DIMAG(p2)/M
          ENDIF
c         Eq.(5.36-5.38) in arXiv:1804.10017
          sqrtterm=sqrt_trajectory(n_seg,p2,m12,m22)
          gamma0=one+m12/p2-m22/p2
          gammap=half*(gamma0+sqrtterm)
          gammam=half*(gamma0-sqrtterm)
          IF(ABS(Ga).EQ.0d0)THEN
             log_trajectory=-LOG(gammap-one)-LOG(gammam-one)+gammap*LOG((gammap-one)/gammap)+gammam*LOG((gammam-one)/gammam)
             RETURN
          ENDIF
c         segments from -DABS(tiny*Ga) to Ga
          Ga_start=-DABS(tiny*Ga)
          dGa=(Ga-Ga_start)/n_seg
          addfactor(1:4)=zero
          Gai=Ga_start
          p2i=dcmplx(M**2,-Gai*M)
          sqrttermi=sqrt_trajectory(n_seg,p2i,m12,m22)
          gamma0i=one+m12/p2i-m22/p2i
          gammapi=half*(gamma0i+sqrttermi)
          gammami=half*(gamma0i-sqrttermi)
          argim1(1)=gammapi-one
          argim1(2)=gammami-one
          argim1(3)=(gammapi-one)/gammapi
          argim1(4)=(gammami-one)/gammami
          DO i=1,n_seg
             Gai=dGa*i+Ga_start
             p2i=dcmplx(M**2,-Gai*M)
             sqrttermi=sqrt_trajectory(n_seg,p2i,m12,m22)
             gamma0i=one+m12/p2i-m22/p2i
             gammapi=half*(gamma0i+sqrttermi)
             gammami=half*(gamma0i-sqrttermi)
             argi(1)=gammapi-one
             argi(2)=gammami-one
             argi(3)=(gammapi-one)/gammapi
             argi(4)=(gammami-one)/gammami
             DO j=1,4
                IF(DIMAG(argi(j))*DIMAG(argim1(j)).LT.0d0)THEN
                   intersection=DIMAG(argim1(j))*(DBLE(argi(j))-DBLE(argim1(j)))
                   intersection=intersection/(DIMAG(argi(j))-DIMAG(argim1(j)))
                   intersection=intersection-DBLE(argim1(j))
                   IF(intersection.GT.0d0)THEN
                      IF(DIMAG(argim1(j)).LT.0)THEN
                         addfactor(j)=addfactor(j)-TWOPII
                      ELSE
                         addfactor(j)=addfactor(j)+TWOPII
                      ENDIF
                   ENDIF
                ENDIF
                argim1(j)=argi(j)
              ENDDO
          ENDDO
          log_trajectory=-(LOG(gammap-one)+addfactor(1))-(LOG(gammam-one)+addfactor(2))
          log_trajectory=log_trajectory+gammap*(LOG((gammap-one)/gammap)+addfactor(3))
          log_trajectory=log_trajectory+gammam*(LOG((gammam-one)/gammam)+addfactor(4))
          RETURN
          end
          
          double complex function arg(comnum)
          implicit none
          double complex comnum
          double complex iim 
          iim = (0.0d0,1.0d0)
          if(comnum.eq.(0.0d0,0.0d0)) then
             arg=(0.0d0,0.0d0)
          else
             arg=log(comnum/abs(comnum))/iim
          endif
          end""")
        if self.opt['mp']:
            fsock.writelines(misc.apply_template("""

              %(complex_mp_format)s function mp_cond(condition,truecase,falsecase)
              implicit none
              %(complex_mp_format)s condition,truecase,falsecase
              if(condition.eq.(0.0e0_16,0.0e0_16)) then
                 mp_cond=truecase
              else
                 mp_cond=falsecase
              endif
              end
              
              %(complex_mp_format)s function mp_condif(condition,truecase,falsecase)
              implicit none
              logical condition
              %(complex_mp_format)s truecase,falsecase
              if(condition) then
                 mp_condif=truecase
              else
                 mp_condif=falsecase
              endif
              end

              %(complex_mp_format)s function mp_recms(condition,expr)
              implicit none
              logical condition
              %(complex_mp_format)s expr
              if(condition)then
                 mp_recms=expr
              else
                 mp_recms=cmplx(real(expr),kind=16)
              endif
              end

              
              %(complex_mp_format)s function mp_reglog(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              if(arg.eq.(0.0e0_16,0.0e0_16)) then
                 mp_reglog=(0.0e0_16,0.0e0_16)
              else
                 mp_reglog=log(arg)
              endif
              end

              %(complex_mp_format)s function mp_reglogp(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              if(arg.eq.(0.0e0_16,0.0e0_16))then
                 mp_reglogp=(0.0e0_16,0.0e0_16)
              else
                 if(real(arg,kind=16).lt.0.0e0_16.and.imagpart(arg).lt.0.0e0_16)then
                    mp_reglogp=log(arg) + TWOPII
                 else
                    mp_reglogp=log(arg)
                 endif
              endif
              end
              
              %(complex_mp_format)s function mp_reglogm(arg_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              if(arg.eq.(0.0e0_16,0.0e0_16))then
                 mp_reglogm=(0.0e0_16,0.0e0_16)
              else
                 if(real(arg,kind=16).lt.0.0e0_16.and.imagpart(arg).gt.0.0e0_16)then
                    mp_reglogm=log(arg) - TWOPII
                 else
                    mp_reglogm=log(arg)
                 endif 
              endif
              end

              %(complex_mp_format)s function mp_regsqrt(arg_in)
              implicit none
              %(complex_mp_format)s arg_in
              %(complex_mp_format)s arg
              arg=arg_in
              if(abs(imagpart(arg)).eq.0.0e0_16)then
                 arg=cmplx(real(arg,kind=16),0.0e0_16)
              endif
              if(abs(real(arg,kind=16)).eq.0.0e0_16)then
                 arg=cmplx(0.0e0_16,imagpart(arg))
              endif
              mp_regsqrt=sqrt(arg)
              end

              %(complex_mp_format)s function mp_grreglog(logsw,expr1_in,expr2_in)
              implicit none
              %(complex_mp_format)s TWOPII
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s expr1_in,expr2_in
              %(complex_mp_format)s expr1,expr2
              %(real_mp_format)s logsw
              %(real_mp_format)s imagexpr
              logical firstsheet
              expr1=expr1_in
              expr2=expr2_in
              if(abs(imagpart(expr1)).eq.0.0e0_16)then
                 expr1=cmplx(real(expr1,kind=16),0.0e0_16)
              endif
              if(abs(real(expr1,kind=16)).eq.0.0e0_16)then
                 expr1=cmplx(0.0e0_16,imagpart(expr1))
              endif
              if(abs(imagpart(expr2)).eq.0.0e0_16)then
                 expr2=cmplx(real(expr2,kind=16),0.0e0_16)
              endif
              if(abs(real(expr2,kind=16)).eq.0.0e0_16)then
                 expr2=cmplx(0.0e0_16,imagpart(expr2))
              endif
              if(expr1.eq.(0.0e0_16,0.0e0_16))then
                 mp_grreglog=(0.0e0_16,0.0e0_16)
              else
                 imagexpr=imagpart(expr1)*imagpart(expr2)
                 firstsheet=imagexpr.ge.0.0e0_16
                 firstsheet=firstsheet.or.real(expr1,kind=16).ge.0.0e0_16
                 firstsheet=firstsheet.or.real(expr2,kind=16).ge.0.0e0_16
                 if(firstsheet)then
                    mp_grreglog=log(expr1)
                 else
                    if(imagpart(expr1).gt.0.0e0_16)then
                       mp_grreglog=log(expr1) - logsw*TWOPII
                    else
                       mp_grreglog=log(expr1) + logsw*TWOPII
                    endif
                 endif
              endif
              end

              module mp_b0f_caching

              type mp_b0f_node
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s value
              type(mp_b0f_node),pointer::parent
              type(mp_b0f_node),pointer::left
              type(mp_b0f_node),pointer::right
              end type mp_b0f_node

              contains

              subroutine mp_b0f_search(item, head, find)
              implicit none
              type(mp_b0f_node),pointer,intent(inout)::head,item
              logical,intent(out)::find
              type(mp_b0f_node),pointer::item1
              integer::icomp
              find=.false.
              nullify(item%parent)
              nullify(item%left)
              nullify(item%right)
              if(.not.associated(head))then
                 head => item
                 return
              endif
              item1 => head
              do
                 icomp=mp_b0f_node_compare(item,item1)
                 if(icomp.lt.0)then
                    if(.not.associated(item1%left))then
                       item1%left => item
                       item%parent => item1
                       exit
                    else
                       item1 => item1%left
                    endif
                 elseif(icomp.gt.0)then
                    if(.not.associated(item1%right))then
                       item1%right => item
                       item%parent => item1
                       exit
                     else
                       item1 => item1%right
                     endif
                 else
                     find=.true.
                     item%value=item1%value
                     exit
                 endif
              enddo
              return
              end

              integer function mp_b0f_node_compare(item1,item2) result(res)
              implicit none
              type(mp_b0f_node),pointer,intent(in)::item1,item2
              res=mp_complex_compare(item1%p2,item2%p2)
              if(res.ne.0)return
              res=mp_complex_compare(item1%m22,item2%m22)
              if(res.ne.0)return
              res=mp_complex_compare(item1%m12,item2%m12)
              return
              end

              integer function mp_real_compare(r1,r2) result(res)
              implicit none
              %(real_mp_format)s r1,r2
              %(real_mp_format)s maxr,diff
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              maxr=max(abs(r1),abs(r2))
              diff=r1-r2
              if(maxr.le.1.0e-99_16.or.abs(diff)/max(maxr,1.0e-99_16).le.abs(tiny))then
                 res=0
                 return
              endif
              if(diff.gt.0.0e0_16)then
                 res=1
                 return
              else
                 res=-1
                 return
              endif
              end

              integer function mp_complex_compare(c1,c2) result(res)
              implicit none
              %(complex_mp_format)s c1,c2
              %(real_mp_format)s r1,r2
              r1=real(c1,kind=16)
              r2=real(c2,kind=16)
              res=mp_real_compare(r1,r2)
              if(res.ne.0)return
              r1=imagpart(c1)
              r2=imagpart(c2)
              res=mp_real_compare(r1,r2)
              return
              end

              end module mp_b0f_caching

              %(complex_mp_format)s function mp_b0f(p2,m12,m22)
              use mp_b0f_caching
              implicit none
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,TWOPII
              parameter (zero=(0.0e0_16,0.0e0_16))
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(real_mp_format)s M,M2,Ga,Ga2
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              %(complex_mp_format)s logterms
              %(complex_mp_format)s mp_log_trajectory
              logical use_caching
              parameter (use_caching=.true.)
              type(mp_b0f_node),pointer::item
              type(mp_b0f_node),pointer,save::b0f_bt
              integer init
              save init
              data init /0/
              logical find
              IF(m12.eq.zero)THEN
                 M=real(p2,kind=16)
                 M2=real(m22,kind=16)
                 IF(M.LT.tiny.OR.M2.LT.tiny)THEN
                 WRITE(*,*)'ERROR:MP_B0F is not well defined when M^2,M2^2<0'
                 STOP
                 ENDIF
                 M=sqrt(abs(M))
                 M2=sqrt(abs(M2))
                 IF(M.EQ.0.0e0_16)THEN
                    Ga=0.0e0_16
                 ELSE
                    Ga=-imagpart(p2)/M
                 ENDIF
                 IF(M2.EQ.0.0e0_16)THEN
                    Ga2=0.0e0_16
                 ELSE
                    Ga2=-imagpart(m22)/M2
                 ENDIF
                 IF(p2.NE.m22.AND.p2.NE.zero.AND.m22.NE.zero)THEN
                    mp_b0f=(m22-p2)/p2*log((m22-p2)/m22)
                    IF(M.GT.M2.AND.Ga*M2.GT.Ga2*M)THEN
                       mp_b0f=mp_b0f-TWOPII
                    ENDIF
                    RETURN
                 ELSE
                    WRITE(*,*)'ERROR:MP_B0F is not supported for a simple form'
                    STOP
                 ENDIF
              ENDIF
              if(use_caching)then
                 if(init.eq.0)then
                    nullify(b0f_bt)
                    init=1
                 endif
                 allocate(item)
                 item%p2=p2
                 item%m12=m12
                 item%m22=m22
                 find=.false.
                 call mp_b0f_search(item, b0f_bt, find)
                 if(find)then
                    mp_b0f=item%value
                    deallocate(item)
                    return
                 else
                    logterms=mp_log_trajectory(100,p2,m12,m22)
                    mp_b0f=-LOG(p2/m22)+logterms
                    item%value=mp_b0f
                    return
                 endif
              else
                 logterms=mp_log_trajectory(100,p2,m12,m22)
                 mp_b0f=-LOG(p2/m22)+logterms
              endif
              RETURN
              end

              %(complex_mp_format)s function mp_sqrt_trajectory(n_seg,p2,m12,m22)
              implicit none
              integer n_seg
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,one
              parameter (zero=(0.0e0_16,0.0e0_16),one=(1.0e0_16,0.0e0_16))
              %(complex_mp_format)s gamma0,gamma1
              %(real_mp_format)s M,Ga,dGa,Ga_start
              %(real_mp_format)s Gai,intersection
              %(complex_mp_format)s argim1,argi,p2i
              %(complex_mp_format)s gamma0i,gamma1i
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-24_16)
              integer i
              %(real_mp_format)s prefactor
              IF(ABS(p2*m12*m22).EQ.0.0e0_16)THEN
              WRITE(*,*)'ERROR:mp_sqrt_trajectory works when p2*m12*m22/=0'
              STOP
              ENDIF
              M=real(p2,kind=16)
              M=sqrt(abs(M))
              IF(M.EQ.0.0e0_16)THEN
                 Ga=0.0e0_16
              ELSE
                 Ga=-imagpart(p2)/M
              ENDIF
              gamma0=one+m12/p2-m22/p2
              gamma1=m12/p2-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2
              IF(abs(Ga).EQ.0.0e0_16)THEN
                mp_sqrt_trajectory=sqrt(gamma0**2-4.0e0_16*gamma1)
                RETURN
              ENDIF
              Ga_start=-abs(tiny*Ga)
              dGa=(Ga-Ga_start)/n_seg
              prefactor=1.0e0_16
              Gai=Ga_start
              p2i=cmplx(M**2,-Gai*M)
              gamma0i=one+m12/p2i-m22/p2i
              gamma1i=m12/p2i-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2i
              argim1=gamma0i**2-4.0e0_16*gamma1i
              DO i=1,n_seg
                 Gai=dGa*i+Ga_start
                 p2i=cmplx(M**2,-Gai*M)
                 gamma0i=one+m12/p2i-m22/p2i
                 gamma1i=m12/p2i-cmplx(0.0e0_16,1.0e0_16)*abs(tiny)/p2i
                 argi=gamma0i**2-4.0e0_16*gamma1i
                 IF(imagpart(argi)*imagpart(argim1).LT.0.0e0_16)THEN
                   intersection=imagpart(argim1)*(real(argi,kind=16)-real(argim1,kind=16))
                   intersection=intersection/(imagpart(argi)-imagpart(argim1))
                   intersection=intersection-real(argim1,kind=16)
                   IF(intersection.GT.0.0e0_16)THEN
                      prefactor=-prefactor
                   ENDIF
                 ENDIF
                 argim1=argi
              ENDDO
              mp_sqrt_trajectory=sqrt(gamma0**2-4.0e0_16*gamma1)*prefactor
              RETURN
              end

              %(complex_mp_format)s function mp_log_trajectory(n_seg,p2,m12,m22)
              implicit none
              integer n_seg
              %(complex_mp_format)s p2,m12,m22
              %(complex_mp_format)s zero,one,half,TWOPII
              parameter (zero=(0.0e0_16,0.0e0_16),one=(1.0e0_16,0.0e0_16))
              parameter (half=(0.5e0_16,0.0e0_16))
              parameter (TWOPII=2.0e0_16*3.14169258478796109557151794433593750e0_16*(0.0e0_16,1.0e0_16))
              %(complex_mp_format)s gamma0,gammap,gammam,sqrtterm
              %(real_mp_format)s M,Ga,dGa,Ga_start
              %(real_mp_format)s Gai,intersection
              %(complex_mp_format)s argim1(4),argi(4),p2i,sqrttermi
              %(complex_mp_format)s gamma0i,gammapi,gammami
              %(real_mp_format)s tiny
              parameter (tiny=-1.0e-14_16)
              integer i,j
              %(complex_mp_format)s addfactor(4)
              %(complex_mp_format)s mp_sqrt_trajectory
              IF(abs(p2*m12*m22).eq.0.0e0_16)THEN
              WRITE(*,*)'ERROR:mp_log_trajectory works when p2*m12*m22/=0'
              STOP
              ENDIF
              M=real(p2,kind=16)
              M=sqrt(abs(M))
              IF(M.eq.0.0e0_16)THEN
                 Ga=0.0e0_16
              ELSE
                 Ga=-imagpart(p2)/M
              ENDIF
              sqrtterm=mp_sqrt_trajectory(n_seg,p2,m12,m22)
              gamma0=one+m12/p2-m22/p2
              gammap=half*(gamma0+sqrtterm)
              gammam=half*(gamma0-sqrtterm)
              IF(abs(Ga).EQ.0.0e0_16)THEN
                 mp_log_trajectory=-LOG(gammap-one)-LOG(gammam-one)+gammap*LOG((gammap-one)/gammap)+gammam*LOG((gammam-one)/gammam)
                 RETURN
              ENDIF
              Ga_start=-abs(tiny*Ga)
              dGa=(Ga-Ga_start)/n_seg
              addfactor(1:4)=zero
              Gai=Ga_start
              p2i=cmplx(M**2,-Gai*M)
              sqrttermi=mp_sqrt_trajectory(n_seg,p2i,m12,m22)
              gamma0i=one+m12/p2i-m22/p2i
              gammapi=half*(gamma0i+sqrttermi)
              gammami=half*(gamma0i-sqrttermi)
              argim1(1)=gammapi-one
              argim1(2)=gammami-one
              argim1(3)=(gammapi-one)/gammapi
              argim1(4)=(gammami-one)/gammami
              DO i=1,n_seg
                 Gai=dGa*i+Ga_start
                 p2i=cmplx(M**2,-Gai*M)
                 sqrttermi=mp_sqrt_trajectory(n_seg,p2i,m12,m22)
                 gamma0i=one+m12/p2i-m22/p2i
                 gammapi=half*(gamma0i+sqrttermi)
                 gammami=half*(gamma0i-sqrttermi)
                 argi(1)=gammapi-one
                 argi(2)=gammami-one
                 argi(3)=(gammapi-one)/gammapi
                 argi(4)=(gammami-one)/gammami
                 DO j=1,4
                    IF(imagpart(argi(j))*imagpart(argim1(j)).LT.0.0e0_16)THEN
                       intersection=imagpart(argim1(j))*(real(argi(j),kind=16)-real(argim1(j),kind=16))
                       intersection=intersection/(imagpart(argi(j))-imagpart(argim1(j)))
                       intersection=intersection-real(argim1(j),kind=16)
                       IF(intersection.GT.0.0e0_16)THEN
                          IF(imagpart(argim1(j)).LT.0.0e0_16)THEN
                             addfactor(j)=addfactor(j)-TWOPII
                          ELSE
                             addfactor(j)=addfactor(j)+TWOPII
                          ENDIF
                       ENDIF
                    ENDIF
                    argim1(j)=argi(j)
                 ENDDO
              ENDDO
              mp_log_trajectory=-(LOG(gammap-one)+addfactor(1))-(LOG(gammam-one)+addfactor(2))
              mp_log_trajectory=mp_log_trajectory+gammap*(LOG((gammap-one)/gammap)+addfactor(3))
              mp_log_trajectory=mp_log_trajectory+gammam*(LOG((gammam-one)/gammam)+addfactor(4))
              RETURN
              end
              
              %(complex_mp_format)s function mp_arg(comnum)
              implicit none
              %(complex_mp_format)s comnum
              %(complex_mp_format)s imm
              imm = (0.0e0_16,1.0e0_16)
              if(comnum.eq.(0.0e0_16,0.0e0_16)) then
                 mp_arg=(0.0e0_16,0.0e0_16)
              else
                 mp_arg=log(comnum/abs(comnum))/imm
              endif
              end""", {'complex_mp_format':self.mp_complex_format,'real_mp_format':self.mp_real_format}))


        #check for the file functions.f
        model_path = self.model.get('modelpath')
        if os.path.exists(pjoin(model_path,'Fortran','functions.f')):
            fsock.write_comment_line(' USER DEFINE FUNCTIONS ')
            input = pjoin(model_path,'Fortran','functions.f')
            fsock.writelines(open(input).read())
            fsock.write_comment_line(' END USER DEFINE FUNCTIONS ')
            
        # check for functions define in the UFO model
        ufo_fct = self.model.get('functions')
        if ufo_fct:
            fsock.write_comment_line(' START UFO DEFINE FUNCTIONS ')
            done = []
            for fct in ufo_fct:
                # already handle by default
                if str(fct.name.lower()) not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc", "condif",
                                    "theta_function", "cond", "reglog", "reglogp", "reglogm", "recms","arg",
                                    "grreglog","regsqrt","B0F","b0f","sqrt_trajectory","log_trajectory"]:

                    ufo_fct_template = """
          double complex function %(name)s(%(args)s)
          implicit none
          double complex %(args)s
          %(definitions)s
          %(name)s = %(fct)s

          return
          end
          """
                    str_fct = self.p_to_f.parse(fct.expr)
                    if not self.p_to_f.to_define:
                        definitions = []
                    else:
                        definitions=[]
                        for d in self.p_to_f.to_define:
                            if d == 'pi':
                                definitions.append(' double precision pi')
                                definitions.append(' data pi /3.1415926535897932d0/')
                            else:
                                definitions.append(' double complex %s' % d)
                                
                    text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", ".join(fct.arguments),                
                                'fct': str_fct,
                                'definitions': '\n'.join(definitions)
                                 }

                    fsock.writelines(text)
            if self.opt['mp']:
                fsock.write_comment_line(' START UFO DEFINE FUNCTIONS FOR MP')
                for fct in ufo_fct:
                    # already handle by default
                    if fct.name not in ["complexconjugate", "re", "im", "sec", "csc", "asec", "acsc","condif",
                                        "theta_function", "cond", "reglog", "reglogp","reglogm", "recms","arg",
                                        "grreglog","regsqrt","B0F","b0f","sqrt_trajectory","log_trajectory"]:

                        ufo_fct_template = """
          %(complex_mp_format)s function mp_%(name)s(mp__%(args)s)
          implicit none
          %(complex_mp_format)s mp__%(args)s
          %(definitions)s
          mp_%(name)s = %(fct)s

          return
          end
          """
                        str_fct = self.mp_p_to_f.parse(fct.expr)
                        if not self.mp_p_to_f.to_define:
                            definitions = []
                        else:
                            definitions=[]
                            for d in self.mp_p_to_f.to_define:
                                if d == 'pi': 
                                    definitions.append(' %s mp__pi' % self.mp_real_format)
                                    definitions.append(' data mp__pi /3.141592653589793238462643383279502884197e+00_16/')
                                else:   
                                    definitions.append(' %s mp_%s' % (self.mp_complex_format,d))
                        text = ufo_fct_template % {
                                'name': fct.name,
                                'args': ", mp__".join(fct.arguments),                
                                'fct': str_fct,
                                'definitions': '\n'.join(definitions),
                                'complex_mp_format': self.mp_complex_format
                                 }
                        fsock.writelines(text)


                    
            fsock.write_comment_line(' STOP UFO DEFINE FUNCTIONS ')                    

        

    def create_makeinc(self):
        """create makeinc.inc containing the file to compile """
        
        fsock = self.open('makeinc.inc', comment='#')
        text = 'MODEL = flavor_couplings.o couplings.o lha_read.o printout.o rw_para.o'
        text += ' model_functions.o get_color.o '
        
        if self.opt['export_format'].startswith('standalone'):
            text += ' alfas_functions.o '

        nb_coup_indep_noloop = 1 + len(self.coups_indep_noloop) // self.nb_def_by_file 
        nb_coup_indep_loop = 1 + len(self.coups_indep_loop) // self.nb_def_by_file
        nb_coup_indep = nb_coup_indep_noloop + nb_coup_indep_loop
        nb_coup_dep = 1 + len(self.coups_dep) // self.nb_def_by_file
        couplings_files=['couplings%s.o' % (i+1) \
                                for i in range(nb_coup_dep + nb_coup_indep) ]
        if self.opt['mp']:
            # this part changed to include also the couplings which do not 
            # depend on the PSP
            couplings_files+=['mp_couplings%s.o' % (i+1) \
                                for i in range(nb_coup_dep + nb_coup_indep) ]
        text += ' '.join(couplings_files)
        fsock.writelines(text)
        
    def create_param_write(self):
        """ create param_write """

        fsock = self.open('param_write.inc', format='fortran')
        
        fsock.writelines("""write(*,*)  ' External Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")
        def format(name):
            return 'write(*,*) \'%(name)s = \', %(name)s' % {'name': name}
        
        # Write the external parameter
        # order them in a smart way
        self.params_ext.sort(key=models.write_param_card.cmp_to_key(models.write_param_card.ParamCardWriter.order_param))

        lines = [format(param.name) for param in self.params_ext]       
        fsock.writelines('\n'.join(lines))        
        
        fsock.writelines("""write(*,*)  ' Internal Params'
                            write(*,*)  ' ---------------------------------'
                            write(*,*)  ' '""")        
        lines = [format(data.name) for data in self.params_indep 
                  if data.name != 'ZERO' and self.check_needed_param(data.name)]
        fsock.writelines('\n'.join(lines))
        fsock.writelines("""write(*,*)  ' Internal Params evaluated point by point'
                            write(*,*)  ' ----------------------------------------'
                            write(*,*)  ' '""")         
        lines = [format(data.name) for data in self.params_dep \
                 if self.check_needed_param(data.name)]
        
        fsock.writelines('\n'.join(lines))                
        
 
    def create_get_color(self):
        """Create get_color.f in Source/MODEL with get_color and get_spin
        functions covering all particles in the model, using select case."""

        fsock = self.open('get_color.f', format='fortran')

        particle_dict = self.model.get('particle_dict')
        particle_ids = sorted(particle_dict.keys())
        dummy_pdg = self.model.get_first_non_pdg()

        lines = "function get_color(ipdg)\n"
        lines += "implicit none\n"
        lines += "integer get_color, ipdg\n"
        lines += "select case (ipdg)\n"
        for pdg in particle_ids:
            lines += "case(%d)\n" % pdg
            lines += "get_color=%d\n" % particle_dict[pdg].get_color()
        lines += "case(%d)\n" % dummy_pdg
        lines += "c This is dummy particle used in multiparticle vertices\n"
        lines += "get_color=2\n"
        lines += "case default\n"
        lines += "write(*,*)'Error: No color given for pdg ',ipdg\n"
        lines += "stop 1\n"
        lines += "end select\n"
        lines += "end\n"

        lines += "\n"
        lines += "function get_spin(ipdg)\n"
        lines += "implicit none\n"
        lines += "integer get_spin, ipdg\n"
        lines += "select case (ipdg)\n"
        for pdg in particle_ids:
            lines += "case(%d)\n" % pdg
            lines += "get_spin=%d\n" % particle_dict[pdg].get('spin')
        lines += "case(%d)\n" % dummy_pdg
        lines += "c This is dummy particle used in multiparticle vertices\n"
        lines += "get_spin=-2\n"
        lines += "case default\n"
        lines += "write(*,*)'Error: No spin given for pdg ',ipdg\n"
        lines += "stop 1\n"
        lines += "end select\n"
        lines += "end\n"

        fsock.writelines(lines)

    def create_ident_card(self):
        """ create the ident_card.dat """
    
        def format(parameter):
            """return the line for the ident_card corresponding to this parameter"""
            colum = [parameter.lhablock.lower()] + \
                    [str(value) for value in parameter.lhacode] + \
                    [parameter.name]
            if not parameter.name:
                return ''
            return ' '.join(colum)+'\n'
    
        fsock = self.open('ident_card.dat')
     
        external_param = [format(param) for param in self.params_ext]
        if self.model['running_elements']:
            scales = set()
            
            for elements in self.model["running_elements"]:
                for params in elements.run_objects:
                    for param in params:
                        scales.add(param.lhablock)

            try:
                scales.remove('SMINPUTS')
            except Exception:
                pass
            #entry should be a parameter ... not a string
            for b in scales:
                param = base_objects.ParamCardVariable(
                    'mdl__%s__scale' % b.lower(),
                     91.188, b, [0])
                external_param.append(format(param))

        fsock.writelines('\n'.join(external_param))

    def create_actualize_mp_ext_param_inc(self):
        """ create the actualize_mp_ext_params.inc code """
        
        # In principle one should actualize all external, but for now, it is
        # hardcoded that only AS and MU_R can by dynamically changed by the user
        # so that we only update those ones.
        # Of course, to be on the safe side, one could decide to update all
        # external parameters.
        update_params_list=[p for p in self.params_ext if p.name in 
                                                          self.PS_dependent_key]
        
        res_strings = ["%(mp_prefix)s%(name)s=%(name)s"\
                        %{'mp_prefix':self.mp_prefix,'name':param.name}\
                                                for param in update_params_list]
        # When read_lha is false, it is G which is taken in input and not AS, so
        # this is what should be reset here too.
        if 'aS' in [param.name for param in update_params_list]:
            res_strings.append("%(mp_prefix)sG=G"%{'mp_prefix':self.mp_prefix})
            
        fsock = self.open('actualize_mp_ext_params.inc', format='fortran')
        fsock.writelines('\n'.join(res_strings))

    def create_param_read(self):    
        """create param_read"""
        
        if self.opt['export_format'] in ['madevent', 'FKS5_default', 'FKS5_optimized'] \
            or self.opt['loop_induced']:
            fsock = self.open('param_read.inc', format='fortran')
            fsock.writelines(' include \'../param_card.inc\'')
            return
    
        def format_line(parameter):
            """return the line for the ident_card corresponding to this 
            parameter"""
            template = \
            """ call LHA_get_real(npara,param,value,'%(name)s',%(name)s,%(value)s)""" \
                % {'name': parameter.name,
                   'value': self.p_to_f.parse(str(parameter.value.real))}
            if self.opt['mp']:
                template = template+ \
                ("\n call MP_LHA_get_real(npara,param,value,'%(name)s',"+
                 "%(mp_prefix)s%(name)s,%(value)s)") \
                % {'name': parameter.name,'mp_prefix': self.mp_prefix,
                   'value': self.mp_p_to_f.parse(str(parameter.value.real))}

            if parameter.lhablock.lower() == 'loop':
                template = template.replace('LHA_get_real', 'LHA_get_real_silent') 
                
            return template        
    
        fsock = self.open('param_read.inc', format='fortran')
        res_strings = [format_line(param) \
                          for param in self.params_ext]
        
        if self.model['running_elements']:
            scales = set()
            
            for elements in self.model["running_elements"]:
                for params in elements.run_objects:
                    for param in params:
                        scales.add(param.lhablock)

            try:
                scales.remove('SMINPUTS')
            except Exception:
                pass
            #entry should be a parameter ... not a string
            for b in scales:
                param = base_objects.ParamCardVariable(
                    'mdl__%s__scale' % b,
                     91.188, b, 0)
                res_strings.append(format_line(param))
        
        # Correct width sign for Majorana particles (where the width
        # and mass need to have the same sign)        
        for particle in self.model.get('particles'):
            if particle.is_fermion() and particle.get('self_antipart') and \
                   particle.get('width').lower() != 'zero':
                
                res_strings.append('%(width)s = sign(%(width)s,%(mass)s)' % \
                 {'width': particle.get('width'), 'mass': particle.get('mass')})
                if self.opt['mp']:
                    res_strings.append(\
                      ('%(mp_pref)s%(width)s = sign(%(mp_pref)s%(width)s,'+\
                       '%(mp_pref)s%(mass)s)')%{'width': particle.get('width'),\
                       'mass': particle.get('mass'),'mp_pref':self.mp_prefix})

        fsock.writelines('\n'.join(res_strings))


    @staticmethod
    def create_param_card_static(model, output_path, rule_card_path=False,
                                 mssm_convert=True, write_special=True):
        """ create the param_card.dat for a givent model --static method-- """
        #1. Check if a default param_card is present:
        done = False
        if hasattr(model, 'restrict_card') and isinstance(model.restrict_card, str):
            restrict_name = os.path.basename(model.restrict_card)[9:-4]
            model_path = model.get('modelpath')
            if os.path.exists(pjoin(model_path,'paramcard_%s.dat' % restrict_name)):
                done = True
                files.cp(pjoin(model_path,'paramcard_%s.dat' % restrict_name),
                         output_path)
        if not done:
            param_writer.ParamCardWriter(model, output_path, write_special=write_special)
         
        if rule_card_path:   
            if hasattr(model, 'rule_card'):
                model.rule_card.write_file(rule_card_path)
        
        if mssm_convert:
            model_name = model.get('name')
            # IF MSSM convert the card to SLAH1
            if model_name == 'mssm' or model_name.startswith('mssm-'):
                import models.check_param_card as translator    
                # Check the format of the param_card for Pythia and make it correct
                if rule_card_path:
                    translator.make_valid_param_card(output_path, rule_card_path)
                translator.convert_to_slha1(output_path)        
    
    def create_param_card(self, write_special=True):
        """ create the param_card.dat """

        rule_card = pjoin(self.dir_path, 'param_card_rule.dat')
        if not hasattr(self.model, 'rule_card'):
            rule_card=False
        write_special = True
        if 'exporter' in self.opt:
            import madgraph.loop.loop_exporters as loop_exporters
            import madgraph.iolibs.export_fks as export_fks
            write_special = False
            if  issubclass(self.opt['exporter'], loop_exporters.LoopProcessExporterFortranSA):
                write_special = True
                if issubclass(self.opt['exporter'],(loop_exporters.LoopInducedExporterME,export_fks.ProcessExporterFortranFKS)):
                     write_special = False
                        
        self.create_param_card_static(self.model, 
                                      output_path=pjoin(self.dir_path, 'param_card.dat'), 
                                      rule_card_path=rule_card, 
                                      mssm_convert=True,
                                      write_special=write_special)
        
def ExportV4Factory(cmd, noclean, output_type='default', group_subprocesses=True, cmd_options={}):
    """ Determine which Export_v4 class is required. cmd is the command 
        interface containing all potential usefull information.
        The output_type argument specifies from which context the output
        is called. It is 'madloop' for MadLoop5, 'amcatnlo' for FKS5 output
        and 'default' for tree-level outputs."""

    opt = dict(cmd.options)
    opt['output_options'] = cmd_options

    # ==========================================================================
    # First check whether Ninja must be installed.
    # Ninja would only be required if:
    #  a) Loop optimized output is selected
    #  b) the process gathered from the amplitude generated use loops

    if len(cmd._curr_amps)>0:
        try:
            curr_proc = cmd._curr_amps[0].get('process')
        except base_objects.PhysicsObject.PhysicsObjectError:
            curr_proc = None
    elif hasattr(cmd,'_fks_multi_proc') and \
                          len(cmd._fks_multi_proc.get('process_definitions'))>0:
        curr_proc = cmd._fks_multi_proc.get('process_definitions')[0]
    else:
        curr_proc = None

    requires_reduction_tool = opt['loop_optimized_output'] and \
                (not curr_proc is None) and \
                (curr_proc.get('perturbation_couplings') != [] and \
                not curr_proc.get('NLO_mode') in [None,'real','tree','LO','LOonly'])

    # An installation is required then, but only if the specified path is the
    # default local one and that the Ninja library appears missing.
    if requires_reduction_tool:
        cmd.install_reduction_library()
        
    # ==========================================================================
    # First treat the MadLoop5 standalone case       
    MadLoop_SA_options = {'clean': not noclean, 
      'complex_mass':cmd.options['complex_mass_scheme'],
      'export_format':'madloop', 
      'mp':True,
      'loop_dir': os.path.join(cmd._mgme_dir,'Template','loop_material'),
      'cuttools_dir': cmd._cuttools_dir,
      'iregi_dir':cmd._iregi_dir,
      'golem_dir':cmd.options['golem'],
      'samurai_dir':cmd.options['samurai'],
      'ninja_dir':cmd.options['ninja'],
      'collier_dir':cmd.options['collier'],
      'fortran_compiler':cmd.options['fortran_compiler'],
      'f2py_compiler':cmd.options['f2py_compiler'],
      'output_dependencies':cmd.options['output_dependencies'],
      'SubProc_prefix':'P',
      'compute_color_flows':cmd.options['loop_color_flows'],
      'mode': 'reweight' if cmd._export_format == "standalone_rw" else '',
      'cluster_local_path': cmd.options['cluster_local_path'],
      'output_options': cmd_options
      }

    if output_type.startswith('madloop'):        
        import madgraph.loop.loop_exporters as loop_exporters
        if os.path.isdir(os.path.join(cmd._mgme_dir, 'Template/loop_material')):
            ExporterClass=None
            if not cmd.options['loop_optimized_output']:
                ExporterClass=loop_exporters.LoopProcessExporterFortranSA
            else:
                if output_type == "madloop":
                    ExporterClass=loop_exporters.LoopProcessOptimizedExporterFortranSA
                    MadLoop_SA_options['export_format'] = 'madloop_optimized'
                elif output_type == "madloop_matchbox":
                    ExporterClass=loop_exporters.LoopProcessExporterFortranMatchBox
                    MadLoop_SA_options['export_format'] = 'madloop_matchbox'
                else:
                    raise Exception("output_type not recognize %s" % output_type)
            return ExporterClass(cmd._export_dir, MadLoop_SA_options)
        else:
            raise MadGraph5Error('MG5_aMC cannot find the \'loop_material\' directory'+\
                                 ' in %s'%str(cmd._mgme_dir))

    # Then treat the aMC@NLO output     
    elif output_type=='amcatnlo':
        import madgraph.iolibs.export_fks as export_fks
        ExporterClass=None
        amcatnlo_options = dict(opt)
        amcatnlo_options.update(MadLoop_SA_options)
        amcatnlo_options['running'] = cmd._curr_model.get('running_elements')
        amcatnlo_options['mp'] = len(cmd._fks_multi_proc.get_virt_amplitudes()) > 0
        if not cmd.options['loop_optimized_output']:
            logger.info("Writing out the aMC@NLO code")
            ExporterClass = export_fks.ProcessExporterFortranFKS
            amcatnlo_options['export_format']='FKS5_default'
        else:
            logger.info("Writing out the aMC@NLO code, using optimized Loops")
            ExporterClass = export_fks.ProcessOptimizedExporterFortranFKS
            amcatnlo_options['export_format']='FKS5_optimized'
        return ExporterClass(cmd._export_dir, amcatnlo_options)

    # Then treat the EW sudakov Standalone output     
    elif output_type=='ewsudsa':
        import madgraph.iolibs.export_fks as export_fks
        ExporterClass=None
        amcatnlo_options = dict(opt)
        amcatnlo_options.update(MadLoop_SA_options)
        amcatnlo_options['mp'] = False
        logger.info("Writing out the EW Sudakov approximation in a standalone format")
        ExporterClass = export_fks.ProcessExporterEWSudakovSA
        amcatnlo_options['export_format']='FKS5_optimized'
        return ExporterClass(cmd._export_dir, amcatnlo_options)


    # Then the default tree-level output
    elif output_type=='default':
        assert group_subprocesses in [True, False]
        
        opt = dict(opt)
        opt.update({'clean': not noclean,
               'complex_mass': cmd.options['complex_mass_scheme'],
               'export_format':cmd._export_format,
               'mp': False,
               'sa_symmetry':False,
               # --use_crossing of the generate/add process command: when off,
               # the standalone matrix.f is written without any crossing
               # machinery (see ProcessExporterFortranSA.write_matrix_element_v4).
               'use_crossing': getattr(cmd, '_use_crossing', True),
               'model': cmd._curr_model.get('name'),
               'v5_model': False if cmd._model_v4_path else True,
               'running': cmd._curr_model.get('running_elements'),
                })

        format = cmd._export_format #shortcut

        if format in ['standalone_msP', 'standalone_msF', 'standalone_rw']:
            opt['sa_symmetry'] = True      
        elif format == 'plugin':
            opt['sa_symmetry'] = cmd._export_plugin.sa_symmetry
    
        loop_induced_opt = dict(opt)
        loop_induced_opt.update(MadLoop_SA_options)
        loop_induced_opt['export_format'] = 'madloop_optimized'
        loop_induced_opt['SubProc_prefix'] = 'PV'
        # For loop_induced output with MadEvent, we must have access to the 
        # color flows.
        loop_induced_opt['compute_color_flows'] = True
        for key in opt:
            if key not in loop_induced_opt:
                loop_induced_opt[key] = opt[key]
    
        # Madevent output supports MadAnalysis5
        if format in ['madevent']:
            opt['madanalysis5'] = cmd.options['madanalysis5_path']
            
        if format == 'matrix' or format.startswith('standalone'):
            return ProcessExporterFortranSA(cmd._export_dir, opt, format=format)
        
        elif format in ['madevent'] and group_subprocesses:
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                import madgraph.loop.loop_exporters as loop_exporters
                return  loop_exporters.LoopInducedExporterMEGroup( 
                                               cmd._export_dir,loop_induced_opt)
            elif cmd._export_plugin:
                return cmd._export_plugin(cmd._export_dir,opt) 
            else:
                return  ProcessExporterFortranMEGroup(cmd._export_dir,opt)                
        elif format in ['madevent']:
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                import madgraph.loop.loop_exporters as loop_exporters
                return  loop_exporters.LoopInducedExporterMENoGroup( 
                                               cmd._export_dir,loop_induced_opt)
            else:
                return  ProcessExporterFortranME(cmd._export_dir,opt)
        elif format in ['matchbox']:
            return ProcessExporterFortranMatchBox(cmd._export_dir,opt)
        elif cmd._export_format in ['madweight'] and group_subprocesses:

            return ProcessExporterFortranMWGroup(cmd._export_dir, opt)
        elif cmd._export_format in ['madweight']:
            return ProcessExporterFortranMW(cmd._export_dir, opt)
        elif format == 'plugin':
            if isinstance(cmd._curr_amps[0], 
                                         loop_diagram_generation.LoopAmplitude):
                return cmd._export_plugin(cmd._export_dir, loop_induced_opt)
            else:
                return cmd._export_plugin(cmd._export_dir, opt)

        else:
            raise Exception('Wrong export_v4 format')
    else:
        raise MadGraph5Error('Output type %s not reckognized in ExportV4Factory.')
    
            


#===============================================================================
# ProcessExporterFortranMWGroup
#===============================================================================
class ProcessExporterFortranMWGroup(ProcessExporterFortranMW):
    """Class to take care of exporting a set of matrix elements to
    MadEvent subprocess group format."""

    matrix_file = "matrix_madweight_group_v4.inc"
    grouped_mode = 'madweight'
    #===========================================================================
    # generate_subprocess_directory
    #===========================================================================
    def generate_subprocess_directory(self, subproc_group,
                                         fortran_model,
                                         group_number, **opt):
        """Generate the Pn directory for a subprocess group in MadEvent,
        including the necessary matrix_N.f files, configs.inc and various
        other helper files."""

        if not isinstance(subproc_group, group_subprocs.SubProcessGroup):
            raise base_objects.PhysicsObject.PhysicsObjectError("subproc_group object not SubProcessGroup")

        if not self.model:
            self.model = subproc_group.get('matrix_elements')[0].\
                         get('processes')[0].get('model')

        pathdir = os.path.join(self.dir_path, 'SubProcesses')

        # Create the directory PN in the specified path
        subprocdir = "P%d_%s" % (subproc_group.get('number'),
                                 subproc_group.get('name'))
        try:
            os.mkdir(pjoin(pathdir, subprocdir))
        except os.error as error:
            logger.warning(error.strerror + " " + subprocdir)

        logger.info('Creating files in directory %s' % subprocdir)
        Ppath = pjoin(pathdir, subprocdir)

        # Create the matrix.f files, auto_dsig.f files and all inc files
        # for all subprocesses in the group

        maxamps = 0
        maxflows = 0
        tot_calls = 0

        matrix_elements = subproc_group.get('matrix_elements')

        for ime, matrix_element in \
                enumerate(matrix_elements):
            filename = pjoin(Ppath, 'matrix%d.f' % (ime+1))
            calls, ncolor = \
               self.write_matrix_element_v4(writers.FortranWriter(filename), 
                                                matrix_element,
                                                fortran_model,
                                                str(ime+1),
                                                subproc_group.get('diagram_maps')[\
                                                                              ime])

            filename = pjoin(Ppath, 'auto_dsig%d.f' % (ime+1))
            self.write_auto_dsig_file(writers.FortranWriter(filename),
                                 matrix_element,
                                 str(ime+1))

            # Keep track of needed quantities
            tot_calls += int(calls)
            maxflows = max(maxflows, ncolor)
            maxamps = max(maxamps, len(matrix_element.get('diagrams')))

            # Draw diagrams
            filename = pjoin(Ppath, "matrix%d.ps" % (ime+1))
            plot = draw.MultiEpsDiagramDrawer(matrix_element.get('base_amplitude').\
                                                                    get('diagrams'),
                                              filename,
                                              model = \
                                                matrix_element.get('processes')[0].\
                                                                       get('model'),
                                              amplitude=True)
            logger.info("Generating Feynman diagrams for " + \
                         matrix_element.get('processes')[0].nice_string())
            plot.draw()

        # Extract number of external particles
        (nexternal, ninitial) = matrix_element.get_nexternal_ninitial()

        # Generate a list of diagrams corresponding to each configuration
        # [[d1, d2, ...,dn],...] where 1,2,...,n is the subprocess number
        # If a subprocess has no diagrams for this config, the number is 0

        subproc_diagrams_for_config = subproc_group.get('diagrams_for_configs')

        filename = pjoin(Ppath, 'auto_dsig.f')
        self.write_super_auto_dsig_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = pjoin(Ppath,'configs.inc')
        nconfigs, s_and_t_channels = self.write_configs_file(\
            writers.FortranWriter(filename),
            subproc_group,
            subproc_diagrams_for_config)

        filename = pjoin(Ppath, 'leshouche.inc')
        self.write_leshouche_file(writers.FortranWriter(filename),
                                   subproc_group)

        filename = pjoin(Ppath, 'phasespace.inc')
        self.write_phasespace_file(writers.FortranWriter(filename),
                           nconfigs)
                           
        nb_flavor_per_proc = matrix_elements.get_nb_flavors()
        self.write_maxamps_file(writers.FortranWriter(filename),
                           maxamps,
                           maxflows,
                           nb_flavor_per_proc,
                           nb_flavor_per_proc, # THis is max(flavor*process) 
                           len(matrix_elements))
        
        #filename = pjoin(Ppath, 'maxamps.inc')
        #self.write_maxamps_file(writers.FortranWriter(filename),
        #                   maxamps,
        #                   maxflows,
        #                   max([len(me.get('processes')) for me in \
        #                        matrix_elements]),
        #                   len(matrix_elements))

        filename = pjoin(Ppath, 'mirrorprocs.inc')
        self.write_mirrorprocs(writers.FortranWriter(filename),
                          subproc_group)

        filename = pjoin(Ppath, 'nexternal.inc')
        self.write_nexternal_file(writers.FortranWriter(filename),
                             nexternal, ninitial)

        filename = pjoin(Ppath, 'pmass.inc')
        self.write_pmass_file(writers.FortranWriter(filename),
                         matrix_element)

        filename = pjoin(Ppath, 'props.inc')
        self.write_props_file(writers.FortranWriter(filename),
                         matrix_element,
                         s_and_t_channels)

#        filename = pjoin(Ppath, 'processes.dat')
#        files.write_to_file(filename,
#                            self.write_processes_file,
#                            subproc_group)

        # Generate jpgs -> pass in make_html
        #os.system(os.path.join('..', '..', 'bin', 'gen_jpeg-pl'))

        linkfiles = ['driver.f', 'cuts.f', 'initialization.f','gen_ps.f', 'makefile', 'coupl.inc','madweight_param.inc', 'run.inc', 'setscales.f', 'dummy_fct.f']

        for file in linkfiles:
            ln('../%s' % file, cwd=Ppath)

        ln('nexternal.inc', '../../Source', cwd=Ppath, log=False)
        ln('leshouche.inc', '../../Source', cwd=Ppath, log=False)
        ln('maxamps.inc', '../../Source', cwd=Ppath, log=False)
        ln('../../Source/vector.inc', cwd=Ppath, log=False)
        ln('../../Source/maxparticles.inc', '.', log=True, cwd=Ppath)
        ln('../../Source/maxparticles.inc', '.', name='genps.inc', log=True, cwd=Ppath)
        ln('phasespace.inc', '../', log=True, cwd=Ppath)
        if not tot_calls:
            tot_calls = 0
        return tot_calls

    #===========================================================================
    # Helper functions
    #===========================================================================
    def modify_grouping(self, matrix_element):
        """allow to modify the grouping (if grouping is in place)
            return two value:
            - True/False if the matrix_element was modified
            - the new(or old) matrix element"""
            
        return True, matrix_element.split_lepton_grouping()
    
    #===========================================================================
    # write_super_auto_dsig_file
    #===========================================================================
    def write_super_auto_dsig_file(self, writer, subproc_group):
        """Write the auto_dsig.f file selecting between the subprocesses
        in subprocess group mode"""

        replace_dict = {}

        # Extract version number and date from VERSION file
        info_lines = self.get_mg5_info_lines()
        replace_dict['info_lines'] = info_lines

        matrix_elements = subproc_group.get('matrix_elements')

        # Extract process info lines
        process_lines = '\n'.join([self.get_process_info_lines(me) for me in \
                                   matrix_elements])
        replace_dict['process_lines'] = process_lines

        nexternal, ninitial = matrix_elements[0].get_nexternal_ninitial()
        replace_dict['nexternal'] = nexternal

        replace_dict['nsprocs'] = 2*len(matrix_elements)

        # Generate dsig definition line
        dsig_def_line = "DOUBLE PRECISION " + \
                        ",".join(["DSIG%d" % (iproc + 1) for iproc in \
                                  range(len(matrix_elements))])
        replace_dict["dsig_def_line"] = dsig_def_line

        # Generate dsig process lines
        call_dsig_proc_lines = []
        for iproc in range(len(matrix_elements)):
            call_dsig_proc_lines.append(\
                "IF(IPROC.EQ.%(num)d) DSIGPROC=DSIG%(num)d(P1,WGT,IMODE) ! %(proc)s" % \
                {"num": iproc + 1,
                 "proc": matrix_elements[iproc].get('processes')[0].base_string()})
        replace_dict['call_dsig_proc_lines'] = "\n".join(call_dsig_proc_lines)

        if writer:
            file = open(os.path.join(_file_path, \
                       'iolibs/template_files/super_auto_dsig_mw_group_v4.inc')).read()
            file = file % replace_dict
            # Write the file
            writer.writelines(file)
        else:
            return replace_dict
        
    #===========================================================================
    # write_mirrorprocs
    #===========================================================================
    def write_mirrorprocs(self, writer, subproc_group):
        """Write the mirrorprocs.inc file determining which processes have
        IS mirror process in subprocess group mode."""

        lines = []
        bool_dict = {True: '.true.', False: '.false.'}
        matrix_elements = subproc_group.get('matrix_elements')
        lines.append("DATA (MIRRORPROCS(I),I=1,%d)/%s/" % \
                     (len(matrix_elements),
                      ",".join([bool_dict[me.get('has_mirror_process')] for \
                                me in matrix_elements])))
        # Write the file
        writer.writelines(lines)

    #===========================================================================
    # write_configs_file
    #===========================================================================
    def write_configs_file(self, writer, subproc_group, diagrams_for_config):
        """Write the configs.inc file with topology information for a
        subprocess group. Use the first subprocess with a diagram for each
        configuration."""

        matrix_elements = subproc_group.get('matrix_elements')
        model = matrix_elements[0].get('processes')[0].get('model')

        diagrams = []
        config_numbers = []
        for iconfig, config in enumerate(diagrams_for_config):
            # Check if any diagrams correspond to this config
            if set(config) == set([0]):
                continue
            subproc_diags = []
            for s,d in enumerate(config):
                if d:
                    subproc_diags.append(matrix_elements[s].\
                                         get('diagrams')[d-1])
                else:
                    subproc_diags.append(None)
            diagrams.append(subproc_diags)
            config_numbers.append(iconfig + 1)

        # Extract number of external particles
        (nexternal, ninitial) = subproc_group.get_nexternal_ninitial()

        return len(diagrams), \
               self.write_configs_file_from_diagrams(writer, diagrams,
                                                config_numbers,
                                                nexternal, ninitial,
                                                matrix_elements[0],model)

    #===========================================================================
    # write_run_configs_file
    #===========================================================================
    def write_run_config_file(self, writer):
        """Write the run_configs.inc file for MadEvent"""

        path = os.path.join(_file_path,'iolibs','template_files','madweight_run_config.inc') 
        text = open(path).read() % {'chanperjob':'2'} 
        writer.write(text)
        return True


    #===========================================================================
    # write_leshouche_file
    #===========================================================================
    def write_leshouche_file(self, writer, subproc_group):
        """Write the leshouche.inc file for MG4"""

        all_lines = []

        for iproc, matrix_element in \
            enumerate(subproc_group.get('matrix_elements')):
            all_lines.extend(self.get_leshouche_lines(matrix_element,
                                                 iproc))

        # Write the file
        writer.writelines(all_lines)

        return True


    
