#!/usr/bin/env python3

import argparse
import atexit
import glob
import os
import re
import collections
from string import Template
from copy import copy
from itertools import product
from functools import reduce 

try:
     import madgraph
except:
     import internal.misc as misc
else:
     import madgraph.various.misc as misc
import mmap
try:
    from tqdm import tqdm
except ImportError:
    tqdm = misc.tqdm

    
def get_num_lines(file_path):
    fp = open(file_path, 'r+')
    buf = mmap.mmap(fp.fileno(),0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


# Default number of fortran statements per amplitude-chunk file; kept in step
# with export_v4.AMP_CHUNK_SIZE_DEFAULT, which this module cannot import (it is
# shipped stand-alone as bin/internal/hel_recycle.py). See the comment there.
AMP_CHUNK_SIZE_DEFAULT = 2000

# The markers the exporter puts around the HELAS block of an amplitude-chunk
# file, so that the unrolling below can read the calls back out of it.
AMP_CHUNK_BEGIN = 'HELAS CALLS BEGIN'
AMP_CHUNK_END = 'HELAS CALLS END'
AMP_CHUNK_CALL_RE = re.compile(r'^\s*CALL\s+ORIGAMP\d+_(\d+)\s*\(', re.IGNORECASE)


_CHUNK_COMMENT_RE = re.compile(r"^(\s*#|c\$|c$|(c\s+([^=]|$))|cf2py|c\-\-|c\*\*|\s*!|!\$)",
                               re.IGNORECASE)
_CHUNK_CONTINUATION_RE = re.compile(r"^(?:     )[$&]")


def chunk_statements(lines, chunk_size):
    """Group column-formatted fortran *lines* into slices of about *chunk_size*
    statements each. A slice boundary may only fall where a new statement
    starts at nesting depth zero: continuation lines stay with their statement,
    comments attach to the statement below them, and an IF(...)THEN block --
    which split_amps puts around a flavor-masked amplitude -- is never cut in
    half. Mirrors export_v4.chunk_fortran_statements.
    """

    def depth_change(line):
        code = line.upper().split('!')[0].strip()
        if code.startswith('IF') and code.endswith('THEN'):
            return 1
        if code.startswith('DO ') or code == 'DO':
            return 1
        if code.startswith(('ENDIF', 'END IF', 'ENDDO', 'END DO')):
            return -1
        return 0

    chunks = []
    current = []
    pending = []
    nb_statements = 0
    depth = 0
    for line in lines:
        if not line.strip() or _CHUNK_COMMENT_RE.search(line):
            pending.append(line)
            continue
        if _CHUNK_CONTINUATION_RE.match(line):
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
        depth = max(0, depth + depth_change(line))
    if pending:
        current.extend(pending)
    if current:
        chunks.append(current)
    return chunks


def get_subroutine_signature(text):
    """(name, argument list) of the first SUBROUTINE statement of *text*, with
    its continuation lines folded back in. The chunk template is written by the
    exporter, which is where the argument list of a chunk is decided (it
    depends on the crossing and flavor-mask holes), so it is read back from
    there rather than repeated here."""

    statement = ''
    for line in text.split('\n'):
        if not statement:
            if 'SUBROUTINE' not in line.upper():
                continue
            statement = line.strip()
        elif line[5:6] in ('$', '&'):
            statement += line[6:].strip()
        else:
            break
        if statement.endswith(')'):
            break
    head, _, args = statement.partition('(')
    return head.split()[-1], args.rsplit(')', 1)[0]


def read_amp_chunk_body(path):
    """Return the HELAS call lines of an amplitude-chunk file, i.e. what used
    to sit inline in the matrix element before the split."""

    body = []
    inside = False
    with open(path) as chunk_file:
        for line in chunk_file:
            if AMP_CHUNK_BEGIN in line.upper():
                inside = True
            elif AMP_CHUNK_END in line.upper():
                break
            elif inside:
                body.append(line)
    return body


def splice_amp_chunks(path):
    """Iterate over the lines of *path*, substituting the body of
    matrix<i>_origamp<k>.f wherever the matrix element calls it.

    The exporter can move the HELAS call sequence of matrix<i>_orig.f into
    files of its own; the unrolling below has to see that sequence, and it sees
    exactly the lines that used to be there.
    """

    directory = os.path.dirname(path) or '.'
    base = os.path.basename(path)[:-len('_orig.f')]
    skipping = False
    with open(path) as input_file:
        for line in input_file:
            if skipping:
                # the call is long enough to be wrapped as soon as the flavor
                # masks are threaded through it; its continuations must go with
                # it, or they would be folded onto the last spliced statement
                if _CHUNK_CONTINUATION_RE.match(line):
                    continue
                skipping = False
            match = AMP_CHUNK_CALL_RE.match(line)
            if not match:
                yield line
                continue
            skipping = True
            chunk = os.path.join(
                directory, '%s_origamp%s.f' % (base, match.group(1)))
            for chunk_line in read_amp_chunk_body(chunk):
                yield chunk_line

class DAG:

    def __init__(self):
        self.all_wavs = []
        self.external_wavs = []
        self.internal_wavs = []
        # all_wavs holds every wavefunction ever built, dead ones included, and
        # grows to hundreds of thousands of entries at high multiplicity. Every
        # question ever asked of it is keyed on old_name, so bucket it: a linear
        # scan per HELAS line is quadratic in the file, and it was the second
        # cost centre of the whole recycling step after find_path.
        self.by_old_name = {}
        # The externals each wavefunction depends on, which is the ONLY thing
        # anyone ever wanted the graph for -- good_helicity asked it as
        # find_path(dep, ext) over every (dep, external) pair, i.e. a fresh DFS
        # per pair, 114 million of them on g g > 5g. It needs no search at all:
        # the edges recorded by store_wav go straight from a wavefunction to the
        # externals under it (that is what its caller passes as ext_deps, itself
        # already a transitive closure), and externals have no outgoing edges,
        # so the graph is two levels deep by construction and reachability is a
        # set membership. Verified against find_path over every top-level pair
        # of g g > g g g and g g > g g g g g: same answer everywhere, and no
        # path longer than two nodes exists.
        self.ext_closure = {}
        # Bit i of comb_masks[wav] is set when good_wav_combs[i] contains wav;
        # compat_masks[node] is the AND over its ext_closure, so "does some good
        # helicity combination cover this subtree" is one big-int test instead
        # of a rescan of the whole comb list. Rebuilt by set_good_wav_combs.
        self.comb_masks = {}
        self.compat_masks = {}
        self.full_mask = 0

    def store_wav(self, wav, ext_deps=[]):
        self.all_wavs.append(wav)
        nature = wav.nature
        if nature == 'external':
            self.external_wavs.append(wav)
            # An external is its own only external dependency: find_path(w, w)
            # returned the one-element path [w], which is truthy.
            self.ext_closure[wav] = frozenset((wav,))
        else:
            if nature == 'internal':
                self.internal_wavs.append(wav)
            self.ext_closure[wav] = frozenset(ext_deps)
        try:
            self.by_old_name[wav.old_name].append(wav)
        except KeyError:
            self.by_old_name[wav.old_name] = [wav]

    def dependencies(self, old_name):
        return list(self.by_old_name.get(old_name, ()))

    def kill_old(self, old_name):
        # Every wavefunction under this name dies at once, so the bucket can be
        # emptied rather than filtered later: dead wavefunctions are never
        # resurrected, and dependencies() would drop them anyway. The name stays
        # a key so old_names() keeps reporting it, exactly as the scan over
        # all_wavs (which also kept the dead entries) used to.
        bucket = self.by_old_name.get(old_name)
        if bucket:
            for wav in bucket:
                wav.dead = True
            del bucket[:]

    def old_names(self):
        '''The old names ever stored, live or dead. Callers only intersect it
        with a set, which leaves it untouched -- do not mutate the result.'''
        return self.by_old_name.keys()

    def set_good_wav_combs(self, good_wav_combs):
        '''Index the good external-wavefunction combinations as bitmasks. Called
        whenever External.get_gwc rebuilds them, which is what invalidates the
        cached per-node masks.'''
        self.comb_masks = comb_masks = {}
        self.compat_masks = {}
        for i, comb in enumerate(good_wav_combs):
            bit = 1 << i
            for wav in comb:
                comb_masks[wav] = comb_masks.get(wav, 0) | bit
        self.full_mask = (1 << len(good_wav_combs)) - 1

    def compat_mask(self, node):
        '''The combinations that cover every external under `node`. Zero when
        none does -- with no combinations at all that is every node, which is
        how the old "no comb was a superset" answer came out for an empty
        good_wav_combs.'''
        try:
            return self.compat_masks[node]
        except KeyError:
            pass
        mask = self.full_mask
        comb_masks = self.comb_masks
        for ext in self.ext_closure[node]:
            mask &= comb_masks.get(ext, 0)
            if not mask:
                break
        self.compat_masks[node] = mask
        return mask

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        branches = [(key, sorted(item, key=lambda w: w.name))
                    for key, item in self.ext_closure.items()
                    if item and key.nature != 'external']
        print_str = 'With new names:\n\t'
        print_str += '\n\t'.join([f'{key} : {item}' for key, item in branches])
        print_str += '\n\nWith old names:\n\t'
        print_str += '\n\t'.join([f'{key.old_name} : {[i.old_name for i in item]}' for key, item in branches])
        return print_str



class MathsObject:
    '''Abstract class for wavefunctions and Amplitudes'''

    # Store here which externals the last wav/amp depends on, so that get_obj
    # and get_number do not have to recompute what good_helicity just worked out.
    ext_deps = None

    def __init__(self, arguments, old_name, nature):
        self.args = arguments
        self.old_name = old_name
        self.nature = nature
        self.name = None
        self.dead = False
        self.nb_used = 0
        self.linkdag = []

    def set_name(self, *args):
        self.args[-1] = self.format_name(*args)
        self.name = self.args[-1]

    def format_name(self, *nums):
        pass

    @staticmethod
    def get_deps(line, graph):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
        matches = graph.old_names() & set([old.replace(' ','') for old in old_args])
        try:
            matches.remove(old_name)
        except KeyError:
            pass
        old_deps = old_args[0:len(matches)]

        # If we're overwriting a wav clear it from graph
        graph.kill_old(old_name)
        return [graph.dependencies(dep) for dep in old_deps]

    @classmethod
    def good_helicity(cls, wavs, graph, diag_number=None, all_hel=[], bad_hel_amp=[]):
        # The externals under this combination of dependencies: the union of the
        # closures the DAG already recorded, not a search per (dep, external)
        # pair. See DAG.ext_closure.
        closure = graph.ext_closure
        ext_deps = set()
        for dep in wavs:
            ext_deps |= closure[dep]
        cls.ext_deps = ext_deps
        # "Is ext_deps covered by some good combination" -- an AND of the
        # per-dependency masks, which is the same answer as testing every
        # combination for a superset (a combination covers the union exactly when
        # it covers each closure) but does not rescan the comb list, and reuses
        # the mask each dependency was given the first time it was seen.
        mask = graph.full_mask
        for dep in wavs:
            mask &= graph.compat_mask(dep)
            if not mask:
                break
        this_comb_good = bool(mask)

        if diag_number and this_comb_good and cls.ext_deps:

            helicity = dict([(a.get_id(), a.hel) for a in cls.ext_deps])
            this_hel = [helicity[i] for i in range(1, len(helicity)+1)]
            hel_number = 1 + External.all_hel_index[tuple(this_hel)]
            
            if (hel_number,diag_number) in bad_hel_amp:        
                this_comb_good = False
            

            
        return this_comb_good and cls.ext_deps

    @staticmethod
    def get_new_args(line, wavs):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
        # Work out if wavs corresponds to an allowed helicity combination
        this_args = copy(old_args)
        wav_names = [w.name for w in wavs]
        this_args[0:len(wavs)] = wav_names
        # This isnt maximally efficient
        # Could take the num from wavs that've been deleted in graph
        return this_args

    @staticmethod
    def get_number():
        pass

    @classmethod
    def get_obj(cls, line, wavs, graph, diag_num = None):
        old_name = get_arguments(line)[-1].replace(' ','')
        new_args = cls.get_new_args(line, wavs)
        num = cls.get_number(wavs, graph)
        
        this_obj = cls.call_constructor(new_args, old_name, diag_num)
        this_obj.set_name(num, diag_num)
        if this_obj.nature != 'amplitude':
            graph.store_wav(this_obj, cls.ext_deps)
        return this_obj


    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

class External(MathsObject):
    '''Class for storing external wavefunctions'''

    good_hel = []
    nhel_lines = ''
    num_externals = 0
    # Could get this from dag but I'm worried about preserving order
    wavs_same_leg = {}
    good_wav_combs = []
    max_wav_num = 0
    # helicity tuple -> its row in the original NHEL table, filled by
    # HelicityRecycler.get_good_hel once that table is complete
    all_hel_index = {}

    def __init__(self, arguments, old_name):
        super().__init__(arguments, old_name, 'external')
        self.hel = int(self.args[2])
        self.mg = int(arguments[0].split(',')[-1][:-1])
        self.hel_ranges = []
        self.raise_num()

    @classmethod
    def raise_num(cls):
        cls.num_externals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        # If graph is passed in Internal it should be done here to so
        # we can set names
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')
        graph.kill_old(old_name)

        if 'NHEL' in old_args[2].upper():
            nhel_index = re.search(r'\(.*?\)', old_args[2]).group()
            ext_num = int(nhel_index[1:-1]) - 1
            new_hels = sorted(list(External.hel_ranges[ext_num]), reverse=True)
            new_hels = [int_to_string(i) for i in new_hels]
        else:
            # Spinor must be a scalar so give it hel = 0
            ext_num = int(re.search(r'\(0,(\d+)\)', old_args[0]).group(1)) -1
            new_hels = [' 0']

        new_wavfuncs = []
        for hel in new_hels:

            this_args = copy(old_args)
            this_args[2] = hel

            this_wavfunc = External(this_args, old_name)
            this_wavfunc.set_name(len(graph.external_wavs) + len(graph.internal_wavs) +1)

            graph.store_wav(this_wavfunc)
            new_wavfuncs.append(this_wavfunc)
        if ext_num in cls.wavs_same_leg:
            cls.wavs_same_leg[ext_num] += new_wavfuncs
        else:
            cls.wavs_same_leg[ext_num] = new_wavfuncs
        
        cls.max_wav_num = max( cls.max_wav_num, len(graph.external_wavs) + len(graph.internal_wavs))
        return new_wavfuncs

    @classmethod
    def get_gwc(cls):
        num_combs = len(cls.good_hel)
        gwc_old = [[] for x in range(num_combs)]
        gwc=[]
        for n, comb in enumerate(cls.good_hel):
            sols = [[]]
            for leg, wavs in cls.wavs_same_leg.items():
                valid = []
                for wav in wavs:
                    if comb[leg] == wav.hel:
                        valid.append(wav)
                        gwc_old[n].append(wav)
                if len(valid) == 1:
                    for sol in sols:
                        sol.append(valid[0])
                else:
                    tmp = []
                    for w in valid:
                        for sol in sols:
                            tmp2 = list(sol)
                            tmp2.append(w)
                            tmp.append(tmp2)
                    sols = tmp
            gwc += sols

        cls.good_wav_combs = gwc

    @staticmethod
    def format_name(*nums):
        return f'W({nums[0]})'
    
    def get_id(self):
        """ return the id of the particle under consideration """
        
        try:
           return self.id 
        except:
            self.id =  int(re.findall(r'P\(0,(\d+)\)', self.args[0])[0])
            return self.id
        
        

class Internal(MathsObject):
    '''Class for storing internal wavefunctions'''

    max_wav_num = 0
    num_internals = 0

    @classmethod
    def raise_num(cls):
        cls.num_internals += 1

    @classmethod
    def generate_wavfuncs(cls, line, graph):
        deps = cls.get_deps(line, graph)
        new_wavfuncs = [ cls.get_obj(line, wavs, graph) 
                         for wavs in product(*deps) 
                         if cls.good_helicity(wavs, graph) ]

        return new_wavfuncs


    # There must be a better way
    @classmethod
    def call_constructor(cls, new_args, old_name, diag_num):
        return Internal(new_args, old_name)

    @classmethod
    def get_number(cls, *args):
        num = External.num_externals + Internal.num_internals + 1
        if cls.max_wav_num < num:
            cls.max_wav_num = num
        return num

    def __init__(self, arguments, old_name):
        super().__init__(arguments, old_name, 'internal')
        self.raise_num()


    @staticmethod
    def format_name(*nums):
        return f'W({nums[0]})'

class Amplitude(MathsObject):
    '''Class for storing Amplitudes'''

    max_amp_num = 0

    def __init__(self, arguments, old_name, diag_num):
        self.diag_num = diag_num
        super().__init__(arguments, old_name, 'amplitude')


    @staticmethod
    def format_name(*nums):
        return f'AMP({nums[0]},{nums[1]})'

    @classmethod
    def generate_amps(cls, line, graph, all_hel=None, all_bad_hel=[]):
        old_args = get_arguments(line)
        old_name = old_args[-1].replace(' ','')

        amp_index = re.search(r'\(.*?\)', old_name).group()
        diag_num = int(amp_index[1:-1])

        deps = cls.get_deps(line, graph)

        new_amps = [cls.get_obj(line, wavs, graph, diag_num) 
                        for wavs in product(*deps) 
                        if cls.good_helicity(wavs, graph, diag_num, all_hel,all_bad_hel)]

        return new_amps

    @classmethod
    def call_constructor(cls, new_args, old_name, diag_num):
        return Amplitude(new_args, old_name, diag_num)

    @classmethod
    def get_number(cls, *args):
        wavs, graph = args
        amp_num = -1
        exts = graph.external_wavs        
        hel_amp = tuple([w.hel for w in sorted(cls.ext_deps, key=lambda x: x.mg)])
        amp_num  = External.map_hel[hel_amp] +1 # Offset because Fortran counts from 1

        if cls.max_amp_num < amp_num:
            cls.max_amp_num = amp_num 
        return amp_num  

class HelicityRecycler():
    '''Class for recycling helicity'''

    def __init__(self, good_elements, bad_amps=[], bad_amps_perhel=[], gauge='U'):

        External.good_hel = []
        External.nhel_lines = ''
        External.num_externals = 0
        External.wavs_same_leg = {}
        External.good_wav_combs = []
        External.all_hel_index = {}

        Internal.max_wav_num = 0
        Internal.num_internals = 0

        Amplitude.max_amp_num = 0
        self.last_category = None
        self.good_elements = good_elements
        # Both are only ever asked "is this one in you?" -- bad_amps once per
        # amplitude line, bad_amps_perhel once per (amplitude, helicity
        # combination). As lists that is a linear scan every time, which was
        # affordable while they held the handful of identically-zero
        # amplitudes. The C-parity de-duplication now adds EVERY amplitude of
        # every dropped mirror row: 128 x 28215 entries on g g > t t~ 4g and
        # 128 x 126630 on g g > 6g, turning the scan into the dominant cost of
        # the whole recycling step. Sets make the same question O(1).
        self.bad_amps = set(bad_amps)
        self.bad_amps_perhel = set(bad_amps_perhel)

        # Default file names
        self.input_file = 'matrix_orig.f'
        self.output_file = 'matrix_orig.f'
        self.template_file = 'template_matrix.f'
        
        self.template_dict = {}
        #initialise everything as for zero matrix element
        self.template_dict['helicity_lines'] = '\n'
        self.template_dict['helas_calls'] = []
        self.template_dict['jamp_lines'] = '\n'
        self.template_dict['amp2_lines'] = '\n'
        self.template_dict['ncomb'] = '0'
        self.template_dict['nwavefuncs'] = '0'
        # C-parity de-duplication: fortran that copies a dropped C-partner's
        # |M|^2 back from its representative (TS(flip)=TS(rep)). Empty unless
        # gen_ximprove supplies C-symmetric pairs: it keeps the partner's
        # helicity row but adds all its amplitudes to bad_amps_perhel, so their
        # HELAS calls are never generated and only the representatives are
        # computed. The indices here are the optim's re-numbered helicities.
        self.template_dict['csym_reuse'] = '\n'
        # Optional IF/ENDIF around the AMP2 (multi-channel) and JAMP2
        # (colour-flow) accumulation of the helicity loop, so a config can
        # contribute to the |M|^2 sum without contributing to either weight.
        # Empty -- every kept config feeds both, as it always did -- unless
        # gen_ximprove is recycling a matrix element SHARED by a crossing, whose
        # config set has to cover every member of the class: a config that is
        # dead for the caller at hand still has non-zero individual diagrams and
        # JAMPs, and those are not the gauge-invariant |M|^2. See
        # gen_ximprove.gensym.get_helicity.
        self.template_dict['dead_row_if'] = '\n'
        self.template_dict['dead_row_endif'] = '\n'

        self.dag = DAG()

        self.diag_num = 1
        self.got_gwc = False

        self.procedure_name = self.input_file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'

        self.old_out_name = ''
        self.loop_var = 'K'

        self.all_hel = []
        self.hel_filt = True
        self.gauge = gauge
        # statements per matrix<i>_optimamp<k>.f; 0 keeps the unrolled sequence
        # inline in matrix<i>_optim.f as it always was
        self.amp_chunk_size = AMP_CHUNK_SIZE_DEFAULT

    def set_input(self, file):
        if 'born_matrix' in file:
            print('HelicityRecycler is currently '
                  f'unable to handle {file}')
            exit(1)
        self.procedure_name = file.split('.')[0].upper()
        self.procedure_kind = 'FUNCTION'
        self.input_file = file

    def set_output(self, file):
        self.output_file = file
        if os.path.islink(self.output_file):
            os.remove(self.output_file)

    def set_template(self, file):
        self.template_file = file

    def function_call(self, line):
        # Check a function is called at all
        if 'CALL' not in line.upper():
            return None

        function = get_called_function(line)
        if not function:
            return None

        # Now check for external spinor
        ext_calls = ['OXXXXX', 'IXXXXX', 'VXXXXX', 'SXXXXX']
        if function.upper() in ext_calls:
            return 'external'

        # Now check for internal
        # Wont find a internal when no externals have been found...
        # ... I assume
        if not self.dag.external_wavs:
            return None

        # Search for internals by looking for calls to the externals
        # Maybe I should just get a list of all internals?
        matches = self.dag.old_names() & set(get_arguments(line))
        try:
            matches.remove(get_arguments(line)[-1])
        except KeyError:
            pass
        # What if [-1] is garbage? Then I'm relying on needs changing.
        # Is that OK?
        if (function.split('_')[-1] != '0'):
            return 'internal'
        elif (function.split('_')[-1] == '0'):
            return 'amplitude'
        else:
            print(f'Ahhhh what is going on here?\n{line}')
            set_trace()

        return None

    # string manipulation

    def add_amp_index(self, matchobj):
        old_pat = matchobj.group()
        new_pat = old_pat.replace('AMP(', 'AMP( %s,' % self.loop_var)
        
        #new_pat = f'{self.loop_var},{old_pat[:-1]}{old_pat[-1]}'
        return new_pat

    def add_indices(self, line):
        '''Add loop_var index to amp and output variable. 
           Also update name of output variable.'''
        # Doesnt work if the AMP arguments contain brackets.
        # The character in front is looked at rather than eaten, so that an
        # AMP( opening the statement is indexed too -- which is what a line
        # like "AMP(31) = AMP(31) + AMP(1)" needs.
        new_line = re.sub(r'(?<![A-Za-z0-9_])AMP\(.*?\)',
                          self.add_amp_index, line)
        new_line = re.sub(r'MATRIX\d+', 'TS(K)', new_line)
        return new_line

    def jamp_finished(self, line):
        # indent_end = re.compile(fr'{self.jamp_indent}END\W')
        # m = indent_end.match(line)
        # if m:
        #     return True
        return 'init_mode' in line.lower() 
        #if f'{self.old_out_name}=0.D0' in line.replace(' ', ''):
        #    return True
        #return False

    def get_old_name(self, line):
        if f'{self.procedure_kind} {self.procedure_name}' in line:
            if 'SUBROUTINE' == self.procedure_kind:
                self.old_out_name = get_arguments(line)[-1]
            if 'FUNCTION' == self.procedure_kind:
                self.old_out_name = line.split('(')[0].split()[-1]

    def get_amp_stuff(self, line_num, line):

        if 'diagram number' in line:
            self.amp_calc_started = True
        # Check if the calculation of this diagram is finished
        if ('AMP' not in get_arguments(line)[-1]
                and self.amp_calc_started and list(line)[0] != 'C'):
            # Check if the calculation of all diagrams is finished
            if self.function_call(line) not in ['external',
                                                'internal',
                                                'amplitude']:
                self.jamp_started = True
            self.amp_calc_started = False
        if self.jamp_started:
            self.get_jamp_lines(line)
        if self.in_amp2:
            self.get_amp2_lines(line)
        if self.find_amp2 and line.startswith('      ENDDO'):
            self.in_amp2 = True
            self.find_amp2 = False

    def get_jamp_lines(self, line):
        if self.jamp_finished(line):
            self.jamp_started = False
            self.find_amp2 = True
        elif not line.isspace():
            self.template_dict['jamp_lines'] += f'{line[0:6]}  {self.add_indices(line[6:])}'

    def get_amp2_lines(self, line):
        if line.startswith('      DO I = 1, NCOLOR'):
            self.in_amp2 = False
        elif not line.isspace() and 'DENOM' not in line:
            self.template_dict['amp2_lines'] += f'{line[0:6]}  {self.add_indices(line[6:])}'

    def prepare_bools(self):
        self.amp_calc_started = False
        self.jamp_started = False
        self.find_amp2 = False
        self.in_amp2 = False
        self.nhel_started = False

    def unfold_helicities(self, line, nature):



        #print('deps',line, deps)
        if nature not in  ['external', 'internal', 'amplitude']:
            raise Exception('wrong unfolding')
        
        if nature == 'external':
            new_objs = External.generate_wavfuncs(line, self.dag)
            for obj in new_objs:
                obj.line = apply_args(line, [obj.args])
        else:
            deps = Amplitude.get_deps(line, self.dag)
            name2dep = dict([(d.name,d) for d in sum(deps,[])])
            
            
        if nature == 'internal':
            new_objs = Internal.generate_wavfuncs(line, self.dag)
            for obj in new_objs:
                obj.line = apply_args(line, [obj.args])
                obj.linkdag = []
                for name in obj.args:
                    if name in name2dep:
                        name2dep[name].nb_used +=1
                        obj.linkdag.append(name2dep[name])
                
        if nature == 'amplitude':
            nb_diag = re.findall(r'AMP\((\d+)\)', line)[0]
            if nb_diag not in self.bad_amps:
                new_objs = Amplitude.generate_amps(line, self.dag, self.all_hel, self.bad_amps_perhel)
                out_line = self.apply_amps(line, new_objs)
                for i,obj in enumerate(new_objs):
                    if i == 0: 
                        obj.line = out_line
                        obj.nb_used = 1
                    else:
                        obj.line = ''
                        obj.nb_used = 1
                    obj.linkdag = []
                    for name in obj.args:
                        if name in name2dep:
                            name2dep[name].nb_used +=1
                            obj.linkdag.append(name2dep[name])
            else:
                return ''

          
        return new_objs
        #return f'{line}\n' if nature == 'external' else line

    def apply_amps(self, line, new_objs):
        if self.amp_splt:
            return split_amps(line, new_objs, gauge=self.gauge)  
        else: 

            return apply_args(line, [i.args for i in new_objs])

    def get_gwc(self, line, category):

        #self.last_category = 
        if category not in ['external', 'internal', 'amplitude']:
            return
        if self.last_category != 'external':
            self.last_category = category
            return

        External.get_gwc()
        # The only place the combinations change, so the only place the DAG's
        # bitmask index has to be rebuilt.
        self.dag.set_good_wav_combs(External.good_wav_combs)
        self.last_category = category

    def get_good_hel(self, line):
        if 'DATA (NHEL' in line:
            self.nhel_started = True
            this_hel = [int(hel) for hel in line.split('/')[1].split(',')]
            self.all_hel.append(tuple(this_hel))
        elif self.nhel_started:
            self.nhel_started = False
            
            if self.hel_filt:
                External.good_hel = dict([ (self.all_hel[int(i)-1],int(i)) for i in self.good_elements ])
            else:
                External.good_hel = dict([(v,i) for i,v in enumerate(self.all_hel)])

            External.map_hel=dict([(hel,i) for i,hel in  enumerate(External.good_hel)])
            # good_helicity needs the position of a helicity tuple in the FULL
            # table (not the filtered one map_hel indexes) once per amplitude it
            # unfolds; that was all_hel.index, a scan of the 128 rows 811 000
            # times over g g > g g g g g. The table is complete by now -- every
            # DATA (NHEL line precedes the first HELAS call.
            External.all_hel_index = dict([(hel,i) for i,hel in enumerate(self.all_hel)])
            External.hel_ranges = [set() for hel in next(iter(External.good_hel))]
            for comb in External.good_hel:
                for i, hel in enumerate(comb):
                    External.hel_ranges[i].add(hel)

            self.counter = 0
            nhel_array = [self.nhel_string(hel)
                          for hel in External.good_hel]
            nhel_lines = '\n'.join(nhel_array)
            self.template_dict['helicity_lines'] += nhel_lines

            self.template_dict['ncomb'] = len(External.good_hel)

    def nhel_string(self, hel_comb):
        old_id = External.good_hel[hel_comb]
        self.counter += 1
        formatted_hel = [f'{hel}' if hel < 0 else f' {hel}' for hel in hel_comb]
        nexternal = len(hel_comb)
        return (f'      DATA (NHEL(I,{self.counter}),I=0,{nexternal}) /{old_id},{",".join(formatted_hel)}/')

    def read_orig(self):

        # The HELAS call sequence may live in matrix<i>_origamp<k>.f rather than
        # inline; splice_amp_chunks puts those lines back where they were.
        input_file = splice_amp_chunks(self.input_file)

        self.prepare_bools()

        for line_num, line in tqdm(enumerate(input_file), total=get_num_lines(self.input_file)):
            if line_num == 0:
                line_cache = line
                continue

            if '!SKIP' in line:
                continue

            char_5 = ''
            try:
                char_5 = line[5]
            except IndexError:
                pass
            if char_5 == '$':
                line_cache = undo_multiline(line_cache, line)
                continue

            line, line_cache = line_cache, line

            self.get_old_name(line)
            self.get_good_hel(line)
            self.get_amp_stuff(line_num, line)
            call_type = self.function_call(line)
            self.get_gwc(line, call_type)


            if call_type in ['external', 'internal', 'amplitude']:
                self.template_dict['helas_calls'] += self.unfold_helicities(
                    line, call_type)

        self.template_dict['nwavefuncs'] = max(External.num_externals, Internal.max_wav_num, External.max_wav_num)
        # filter out uselless call
        for i in range(len(self.template_dict['helas_calls'])-1,-1,-1):
            obj = self.template_dict['helas_calls'][i]
            if obj.nb_used == 0:
                obj.line = ''
                for dep in obj.linkdag:
                    dep.nb_used -= 1

        
        
        self.template_dict['helas_calls'] = '\n'.join([f'{obj.line.rstrip()} ! count {obj.nb_used}' 
                                 for obj in self.template_dict['helas_calls']
                                 if obj.nb_used > 0 and obj.line])

    def read_template(self):
        out_file = open(self.output_file, 'w+')
        with open(self.template_file, 'r') as file:
            for line in file:
                s = Template(line)
                line = s.safe_substitute(self.template_dict)
                line = '\n'.join([do_multiline(sub_lines) for sub_lines in line.split('\n')])
                out_file.write(line)
        out_file.close()

    def amp_chunk_paths(self):
        """(chunk template, chunk file stem) for this matrix element, or None
        when the exporter did not write a chunk template for it."""

        if not self.output_file.endswith('_optim.f'):
            return None
        template_file = '%s_ampchunk.f' % self.template_file[:-len('.f')]
        if not os.path.exists(template_file):
            return None
        return template_file, self.output_file[:-len('_optim.f')]

    def write_amp_chunks(self):
        """Move the unrolled HELAS call sequence out of matrix<i>_optim.f and
        into matrix<i>_optimamp<k>.f, one subroutine per amp_chunk_size
        statements, leaving the calls to them behind.

        That sequence is essentially the whole recycled matrix element at high
        multiplicity, and as one basic block inside one routine it is what
        makes the file uncompilable; split up it also gets to be compiled
        apart from -- and at a lower optimisation level than -- the JAMP and
        colour blocks, which are the only part -O has anything to do on.

        Returns the number of chunk files written; 0 leaves the sequence inline
        and matrix<i>_optim.f exactly as it was before.
        """

        paths = self.amp_chunk_paths()
        if not paths:
            return 0
        template_file, stem = paths
        # a shorter sequence than last time must not leave live orphans behind
        for stale in glob.glob('%s_optimamp*.f' % stem):
            os.remove(stale)

        lines = self.template_dict['helas_calls'].split('\n')
        if self.amp_chunk_size <= 0 or len(lines) <= self.amp_chunk_size:
            return 0
        chunks = chunk_statements(lines, self.amp_chunk_size)
        if len(chunks) < 2:
            return 0

        template = open(template_file).read()
        name, args = get_subroutine_signature(template)
        # the leading blank puts the comments below in column 1: the template
        # hole itself is indented, and a comment marker has to start the line
        driver = ['',
                  'C     The unrolled HELAS call sequence lives in '
                  '%s_optimamp<k>.f,' % os.path.basename(stem),
                  'C     one subroutine per %d statements.' % self.amp_chunk_size]
        for i, chunk in enumerate(chunks):
            chunk_dict = dict(self.template_dict)
            chunk_dict['chunk_id'] = str(i + 1)
            # the template hole is indented; the leading newline keeps the
            # first call of the slice in the same columns as all the others,
            # which a long one would otherwise be split out of
            chunk_dict['helas_calls'] = '\n' + '\n'.join(chunk)
            text = Template(template).safe_substitute(chunk_dict)
            text = '\n'.join([do_multiline(sub) for sub in text.split('\n')])
            with open('%s_optimamp%d.f' % (stem, i + 1), 'w') as chunk_file:
                chunk_file.write(text)
            driver.append('      CALL %s(%s)'
                          % (Template(name).safe_substitute(chunk_id=i + 1),
                             args))
        self.template_dict['helas_calls'] = '\n'.join(driver)
        return len(chunks)

    def write_zero_matrix_element(self):
        paths = self.amp_chunk_paths()
        if paths:
            for stale in glob.glob('%s_optimamp*.f' % paths[1]):
                os.remove(stale)
        try:
      	    os.remove(self.output_file)
        except Exception:
            pass
        input_file = self.output_file.replace("_optim.f", "_orig.f")
        os.symlink(input_file, self.output_file)


    def generate_output_file(self):
        if not self.good_elements:
            misc.sprint("No helicity", self.input_file)
            self.write_zero_matrix_element()
            return
        
        atexit.register(self.clean_up)
        self.read_orig()
        self.write_amp_chunks()
        self.read_template()
        atexit.unregister(self.clean_up)

    def clean_up(self):
        pass


# get_arguments walks its line character by character, and unfold_helicities
# asks it again for every object it unfolds out of that line: 905 351 calls over
# the 8 143 HELAS lines of g g > g g g g g, all but ~50 000 of them a repeat of
# the line just parsed, and 3.7 s of a 16.8 s (profiled) recycling step. Keep the
# last few answers -- the callers walk the file one line at a time, so a handful
# of slots is all it takes -- and hand out a copy, since a shared mutable list is
# not what a caller that goes on to substitute arguments into it expects.
_ARGUMENT_CACHE = {}
_ARGUMENT_CACHE_SIZE = 32


def get_arguments(line):
    '''Find the substrings separated by commas between the first
    closed set of parentheses in 'line'.
    '''
    try:
        return list(_ARGUMENT_CACHE[line])
    except KeyError:
        pass
    arguments = parse_arguments(line)
    if len(_ARGUMENT_CACHE) >= _ARGUMENT_CACHE_SIZE:
        _ARGUMENT_CACHE.clear()
    _ARGUMENT_CACHE[line] = arguments
    return list(arguments)


def parse_arguments(line):
    '''The uncached get_arguments.'''
    start_idx = None
    call_idx = line.upper().find('CALL ')
    if call_idx != -1:
        start_idx = line.find('(', call_idx)
    if start_idx is None or start_idx == -1:
        start_idx = line.find('(')
    if start_idx == -1:
        return ['']

    bracket_depth = 0
    element = 0
    arguments = ['']
    for i, char in enumerate(line):
        if i < start_idx:
            continue
        if char == '(':
            bracket_depth += 1
            if bracket_depth - 1 == 0:
                # This is the first '('. We don't want to add it to
                # 'arguments'
                continue
        if char == ')':
            bracket_depth -= 1
            if bracket_depth == 0:
                # We've reached the end
                break
        if char == ',' and bracket_depth == 1:
            element += 1
            arguments.append('')
            continue
        if bracket_depth > 0 and char != ' ':
            arguments[element] += char
    return arguments


def apply_args(old_line, all_the_args):
    call_idx = old_line.upper().find('CALL ')
    if call_idx == -1:
        function = (old_line.split('(')[0]).split()[-1]
        old_args = old_line.split(function)[-1]
        new_lines = [old_line.replace(old_args, f'({",".join(x)})\n')
                     for x in all_the_args]
        return ''.join(new_lines)

    call_arg_start = old_line.find('(', call_idx)
    if call_arg_start == -1:
        return old_line

    bracket_depth = 0
    call_arg_end = -1
    for i, char in enumerate(old_line[call_arg_start:], start=call_arg_start):
        if char == '(':
            bracket_depth += 1
        elif char == ')':
            bracket_depth -= 1
            if bracket_depth == 0:
                call_arg_end = i
                break
    if call_arg_end == -1:
        return old_line

    call_head = old_line[:call_arg_start]
    call_tail = old_line[call_arg_end+1:]
    new_lines = [f'{call_head}({",".join(args)}){call_tail}'
                 for args in all_the_args]
    
    return ''.join(new_lines)

def get_called_function(line):
    call_idx = line.upper().find('CALL ')
    if call_idx == -1:
        return None
    after_call = line[call_idx+5:]
    if '(' not in after_call:
        return None
    return after_call.split('(', 1)[0].strip().split()[-1]

def split_amps(line, new_amps, gauge):
    if not new_amps:
        return ''
    call_idx = line.upper().find('CALL ')
    call_arg_start = line.find('(', call_idx) if call_idx != -1 else -1
    called_function = get_called_function(line)
    if call_idx == -1 or call_arg_start == -1 or not called_function:
        return ''
    call_prefix = line[:call_idx]
    call_keyword = line[call_idx:call_idx+5]
    function_root = called_function.split('_0')[0]
    indent = re.match(r'\s*', call_prefix).group(0)
    guard_stmt = call_prefix.strip()
    guarded_call = guard_stmt.upper().startswith('IF')
    call_stmt_prefix = call_prefix if not guarded_call else (indent + '  ')
    fct = '%s%s%s' % (call_stmt_prefix, call_keyword, function_root)
    for i,amp in enumerate(new_amps):
        if i == 0:
            occur = []
            for a in amp.args:
                if "W(" in a:
                    tmp = collections.defaultdict(int)
                    tmp[a] += 1
                    occur.append(tmp)
        else:
            for i in range(len(occur)):
                a = amp.args[i]
                occur[i][a] +=1
    # Each element in occur is the wavs that appear in a column, with
    # the number of occurences
    nb_wav =  [len(o) for o in occur]
    to_remove = nb_wav.index(max(nb_wav)) 
    # Remove the one that occurs the most
    occur.pop(to_remove)
    
    lines = [] 
    # Get the wavs per column
    wav_name = [o.keys() for o in occur]          
    for wfcts in product(*wav_name):
        # Select the amplitudes produced by wfcts
        sub_amps = [amp for amp in new_amps 
                    if all(w in amp.args for w in wfcts)]
        if not sub_amps:
            continue
        if len(sub_amps) ==1:
            lines.append(apply_args(line, [i.args for i in sub_amps]).replace('\n',''))
            
            continue
                         
        # the next line is to make the code nicer 
        sub_amps.sort(key=lambda a: int(a.args[-1][:-1].split(',',1)[1]))
        windices = []
        hel_calculated = []
        iamp = 0
        local_lines = []
        for i,amp in enumerate(sub_amps):
            args = amp.args[:]   
            # Remove wav and get its index
            wcontract = args.pop(to_remove)
            windex = wcontract.split('(')[1].split(')')[0]
            windices.append(windex)
            amp_result,  args[-1]  =  args[-1], 'TMP(1)'
            
            if i ==0:
                # Call the original fct with P1N_...
                # Final arg is replaced with TMP(1)
                spin = function_root[to_remove]
                local_lines.append('%sP1N_%s(%s)' % (fct, to_remove+1, ', '.join(args)))

            hel, iamp = re.findall(r'AMP\((\d+),(\d+)\)', amp_result)[0]
            hel_calculated.append(hel)
            #lines.append(' %(result)s = TMP(3) * W(3,%(w)s) + TMP(4) * W(4,%(w)s)+'
            #             % {'result': amp_result, 'w':  windex}) 
            #lines.append('     &             TMP(5) * W(5,%(w)s)+TMP(6) * W(6,%(w)s)'
            #             % {'result': amp_result, 'w':  windex})
        if spin == "F" or ( spin == "V" and gauge !='FD'):
            suffix = ''
        elif spin == "S":
            suffix = 'S'
        elif spin == "V" and  gauge == "FD":
            suffix = "FD"
        else:
            raise Exception("split amp not supported for spin2, 3/2")

        local_lines.append("""%(call_prefix)s%(call_keyword)sCombineAmp%(suffix)s(%(nb)i,
     & (/%(hel_list)s/), 
     & (/%(w_list)s/),
     & TMP, W, AMP(1,%(iamp)s))""" % {'suffix':suffix,
                                      'call_prefix': call_stmt_prefix,
                                      'call_keyword': call_keyword,
                                      'nb': len(sub_amps),
                                      'hel_list': ','.join(hel_calculated),
                                      'w_list': ','.join(windices),
                                      'iamp': iamp
                                     })
        if guarded_call:
            if not guard_stmt.upper().endswith('THEN'):
                guard_stmt = '%s THEN' % guard_stmt
            lines.append('%s%s' % (indent, guard_stmt))
            lines.extend(local_lines)
            lines.append('%sENDIF' % indent)
        else:
            lines.extend(local_lines)

            
    #lines.append('')
    return '\n'.join(lines)

def get_num(wav):
    name = wav.name
    between_brackets = re.search(r'\(.*?\)', name).group()
    num = int(between_brackets[1:-1].split(',')[-1])    
    return num

def undo_multiline(old_line, new_line):
    new_line = new_line[6:]
    old_line = old_line.replace('\n','')
    return f'{old_line}{new_line}'

def do_multiline(line):
    if "!" in line:
        line,comment  = line.split("!",1)
    else: 
        comment = None
    char_limit = 72
    if len(line) > char_limit:
        indent = ''
        for char in line[6:]:
            if char == ' ':
                indent += char
            else:
                break

        # The split must leave at least one character of the statement on the
        # first line. Searching from column 0 lets a statement with no internal
        # blank before the limit -- JAMPF(2,1)=+2D0*(-IMAG1*JAMP(3,1)-...) is
        # one, and so is any long JAMP -- split inside its own indent: the
        # first line comes out blank and the continuation after it then
        # attaches to the PREVIOUS statement, which fortran rejects.
        first_split = 6 + len(indent)

        split_line = []
        remaining = line
        floor = first_split
        while len(remaining) > char_limit:
            split_at = remaining.rfind(' ', floor + 1, char_limit + 1)
            if split_at <= floor:
                split_line.append(remaining[:char_limit])
                remaining = remaining[char_limit:]
            else:
                split_line.append(remaining[:split_at+1])
                remaining = remaining[split_at+1:]
            # the continuations carry no indent of their own, it is prepended
            # by the join below
            floor = 0
        split_line.append(remaining)

        line = f'\n     ${indent}'.join(split_line)
    if not comment:
        return line
    else:
        return f"{line} ! {comment}"
def int_to_string(i):
    if i == 1:
        return '+1'
    if i == 0:
        return ' 0'
    if i == -1:
        return '-1'
    else:
        print(f'How can {i} be a helicity?')
        set_trace()
        exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='The file containing the '
                                          'original matrix calculation')
    parser.add_argument('hel_file', help='The file containing the '
                                         'contributing helicities')
    parser.add_argument('--hf-off', dest='hel_filt', action='store_false', default=True, help='Disable helicity filtering')
    parser.add_argument('--as-off', dest='amp_splt', action='store_false', default=True, help='Disable amplitude splitting')

    args = parser.parse_args()

    with open(args.hel_file, 'r') as file:
        good_elements = file.readline().split()

    recycler = HelicityRecycler(good_elements)

    recycler.hel_filt = args.hel_filt
    recycler.amp_splt = args.amp_splt

    recycler.set_input(args.input_file)
    recycler.set_output('green_matrix.f')
    recycler.set_template('template_matrix1.f')

    recycler.generate_output_file()

if __name__ == '__main__':
    main()
