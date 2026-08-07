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
"""Language neutral part of the color flow (JAMP) optimisation.

Every backend writes the same object: the matrix of coefficients giving each
color flow as a combination of the amplitudes,

    JAMP(i) = sum_j A(i,j) * AMP(j)

Written out as it stands that is one line per non zero entry, which for a
multi-gluon process is tens of thousands of them. The search below replaces the
repeated pieces by shared sub-expressions, so that the matrix is left with far
fewer entries and a list of definitions to compute first.

Nothing here knows about fortran or C++: it takes the coefficient matrix and
gives back the reduced matrix and the definitions. The exporters print that in
their own language (get_JAMP_lines for fortran, get_jamp_accumulation_lines for
the C++/cudacpp writer).
"""

from __future__ import absolute_import

import bisect
import collections
import fractions
import logging
import time

import madgraph.various.banner as banner_mod

logger = logging.getLogger('madgraph.export_v4')


class JampOptimiser(object):
    """The common sub-expression search over the JAMP coefficient matrix.

    Mixed into the exporters, which supply the printing. A subclass that hands
    a symmetry to optimise_jamp must also provide optimise_jamp_best (only the
    fortran exporter does, see export_v4)."""

    # Off by default: the plain output of a backend is the expanded one, and
    # each exporter switches this on for itself. 'jamp_optim' in cmd_options
    # (i.e. --jamp_optim=True|False at output time) wins over the class value.
    jamp_optim = False
    # how many times the JAMP optimisation called itself, for the record
    myjamp_count = 0
    # take the power of i shared by every coefficient out before searching, so
    # that the search walks over whole numbers (see optimise_jamp_matrix)
    jamp_integer_walk = True

    def jamp_optim_enabled(self):
        """Whether to run the optimisation, --jamp_optim first."""

        cmd_options = getattr(self, 'cmd_options', None) or {}
        if 'jamp_optim' in cmd_options:
            return banner_mod.ConfigFile.format_variable(
                            cmd_options['jamp_optim'], bool, 'jamp_optim')
        return self.jamp_optim

    @staticmethod
    def jamp_matrix(color_amplitudes):
        """The coefficient matrix all_element[(color flow, amplitude)] = value
        of the color amplitudes, color flows numbered from 1 and amplitudes as
        they number themselves. This is the input of the optimisation, and the
        same value the expanded lines are written with."""

        all_element = {}
        # Every single amplitude carries a power of the number of colors in its
        # coefficient, but a process only uses a handful of distinct powers, so
        # build the corresponding fractions once instead of once per amplitude.
        nc_powers = {}
        for i, coeff_list in enumerate(color_amplitudes):
            for (coefficient, amp_number) in coeff_list:
                if not coefficient:
                    continue
                try:
                    nc_power = nc_powers[coefficient[3]]
                except KeyError:
                    nc_power = fractions.Fraction(3)**coefficient[3]
                    nc_powers[coefficient[3]] = nc_power
                value = (1j if coefficient[2] else 1) * \
                        coefficient[0] * coefficient[1] * nc_power
                key = (i + 1, amp_number)
                if key not in all_element:
                    all_element[key] = value
                else:
                    all_element[key] += value
        return all_element

    def jamp_walk_integers(self, all_element):
        """Take the power of i shared by every coefficient out of the matrix
        and return it, leaving all_element with whole numbers -- they compare
        and hash exactly, and nothing has to be widened to complex. Returns
        None when there is no such phase, all_element then being complex
        throughout. The phase goes back on with jamp_apply_phase, so the lines
        written are the same either way."""

        phase = self.jamp_global_phase(all_element) \
                                        if self.jamp_integer_walk else None
        if phase is not None:
            whole = {}
            for key, value in all_element.items():
                number = value / phase if phase != 1 else value
                if isinstance(number, complex):
                    number = number.real
                number = fractions.Fraction(number).limit_denominator(10**9)
                if number.denominator != 1:
                    break
                whole[key] = int(number)
            else:
                all_element.clear()
                all_element.update(whole)
                return phase
        for key in all_element:
            all_element[key] = complex(all_element[key])
        return None

    @staticmethod
    def jamp_apply_phase(new_mat, phase):
        """Put back the phase jamp_walk_integers took out. The definitions hold
        ratios, which it cancels out of; only the coefficients left in the
        matrix carry it."""

        if phase is None or phase == 1:
            return
        for key in new_mat:
            new_mat[key] = new_mat[key] * phase

    def optimise_jamp_matrix(self, all_element, symmetry=None):
        """Run the optimisation over the coefficient matrix and return
        (new_mat, defs):
          - defs is a list of (i, op1, op2, frac, nb): definition number i is
            op1 + frac*op2, where a positive operand is an amplitude and a
            negative one is the definition number -op;
          - new_mat is what the matrix is left with, keyed the same way as the
            input except that a negative amplitude index means the definition
            of that number.

        all_element is consumed (the optimisation works in place). The fortran
        exporter runs the three steps itself, since it has to look at the
        walked matrix to work out the color basis symmetry in between."""

        if len(all_element) > 1000:
            logger.info("Computing Color-Flow optimization [%s term]",
                        len(all_element))
            start_time = time.time()
        else:
            start_time = 0

        self.myjamp_count = 0
        phase = self.jamp_walk_integers(all_element)
        new_mat, defs = self.optimise_jamp(all_element, symmetry=symmetry)
        self.jamp_apply_phase(new_mat, phase)
        if start_time:
            logger.info("Color-Flow passed to %s term in %ss. Introduce %i contraction",
                        len(new_mat), int(time.time()-start_time), len(defs))
        return new_mat, defs

    @staticmethod
    def jamp_global_phase(all_element):
        """The power of i every coefficient carries, when they all carry the
        same one. A pure gluon process picks up one factor of i per f^abc, the
        same for every term, so the whole matrix is real or wholly imaginary;
        a quark line mixes the two and there is nothing to take out."""

        phase = None
        for value in all_element.values():
            if not value:
                continue
            number = complex(value)
            if number.imag == 0:
                here = 1
            elif number.real == 0:
                here = 1j
            else:
                return None
            if phase is None:
                phase = here
            elif phase != here:
                return None
        return phase

    @staticmethod
    def index_jamp_matrix(all_element, nb_col):
        """Sorted lists of the positions of the non zero entries of the matrix,
        by line and by column. An entry which is present but zero does not
        count, and neither does a column outside the 0..nb_col range, so that
        these indices list exactly the entries the plain scan would look at."""

        lines = collections.defaultdict(list)
        columns = collections.defaultdict(list)
        for (i, j), value in all_element.items():
            if value and j < nb_col:
                lines[i].append(j)
                columns[j].append(i)
        for line in lines.values():
            line.sort()
        for column in columns.values():
            column.sort()
        return lines, columns

    @staticmethod
    def common_jamp_lines(columns, nb_line, j1, j2):
        """Lines, in increasing order, where both columns j1 and j2 are non
        zero. Both column lists are sorted, so this is a plain merge."""

        left, right = columns.get(j1, []), columns.get(j2, [])
        res = []
        pos1 = pos2 = 0
        while pos1 < len(left) and pos2 < len(right):
            if left[pos1] == right[pos2]:
                if left[pos1] < nb_line:
                    res.append(left[pos1])
                pos1 += 1
                pos2 += 1
            elif left[pos1] < right[pos2]:
                pos1 += 1
            else:
                pos2 += 1
        return res

    def optimise_jamp(self, all_element, nb_line=0, nb_col=0, added=0,
                      symmetry=None):
        """ optimise problem of type Y = A X
                A is a matrix (all_element)
                X is the fortran name of the input.
            The code iteratively add sub-expression jtemp[sub_add]
            and recall itself (this is add to the X size)

            With a symmetry (see get_jamp_symmetry) the sub-expressions are
            introduced by whole orbits of that symmetry instead of one at a
            time, so that the result can be written as one recipe per orbit.
            The orbits are then left in self.jamp_orbits.
        """
        if symmetry:
            return self.optimise_jamp_best(all_element, symmetry)

        self.myjamp_count +=1

        if not nb_line:
            for i,j in all_element:
                if i+1 > nb_line:
                    nb_line = i+1
                if j+1> nb_col:
                    nb_col = j+1
            if nb_col > 600 and added==0:
                all_element1, all_element2 = {}, {}
                for (k1,k2) in all_element:
                    if k2 >= nb_col//2:
                        all_element2[(k1,1+k2-(nb_col//2))] = all_element[(k1,k2)]
                    else:
                        all_element1[(k1,k2)] = all_element[(k1,k2)]

                all_element1, newdef1 = self.optimise_jamp(all_element1)
                nb_added1 = len(newdef1)

                all_element2, newdef2 = self.optimise_jamp(all_element2)

                for (k1,k2) in all_element2:
                    if k2 >= 0:
                        all_element1[(k1,k2+(nb_col//2)-1)] = all_element2[(k1,k2)]
                    if k2 < 0:
                        all_element1[(k1,k2-nb_added1)] = all_element2[(k1,k2)]
                # new_def format: added,j1,j2,R, max_count
                for k, j1,j2, R, c in newdef2:
                    if j2 > 0:
                        k2 = j2+nb_col//2 -1
                    else:
                        k2 = j2-nb_added1
                    if j1 > 0:
                        k1 = j1+nb_col//2 -1
                    else:
                        k1 = j1-nb_added1
                    newdef1.append((k+nb_added1, k1, k2, R, c))
                if newdef1:
                    all_element, new_def = self.optimise_jamp(all_element1, nb_line=0, nb_col=0, added=len(newdef1))
                    newdef1 = newdef1 + new_def
                return all_element, newdef1

        # Index of the non zero entries, by line and by column. The matrix is
        # very sparse (a color flow only gets a small share of the amplitudes)
        # so walking the whole 0..nb_col range for every entry, as looking the
        # columns up one by one in the matrix amounts to, spends nearly all of
        # its time discovering zeros.
        lines, columns = self.index_jamp_matrix(all_element, nb_col)

        max_count = 0
        all_index = []
        # how many lines have the same ratio between two given columns, keyed
        # by the two columns and the ratio at once rather than by nested
        # dictionaries: this is the innermost loop of the whole optimisation
        operation = collections.defaultdict(int)
        for (i,j1), v1 in all_element.items():
            line = lines.get(i)
            if not line:
                continue
            for j2 in line[bisect.bisect_right(line, j1):]:
                key = (j1, j2, all_element[(i,j2)]/v1)
                operation[key] += 1
                count = operation[key]
                if count > max_count:
                    max_count = count
                    all_index = [key]
                elif count == max_count:
                    all_index.append(key)

        if max_count <= 1:
            return all_element, []

        to_add = []
        for index in all_index:
            j1,j2,R = index
            first = True
            # only the lines where both columns are filled can contribute; the
            # substitutions done here can empty some of them, so the values
            # still have to be read back from the matrix
            for i in self.common_jamp_lines(columns, nb_line, j1, j2):
                v1 = all_element.get((i,j1), 0)
                v2 = all_element.get((i,j2), 0)
                if not v1 or not v2:
                    continue
                if v2/v1 == R:
                    if first:
                        first = False
                        added +=1
                        to_add.append((added,j1,j2,R, max_count))

                    all_element[(i,-added)] = v1
                    del all_element[(i,j1)] #= 0
                    del all_element[(i,j2)] #= 0

        logger.log(5,"Define %d new shortcut reused %d times", len(to_add), max_count)
        new_element, new_def =  self.optimise_jamp(all_element, nb_line=nb_line, nb_col=nb_col, added=added)
        for one_def in to_add:
            new_def.insert(0, one_def)
        return new_element, new_def

    @staticmethod
    def jamp_operation_count(new_mat, defs):
        """Additions the result asks for: one per definition, plus what is left
        in each line of the matrix."""

        terms = collections.Counter()
        for jamp, _var in new_mat:
            terms[jamp] += 1
        return len(defs) + sum(max(0, count - 1) for count in terms.values())

    @staticmethod
    def jamp_definition_order(defs):
        """The definitions in the order they can be computed while the
        amplitudes are produced one at a time, together with the amplitude each
        one is ready after.

        Returns (order, ready): order lists the definition numbers, ready maps
        a definition number onto the last amplitude it needs (transitively).
        Sorting by that amplitude keeps the list topological, since a
        definition never needs fewer amplitudes than the ones it is built
        from."""

        ready = {}
        rank = {}
        for position, (i, amp1, amp2, _frac, _nb) in enumerate(defs):
            last = 0
            for amp in (amp1, amp2):
                last = max(last, amp if amp > 0 else ready[-amp])
            ready[i] = last
            rank[i] = position
        order = sorted(ready, key=lambda i: (ready[i], rank[i]))
        return order, ready
