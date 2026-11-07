"""Assemble the madspace objects a dressed-lepton run needs.

`beam.lepton_density` names a directory of `Source/PDF/lep_densities`. That
directory ships the density as Fortran (`eepdf.f`), which madspace cannot
read, so the grid is re-tabulated into the LHAPDF format the first time it is
needed and cached next to the run. Re-tabulating takes a fraction of a second
and needs no compiler, which is why the tables are generated rather than
committed.

Four things come out of it, and they have to agree with each other:

* `luminosity`  -- `EELuminosity`, the two-beam density `sum_n f1_n f2_n`;
* `mapping`     -- `LeptonLuminosity`, which draws x from the `(1-x)`
                   endpoint the density lives on;
* `sampler`     -- `BeamstrahlungSampler`, or None for a pure-ISR density;
* `pdf_factor`  -- what to multiply the luminosity by before handing it to
                   `DifferentialCrossSection`.

The agreement that matters is the sampling exponent. The density is stored
reduced by `ps_expo` and the mapping's Jacobian omits exactly that power so
the two cancel; a mismatch leaves the cross section wrong by `(1-x)**delta`
with nothing to flag it. Both are built here from the same constant for that
reason.
"""

import gzip
import logging
import os
import sys

logger = logging.getLogger('madevent.lepton_density')

# MadEvent's get_ee_expo(). The density's reduction and the mapping's
# importance sampling must use the same value -- see the module docstring.
PS_EXPO = 0.96

_QTERM_NAMES = {
    'sqrt': 'sqrt_u',
    'u15': 'u_three_halves',
    'ym15': 'y_minus_three_halves',
}


def _generator(mg5_root):
    """The re-tabulation script, which lives with the grids it reads."""
    path = os.path.join(mg5_root, 'vendor', 'eepdfgrid')
    if path not in sys.path:
        sys.path.insert(0, path)
    import gen_eepdf_lhagrid
    return gen_eepdf_lhagrid


def available(source_dir):
    """Density directories that carry a grid this can use."""
    base = os.path.join(source_dir, 'PDF', 'lep_densities')
    if not os.path.isdir(base):
        return []
    return sorted(name for name in os.listdir(base)
                  if os.path.isfile(os.path.join(base, name, 'eepdf.f')))


class LeptonDensity:
    """Everything a dressed-lepton run needs from madspace."""

    def __init__(self, name, source_dir, mg5_root, madspace, context,
                 e_cm, beam_pdgs=(11, -11), s_hat_min=0.0, prefix='ee_',
                 cache_dir=None):
        self.name = name
        self.ms = madspace
        gen = _generator(mg5_root)

        set_dir = os.path.join(source_dir, 'PDF', 'lep_densities', name)
        if not os.path.isdir(set_dir):
            raise ValueError(
                'unknown lepton density %r; %s has %s'
                % (name, os.path.join(source_dir, 'PDF', 'lep_densities'),
                   ', '.join(available(source_dir)) or 'none'))

        work = cache_dir or set_dir
        dat = os.path.join(work, '%s.dat' % name)
        meta_path = os.path.join(work, '%s.json' % name)
        if not (os.path.exists(dat) and os.path.exists(meta_path)):
            logger.info('re-tabulating the %s density for madspace', name)
            if work != set_dir:
                import shutil
                if not os.path.isdir(work):
                    os.makedirs(work)
                shutil.copy(os.path.join(set_dir, 'eepdf.f'), work)
            try:
                dat, meta_path, _, _ = gen.generate(work, name)
            except gen.AnalyticDensity:
                raise ValueError(
                    '%s defines its density in closed form and has no grid; '
                    'madspace cannot use it yet' % name)

        import json
        with open(meta_path) as fsock:
            self.meta = json.load(fsock)

        # component order has to be the same on both beams: the luminosity
        # pairs them term by term, not as a product of sums
        self.components = (1, 2, 3, 4)
        pids = [self._pids(pdg) for pdg in beam_pdgs]
        offsets = [self._offsets(pdg) for pdg in beam_pdgs]

        self._host_pids = pids
        self._host_offsets = offsets
        self._host_lum = None
        # kept for _host_luminosity: deriving its prefix from the set name
        # instead would collide as soon as one process builds two densities
        # for the same set, which the context refuses
        self._prefix = prefix
        grid = madspace.PdfGrid(dat)
        grid.initialize_globals(context, prefix)
        self.grid = grid
        self.luminosity = madspace.EELuminosity(
            grid, pids[0], pids[1], offsets[0], offsets[1], prefix=prefix)

        self.mapping_e_cm = e_cm
        self.mapping = madspace.LeptonLuminosity(e_cm * e_cm, s_hat_min, PS_EXPO)
        self.sampler = self._build_sampler(beam_pdgs)

    # -- component bookkeeping ------------------------------------------
    @staticmethod
    def _beam_index(pdg):
        """The grid's beam index: 1 for a positron, 2 for an electron."""
        if pdg == 11:
            return 2
        if pdg == -11:
            return 1
        raise ValueError(
            'the dressed-lepton densities are fitted for e+ e-, not %d' % pdg)

    def _pids(self, pdg):
        index = self._beam_index(pdg)
        return [1000 * n + 10 * index + index for n in self.components]

    def _offsets(self, pdg):
        """`ps_expo - kappa0` per component: what EEPartonDensity adds on top
        of the Q-dependent part of the reduction."""
        out = []
        for n in self.components:
            k = self.meta['kappa']['%d_%d' % (n, pdg)]
            out.append(PS_EXPO - (k['c_b'] * self.meta['b_exp'] + k['c_1']))
        return out

    # -- beamstrahlung ---------------------------------------------------
    def _build_sampler(self, beam_pdgs):
        info = self.meta.get('beamstrahlung')
        if info is None:
            return None
        pars = info['parameters']

        def per_beam(pdg):
            out = []
            for n in self.components:
                key = '%d_%d' % (n, pdg)
                if key in pars:
                    p = pars[key]
                    out.append(self.ms.BeamstrahlungParameters(
                        p['a'], p['p'], p['q'], True))
                else:
                    out.append(self.ms.BeamstrahlungParameters())
            return out

        # the collider-dependent factor is carried with the grid rather than
        # assumed: getting it wrong drifts the normalisation silently
        q_term = getattr(self.ms.BeamstrahlungQTerm,
                         _QTERM_NAMES[info['q_term']])
        return self.ms.BeamstrahlungSampler(
            per_beam(beam_pdgs[0]), per_beam(beam_pdgs[1]), q_term)

    @property
    def has_beamstrahlung(self):
        return self.sampler is not None and self.sampler.has_beamstrahlung()

    # -- beamstrahlung at event-writing time ------------------------------
    # 1-x is not recoverable below this from an x stored as a double; a beam
    # that close to 1 did not radiate measurably, and clamping here makes the
    # component sampling pick one of the delta pieces, which is what says so.
    OMX_FLOOR = 1e-16

    def _host_luminosity(self):
        """A luminosity bound to the default context.

        The integration's copy lives on whatever device context the run uses;
        calling a FunctionGenerator from Python goes through the default one,
        which would not have the grid.
        """
        if getattr(self, '_host_lum', None) is None:
            context = self.ms.default_context()
            prefix = '%shost_%s_' % (self._prefix, self.name)
            self.grid.initialize_globals(context, prefix)
            self._host_lum = self.ms.EELuminosity(
                self.grid, self._host_pids[0], self._host_pids[1],
                self._host_offsets[0], self._host_offsets[1], prefix=prefix)
        return self._host_lum

    def annotate_lhe(self, path, seed=0, fact_scale=None):
        """Write <beamstrahlung> into every event of an LHE file.

        The split is sampled once per *written* event rather than per
        phase-space point: it does not enter the weight, so there is nothing
        to gain from doing it during integration and a great deal of work to
        avoid.

        x comes back from the initial-state momenta, which carry it exactly --
        x1 = (E+pz)/sqrt(s), x2 = (E-pz)/sqrt(s) -- so this needs nothing
        threaded through the integrand or the event record.

        `fact_scale` is the scale the density was evaluated at, and has to be
        passed in: the only scale an LHE event carries is SCALUP, which is
        the shower starting scale. MadEvent happens to write mu_F there, so
        reading it looks right when the two files are compared, but mg7 does
        not -- it writes a per-event shower scale, 10 to 120 GeV where mu_F
        is 91.188 -- and sampling the split at that scale draws it from a
        different spectrum than the density that produced the event. None
        falls back to SCALUP with a warning, which is all a run with a
        dynamical factorisation scale can do.

        Returns the number of events annotated.
        """
        if self.sampler is None or not self.has_beamstrahlung:
            return 0

        import numpy as np
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as fsock:
            text = fsock.read()
        head, _, rest = text.partition('<event>')
        if not rest:
            return 0
        events = ('<event>' + rest).split('<event>')[1:]

        e_cm = float(self.mapping_e_cm)
        x1, x2, scale = [], [], []
        for event in events:
            body = event.split('</event>')[0]
            incoming = []
            header = None
            for line in body.split('\n'):
                field = line.split()
                # The header is the FIRST six-field line of this event.
                # Collecting every six-field line in the file instead lets an
                # <mgrwt> or <scales> payload -- or anything another LHEF
                # writer added -- shift the scales against the events, and
                # the only symptom would be the length check below firing and
                # silently putting every event at e_cm/2.
                if header is None and len(field) == 6:
                    header = field
                elif len(field) >= 13 and field[1] == '-1':
                    incoming.append((float(field[9]), float(field[8])))
            if len(incoming) == 2:
                x1.append((incoming[0][0] + incoming[0][1]) / e_cm)
                x2.append((incoming[1][0] - incoming[1][1]) / e_cm)
            else:
                x1.append(1.0)
                x2.append(1.0)
            scale.append(float(header[3]) if header else e_cm / 2.)

        n = len(events)
        x1 = np.clip(np.asarray(x1), 0.0, 1.0)
        x2 = np.clip(np.asarray(x2), 0.0, 1.0)
        omx1 = np.maximum(1.0 - x1, self.OMX_FLOOR)
        omx2 = np.maximum(1.0 - x2, self.OMX_FLOOR)
        if fact_scale is not None:
            q = np.full(n, float(fact_scale))
        else:
            logger.warning(
                'the factorisation scale is dynamical, so the beamstrahlung '
                'split falls back to the shower scale in SCALUP; it is not '
                'the scale the density was evaluated at')
            q = np.asarray(scale, dtype=float)

        # A list seed gives a stream independent of default_rng(run_seed),
        # which the event generation already used; the mask keeps any integer
        # the run card allows -- including the -1 that means "draw one" before
        # it is resolved -- a legal non-negative seed rather than a ValueError.
        rng = np.random.default_rng([int(seed) & ((1 << 64) - 1), 0xB57A])
        component = np.asarray(
            self._host_luminosity()(x1, omx1, x2, omx2, q,
                                    rng.uniform(0, 1, n)).component_index)
        deviates = rng.uniform(0, 1, (n, 4))

        out = [head]
        annotated = 0
        for i, event in enumerate(events):
            fractions = self.sampler.sample(
                float(x1[i]), float(x2[i]), float(q[i]) ** 2,
                int(component[i]), *deviates[i].tolist())
            body, sep, tail = event.partition('</event>')
            element = '<beamstrahlung>%15.7e%15.7e </beamstrahlung>\n' % fractions
            out.append('<event>' + body + element + sep + tail)
            annotated += 1
        with opener(path, 'wt') as fsock:
            fsock.write(''.join(out))
        return annotated

    # -- cross-section plumbing ------------------------------------------
    @staticmethod
    def pdf_factor(builder, luminosity, x1, x2):
        """Convert the luminosity to what `diff_cross_section` expects.

        That instruction computes `ME * pdf1 * pdf2 / (2 s x1^2 x2^2)`, which
        is the LHAPDF `x f(x)` convention. The dressed densities are plain
        `f(x)` (MadEvent's `pdg2pdf` divides by x), and the luminosity already
        pairs the two beams, so it goes in as `L * x1 * x2` on one beam with
        nothing on the other -- giving `L * ME / (2 s_hat)`.
        """
        return builder.mul(luminosity, builder.mul(x1, x2))
