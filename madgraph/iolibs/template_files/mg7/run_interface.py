"""Reuse the madevent post-processing tool drivers from the standard (mg7)
output.

The mg7 output is a lean C++/madspace directory: it has no Fortran ``Source``
to compile, its run card is ``run_card.toml`` (not ``run_card.dat``) and its
events live in ``Events/<run>/events.lhe``. The madevent interface
(:class:`~madgraph.interface.madevent_interface.MadEventCmd`) already knows how
to run Pythia8 (in parallel, over the cluster/multicore backend), MadSpin,
reweighting, Delphes, Rivet and MadAnalysis5 -- but it assumes the madevent
directory layout.

:class:`MG7RunCmd` is a thin adapter subclass that reconciles the two so that
the mg7 launcher can simply do ``cmd.exec_cmd('pythia8 --no_default run_01')``
and inherit the real (parallel) shower code, etc., instead of re-implementing
each tool.
"""

import gzip
import json
import logging
import math
import os
import shutil

import madgraph.interface.madevent_interface as madevent_interface
import madgraph.madevent.gen_crossxhtml as gen_crossxhtml
import madgraph.various.banner as banner_mod
import madgraph.various.lhe_parser as lhe_parser

pjoin = os.path.join
logger = logging.getLogger('madevent')


class MG7RunCmd(madevent_interface.MadEventCmd):
    """A :class:`MadEventCmd` pointed at an mg7 output directory.

    Only the pieces that are genuinely madevent-specific are overridden:

    * ``configure_directory`` -- the base version compiles the Fortran
      ``Source`` and links LHAPDF; none of that exists / is needed here.
    * ``set_run_name`` -- loads the run card from ``run_card.toml``
      (:class:`RunCardMG7`) and takes the banner straight from the mg7 LHE,
      instead of reading ``run_card.dat`` / juggling banner files.
    * ``do_treatcards`` -- writes Fortran ``.inc`` files; not applicable.
    * ``getSysSummaryFromLog`` -- madspace computes the scale/PDF systematics
      itself and reports them in ``info.json``; the base version parses the
      ``parton_systematics.log`` that systematics.py used to write.
    """

    def getSysSummaryFromLog(self, kpath=None, knext_name=None):
        """Scale and PDF variation for the scan summary, as [Hi, Lo] percents.

        madspace computes the variations itself while writing the events and
        reports them in ``Events/<run>/info.json`` under "systematics"; there
        is no ``parton_systematics.log`` unless the legacy post-processing path
        ([postprocessing] systematics) was used, so fall back to the base
        parser for that case.

        The strings match what the base parser returns for a madevent run --
        signed percentages like "+29.6" / "-21.5" -- so the scan summary column
        is the same whichever path produced them.
        """

        summary = self._read_native_systematics(kpath, knext_name)
        if summary is None:
            return madevent_interface.MadEventCmd.getSysSummaryFromLog(
                self, kpath=kpath, knext_name=knext_name)

        nominal = summary.get('nominal', {}).get('cross_section')
        if not nominal:
            return [], []

        scale = []
        band = summary.get('scale')
        if band and band.get('max') is not None and band.get('min') is not None:
            # systematics.py reports +(max-nom)/nom and -(nom-min)/nom, both as
            # positive magnitudes carrying an explicit sign
            scale = [self._as_percent(band['max'] - nominal, nominal),
                     self._as_percent(nominal - band['min'], nominal,
                                      negative=True)]

        pdf = []
        for entry in summary.get('pdf') or []:
            up, down = entry.get('uncertainty_up'), entry.get('uncertainty_down')
            if up is None or down is None:
                continue
            # the uncertainties are absolute. Divide by the set's own central
            # value, which is what log_systematics_summary prints in the same
            # run -- for a replicas set that is the replica mean, not the
            # nominal member, and the two numbers should not disagree.
            reference = entry.get('central') or nominal
            pdf = [self._as_percent(up, reference),
                   self._as_percent(down, reference, negative=True)]
            break          # the nominal set's entry comes first

        return scale, pdf

    def _read_native_systematics(self, kpath, knext_name):
        """The "systematics" block of the run's info.json, or None.

        None means "no native summary here" -- either the file is missing or
        the run did not compute them -- and the caller falls back.
        """

        if not kpath or not knext_name:
            return None
        me_dir = kpath.rsplit('/', 2)[0]
        info_path = pjoin(me_dir, 'Events', knext_name, 'info.json')
        try:
            with open(info_path) as handle:
                info = json.load(handle)
        except (OSError, ValueError) as error:
            logger.debug('no native systematics in %s: %s', info_path, error)
            return None
        summary = info.get('systematics')
        if not summary:
            return None
        for warning in summary.get('warnings') or []:
            logger.debug('systematics: %s', warning)
        return summary

    @staticmethod
    def _as_percent(value, nominal, negative=False):
        """A cross-section difference as a signed percentage of the nominal.

        Same shape as the strings the base parser pulls out of
        parton_systematics.log ("+29.6", "-21.5"): a magnitude with an explicit
        sign, three significant digits.
        """

        if value is None or nominal in (None, 0):
            return ''
        if isinstance(value, float) and math.isnan(value):
            return ''
        return '%s%.3g' % ('-' if negative else '+',
                           abs(value) / abs(nominal) * 100)

    def __init__(self, me_dir, options, run_name, lhe_path):
        self._mg7_run_name = run_name
        self._mg7_lhe_path = os.path.abspath(lhe_path)
        self._mg7_banner_cache = {}
        # Seed the full madevent default option set (run_mode, nb_core,
        # cluster_*, tool paths, ...) before layering the caller's options on
        # top: CommonRunCmd only fills these defaults when it receives an empty
        # options dict, and the tool drivers read e.g. options['run_mode'].
        merged = dict(self.options_configuration)
        merged.update(self.options_madgraph)
        merged.update(self.options_madevent)
        merged.update(options or {})
        # Make the run directory look enough like a madevent run for the
        # results-db scan (and the tools' event-file lookup) to be happy.
        self._prepare_run_dir()
        super(MG7RunCmd, self).__init__(me_dir, merged, force_run=True)

    # ------------------------------------------------------------------
    # directory / run-state adaptation
    # ------------------------------------------------------------------
    def _prepare_run_dir(self):
        """Provide the file names the madevent tools look for: a gzipped
        ``unweighted_events.lhe.gz`` next to the mg7 event file.

        The mg7 event file may already be gzipped (``events.lhe.gz``, the
        default) or plain (``events.lhe``). Compressing an already-gzipped
        source would produce a doubly-gzipped file: every consumer would
        decompress it once, get gzip bytes instead of LHE text, and read zero
        events. So copy when the source is already compressed and only gzip a
        plain source."""
        run_dir = os.path.dirname(self._mg7_lhe_path)
        gz = pjoin(run_dir, 'unweighted_events.lhe.gz')
        if not os.path.exists(gz) and os.path.exists(self._mg7_lhe_path):
            if self._mg7_lhe_path.endswith('.gz'):
                shutil.copyfile(self._mg7_lhe_path, gz)
            else:
                with open(self._mg7_lhe_path, 'rb') as fin, \
                        gzip.open(gz, 'wb') as fout:
                    shutil.copyfileobj(fin, fout)

    def load_results_db(self):
        """Fresh results database without recreating old runs from banners: the
        mg7 LHE banner lacks the <MGGenerationInfo> block that madevent's
        recreate() expects. The run we post-process is registered explicitly in
        :meth:`set_run_name`."""
        model = self.find_model_name()
        self.results = gen_crossxhtml.AllResults(model, self.process,
                                                 self.me_dir, recreateold=False)
        self.last_mode = ''
        return self.results

    def _event_file_for_run(self, name):
        """The LHE of run ``name``; falls back to the original mg7 LHE. Lets a
        run created downstream (e.g. ``<run>_decayed_1`` from MadSpin) provide
        its own events/banner so the shower runs on the decayed events."""
        run_dir = pjoin(self.me_dir, 'Events', name)
        for fname in ('unweighted_events.lhe.gz', 'unweighted_events.lhe',
                      'events.lhe.gz', 'events.lhe'):
            path = pjoin(run_dir, fname)
            if os.path.exists(path):
                return path
        return self._mg7_lhe_path

    def _banner_for_run(self, name):
        """Banner (slha + MG7RunCard + init + proc card) of run ``name``,
        parsed once per run."""
        if name not in self._mg7_banner_cache:
            lhe = lhe_parser.EventFile(self._event_file_for_run(name))
            self._mg7_banner_cache[name] = banner_mod.Banner(lhe.banner)
        return self._mg7_banner_cache[name]

    def set_run_name(self, name, tag=None, level='parton', reload_card=False,
                     allow_new_tag=True):
        """mg7 flavour of :meth:`MadEventCmd.set_run_name`.

        Loads the run card from the TOML card and takes the banner from the
        run's own LHE, then registers the run with the results database so the
        standard tool drivers (which store their output there) work unchanged.
        """
        self.run_name = name
        self.run_card = banner_mod.RunCardMG7(
            pjoin(self.me_dir, 'Cards', 'run_card.toml'))
        self.run_tag = tag or 'tag_1'
        self.run_card['run_tag'] = self.run_tag

        if name not in self.results:
            self.results.add_run(name, self.run_card)
        else:
            self.results.def_current(name, self.run_tag)

        self.banner = self._banner_for_run(name)
        return None

    def configure_directory(self, html_opening=True):
        """Lightweight replacement: the mg7 output has no Fortran ``Source`` to
        compile and no ``run_card.dat``. Set only what the tools consult."""
        if self.options.get('heptools_install_dir'):
            libdir = os.path.abspath(
                pjoin(self.options['heptools_install_dir'], 'lib'))
            for var in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH'):
                os.environ[var] = libdir + ':' + os.environ.get(var, '')
        self.make_opts_var = {}
        if not hasattr(self, 'random'):
            try:
                self.random = int(self.run_card['iseed']) or 1234
            except Exception:
                self.random = 1234
        return

    def do_treatcards(self, line, mode=None, opt=None):
        """No Fortran include files to generate for the mg7 output."""
        return

    def do_set(self, line, log=True):
        """Intercept the compiler options: the base handler patches
        ``Source/make_opts`` to switch Fortran/C++ compilers, which does not
        exist in the mg7 (C++/madspace) output. Everything else is delegated."""
        args = self.split_arg(line)
        if args and args[0] in ('fortran_compiler', 'cpp_compiler',
                                'f2py_compiler'):
            self.options[args[0]] = None if args[1] == 'None' else args[1]
            return
        return super(MG7RunCmd, self).do_set(line, log=log)
