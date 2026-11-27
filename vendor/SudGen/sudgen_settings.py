#! /usr/bin/env python3
"""Pythia8 settings for the MC@NLO-Delta no-emission tables.

The shower settings are not repeated here: they are taken from the MC@NLO
shower interface itself (madgraph/various/mcatnlo_shower.pythia8_settings),
for a default shower card and a proton-proton run with the internal PDF.
Only what the table definition requires is added on top:

  * the merging machinery that gives access to the trial showers
    (Merging::generateSingleSudakov) and a process to initialise Pythia;
  * SpaceShower:pdfMode = 1: PDF-free initial-state Sudakovs (momentum sum
    rule). The Delta code multiplies the PDF ratio of each incoming leg
    itself (Fk in delta_leg_probability), and DeltaBackwardISR requires it;
  * no QED radiation, MPI or hadronisation: the tables are QCD no-emission
    probabilities of single dipoles;
  * no matrix-element corrections: the shower applies them only in
    resonance decays, never to the production dipoles of the tables;
  * no shower uncertainty or merging-scale variations, which would defer
    the trial-shower accept/reject steps (see README_grid_clust).

Usage: sudgen_settings.py <pythia8 path> <output .cmd> [<output .json>]
"""
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, ROOT)

# Top-quark mass of the MG5_aMC standard-model parameter cards, used for the
# top-quark tables (the shower interface takes it from the param_card).
MT_DEFAULT = 173.0

# Sections of the shower settings that describe the shower itself. Run
# control, beams and event input, FxFx merging, decays and random numbers
# belong to an event run, not to a dipole Sudakov.
SHOWER_SECTIONS = ('matching', 'mec', 'couplings', 'pdf', 'masses')

TABLE_SETTINGS = [
    ('Print:quiet', 'on'),
    ('Beams:eCM', 13000.),
    ('HardQCD:all', 'on'),
    ('PhaseSpace:pTHatMin', 20.),
    # Merging machinery: gives Pythia a Merging object with trial showers.
    ('Merging:doPTLundMerging', 'on'),
    ('Merging:Process', 'pp>jj'),
    ('Merging:TMS', -1.0),
    ('Merging:includeWeightInXSection', 'off'),
    ('Merging:nJetMax', 1000),
    ('Merging:applyVeto', 'off'),
    ('Merging:nRequested', 0),
    ('Check:abortIfVeto', 'on'),
    # No uncertainty or merging variations in the trial showers.
    ('UncertaintyBands:doVariations', 'off'),
    ('Merging:muRfactors', '{}'),
    ('Merging:doSchemeVariation', 'off'),
    # QCD no-emission probabilities of single dipoles.
    ('PartonLevel:MPI', 'off'),
    ('HadronLevel:all', 'off'),
    ('TimeShower:QEDshowerByQ', 'off'),
    ('TimeShower:QEDshowerByL', 'off'),
    ('TimeShower:QEDshowerByOther', 'off'),
    ('TimeShower:QEDshowerByGamma', 'off'),
    ('SpaceShower:QEDshowerByQ', 'off'),
    ('SpaceShower:QEDshowerByL', 'off'),
    # PDF-free initial-state Sudakovs: the Delta code applies the PDF ratio.
    ('SpaceShower:pdfMode', 1),
    # The shower applies matrix-element corrections only in resonance decays
    # (TimeShower:MEextended = off), never to the production dipoles that the
    # tables describe, and the PYTHIA8 MC counterterms use the plain kernels.
    # The dummy q qbar pair of noEmissionProbability hangs off a colour
    # singlet, so with MEcorrections = on it would get the singlet-decay
    # correction (up to 0.1 in the quark FF tables).
    ('TimeShower:MEcorrections', 'off'),
]


def read_mc_masses(path):
    masses = {}
    for line in open(path):
        line = line.strip()
        if line.startswith('mcmass('):
            pdg = int(line[len('mcmass('):line.index(')')])
            masses[pdg] = float(line.split('=')[1].replace('d', 'e').replace('D', 'e'))
    return masses


def table_settings(pythia8_path, mc_masses_path=None):
    """Return (list of 'Key = value' lines, metadata dict)."""
    from madgraph.various import mcatnlo_shower, shower_card
    mc_masses_path = mc_masses_path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                    'MCmasses_PYTHIA8.inc')
    installation = mcatnlo_shower.Pythia8Installation(pythia8_path)
    # The defaults of a newly exported shower_card.dat.
    card = shower_card.ShowerCard()
    contract = mcatnlo_shower.MatchingContract(parton_shower='PYTHIA8', lpp1=1, lpp2=1)
    mc_masses = read_mc_masses(mc_masses_path)
    run = {'nevents': 1, 'nevents_total': 1, 'event_norm': 'average',
           'mc_masses': mc_masses, 'masses': {6: MT_DEFAULT}, 'widths': {},
           'pdf': ('internal', '', False), 'pdf_fallback': True, 'seed': 1}
    settings = mcatnlo_shower.pythia8_settings(contract, card, run, installation)
    lines, used = [], {}
    overridden = set(key.lower() for key, _ in TABLE_SETTINGS)
    for section in SHOWER_SECTIONS:
        for setting in settings.entries[section]:
            if setting.key.lower() in overridden:
                continue
            lines.append('%s = %s' % (setting.key, setting.value))
            used[setting.key] = setting.value
    for key, value in TABLE_SETTINGS:
        if isinstance(value, float):
            value = repr(value)
        lines.append('%s = %s' % (key, value))
        used[key] = value
    metadata = {
        'pythia8_version': installation.version,
        'pythia8_path': installation.path,
        'mc_masses': mc_masses,
        'top_mass': MT_DEFAULT,
        'settings': used,
    }
    return lines, metadata


def main(argv):
    if len(argv) < 3:
        sys.stderr.write(__doc__)
        return 1
    lines, metadata = table_settings(argv[1])
    with open(argv[2], 'w') as out:
        out.write('! Pythia8 settings for the MC@NLO-Delta Sudakov tables\n')
        out.write('! Pythia %s; shower settings from mcatnlo_shower.pythia8_settings\n'
                  % metadata['pythia8_version'])
        out.write('\n'.join(lines) + '\n')
    if len(argv) > 3:
        with open(argv[3], 'w') as out:
            json.dump(metadata, out, indent=1, sort_keys=True)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
