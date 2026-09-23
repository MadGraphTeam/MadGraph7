"""Validation figure: beamstrahlung sampling and its parton-shower interface.

Columns of the data files (one line per showered event):
  xbs1 xbs2 x1_pythia x2_pythia eCM sum(E_final) pT(mumu) sum(E_forward_gamma)
"""
import math
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

S = os.path.dirname(os.path.abspath(__file__)) + '/'

# CIRCE parameters of fcce240ll, as shipped in its bstrahl.f
P1, Q1 = -901.2, -19.30
P3, Q3 = -899.2, -18.03


def circe(omy, p, q):
    """Single-beam beamstrahlung density in 1-y, up to normalisation.

    Obtained by undoing the (1-x)^k reduction of eepdf_tilde_power, which
    turns the Iapq integrand into a plain Mellin convolution D_BS x phi_ISR.
    """
    return np.exp(p * omy) * np.exp(q * np.sqrt(omy)) * omy ** (-2.0 / 3.0)


def load(name):
    d = np.loadtxt(S + name)
    return dict(y1=d[:, 0], y2=d[:, 1], x1=d[:, 2], x2=d[:, 3], ecm=d[:, 4],
                esum=d[:, 5], pt=d[:, 6], egam=d[:, 7])


fcc_p, fcc_d = load('fcc_plain.dat'), load('fcc_degr.dat')
ilc_p, ilc_d = load('ilc_plain.dat'), load('ilc_degr.dat')
EF, EI = 120.0, 250.0

fig, ax = plt.subplots(2, 3, figsize=(16.6, 8.8))
fig.suptitle('Beamstrahlung + parton-shower validation   '
             r'$e^-e^+\to\mu^-\mu^+$   '
             '| FCC-ee 240 GeV (fcce240ll), 20k events   '
             '| ILC 500 GeV (ilc500ll), 15k events', fontsize=12.5)

# ---- A: does the sampler reproduce the density it was built from? -------
a = ax[0][0]
y = np.concatenate([fcc_d['y1'], fcc_d['y2']])
omy = 1.0 - y[y < 1 - 1e-9]
bins = np.logspace(math.log10(np.percentile(omy, 0.3)),
                   math.log10(omy.max()), 38)
a.hist(omy, bins=bins, histtype='step', lw=2.0, color='#1f77b4',
       density=True, label='sampled by MadGraph')
c = np.sqrt(bins[1:] * bins[:-1])
ref = 0.5 * (circe(c, P1, Q1) + circe(c, P3, Q3))
a.plot(c, ref / np.sum(ref * np.diff(bins)), 'r--', lw=1.7,
       label='CIRCE spectrum of the grid')
a.set(xscale='log', yscale='log',
      xlabel=r'$1-x_{\rm BS}$  (fraction of beam energy radiated)',
      ylabel='normalised density')
a.set_title('A. generator: sampled split vs the shipped density', fontsize=10)
a.legend(fontsize=8, loc='lower left')
a.grid(alpha=.25)
a.text(.96, .94, '%.1f%% of beams radiate;\nthe rest are the $\\delta$ peak'
       % (100.0 * len(omy) / len(y)), transform=a.transAxes, ha='right',
       va='top', fontsize=8, bbox=dict(fc='w', ec='.7', alpha=.9))

# ---- B: does Pythia backward-evolve only the ISR part? ------------------
a = ax[0][1]
res = np.concatenate([fcc_d['x1'] - fcc_p['x1'] / fcc_d['y1'],
                      fcc_d['x2'] - fcc_p['x2'] / fcc_d['y2'],
                      ilc_d['x1'] - ilc_p['x1'] / ilc_d['y1'],
                      ilc_d['x2'] - ilc_p['x2'] / ilc_d['y2']])
a.hist(res, bins=70, histtype='stepfilled', color='#2ca02c', alpha=.8)
a.set(xlabel=r'$x^{\rm Pythia} - x^{\rm LHE}/x_{\rm BS}$', ylabel='beams',
      yscale='log')
a.set_title('B. shower: Pythia sees $x_{\\rm ISR}$, not $x$', fontsize=10)
a.grid(alpha=.25)
a.text(.5, .94, 'max |residual| = %.1e\nset by the 11 digits of the LHE momenta'
       % np.abs(res).max(), transform=a.transAxes, ha='center', va='top',
       fontsize=8, bbox=dict(fc='w', ec='.7', alpha=.9))

# ---- C: beam energy and energy bookkeeping ------------------------------
a = ax[0][2]
sn_f, sn_i = 4 * EF ** 2, 4 * EI ** 2
series = [
    (np.abs(fcc_d['ecm'] / np.sqrt(fcc_d['y1'] * fcc_d['y2'] * sn_f) - 1),
     '#d62728', r'$E_{\rm CM}$ vs $\sqrt{x_{\rm BS1}x_{\rm BS2}s}$'),
    (np.abs(fcc_d['esum'] / ((fcc_d['y1'] + fcc_d['y2']) * EF) - 1),
     '#9467bd', r'$\Sigma E_{\rm final}$ vs $(x_{\rm BS1}+x_{\rm BS2})E$'),
    (np.abs(fcc_p['esum'] / (2 * EF) - 1),
     '#7f7f7f', r'$\Sigma E_{\rm final}$ vs $2E$  (no degradation)')]
for v, col, lab in series:
    a.hist(np.maximum(v, 1e-16), bins=np.logspace(-16, -6, 55),
           histtype='step', lw=1.8, color=col, label=lab)
a.set(xscale='log', xlabel='relative deviation', ylabel='events')
a.set_title('C. beam energy and energy balance (FCC-ee)', fontsize=10)
a.legend(fontsize=7.5, loc='upper left')
a.grid(alpha=.25)


# ---- D, E: the ISR energy the shower no longer invents ------------------
def isr_panel(a, pl, dg, ebeam, title, note):
    sig = dg['y1'] * dg['y2'] < 1 - 1e-9
    # cumulative fraction: the distribution spans zero to the beam energy,
    # so an ECDF shows the shift without binning artefacts, and the events
    # with no forward photon at all sit naturally at the left edge
    for v, col, lab in ((pl['egam'][sig], '#d62728', 'beams NOT degraded'),
                        (dg['egam'][sig], '#1f77b4', 'beams degraded')):
        x = np.sort(np.maximum(v, 1e-3))
        a.step(x, np.arange(1, len(x) + 1) / len(x), where='post', lw=2.0,
               color=col, label=lab)
        a.axvline(np.median(v), color=col, ls=':', lw=1.3)
    a.set(xscale='log', xlim=(1e-2, 2.5 * ebeam), ylim=(0, 1.02),
          xlabel=r'$\Sigma E$ of forward photons [GeV]',
          ylabel='fraction of events below')
    a.set_title(title, fontsize=10)
    a.legend(fontsize=7.5, loc='upper left')
    a.grid(alpha=.25)
    dmean = pl['egam'][sig].mean() - dg['egam'][sig].mean()
    removed = ((2 - dg['y1'][sig] - dg['y2'][sig]) * ebeam).mean()
    a.text(.97, .66,
           '            mean  median\n  not degr. %6.2f %6.2f\n'
           '  degraded  %6.2f %6.2f\n  drop      %6.2f\n'
           'beamstrahlung removed %6.2f'
           % (pl['egam'][sig].mean(), np.median(pl['egam'][sig]),
              dg['egam'][sig].mean(), np.median(dg['egam'][sig]),
              dmean, removed),
           transform=a.transAxes, ha='right', va='top', fontsize=7.5,
           family='monospace', bbox=dict(fc='w', ec='.7', alpha=.92))
    a.text(.97, .22, note, transform=a.transAxes, ha='right', va='top',
           fontsize=7.5, style='italic', color='#333',
           bbox=dict(fc='#f2f2f2', ec='.8', alpha=.92))
    return dmean, removed


di = isr_panel(ax[1][0], ilc_p, ilc_d, EI,
               'D. ILC 500: the shower stops inventing that ISR',
               'the drop tracks the energy\nthat left with the beam')
df = isr_panel(ax[1][1], fcc_p, fcc_d, EF,
               'E. FCC-ee 240: nothing to remove',
               '0.03 GeV is far below the\nevent-to-event spread')

# ---- F: why D and E differ ---------------------------------------------
a = ax[1][2]
for d, lab, col, e in ((fcc_d, 'FCC-ee 240 (fcce240ll)', '#1f77b4', EF),
                       (ilc_d, 'ILC 500 (ilc500ll)', '#d62728', EI)):
    yy = np.concatenate([d['y1'], d['y2']])
    r = 1.0 - yy[yy < 1 - 1e-9]
    a.hist(r, bins=np.logspace(-7, 0, 46), histtype='step', lw=2.0,
           color=col, density=True,
           label='%s\n   %.1f%% of beams radiate, $\\langle\\delta E\\rangle$ = %.4f%%'
                 % (lab, 100.0 * len(r) / len(yy), 100.0 * np.mean(1 - yy)))
a.set(xscale='log', yscale='log', xlabel=r'$1-x_{\rm BS}$',
      ylabel='normalised density')
a.set_title('F. why D shows an effect and E does not', fontsize=10)
a.legend(fontsize=7, loc='lower left')
a.grid(alpha=.25)

fig.tight_layout(rect=(0, 0, 1, .955))
out = S + 'beamstrahlung_validation.png'
fig.savefig(out, dpi=140)
print('wrote', out)
print('ILC  : ISR drop %.2f GeV vs %.2f GeV of beamstrahlung removed' % di)
print('FCC  : ISR drop %.2f GeV vs %.2f GeV of beamstrahlung removed' % df)
print('max x residual %.2e' % np.abs(res).max())
