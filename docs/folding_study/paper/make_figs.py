"""Illustrative figures for the folding note: the cubic map, the images for n=2, convergence."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333',
                     'xtick.color': '#555555', 'ytick.color': '#555555', 'axes.spines.top': False,
                     'axes.spines.right': False})
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#999999'
phi = lambda t: t * t * (3 - 2 * t)
dphi = lambda t: 6 * t * (1 - t)

# Figure 1: the map and the images for n = 2
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4))
t = np.linspace(0, 1, 400)
ax[0].plot(t, t, color=GREY, lw=1, ls='--', label='lattice: $s=t$')
ax[0].plot(t, phi(t), color=ORANGE, lw=2, label='cubic: $s=t^2(3-2t)$')
ax[0].plot(t, dphi(t), color=AQUA, lw=1.5, label="weight $s'(t)=6t(1-t)$")
ax[0].set_xlabel('$t$ (flat coordinate)'); ax[0].set_ylim(0, 1.6); ax[0].legend(fontsize=6.5, frameon=False, loc='upper left')
ax[0].set_title('(a) the map', fontsize=9)
U = 0.3
for k, (name, f, w, col) in enumerate([('lattice', lambda t: t, lambda t: np.ones_like(t), BLUE),
                                        ('cubic', phi, dphi, ORANGE)]):
    a = ax[1 + k]
    tk = (U + np.arange(2)) / 2
    sk = f(tk); wk = w(tk) / 2
    a.bar(sk, wk, width=0.025, color=col)
    for x, y, tt in zip(sk, wk, tk):
        a.annotate('$t=%.2f$\n$s=%.3f$\n$w=%.3f$' % (tt, x, y), (x, y), textcoords='offset points', xytext=(0, 4),
                   ha='center', fontsize=6.5, color='#333333')
    a.set_xlim(0, 1); a.set_ylim(0, 1.0)
    a.set_xlabel('$s$ (vegas-flat coordinate)'); a.set_ylabel('weight of the image')
    a.set_title('(%s) %s, $n=2$, $U=%.1f$' % ('bc'[k], name, U), fontsize=9)
fig.tight_layout()
fig.savefig('fig_cubic_map.pdf')

# Figure 2: convergence of the randomly shifted rule
rng = np.random.default_rng(1)
Us = rng.random(20000)
maps = {'lattice': (lambda t: t, lambda t: np.ones_like(t), BLUE, 'o'),
        'cubic': (phi, dphi, ORANGE, 's'),
        'quintic (not used)': (lambda t: t**3 * (10 - 15 * t + 6 * t * t), lambda t: 30 * t * t * (1 - t)**2, GREY, '^')}
fs = [('smooth: $e^x$', lambda x: np.exp(x), np.e - 1),
      ('endpoint: $1/(2\\sqrt{x})$', lambda x: 0.5 / np.sqrt(x), 1.0),
      ('step: $\\theta(x-0.3)$', lambda x: (x > 0.3) * 1.0, 0.7)]
ns = np.array([1, 2, 4, 8, 16, 32, 64])
fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
for a, (fname, f, I) in zip(ax, fs):
    for mname, (p, dp, col, mk) in maps.items():
        err = []
        for n in ns:
            tt = (Us[:, None] + np.arange(n)[None, :]) / n
            est = (f(np.clip(p(tt), 1e-300, 1)) * dp(tt)).mean(axis=1)
            err.append(np.sqrt(np.mean((est - I)**2)) / abs(I))
        a.loglog(ns, err, marker=mk, color=col, lw=1.2, ms=3.5, label=mname)
    a.set_title(fname, fontsize=9); a.set_xlabel('images $n$')
    a.set_xticks(ns); a.set_xticklabels([str(n) for n in ns])
ax[0].set_ylabel('RMS relative error'); ax[0].legend(fontsize=6.5, frameon=False)
fig.tight_layout()
fig.savefig('fig_convergence.pdf')
