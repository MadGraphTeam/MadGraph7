# Fitted surrogate of the virtual for the virt-trick

In aMC@NLO the finite part of the one-loop virtual V is computed only at a
fraction f of the phase-space points (the *virt-trick*). Everywhere else a
cheap surrogate A replaces it, and the difference is reweighted:

    A + Θ(f - u) (V - A) / f        (u uniform in [0,1])

This is exact for any A. A better A gives a smaller variance in fixed order.
In MC@NLO event generation it also gives fewer negative events, because the
difference is carried by separate "virt" events, which are negative whenever
A > V.

Until now A was a grid of V/B in the Born phase space (or a polynomial fit,
`UsePolyVirtual`). This change adds a surrogate fitted per integration channel,
selected by `VirtTrickSurrogate` in `Cards/FKS_params.dat`:

| value | surrogate |
|---|---|
| 0 | the grid (or the polynomial fit), as before |
| 1 | colour-dipole surrogate |
| 2 | colour dipole + kinematic logarithms of all Born pairs (**default**) |

## The surrogate

With B the Born, B_kl the colour-linked Borns of the I_reg terms (already
computed in `bornsoftvirtual`), and L_kl = log(|2 p_k·p_l| / Q_ES²):

    A_1 = t1 B + t2 Σ_{k<l} B_kl L_kl + t3 Σ_{k<l} B_kl L_kl² + t4 Σ_k B_kk
    A_2 = A_1 + B Σ_{k<l} (a_kl L_kl + b_kl L_kl²)     (all Born pairs, coloured or not)

The kinematic logarithms of mode 2 matter when the Born has few colour dipoles.
For p p > W+ Z, for example, the dipole terms are functions of s-hat only.

The coefficients are fitted per channel during the grid setup by weighted least
squares. The weight is the squared integrand weight of the virtual. They are
refitted after every iteration, frozen within it, and written to `mint_grids`
for the later steps. The fit costs no extra matrix elements: it uses the points
where V is computed anyway, and the colour-linked Borns of the I_reg terms.

## Fallbacks: the grid and the polynomial fit

The grid and the polynomial fit (`UsePolyVirtual`) are both kept, next to the
fit. Each channel uses the best of the three on out-of-sample points. The grid
and the polynomial fit are frozen within an iteration, so they are compared from
the first point. The fit enters the comparison from the iteration after its
first fit.

- **Event generation:** the smallest score
  Σ |c| max(0, A − V) + Σ |c| |A − V|.
  The first term is the weight of the negative residual, which produces the
  negative virt events. The second is the total residual, which sets the
  number of virt events and hence of loop calls. A negative residual thus
  counts twice as much as a positive one.
- **Fixed order:** the smallest residual variance.

A comparison of variances does not see the sign of the residual. In event
generation it was seen to keep the grid where the grid gave more negative
events.

## Effect

The table gives the negative events that the trick itself adds,
E = r − r_SH = V_ev (r_V − r_SH), as a percentage of all events, at folding
1,1,1. Here r_SH is the negative fraction of the S and H events, and V_ev and
r_V are the share and the negative fraction of the separate virt events. The
cost column gives the MadLoop calls per unweighted event in the
event-generation step. The fixed costs of the grid setup and the envelope
are excluded, since they do not grow with the number of events. The table
compares the previous default (grid) with the new one.

| process | E grid → new [%] | MadLoop calls per event (÷ grid) |
|---|---|---|
| p p > t t~ | 0.37 → 0.12 | 0.198 → 0.0665 (÷3.0) |
| p p > e+ e- j | 0.28 → 0.20 | 0.276 → 0.198 (÷1.4) |
| p p > t t~ j | 0.35 → 0.06 | 2.12 → 0.33 (÷6.4) |
| e+ e- > b b~ b b~ | 0.88 → 0.31 | 1.43 → 0.375 (÷3.8) |
| p p > w+ z (4FS) | 1.87 → 0.58 | 1.66 → 0.305 (÷5.4) |
| p p > z j j (4FS) | 0.24 → 0.22 | 0.906 → 0.532 (÷1.7) |
| p p > b b~ (4FS) | 0.30 → 0.14 | 0.675 → 0.366 (÷1.8) |
| p p > e+ ve (4FS) | 0.93 → 0.00 | 0.432 → 0.0004 (÷1200) |
| p p > e+ ve j (4FS) | 0.24 → 0.18 | 0.318 → 0.242 (÷1.3) |
| p p > e+ ve j j (4FS) | 0.00 → 0.07 | 1.25 → 0.564 (÷2.2) |

- **p p > e+ ve:** the fit is exact. The finite virtual is B times a quadratic
  polynomial in log(ŝ/Q_ES²), so the virt events disappear.
- **p p > w+ z:** the trick's negatives dominate. The total negative fraction
  drops from 0.053 to about 0.041, and N_eff gains ×1.06 at equal N.
- **Other processes:** the trick's negatives are a small part of the total.
  The total negative fraction and N_eff change by at most a few percent.
- **Fixed order:** the figure of merit does not change within errors.
- **Training cost:** no measurable overhead in the grid setup.
- **Cross sections:** they agree in every case, since the method is exact.
- **E errors:** E varies from seed to seed more than its counting error;
  differences of a few 0.01% are not significant.

Samples: 3 × 50k events (t t~, W+Z, b b~, e+ ve, e+ ve j), 2 × 10k (e+ e- j,
t t~ j, b b~ b b~), 2 × 20k (e+ ve j j), and 1 × 2k for Z j j. Jets: anti-kT
R = 0.4, pT > 30 GeV for Z j j and the e+ ve processes.
