# Closed run5 returned-coefficient discrepancies

Pure arithmetic over the pinned, closed analysis; no native execution. All 16 workers remained status 120, accuracy [-1,-1]. These are discrepancies of returned REAL*8 values, with no independent oracle, true-error estimate, or scientific/performance acceptance.

The common-input gate tests consistency of physical/model/Born/repair/QP numerator inputs. It does not test numerical agreement. Native bases and precision paths differ: CutTools retains DP denominators/native returns with genuine QP numerator callbacks; Ninja uses binary128 transport; NinjaRS and TIRust recover 113-bit decimal inputs and use 256-bit engines.

The eight ANS(0:3,0:1) slots are Born, finite, single pole, double pole for the total, then the same four for split order 1. Total and split-order-1 values are identical for all 24 observed arm/point rows in this dataset.

Symmetric discrepancy = abs(a-b)/abs(a+b); relative-to-left = abs(a-b)/abs(a). Undefined zero/opposite cases remain explicit in JSON. Counts use strict >. All 384 frozen supplied scalar deltas were independently reproduced.

| Total coefficient | Max symmetric | Max relative-to-left | Max absolute | >1e-10 /48 | >1e-8 /48 | >3e-6 /48 |
|---|---:|---:|---:|---:|---:|---:|
| Born | 0 | 0 | 0 | 0 | 0 | 0 |
| finite | 4.94340275e-11 | 9.8868055e-11 | 1.62382315e-15 | 0 | 0 | 0 |
| single_pole | 6.84519599e-14 | 1.3690392e-13 | 4.60785923e-18 | 0 | 0 | 0 |
| double_pole | 1.24827535e-15 | 2.49655071e-15 | 2.54109884e-20 | 0 | 0 | 0 |

## Mode 4 versus same-mode CutTools

CutTools is the comparison reference, not a truth oracle. Values below are total finite/single/double; split order 1 duplicates them.

| Point | Backend | Finite | Single pole | Double pole | Sym finite | Sym single | Sym double |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | CutTools | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 0 | 0 | 0 |
| 1 | Ninja | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 5.42365691e-16 | 1.55091747e-16 | 0 |
| 1 | NinjaRS | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 8.13548537e-16 | 1.55091747e-16 | 0 |
| 1 | TIRust | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 7.23154255e-16 | 1.55091747e-16 | 0 |
| 93 | CutTools | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 0 | 0 | 0 |
| 93 | Ninja | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.81643876e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 93 | NinjaRS | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.88586271e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 93 | TIRust | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.88586271e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 39 | CutTools | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 0 | 0 | 0 |
| 39 | Ninja | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.89682254e-11 | 6.84519599e-14 | 1.165057e-15 |
| 39 | NinjaRS | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.94339244e-11 | 1.40930506e-14 | 1.24827535e-15 |
| 39 | TIRust | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.94340275e-11 | 1.39923859e-14 | 1.24827535e-15 |

## Mode 5 versus same-mode CutTools

CutTools is the comparison reference, not a truth oracle. Values below are total finite/single/double; split order 1 duplicates them.

| Point | Backend | Finite | Single pole | Double pole | Sym finite | Sym single | Sym double |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | CutTools | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 0 | 0 | 0 |
| 1 | Ninja | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 4.51971409e-16 | 1.55091747e-16 | 0 |
| 1 | NinjaRS | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 8.13548537e-16 | 1.55091747e-16 | 0 |
| 1 | TIRust | -9.3704262e-06 | -2.18459837e-05 | -6.44155015e-06 | 7.23154255e-16 | 1.55091747e-16 | 0 |
| 93 | CutTools | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 0 | 0 | 0 |
| 93 | Ninja | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.81461182e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 93 | NinjaRS | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.88586271e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 93 | TIRust | 2.17328089e-07 | -6.64991327e-06 | -2.3704671e-06 | 4.88586271e-13 | 1.27375037e-16 | 8.93318606e-17 |
| 39 | CutTools | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 0 | 0 | 0 |
| 39 | Ninja | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.89682254e-11 | 6.84519599e-14 | 1.165057e-15 |
| 39 | NinjaRS | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.94339244e-11 | 1.40930506e-14 | 1.24827535e-15 |
| 39 | TIRust | -1.64241437e-05 | -3.36576136e-05 | -1.01784387e-05 | 4.94340275e-11 | 1.39923859e-14 | 1.24827535e-15 |

## All admissible pair maxima

Pairs include the six same-mode backend pairs in each mode and four same-backend mode pairs. Cross-backend, cross-mode pairs are outside the frozen comparison set.

| Left (backend,mode) | Right | Max sym finite | Max sym single | Max sym double |
|---|---|---:|---:|---:|
| [1, 4] | [1, 5] | 0 | 0 | 0 |
| [1, 4] | [6, 4] | 4.89682254e-11 | 6.84519599e-14 | 1.165057e-15 |
| [1, 4] | [8, 4] | 4.94339244e-11 | 1.40930506e-14 | 1.24827535e-15 |
| [1, 4] | [9, 4] | 4.94340275e-11 | 1.39923859e-14 | 1.24827535e-15 |
| [1, 5] | [6, 5] | 4.89682254e-11 | 6.84519599e-14 | 1.165057e-15 |
| [1, 5] | [8, 5] | 4.94339244e-11 | 1.40930506e-14 | 1.24827535e-15 |
| [1, 5] | [9, 5] | 4.94340275e-11 | 1.39923859e-14 | 1.24827535e-15 |
| [6, 4] | [6, 5] | 1.82694605e-16 | 0 | 0 |
| [6, 4] | [8, 4] | 4.65699012e-13 | 5.43589093e-14 | 8.3218357e-17 |
| [6, 4] | [9, 4] | 4.65802157e-13 | 5.44595739e-14 | 8.3218357e-17 |
| [6, 5] | [8, 5] | 4.65699012e-13 | 5.43589093e-14 | 8.3218357e-17 |
| [6, 5] | [9, 5] | 4.65802157e-13 | 5.44595739e-14 | 8.3218357e-17 |
| [8, 4] | [8, 5] | 0 | 0 | 0 |
| [8, 4] | [9, 4] | 1.03144853e-16 | 1.00664647e-16 | 0 |
| [8, 5] | [9, 5] | 1.03144853e-16 | 1.00664647e-16 | 0 |
| [9, 4] | [9, 5] | 0 | 0 | 0 |

## Mode 4 versus 5 by point

| Backend | Point | Sym finite | Sym single | Sym double |
|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 | 0 |
| 1 | 93 | 0 | 0 | 0 |
| 1 | 39 | 0 | 0 | 0 |
| 6 | 1 | 9.03942819e-17 | 0 | 0 |
| 6 | 93 | 1.82694605e-16 | 0 | 0 |
| 6 | 39 | 0 | 0 | 0 |
| 8 | 1 | 0 | 0 | 0 |
| 8 | 93 | 0 | 0 | 0 |
| 8 | 39 | 0 | 0 | 0 |
| 9 | 1 | 0 | 0 | 0 |
| 9 | 93 | 0 | 0 | 0 |
| 9 | 39 | 0 | 0 | 0 |

No coefficient in any admissible pair exceeds 1e-10 symmetric discrepancy. This statement is confined to these three closed points and the stated returned-value diagnostic; it is not a general accuracy claim. Raw native integral comparisons are not used as truth votes.
