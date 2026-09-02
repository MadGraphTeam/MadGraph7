# Writing API documentation for `madspace`

The public C++ headers under `madspace/include/madspace/` are the single source of
the API documentation. Doxygen comments there feed two outputs:

- **C++ API pages** — Doxygen XML → Breathe → Sphinx (`docs/`).
- **Python API pages** — `madspace/generate_docstrings.py` turns the same Doxygen XML
  into a `consteval pydoc::doc()` lookup; `src/python/madspace.cpp` attaches those
  strings to the pybind11 classes; Sphinx `autodoc` renders them.

`docs/generate_api_pages.py` writes one page per class automatically (globbed toctree in
`docs/source/madspace/{cpp,python}-api.rst`) — nothing to register by hand.

The worked reference example is `madspace/include/madspace/phasespace/rambo.hpp`
(`FastRamboMapping`).

## Class comment layout (`/** ... */` above the class)

In this order:

1. **One-line brief** — the first sentence (`JAVADOC_AUTOBRIEF` picks it up).
2. **Prose** describing the algorithm in the paper's terms. Refer to constructor
   parameters as `@p name`. Cite with numbered `[n]` markers.
3. **Equations** — `@f[ ... @f]` for display, `@f$ ... @f$` for inline. Only equations
   that appear in the paper or the code; invent nothing. Prefer several single-line
   `@f[ ]` blocks over one multi-line `aligned`/`cases`/`split` environment (the Python
   path re-indents display math into a `.. math::` block and multi-line alignment is
   not exercised).
4. **Index/batch sentence** (if the class uses a repeated index):
   *"Entries marked with an index `i` are repeated for `i = 0 … n - 1`; `batch` is the
   batch dimension."*
5. For a `Mapping` subclass: **`**Inputs**`**, **`**Conditions**`**, **`**Outputs**`**
   bulleted lists (see below), always all three headings — write a single
   `- None.` bullet when a list is empty. For a `FunctionGenerator` subclass:
   **`**Arguments**`** and **`**Returns**`** with the same bullet format.
6. **Weight note**, verbatim, on every `Mapping` subclass:
   *"In addition every mapping returns a `weight` (`float`, shape `(batch,)`), the
   Jacobian of the transformation."*
7. **`**References**`** list — see below.

Constructors get their own `/** */` with one `@param` line per parameter. Every public
method gets at least a `///` brief (undocumented members render as bare stubs, since
`:undoc-members:` is on during the rollout).

**American spelling.** **Short sentences.** Prefer several plain sentences over one long
clause chain joined by semicolons.

Cross-reference other classes with `@ref ClassName` — it renders as a link on the C++
side and degrades to plain label text in the Python docstring.
**Never** use `\copydoc` (copies the entire description or the whole `@param` list) or
`\rst` (the alias only exists in `docs/Doxyfile`, not in the generator's Doxyfile, so it
produces raw text in the Python docstring).

Two more traps:

- **No `@f[ ]` display math inside a bullet list.** `text_of()` flattens list nesting,
  so the `.. math::` block lands back at column 0 and detaches from its bullet. Put the
  cases in prose paragraphs with the equations between them instead.
- **No `--` / `---` in prose.** Doxygen's Markdown turns them into dashes and the
  generator then drops them, so `Breit--Wigner` renders as `BreitWigner`. Write a plain
  hyphen (`Breit-Wigner`) or a literal en-dash `–`.

## Inputs / Conditions / Outputs bullets

Derive the actual names, types and shapes by **reading the constructor in
`madspace/src/phasespace/<x>.cpp`**: the `Mapping(name, input_types, output_types,
condition_types)` base call builds the three `NamedVector<Type>`s, often in
immediately-invoked lambdas with conditional `push_back`s. Note the base-call order is
inputs, **outputs, then conditions**.

Bullet format (en-dash `–`, U+2013, **not** a hyphen):

```
- `name` – `float`, shape `(batch,)` – short description.
```

`Type` constant → shape string:

| constant | bullet type/shape |
|---|---|
| `batch_float` | `` `float`, shape `(batch,)` `` |
| `batch_int`   | `` `int`, shape `(batch,)` `` |
| `batch_four_vec` | `` `float`, shape `(batch, 4)` `` |

Each `if (...) push_back(...)` becomes a trailing *"Present only when @p X is false."*
(or true) on that bullet.

## References

Per-class local numbering, dense from 1. **`[1]` is always the MadSpace paper.** Format:

```
- [1] T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace",
  https://arxiv.org/abs/2602.06895
- [2] Authors, "Title", https://arxiv.org/abs/XXXX
```

Use the bare journal / DOI reference only when there is no arXiv entry.

There is **no shared bibliography** — `\cite` is not configured in either Doxyfile, and a
generated citelist page would not survive into the Python docstrings. Keep the citation
strings identical across classes by copying from the table below.

### Citation strings

| short | reference string |
|---|---|
| MadSpace | T. Heimel, O. Mattelaer, R. Winterhalder, "MadSpace", https://arxiv.org/abs/2602.06895 |
| RAMBO | R. Kleiss, W. J. Stirling, S. D. Ellis, Comput. Phys. Commun. 40 (1986) 359, https://doi.org/10.1016/0010-4655(86)90119-0 |
| RAMBO on diet | S. Plätzer, "RAMBO on diet", https://arxiv.org/abs/1308.2922 |
| HICOM | A. van Hameren, "Adaptive channels for data analysis and importance sampling", https://arxiv.org/abs/hep-ph/0301036 |
| Neural spline flows | C. Durkan et al., "Neural spline flows", https://arxiv.org/abs/1906.04032 |
| Multichannel weights | R. Kleiss, R. Pittau, "Weight optimization in multichannel Monte Carlo", https://arxiv.org/abs/hep-ph/9405257 |
| MadEvent | F. Maltoni, T. Stelzer, "MadEvent: automatic event generation with MadGraph", https://arxiv.org/abs/hep-ph/0208156 |
| Speeding up MG5aMC | O. Mattelaer, K. Ostrolenk, "Speeding up MadGraph5_aMC@NLO", https://arxiv.org/abs/2102.00773 |
| MadNIS | T. Heimel et al., "MadNIS – Neural multi-channel importance sampling", https://arxiv.org/abs/2212.06172 |
| MadNIS Reloaded | T. Heimel et al., "The MadNIS Reloaded", https://arxiv.org/abs/2311.01548 |
| MadNIS-Lite | T. Heimel et al., "Differentiable MadNIS-Lite", https://arxiv.org/abs/2408.01486 |
| MG5aMC | J. Alwall et al., "The automated computation of tree-level and next-to-leading order differential cross sections", https://arxiv.org/abs/1405.0301 |
| PYTHIA 8.2 | T. Sjöstrand et al., "An introduction to PYTHIA 8.2", https://arxiv.org/abs/1410.3012 |
| Sherpa 2.2 | E. Bothmann et al., "Event generation with Sherpa 2.2", https://arxiv.org/abs/1905.09127 |
| Byckling–Kajantie | E. Byckling, K. Kajantie, "Reductions of the phase-space integral in terms of simpler processes", Phys. Rev. 187 (1969) 2008, https://doi.org/10.1103/PhysRev.187.2008 |
| Leading-colour generation | R. Frederix, T. Vitos, "Leading-colour-based unweighted event generation for multi-parton tree-level processes", https://arxiv.org/abs/2409.12128 |
| Knippen thesis | G. Knippen, PhD thesis, University of Freiburg (2019), https://doi.org/10.6094/UNIFR/154629 |
| Chili | E. Bothmann et al., "Efficient phase-space generation for hadron collider event simulation", https://arxiv.org/abs/2302.10449 |
| Pepper | E. Bothmann et al., "A portable parton-level event generator for the high-luminosity LHC", https://arxiv.org/abs/2311.06198 |
| ALPGEN | M. L. Mangano et al., "ALPGEN, a generator for hard multiparton processes in hadronic collisions", https://arxiv.org/abs/hep-ph/0206293 |
| VEGAS enhanced | G. P. Lepage, "Adaptive multidimensional integration: VEGAS enhanced", https://arxiv.org/abs/2009.05112 |
| LHAPDF6 | A. Buckley et al., "LHAPDF6: parton density access in the LHC precision era", https://arxiv.org/abs/1412.7420 |
| cudacpp | S. Hageböck et al., "Data-parallel leading-order event generation in MadGraph5_aMC@NLO", https://arxiv.org/abs/2507.21039 |
| Loop-induced / scales | V. Hirschi, O. Mattelaer, "Automated event generation for loop-induced processes", https://arxiv.org/abs/1507.00020 |
| kT-clustering scale | S. Catani, F. Krauss, R. Kuhn, B. R. Webber, "QCD matrix elements + parton showers", https://arxiv.org/abs/hep-ph/0109231 |

## Canonical constructor-argument wordings

Reuse these verbatim wherever the parameter appears.

- **`bool com`** —
  *"If true the momenta are generated in the center-of-mass frame, otherwise the total
  incoming momentum is taken from the `com_momentum` input."*
- **`bool has_cut`** —
  *"If true, extra conditions carrying the cut boundaries are consumed and the sampled
  invariants are restricted to the physical region; see @ref Cuts."*
- **`invariant_power` / `mass` / `width` triple** —
  *"`@p invariant_power`, `@p mass` and `@p width` select how the propagator invariant is
  sampled: Breit–Wigner around `@p mass` when `@p width` is non-zero, a `1/s^p` power law
  otherwise (flat for `p = 0`). See @ref Invariant."*
- **bare `invariant_power` (default 0.8)** —
  *"Exponent of the `1/s^p` sampling applied to every propagator invariant; see
  @ref Invariant."*
- **`std::vector<std::size_t>` order / permutation list** —
  *"0-based particle indices; the two incoming beams are 0 and 1. See @ref Topology for
  the ordering convention."*
- **cut vectors `pt_min` / `m_inv_min` / `dr_min`** —
  *"Per-particle (`pt_min`) or per-pair (`m_inv_min`, `dr_min`) cut values; an empty
  vector disables the corresponding cut. See @ref Cuts for the layout."*
- **NN block `prefix` / `hidden_dim` / `layers` / `activation`** —
  *"`@p prefix` namespaces the trainable globals; `@p hidden_dim`, `@p layers` and
  `@p activation` size the subnetwork. Call `initialize_globals(context)` once before
  use."*
- **`option_counts` / `dims_with_prior`** —
  *"`@p option_counts[d]` is the number of choices for discrete dimension `d`;
  `@p dims_with_prior` lists the dimensions whose probabilities are conditioned on a
  prior."*
- **`std::vector<std::shared_ptr<Base>>` + `return_*` bool** —
  *"The per-channel sub-mappings; `@p return_*` also emits the per-channel batch sizes."*

## Verification

Fast loop, no rebuild:

```
cd docs && doxygen 2>&1 | grep -i warn
python madspace/generate_docstrings.py --include-dir madspace/include/madspace \
  --out /tmp/ds.hpp --require-doxygen
grep -n '"<Class>' /tmp/ds.hpp          # exact Python-side RST; reveals #2 overload keys
python docs/check_doc_convention.py
```

Once per phase:

```
python madspace/install.py --source --docs -y      # consteval key typo => compile error
PYTHONPATH=madspace/install python -c "import madspace; print(madspace.<Class>.__doc__)"
cd docs && sphinx-build -q -b html source /tmp/x    # expect 0 `warning:` lines
```

(`sphinx-build -n -W` is *not* the gate: nitpicky mode flags ~500 pre-existing
unresolved cross-references from the `doxygenindex` C++ dump.)

`docs/check_doc_convention.py` needs the Doxygen XML and is a local / pre-PR
check, not a CI job (the wheel-test environment has no doxygen). The CI gate is
`madspace/tests/test_doc_convention.py`, which only needs the installed module.

## Status

**Every class under `phasespace/` is documented** — all `Mapping` and
`FunctionGenerator` subclasses and the plain helper structs.
`python docs/check_doc_convention.py` prints
`check_doc_convention: all phasespace/ classes pass`.

Outside `phasespace/`, the `driver/`, `compgraphs/` and top-level headers are
still undocumented.
