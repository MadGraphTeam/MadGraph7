UMAMI API
=========

MadMatrix real-emission libraries
---------------------------------

A generated NLO real library accepts the momenta of one real matrix element and
a library-local physical flavour index. The FKS caller selects the library and
maps its local coupling orders into the global NLO order table. With no helicity
input the library sums helicities. Its components include the physical spin and
colour averages and identical-particle normalization. The caller supplies any
subtraction factors separately.

The real-emission extensions in ABI 1.5 preserve the released ABI 1.3 flavour
selection keys: random flavour and parton densities are inputs 15 and 16, and
selected flavour is output 13. Direct strong/EW couplings are appended as inputs
17 and 18, and local squared orders as output 14. Discover inputs and outputs
through the capability functions.

The draft NLO ABI 1.4 used the same slots for different quantities and is not
compatible with the released flavour interface. Regenerate its process outputs
and libraries. The native bridge requires ABI 1.5 and the updated process
fingerprint; MadSpace rejects providers explicitly reporting the draft ABI 1.4.
Released ABI 1.3 libraries remain supported.

* ``UMAMI_META_COUPLING_NAMES`` returns a library-owned comma-separated string.
  ``UMAMI_META_SQUARED_ORDER_POWERS`` fills a caller-allocated integer array
  with one column per name. For ``n`` local orders, entry ``i + n*j`` is the
  power of coupling ``j`` in order ``i``.
* ``UMAMI_OUT_SQUARED_ORDERS`` contains every local component, including signed
  interference terms. Entry ``event + stride*i`` holds order ``i``. The
  ``UMAMI_OUT_MATRIX_ELEMENT`` of an NLO real library is the sum of that full
  block: its ``UMAMI_META_SQUARED_ORDER_SELECTED`` mask is all true. The NLO
  caller applies its own order selection to the full block. Ordinary LO
  libraries retain their process-selected mask and total. The library
  fingerprint distinguishes the NLO full-block contract from older libraries.
* ``UMAMI_META_CHARGES`` and ``UMAMI_META_COLOR_REPRESENTATIONS`` give the
  physical external-leg charges and signed colour representations in input leg
  order for an ungrouped process. Incoming charges retain their physical signs.
  A flavour-dependent leg table can return ``UMAMI_ERROR_UNSUPPORTED_META``.
  ``UMAMI_META_TAGGED_PHOTONS`` gives the number of tagged external photons.
* ``UMAMI_META_ALPHA_SCHEME`` gives the model's identified EW input scheme
  when available. A generic inverse-alpha card input alone leaves the
  renormalization scheme unspecified. Such a query returns
  ``UMAMI_ERROR_UNSUPPORTED_META``, as does a loop-scheme query on a real tree.

Supply ``UMAMI_IN_G_STRONG`` and ``UMAMI_IN_G_EW`` as per-event doubles set to
one to request QCD- and QED-stripped components. The equivalent unit strong
coupling through ``UMAMI_IN_ALPHA_S`` is ``1 / (4 pi)``. Supply just one of
the two strong-coupling inputs. For an order with powers ``(p, q)``, multiply
the stripped value by ``g_s**p * e**q`` to obtain the physical result. Other
model parameters enter at their card values. The electric-coupling input scales
the explicit QED power at fixed masses, widths and mixing. Omitting it uses
the parameter card's electric coupling. MadMatrix requires a resolved QED order
and a known nonzero card charge for a nonzero QED power.

The scalar and SIMD providers return all generated orders. Multi-order
reweighting with a new electric coupling supports component and total weights.
Multi-amplitude-order component requests retain all helicities, including ones
that cancel in a process-selected total. These requests use the plain amplitude
code and retain it for subsequent total-only requests on the same instance,
since a recycling table may have omitted amplitudes needed by individual orders.
Requests for auxiliary diagram, colour or sampled-helicity decisions with those
reweighted components return ``UMAMI_ERROR_UNSUPPORTED_OUTPUT``. CUDA and HIP
support a single squared order. The production bridge requests unit components,
restores the caller's physical strong and electroweak powers for each component,
and preserves the native order projection. Born correlations and virtual
corrections continue through the native NLO providers.

.. doxygenfile:: umami.h
