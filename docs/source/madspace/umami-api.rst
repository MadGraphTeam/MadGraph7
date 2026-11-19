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

The real-emission extensions in ABI 1.4 preserve the numeric keys of the
MadGraph7 feature branch's ABI 1.3. Discover inputs and outputs through the
capability functions.

* ``UMAMI_META_COUPLING_NAMES`` returns a library-owned comma-separated string.
  ``UMAMI_META_SQUARED_ORDER_POWERS`` fills a caller-allocated integer array
  with one column per name. For ``n`` local orders, entry ``i + n*j`` is the
  power of coupling ``j`` in order ``i``.
* ``UMAMI_OUT_SQUARED_ORDERS`` contains every local component, including signed
  interference terms. Entry ``event + stride*i`` holds order ``i``. The
  ``UMAMI_META_SQUARED_ORDER_SELECTED`` Boolean mask gives the components whose
  sum forms the process's ``UMAMI_OUT_MATRIX_ELEMENT``. The NLO caller applies
  its own order selection to the full block.
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
Requests for auxiliary diagram, colour or sampled-helicity decisions with those
reweighted components return ``UMAMI_ERROR_UNSUPPORTED_OUTPUT``. CUDA and HIP
support a single squared order. The production bridge requests unit components,
restores the caller's physical strong and electroweak powers for each component,
and preserves the native order projection. Born correlations and virtual
corrections continue through the native NLO providers.

.. doxygenfile:: umami.h
