complex_mass = False  # Tag for activating the complex mass scheme
t_channel_width = False  # Whether to keep the width i*M*Gamma in the propagator
                         # denominator for spacelike (t-channel, P^2<0) momenta.
                         # False (default): drop it there -- the correct tree-level
                         # treatment outside the complex-mass scheme (a t-channel
                         # propagator has no pole to regulate, and the spurious
                         # width breaks gauge cancellations). True: keep the width
                         # in every propagator (legacy behaviour). Ignored when
                         # complex_mass is True (the width lives in the mass then).
unitary_gauge = True  # Tag choosing between Feynman Gauge or unitary gauge
                      # 0/False: Feynman
                      # 1/True: unitary
                      # 2: axial
                      # 3: Feynman Diagram gauge (5D aloha)
loop_mode = False     # Tag for encoding momenta with complex number.
mp_precision = False  # Tag for passing parameter in quadruple precision
aloha_prefix = 'mdl_'


class ALOHAERROR(Exception): pass
