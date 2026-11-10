complex_mass = False  # Tag for activating the complex mass scheme
unitary_gauge = True  # Tag choosing between Feynman Gauge or unitary gauge
                      # 0/False: Feynman
                      # 1/True: unitary
                      # 2: axial
                      # 3: Feynman Diagram gauge (5D aloha)
loop_mode = False     # Tag for encoding momenta with complex number.
dual_mode = 0         # Number of P-wave onia in the current process (0/False
                      # when none): selects the dual-number HELAS routines.
npwave = [0]          # P-wave multiplicities the exporter has to write HELAS
                      # libraries for. 0 is the plain (non-dual) library.
mp_precision = False  # Tag for passing parameter in quadruple precision
aloha_prefix = 'mdl_'


class ALOHAERROR(Exception): pass


# the value of unitary_gauge for each gauge name of 'set gauge'
GAUGE_VALUES = {'unitary': True, 'Feynman': False, 'axial': 2, 'FD': 3}


class gauge_scope(object):
    """`with gauge_scope('FD'):` sets unitary_gauge for the block and restores
    it after, cleaning the ALOHA kernel whenever the value changes: the steps
    of an output that mixes gauges run each matrix element in its own."""

    def __init__(self, gauge):
        self.value = GAUGE_VALUES[gauge] if gauge in GAUGE_VALUES else gauge

    def __enter__(self):
        global unitary_gauge
        self.old = unitary_gauge
        if self.value != self.old:
            from aloha import aloha_lib
            aloha_lib.KERNEL.clean()
            unitary_gauge = self.value
        return self

    def __exit__(self, *args):
        global unitary_gauge
        if unitary_gauge != self.old:
            from aloha import aloha_lib
            aloha_lib.KERNEL.clean()
            unitary_gauge = self.old
        return False
