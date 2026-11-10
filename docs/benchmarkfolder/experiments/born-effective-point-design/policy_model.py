"""Executable control-flow specification, not MadLoop or numerical code."""
from dataclasses import dataclass, field


@dataclass
class RescueCall:
    supported: bool = True
    loop_induced: bool = False
    ready: bool = False
    failed: bool = False
    nonfinite_born_trigger: bool = False
    policy: str | None = None
    canonical: object = None
    rounded: object = None
    published_born: object = None
    reference: object = None
    interference_view: object = None
    numerator_done: bool = False
    numerator_called: bool = False
    checked: bool = False
    cache_invalidations: int = 0
    prepare_count: int = 0
    born_count: int = 0
    events: list = field(default_factory=list)

    def original_born(self, result, reference, finite=True):
        self.published_born, self.reference = result, reference
        self.events.append('original_dp_born')
        self.nonfinite_born_trigger = not finite

    def prepare(self, canonical, rounded, *, repair_ok, conversion_ok=True):
        if self.loop_induced or not self.supported:
            return 'legacy'
        if self.ready:
            return 'ready'
        self.prepare_count += 1
        if not repair_ok or not conversion_ok:
            self.reject('repair_or_conversion')
            return 'failed'
        self.canonical, self.rounded = canonical, rounded
        self.ready = True
        self.numerator_done = False
        self.events.append('coordinates_ready')
        return 'ready'

    def stage_policy(self, genuine_qp, born, reference, *, born_ok=True):
        if self.loop_induced or not self.supported:
            return 'legacy'
        if self.failed or not self.ready:
            raise ValueError('No valid effective point')
        requested = 'genuine_mp' if genuine_qp else 'promoted_dp'
        if requested == self.policy:
            return 'reuse'
        self.born_count += 1
        if not born_ok:
            self.reject('born_or_conversion')
            return 'failed'
        self.policy = requested
        self.published_born, self.reference = born, reference
        self.interference_view = self.canonical if genuine_qp else self.rounded
        self.numerator_done = False
        self.checked = False
        self.cache_invalidations += 1
        self.events.append(('publish_policy', requested))
        return 'staged'

    def finish_numerator(self):
        if self.failed or not self.ready or self.policy is None:
            raise ValueError('No complete input/Born contract')
        self.numerator_called = self.numerator_done = True
        self.events.append('actual_numerator_return')

    def rotate(self, rotation):
        if not self.ready:
            raise ValueError('No canonical point')
        self.numerator_done = False
        self.cache_invalidations += 1
        self.events.append(('rotate_from_canonical', self.canonical, rotation))

    def reject(self, reason):
        self.failed = True
        self.ready = False
        self.policy = None
        self.checked = False
        self.numerator_done = False
        self.published_born = self.reference = None
        self.events.append(('reject', reason))

    def accept(self, *, independently_checked):
        if self.failed or not self.numerator_done:
            raise ValueError('Incomplete input or numerator work')
        self.checked = independently_checked
        return self.checked
