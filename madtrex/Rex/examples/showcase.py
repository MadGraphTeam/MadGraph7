#!/usr/bin/env python3
"""
Rex / teaRex Python bindings -- a runnable tour of everything that is bound.

This file exists as a living reference for what the `pyrex` package exposes,
organised to mirror Rex.h/teaRex.h section by section. Run it directly
(`python3 showcase.py`) to exercise every call end to end -- it is
self-contained (no external .lhe/SLHA files required) and prints its way
through each section.

SCOPE: the bindings cover a pythonic "core" of Rex/teaRex, not a 1:1 mirror
of the C++ API. Things that are deliberately NOT bound (see the bottom of
this file for the full list): LHEF alias accessors (eg xWgtUP -- same field
as weight, just a historical alternate name), the generic arrN/vecArrN
container machinery (momenta etc. are plain Python lists instead), the
xmlNode/xmlDoc/lheReader/lheWriter parsing/writing plumbing (load_lhef/
write_lhef are the practical entry points), REX::tea::threadPool, and the
rwgt_slha/rwgt_card/rwgt_block SLHA-reweight-card internals (only
ParamRwgt.read_slha_rwgt(slha_file, rwgt_file) is exposed).

Everything above needs only `pyrex` and is otherwise self-contained (no
external .lhe/SLHA files required). The one exception is madtrex_demo() at
the bottom: it exercises madspace.madtrex.TupperWare/WareHouse -- the
MatrixElementApi <-> REX::tea bridge that lives in the separate `madspace`
package (MadGraph7/madspace) -- which is optional here and skips itself with
a message if `madspace` isn't installed or no C++ compiler is available to
build its (also self-contained, compiled on the fly) stub matrix element.
"""

import shutil
import subprocess
import tempfile
import os

import pyrex
import pyrex.tea as tea


def section(title):
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")


# ---------------------------------------------------------------------------
def parton_demo():
    section("pyrex.Parton")

    p = pyrex.Parton()
    p.momentum = [50.0, 0.0, 0.0, 50.0]  # [E, px, py, pz]
    p.mass = 0.0
    p.vtim = 0.0
    p.spin = 1.0
    p.pdg = 21
    p.status = 1
    p.mother = [0, 0]
    p.icol = [501, 0]

    print("momentum:", p.momentum, "pdg:", p.pdg)
    # Read-only computed kinematic observables:
    print(
        "pT={:.3f} eta={:.3f} phi={:.3f} m2={:.3f}".format(p.pT, p.eta, p.phi, p.m2)
    )


# ---------------------------------------------------------------------------
def event_demo():
    section("pyrex.Event")

    ev = pyrex.Event(2)  # pre-sizes all per-particle arrays to 2 entries
    ev.proc_id = 1
    ev.weight = 0.75
    ev.scale = 91.19
    ev.alphaS = 0.118
    ev.alphaEW = 1.0 / 127.9
    ev.helicity = -1  # unset sentinel (per REX convention: -1 == "unset")
    ev.flavor = -1

    # Bulk per-particle fields, all plain Python lists:
    ev.pdg = [21, 21]
    ev.status = [-1, 1]  # LHEF status: -1 incoming, +1 outgoing
    ev.momenta = [[50.0, 0.0, 0.0, 50.0], [50.0, 0.0, 0.0, -50.0]]
    ev.mass = [0.0, 0.0]
    ev.vtim = [0.0, 0.0]
    ev.spin = [1.0, 1.0]
    ev.mother = [[0, 0], [0, 0]]
    ev.icol = [[501, 0], [0, 501]]
    ev.wgts = []  # extra (eg reweighted) weights; empty until a reweightor fills it

    print("size:", ev.size(), "== len():", len(ev))
    ev.validate()  # raises if any field's length disagrees with the others
    print("validate() passed")

    # add_particle() appends a Parton and grows n by one:
    extra = pyrex.Parton()
    extra.momentum = [10.0, 1.0, 1.0, 1.0]
    extra.pdg = 22
    extra.status = 1
    ev.add_particle(extra)
    print("after add_particle: size =", ev.size(), "pdg =", ev.pdg)


# ---------------------------------------------------------------------------
def make_event(pdg, e_val, status=1, helicity=-1, flavor=-1, scale=0.0):
    """Small helper used throughout the rest of this file."""
    ev = pyrex.Event(1)
    ev.pdg = [pdg]
    ev.status = [status]
    ev.weight = 1.0
    ev.scale = scale
    ev.momenta = [[e_val, 0.0, 0.0, 0.0]]
    ev.mass = [0.0]
    ev.vtim = [0.0]
    ev.spin = [0.0]
    ev.mother = [[0, 0]]
    ev.icol = [[0, 0]]
    ev.helicity = helicity
    ev.flavor = flavor
    return ev


# ---------------------------------------------------------------------------
def comparator_demo():
    section("pyrex.EventComparatorConfig + compare_* factories")

    cfg = pyrex.EventComparatorConfig()
    cfg.compare_pdg = True
    cfg.compare_status = True
    cfg.compare_momentum = True
    cfg.momentum_tol = 1e-6
    cfg.status_filter = {1}  # a Python set of statuses to restrict comparison to
    print(
        "custom config: compare_pdg =",
        cfg.compare_pdg,
        "status_filter =",
        cfg.status_filter,
    )

    # Three ready-made configs:
    print("compare_legs_only():", type(pyrex.compare_legs_only()).__name__)
    print("compare_final_state_only():", type(pyrex.compare_final_state_only()).__name__)
    print("compare_physics_fields():", type(pyrex.compare_physics_fields()).__name__)


# ---------------------------------------------------------------------------
def event_belongs_demo():
    section("pyrex.EventBelongs")

    reference = make_event(21, 10.0)
    belongs = pyrex.EventBelongs(reference)

    same_flavor = make_event(21, 20.0)
    other_flavor = make_event(22, 20.0)

    # Default comparator (external_legs_comparator) compares external legs;
    # here we instead install a trivial Python predicate for clarity:
    belongs.set_comparator(lambda a, b: a.pdg[0] == b.pdg[0])
    print("belongs(same_flavor):", belongs.belongs(same_flavor))
    print("belongs(other_flavor):", belongs.belongs(other_flavor))
    print("__call__ form:", belongs(same_flavor))

    # A config-driven comparator instead of a raw callable:
    belongs2 = pyrex.EventBelongs(reference)
    belongs2.set_comparator(pyrex.compare_legs_only())

    # Module-level convenience predicates/factories:
    print("all_events_belong():", type(pyrex.all_events_belong()).__name__)
    print("external_legs_comparator is callable:", callable(pyrex.external_legs_comparator))
    print("always_true is callable:", callable(pyrex.always_true))


# ---------------------------------------------------------------------------
def event_sorter_demo():
    section("pyrex.EventSorter")

    checker_a = lambda e: e.pdg[0] == 21
    checker_b = lambda e: e.pdg[0] == 22
    sorter = pyrex.EventSorter([checker_a, checker_b])
    print("size():", sorter.size())

    ev_a, ev_b, ev_c = make_event(21, 1.0), make_event(22, 2.0), make_event(99, 3.0)
    print("position(class A event):", sorter.position(ev_a))
    print("position(class B event):", sorter.position(ev_b))
    unmatched_pos = sorter.position(ev_c)
    print("position(unmatched event):", unmatched_pos, "== pyrex.npos:", unmatched_pos == pyrex.npos)
    print("sort([a, b, c]):", sorter.sort([ev_a, ev_b, ev_c]))


# ---------------------------------------------------------------------------
def process_demo():
    section("pyrex.Process")

    events = [make_event(21, float(i)) for i in range(3)]
    proc = pyrex.Process(events, filter_partons=False, column_major=False)

    print("size():", proc.size(), "== len():", len(proc))
    print("weight:", list(proc.weight), "pdg:", list(proc.pdg))
    print("momenta (list of [E,px,py,pz]):", proc.momenta)

    proc.to_umami()  # transposes into the column-major UMAMI layout
    print("umami_momenta (flat, column-major):", proc.umami_momenta)
    print("umami_pdg:", proc.umami_pdg)

    proc.from_umami()  # back to row-major LHEF layout
    proc.validate()
    print("validate() passed after round trip")

    proc.add_event(make_event(21, 99.0))
    print("size() after add_event:", proc.size())


# ---------------------------------------------------------------------------
def init_node_demo():
    section("pyrex.InitNode")

    init = pyrex.InitNode()
    init.idBm = [2212, 2212]  # beam PDG ids (proton-proton)
    init.eBm = [6500.0, 6500.0]  # beam energies
    init.pdfG = [0, 0]
    init.pdfS = [-1, -1]
    init.idWgt = 3
    init.nProc = 1
    init.xSec = [1.5]
    init.xSecErr = [0.01]
    init.xMax = [2.0]
    init.lProc = [1]
    print("idBm:", init.idBm, "xSec:", init.xSec)


# ---------------------------------------------------------------------------
def lhe_demo():
    section("pyrex.Lhe + load_lhef/write_lhef")

    doc = pyrex.Lhe()  # Lhe subclasses InitNode -- all InitNode fields apply too
    doc.idBm = [2212, 2212]
    doc.eBm = [6500.0, 6500.0]
    doc.idWgt = 3
    for i in range(5):
        doc.add_event(make_event(21, float(i)))

    doc.set_sorter()  # no-arg: auto-derive classes from the sample (external-legs comparator)
    doc.sort_events()
    doc.events_to_processes()
    print("processes after events_to_processes():", len(doc.processes))
    print("process_group (equivalence-class index per process):", list(doc.process_group))

    # max_batch: cap events per REX::process; oversized classes split into
    # several processes that share the same underlying events, and
    # process_group records which original class each batch belongs to.
    doc.max_batch = 2
    doc.events_to_processes()
    print("with max_batch=2, processes:", len(doc.processes), "sizes:", [p.size() for p in doc.processes])
    print("process_group:", list(doc.process_group))
    doc.max_batch = pyrex.npos  # reset to "unset"

    # Round-trip through an actual LHE file on disk:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "showcase.lhe")
        pyrex.write_lhef(doc, path, include_ids=False)
        reloaded = pyrex.load_lhef(path)
        print("write_lhef/load_lhef round trip: events =", len(reloaded.events))


# ---------------------------------------------------------------------------
def histogram_demo():
    section("pyrex.Histogram")

    # A genuinely unweighted sample (event.weight == 1.0 for every event,
    # via make_event()'s default), sorted into 2 bins by pdg id:
    events = [make_event(21, float(i)) for i in range(3)] + [
        make_event(22, float(i)) for i in range(2)
    ]
    indices = [0, 0, 0, 1, 1]

    hist = pyrex.Histogram(events, indices)
    print("bins():", hist.bins(), "size():", hist.size())
    print("nominal (plain unweighted counts) == counts():", list(hist.nominal), list(hist.counts()))
    # mult[0]: the base sample's own deviation from the unweighted
    # assumption nominal makes -- exactly 1.0 here since weight() is
    # uniform, so heights(0) reduces to nominal itself.
    print("mult[0] (==1.0, sample is genuinely unweighted):", hist.mult[0])
    print("heights(0) == nominal here:", hist.heights(0))

    # Built from an EventSorter instead of explicit indices -- an event the
    # sorter can't place lands in a trailing catch-all bin:
    sorter = pyrex.EventSorter([lambda e: e.pdg[0] == 21, lambda e: e.pdg[0] == 22])
    stray = make_event(23, 9.0)
    hist2 = pyrex.Histogram(events + [stray], sorter)
    print("bins() with a catch-all for the unmatched event:", hist2.bins())
    print("counts():", list(hist2.counts()))

    # A non-uniform-weight sample instead: mult[0] wanders away from 1.0
    # wherever a bin's events happen to average above/below the whole
    # sample's mean event.weight().
    weighted = [make_event(21, 1.0), make_event(21, 2.0), make_event(22, 3.0), make_event(22, 4.0)]
    for ev, w in zip(weighted, [1.0, 1.0, 3.0, 3.0]):
        ev.weight = w
    hist_w = pyrex.Histogram(weighted, [0, 0, 1, 1])
    print("weights [1,1,3,3] in bins [0,0,1,1] -> mult[0]:", hist_w.mult[0])

    # Calibrating against an InitNode's IDWTUP/xSec convention -- the same
    # check REX::tea::Reweightor.calc_norm() runs for reweighted cross
    # sections (see weight_norm() below):
    init = pyrex.InitNode()
    init.idWgt = 3  # "properly unweighted" LHEF convention
    init.xSec = [10.0]
    hist.normalise(init)
    print(
        "norm after normalise() (idWgt==3 -> xSec/n_events):",
        hist.norm,
        "==",
        10.0 / len(events),
    )

    # Append a reweighted column per event (as a REX::tea::Reweightor
    # iteration would) and sync it in:
    for ev in events:
        ev.wgts = list(ev.wgts) + [2.0 * ev.weight]  # new = 2x original everywhere
    print("sync_wgts():", hist.sync_wgts(), "-> rows():", hist.rows())
    print("heights(1) (the new weight set's bin heights):", hist.heights(1))
    print("heights() (full nominal*mult 2D array, one row per weight set):", hist.heights())

    # A second sync_wgts() with no new columns is a safe no-op; a shrinking
    # wgts_ is refused (with a warning) rather than silently corrupting
    # already-synced rows:
    print("sync_wgts() again (no new columns):", hist.sync_wgts())
    saved = list(events[0].wgts)
    events[0].wgts = []
    print(
        "sync_wgts() after an event lost its weights:",
        hist.sync_wgts(),
        "-> rows() unchanged:",
        hist.rows(),
    )
    events[0].wgts = saved

    # weight_norm() is also exposed standalone, without needing a Histogram:
    print("weight_norm(init, events):", pyrex.weight_norm(init, events))

    # Lhe integration: make_histogram() builds one against doc.events and
    # auto-normalises it against doc's own InitNode fields.
    lhe_events = [make_event(21, float(i)) for i in range(3)] + [
        make_event(22, float(i)) for i in range(2)
    ]
    doc = pyrex.Lhe(lhe_events)
    doc.idWgt = 3
    doc.xSec = [10.0]
    doc.make_histogram("pdg", indices)
    print("has_histogram('pdg'):", doc.has_histogram("pdg"))
    owned = doc.hist("pdg")
    print("doc.hist('pdg').norm (auto-normalised by make_histogram()):", owned.norm)
    print("doc.histograms.keys():", list(doc.histograms.keys()))

    for ev in doc.events:
        ev.wgts = [4.0 * ev.weight]
    print("sync_histograms():", doc.sync_histograms())
    # doc.hist('pdg') always returns the SAME underlying Histogram (see
    # rex.cpp), so `owned` above already reflects this sync:
    print("owned.rows() after sync_histograms():", owned.rows())

    doc.remove_histogram("pdg")
    print("has_histogram('pdg') after remove_histogram:", doc.has_histogram("pdg"))


# ---------------------------------------------------------------------------
def slha_demo():
    section("pyrex.Slha")

    card = pyrex.Slha()
    card.set("MASS", 6, 172.5)  # top mass
    card.set("MASS", 25, 125.0)  # Higgs mass
    card.set_decay(25, 0.004)  # Higgs width

    print("get(MASS, 6):", card.get("MASS", 6))
    print("get(MASS, 999, fallback=-1.0):", card.get("MASS", 999, -1.0))
    print("get_decay(25):", card.get_decay(25))
    print("has_block('MASS'):", card.has_block("MASS"))
    print("has_block('NONEXISTENT'):", card.has_block("NONEXISTENT"))

    text = card.write()
    print("write() ->\n" + text)

    reparsed = pyrex.to_slha(text)
    print("to_slha() round trip, get(MASS, 6):", reparsed.get("MASS", 6))


# ---------------------------------------------------------------------------
def tea_helpers_demo():
    section("pyrex.tea helper predicates")

    print("true_function():", tea.true_function())
    hel_sel = tea.helicity_selector(1)
    fla_sel = tea.flavor_selector(2)
    both_sel = tea.helicity_flavor_selector(1, 2)

    e1 = make_event(21, 1.0, helicity=1, flavor=2)
    e2 = make_event(21, 1.0, helicity=-1, flavor=2)
    print("helicity_selector(1)(matching event):", hel_sel(e1))
    print("helicity_selector(1)(non-matching event):", hel_sel(e2))
    print("flavor_selector(2)(matching event):", fla_sel(e1))
    print("helicity_flavor_selector(1, 2)(matching event):", both_sel(e1))


# ---------------------------------------------------------------------------
def proc_reweightor_demo():
    section("pyrex.tea.ProcReweightor")

    def times_two(p):
        return [w * 2.0 for w in p.weight]

    def identity(p):
        return list(p.weight)

    def is_gluon(e):
        return e.pdg[0] == 21

    # Several constructor forms are available; this is the fullest one
    # (reweight functions + event selector + explicit normaliser):
    pr = tea.ProcReweightor([times_two], pyrex.EventBelongs(), identity)
    pr.set_event_checker(is_gluon)  # overwrites the selector passed to __init__
    # set_event_checker also accepts an EventBelongs directly:
    # pr.set_event_checker(pyrex.EventBelongs(some_reference_event))

    events = [make_event(21, float(i)) for i in range(3)]
    proc = pyrex.Process(events)
    pr.set_process(proc)
    pr.initialise()
    pr.evaluate(0)  # evaluate amplitude index 0 (the first reweight function)
    pr.append_backlog()

    print("proc.wgts after evaluate+append_backlog:", list(proc.wgts))
    print("normalisation:", list(pr.normalisation))
    print("proc is the bound process:", pr.proc.size() == proc.size())


# ---------------------------------------------------------------------------
def reweightor_demo():
    section("pyrex.tea.Reweightor -- full pipeline")

    doc = pyrex.Lhe()
    for i in range(7):
        doc.add_event(make_event(21, 10.0 + i))  # class 0: 7 events
    for i in range(2):
        doc.add_event(make_event(22, float(i)))  # class 1: 2 events

    def times_three(p):
        return [w * 3.0 for w in p.weight]

    def identity(p):
        return list(p.weight)

    pr_gluon = tea.ProcReweightor(times_three)
    pr_gluon.set_normaliser(identity)
    pr_gluon.set_event_checker(lambda e: e.pdg[0] == 21)

    pr_photon = tea.ProcReweightor(times_three)
    pr_photon.set_normaliser(identity)
    pr_photon.set_event_checker(lambda e: e.pdg[0] == 22)

    rw = tea.Reweightor(doc, [pr_gluon, pr_photon])
    rw.max_batch = 3  # class 0 (7 events) splits into batches of 3, 3, 1
    # NOTE: rw.launch_names only works when doc's `header` came from an
    # actual parsed XML file (ie via load_lhef()) -- finalise_reweighting()
    # calls extract_weight_ids(), which needs a real xmlNode header to scan
    # for existing <weightgroup> ids, and `header`/xmlNode aren't part of
    # this binding's scope. Leave launch_names unset for a synthetic Lhe
    # like this one.
    rw.set_norm(1.0)  # skip calc_norm(); use a fixed normalisation factor

    rw.run()  # setup() -> run_all_iterations() -> finalise_reweighting()

    print("reweightors after run() (1 clone per max_batch overflow batch):", len(rw.reweightors))
    print("process_group:", list(rw.process_group))
    print("per-event appended weight:", [ev.wgts[-1] for ev in rw.events])
    print("rwgt_xSec:", list(rw.rwgt_xSec))

    # The finer-grained methods setup()/run_iteration()/run_all_iterations()/
    # finalise_reweighting()/extract_sorter()/initialise_reweightors() are
    # also exposed individually for callers that want to drive the loop by
    # hand instead of calling run().


# ---------------------------------------------------------------------------
def _make_streaming_sample(path, n_events):
    """Writes an n_events-event .lhe file to path; shared by streaming_demo()'s
    two halves so each gets an independent, freshly-openable eventSource."""
    doc = pyrex.Lhe()
    doc.idBm = [2212, 2212]
    doc.eBm = [6500.0, 6500.0]
    doc.idWgt = -4  # "unweighted events" convention -> norm_factor = 1/n_events
    doc.nProc = 1
    doc.xSec = [2.5]  # a real header cross section, as a genuine LHE file would carry
    doc.xSecErr = [0.1]
    doc.xMax = [1.0]
    doc.lProc = [1]
    for i in range(n_events):
        # scale varies per event so the reweight below is a genuine,
        # per-event-varying function rather than a constant factor.
        doc.add_event(make_event(21, 10.0 + i, scale=10.0 + i))
    pyrex.write_lhef(doc, path, include_ids=False)


def streaming_demo():
    section("pyrex.EventSource / EventSink + Lhe/Reweightor streaming")

    def scale_rwgt(p):
        return [w * (1.0 + 0.01 * s) for w, s in zip(p.weight, p.scale)]

    def identity(p):
        return list(p.weight)

    with tempfile.TemporaryDirectory() as tmp:
        src_path = os.path.join(tmp, "streaming_in.lhe")
        _make_streaming_sample(src_path, n_events=23)

        # ---- the convenient path: Reweightor.run_streaming() ----
        source = pyrex.open_lhe_source(src_path)
        print("EventSource.seekable (forward-only):", source.seekable)

        doc = pyrex.Lhe.open_streaming(source, batch_size=5)
        print("doc.streaming:", doc.streaming, "batch_size:", doc.batch_size)

        rw = tea.Reweightor(doc)
        # Explicit identity normaliser, same reasoning as proc_reweightor_demo:
        # otherwise the default normaliser (the reweight function itself)
        # would cancel scale_rwgt's own scale-dependence back out to 1.0.
        rw.add_reweightor(tea.ProcReweightor([scale_rwgt], pyrex.all_events_belong(), identity))
        rw.verbose = False

        out_path = os.path.join(tmp, "streaming_out.lhe")
        sink = pyrex.open_lhe_sink(out_path, include_ids=False)
        rw.run_streaming(sink, include_ids=False)  # pull_batch() -> reweight -> write -> repeat

        print("events_pulled_total:", rw.events_pulled_total)
        print("rwgt_xSec:", list(rw.rwgt_xSec), "rwgt_xErr:", list(rw.rwgt_xErr))

        reloaded = pyrex.load_lhef(out_path)
        print("streamed output round-trips: events =", len(reloaded.events))

        # ---- the finer-grained path: driving the same loop by hand ----
        # pull_batch()/extract_sorter_stream()/XSecAccumulator are exactly
        # what run_streaming() calls internally (see its C++ doc comment in
        # teaRex.h) -- exposed individually for callers who want to interleave
        # their own logic per batch (eg progress reporting, early stopping).
        # Re-runs the *same* sample/batch_size as above (a fresh EventSource,
        # since the first was already exhausted) specifically so the results
        # can be checked against run_streaming()'s own, above -- hand-driving
        # this loop is meant to be exactly equivalent, not just similar.
        doc2 = pyrex.Lhe.open_streaming(pyrex.open_lhe_source(src_path), batch_size=5)
        rw2 = tea.Reweightor(doc2)
        rw2.add_reweightor(tea.ProcReweightor([scale_rwgt], pyrex.all_events_belong(), identity))
        rw2.verbose = False

        base_reweightors = list(rw2.reweightors)  # the template extract_sorter_stream() clones per batch
        acc = tea.XSecAccumulator()
        n_batches = 0
        while rw2.pull_batch() > 0:
            rw2.extract_sorter_stream(base_reweightors)
            rw2.initialise_reweightors()
            rw2.sync_n_amps()
            rw2.curr_iter = 0  # run_all_iterations() below loops while curr_iter < len(iterators)
            rw2.run_all_iterations()
            for r in rw2.reweightors:
                r.pull_deferred_backlog()
            for p in rw2.processes:
                p.transpose_wgts()  # proc.wgts -> each *existing* event's own .wgts, which absorb() below reads
                p.validate()
            acc.absorb(rw2.events)
            n_batches += 1
        acc.finalise_into(rw2)

        print("manual loop: n_batches:", n_batches, "acc.n_events:", acc.n_events)
        print(
            "manual loop rwgt_xSec/xErr matches run_streaming's:",
            list(rw2.rwgt_xSec) == list(rw.rwgt_xSec) and list(rw2.rwgt_xErr) == list(rw.rwgt_xErr),
        )


# ---------------------------------------------------------------------------
def param_rwgt_demo():
    section("pyrex.tea.ParamRwgt (construction only -- needs real SLHA/rwgt-card files to run)")

    doc = pyrex.Lhe()
    doc.add_event(make_event(21, 10.0))
    pr = tea.ProcReweightor(lambda p: list(p.weight))
    pr.set_event_checker(lambda e: True)

    param_rw = tea.ParamRwgt(doc, [pr])
    print("constructed:", param_rw)
    print(
        "param_rw.read_slha_rwgt(slha_file, rwgt_file) would parse a\n"
        "  '# launch rwgt_name=...' / '# set BLOCK ID VALUE' style card and\n"
        "  drive one reweighting iteration per launch block -- skipped here\n"
        "  since it needs real files on disk."
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# A minimal UMAMI-ABI stub matrix element (madspace/include/madspace/umami.h),
# compiled on the fly so this section stays runnable without a real generated
# process library: 1 massless particle (matching make_event() above), matrix
# element = the particle's energy, so distinct events give distinct weights.
_UMAMI_STUB_SOURCE = r"""
#include "madspace/umami.h"
namespace {
bool supported_in[UMAMI_INPUT_KEY_COUNT] = {};
bool required_in[UMAMI_INPUT_KEY_COUNT] = {};
bool supported_out[UMAMI_OUTPUT_KEY_COUNT] = {};
struct Init {
    Init() {
        supported_in[UMAMI_IN_MOMENTA] = true;
        required_in[UMAMI_IN_MOMENTA] = true;
        supported_out[UMAMI_OUT_MATRIX_ELEMENT] = true;
    }
} init;
}
extern "C" {
UmamiStatus umami_get_meta(UmamiMetaKey key, void* result) {
    switch (key) {
    case UMAMI_META_DEVICE: *reinterpret_cast<UmamiDevice*>(result) = UMAMI_DEVICE_CPU; return UMAMI_SUCCESS;
    case UMAMI_META_PARTICLE_COUNT: *reinterpret_cast<int*>(result) = 1; return UMAMI_SUCCESS;
    case UMAMI_META_DIAGRAM_COUNT: *reinterpret_cast<int*>(result) = 1; return UMAMI_SUCCESS;
    case UMAMI_META_HELICITY_COUNT: *reinterpret_cast<int*>(result) = 1; return UMAMI_SUCCESS;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
}
UmamiStatus umami_supported_inputs(bool const** s, int* c) { *s = supported_in; *c = UMAMI_INPUT_KEY_COUNT; return UMAMI_SUCCESS; }
UmamiStatus umami_required_inputs(bool const** r, int* c) { *r = required_in; *c = UMAMI_INPUT_KEY_COUNT; return UMAMI_SUCCESS; }
UmamiStatus umami_supported_outputs(bool const** s, int* c) { *s = supported_out; *c = UMAMI_OUTPUT_KEY_COUNT; return UMAMI_SUCCESS; }
UmamiStatus umami_initialize(UmamiHandle* handle, char const*) { *handle = new int(0); return UMAMI_SUCCESS; }
UmamiStatus umami_set_parameter(UmamiHandle, char const*, double, double) { return UMAMI_SUCCESS; }
UmamiStatus umami_get_parameter(UmamiHandle, char const*, double* re, double* im) {
    *re = 0.0;
    if (im) *im = 0.0;
    return UMAMI_SUCCESS;
}
UmamiStatus umami_matrix_element(
    UmamiHandle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const* input_keys, void const* const* inputs,
    size_t output_count, UmamiOutputKey const* output_keys, void* const* outputs
) {
    const double* momenta = nullptr;
    for (size_t i = 0; i < input_count; ++i)
        if (input_keys[i] == UMAMI_IN_MOMENTA) momenta = reinterpret_cast<const double*>(inputs[i]);
    for (size_t o = 0; o < output_count; ++o) {
        if (output_keys[o] != UMAMI_OUT_MATRIX_ELEMENT) continue;
        double* out = reinterpret_cast<double*>(outputs[o]);
        for (size_t e = 0; e < count; ++e)
            out[e] = momenta ? momenta[/* particle 0, component E=0 */ stride * 0 + offset + e] : 1.0;
    }
    return UMAMI_SUCCESS;
}
UmamiStatus umami_free(UmamiHandle handle) { delete reinterpret_cast<int*>(handle); return UMAMI_SUCCESS; }
}
"""


def _compile_umami_stub(tmpdir, madspace_include_dir):
    compiler = shutil.which("c++") or shutil.which("g++")
    if compiler is None:
        return None
    src_path = os.path.join(tmpdir, "umami_stub.cpp")
    so_path = os.path.join(tmpdir, "libumami_stub.so")
    with open(src_path, "w") as f:
        f.write(_UMAMI_STUB_SOURCE)
    result = subprocess.run(
        [
            compiler, "-std=c++17", "-O0", "-fPIC", "-shared",
            "-I", madspace_include_dir,
            src_path, "-o", so_path,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("stub matrix element failed to compile:\n" + result.stderr)
        return None
    return so_path


def madtrex_demo():
    section("madspace.madtrex.TupperWare / WareHouse (needs the `madspace` package)")

    try:
        import madspace
        import madspace.madtrex as mt
    except ImportError:
        print("madspace is not installed -- skipping (pip install the sibling madspace/ package)")
        return

    madspace_include_dir = os.path.join(os.path.dirname(madspace.__file__), "include")
    with tempfile.TemporaryDirectory() as tmpdir:
        so_path = _compile_umami_stub(tmpdir, madspace_include_dir)
        if so_path is None:
            print("no C++ compiler found -- skipping (need c++/g++ on PATH)")
            return

        ctx = madspace.Context(1)
        api = ctx.load_matrix_element(so_path, "")

        # TupperWare auto-derives which UMAMI keys it needs straight from the
        # api -- the same query methods MatrixElementApi doesn't expose to
        # Python directly, only through these helpers:
        print("default_input_keys(api):", mt.default_input_keys(api))
        print("default_output_keys(api):", mt.default_output_keys(api))

        events = [make_event(21, 10.0 + i) for i in range(3)]
        doc = pyrex.Lhe()
        for ev in events:
            doc.add_event(ev)

        # TupperWare: one physical process = one or more MatrixElementApis
        # (evaluated as procReweightor.reweight_functions) + an event checker.
        tw = mt.TupperWare()
        tw.add_api(api)  # auto-derived keys; also becomes the normaliser by default
        tw.set_event_checker(lambda e: True)
        proc_rwgt = tw.build()
        print("TupperWare.build() ->", type(proc_rwgt).__name__, "(a real pyrex.tea.ProcReweightor)")

        # WareHouse: aggregates one TupperWare per physical process into an
        # owned pyrex.tea.Reweightor, ready to run() against a real Lhe.
        wh = mt.WareHouse()
        wh.add_process(mt.TupperWare().add_api(api).set_event_checker(lambda e: True))
        rw = wh.build(doc)
        print("WareHouse.build() ->", type(rw).__name__, "(a real pyrex.tea.Reweightor)")

        rw.run()
        print("per-event appended weight:", [ev.wgts[-1] for ev in rw.events])
        # Self-normalized to 1.0: no separate normaliser was set, so
        # procReweightor divides the matrix element by itself.


# ---------------------------------------------------------------------------
def unavailable_reference():
    section("NOT bound (by design) -- see Rex.h/teaRex.h directly if you need these")
    print(
        """
  - LHEF alias accessors: xWgtUP/idPrUP/aQCDUP/... etc. all collapse to a
    single canonical Python name (weight/proc_id/alphaS/...).
  - arrN<T,N> / arrNRef<T,N> / vecArrN<T,N> / nStrideIter: the generic
    fixed-size-tuple container machinery. Momenta/mother/icol are plain
    Python lists (or lists of lists) instead -- read to get a snapshot,
    assign a new list to write back.
  - process::E()/px()/... and the 25+ transpose_*() partial-transposition
    methods: redundant with the bulk list properties + transpose(). The one
    exception is transpose_wgts() (see streaming_demo()): unlike the rest,
    it isn't a redundant narrower transpose() -- transpose() rebuilds
    process.events as brand-new Event objects, while transpose_wgts()
    writes wgts_ into the existing ones in place, which matters for a
    Reweightor's own .events (and anything reading through it, eg
    XSecAccumulator.absorb()) to see the update.
  - process device_data/devptr GPU-pointer plumbing.
  - event/process `extra` (std::any-backed side-channel key/value storage).
  - xmlNode / xmlDoc / lheReader / lheWriter / lheRaw / to_xml / to_lhe /
    load_xml: the templated XML parsing/writing layer. Use load_lhef() /
    write_lhef() instead.
  - REX::tea::threadPool (internal to Reweightor.run()).
  - rwgt_slha / rwgt_card / rwgt_block internals: only
    ParamRwgt.read_slha_rwgt(slha_file, rwgt_file) is exposed; the parsed
    card structure itself isn't.
  - slha.get/set/has_entry's multi-index (initializer_list<int>) overloads:
    only the single-int-index overloads are bound (a pybind11 limitation --
    std::initializer_list can't be built from a runtime Python list).
  - Any cevent_equal_fn/cevent_bool_fn (const-event) comparator overload
    where a mutable-event twin also exists: only the mutable form is bound,
    since pybind11 can't distinguish which std::function signature a plain
    Python callable was meant for.
  - Histogram's/Lhe.make_histogram()'s EventBelongs/event_bool_fn/
    cevent_bool_fn/vector<...> constructor and add_event(s) overloads:
    these are pure convenience wrappers in C++ that build an EventSorter
    internally, so build one explicitly (EventSorter(...)) and pass that
    instead -- nothing is lost, and it sidesteps the same event_bool_fn/
    cevent_bool_fn ambiguity as the bullet above.
  - madtrex.make_weightor()/default input+output key overload taking
    explicit UMAMI keys: TupperWare.add_api()/build() cover the actual
    construction path; make_weightor() itself returns a raw
    shared_ptr<REX::tea::weightor> that nothing bound can consume back from
    Python (same reasoning as leaving REX::tea::weightor itself unbound).
"""
    )


def main():
    parton_demo()
    event_demo()
    comparator_demo()
    event_belongs_demo()
    event_sorter_demo()
    process_demo()
    init_node_demo()
    lhe_demo()
    histogram_demo()
    slha_demo()
    tea_helpers_demo()
    proc_reweightor_demo()
    reweightor_demo()
    streaming_demo()
    param_rwgt_demo()
    madtrex_demo()
    unavailable_reference()
    print("\nAll sections ran successfully.")


if __name__ == "__main__":
    main()
