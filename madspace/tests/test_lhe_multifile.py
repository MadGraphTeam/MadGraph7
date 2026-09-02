"""Tests for LHEMultiFileWriter -- the fan-out behind the multi-file form of
EventGenerator.combine_to_lhe.

The contract a consumer relies on has three parts, and each is checked here:

  * every output file is a *complete* LHE file, with its own <header> and its
    own <init> block carrying the process cross sections, so it can be opened
    by path alone with nothing published beside it;
  * the events are dealt out round-robin, so the files are balanced to within
    one event however the stream happens to be chunked;
  * the same mapping holds whether events are written one at a time or through
    the reserve()/file_index()/write_string() path the generator uses to format
    chunks off-thread.
"""

import os
import re
import tempfile

import pytest

import madspace as ms

# A recognisable <init> block: two processes with distinct cross sections. The
# decay pools MadSpin reads care about exactly this number -- for a 1 -> n
# process the "cross section" of the init block is the channel's partial width,
# which is what picks a channel among a pdg's and sets the branching ratio.
PROCESSES = [
    ms.LHEProcess(
        cross_section=1.4602897691,
        cross_section_error=2.5e-3,
        max_weight=7.25,
        process_id=66,
    ),
    ms.LHEProcess(
        cross_section=0.5,
        cross_section_error=1.0e-3,
        max_weight=1.5,
        process_id=67,
    ),
]


def make_meta():
    return ms.LHEMeta(
        beam1_pdg_id=2212,
        beam2_pdg_id=2212,
        beam1_energy=6500.0,
        beam2_energy=6500.0,
        beam1_pdf_authors=0,
        beam2_pdf_authors=0,
        beam1_pdf_id=247000,
        beam2_pdf_id=247000,
        weight_mode=3,
        processes=PROCESSES,
        headers=[
            ms.LHEHeader(name="MGRunCard", content="a = b", escape_content=True)
        ],
    )


def make_event(index):
    """An event whose process_id is its position in the stream, so a file's
    contents identify which events were dealt to it."""
    return ms.LHEEvent(
        process_id=index,
        weight=1.0,
        scale=91.188,
        alpha_qed=0.0078,
        alpha_qcd=0.118,
        particles=[
            ms.LHEParticle(
                pdg_id=21,
                status_code=ms.LHEParticle.status_incoming,
                energy=100.0,
            ),
            ms.LHEParticle(
                pdg_id=21,
                status_code=ms.LHEParticle.status_outgoing,
                energy=100.0,
            ),
        ],
    )


def event_ids(path):
    """The process_ids of the events in an LHE file, in file order."""
    with open(path) as fsock:
        text = fsock.read()
    ids = []
    for body in re.findall(r"<event>\n(.*?)</event>", text, re.S):
        # first line of an event body: nup idprup xwgtup scalup aqedup aqcdup
        ids.append(int(body.split("\n")[0].split()[1]))
    return ids


def paths(tmpdir, count):
    return [os.path.join(tmpdir, "unweighted_events_%d.lhe" % i) for i in range(count)]


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as directory:
        yield directory


# --------------------------------------------------------------------------
# Every file is complete on its own
# --------------------------------------------------------------------------


def test_every_file_carries_its_own_init_block(tmpdir):
    targets = paths(tmpdir, 4)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    for i in range(40):
        writer.write(make_event(i))
    del writer

    for target in targets:
        with open(target) as fsock:
            text = fsock.read()
        assert text.startswith('<LesHouchesEvents version="3.0">')
        assert text.rstrip().endswith("</LesHouchesEvents>")
        init = re.search(r"<init>\n(.*?)</init>", text, re.S)
        assert init is not None, "no <init> block in %s" % target
        lines = init.group(1).strip().split("\n")
        # first line ends with the number of processes, then one line each
        assert int(lines[0].split()[-1]) == len(PROCESSES)
        assert len(lines) == 1 + len(PROCESSES)
        for line, process in zip(lines[1:], PROCESSES):
            assert float(line.split()[0]) == pytest.approx(process.cross_section)
            assert int(line.split()[3]) == process.process_id


def test_every_file_carries_the_header(tmpdir):
    targets = paths(tmpdir, 3)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    writer.write(make_event(0))
    del writer

    for target in targets:
        with open(target) as fsock:
            text = fsock.read()
        assert "<MGRunCard>" in text
        assert "a = b" in text


def test_a_file_with_no_events_is_still_a_valid_lhe_file(tmpdir):
    # More files than events: the tail files get a banner and nothing else, and
    # a reader must still be able to open them and find the cross section.
    targets = paths(tmpdir, 5)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    writer.write(make_event(0))
    writer.write(make_event(1))
    del writer

    assert [event_ids(t) for t in targets] == [[0], [1], [], [], []]
    for target in targets[2:]:
        with open(target) as fsock:
            text = fsock.read()
        assert "<init>" in text
        assert text.rstrip().endswith("</LesHouchesEvents>")


# --------------------------------------------------------------------------
# The events are dealt out round-robin
# --------------------------------------------------------------------------


def test_events_are_dealt_round_robin(tmpdir):
    targets = paths(tmpdir, 4)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    for i in range(23):
        writer.write(make_event(i))
    del writer

    for i, target in enumerate(targets):
        assert event_ids(target) == list(range(i, 23, 4))


def test_files_are_balanced_to_within_one_event(tmpdir):
    targets = paths(tmpdir, 7)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    for i in range(100):
        writer.write(make_event(i))
    del writer

    counts = [len(event_ids(t)) for t in targets]
    assert sum(counts) == 100
    assert max(counts) - min(counts) <= 1


def test_no_event_is_lost_or_duplicated(tmpdir):
    targets = paths(tmpdir, 5)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    for i in range(53):
        writer.write(make_event(i))
    del writer

    seen = sorted(sum((event_ids(t) for t in targets), []))
    assert seen == list(range(53))


# --------------------------------------------------------------------------
# The chunked path the generator uses agrees with the one-at-a-time path
# --------------------------------------------------------------------------


def test_reserve_hands_out_consecutive_positions(tmpdir):
    writer = ms.LHEMultiFileWriter(paths(tmpdir, 3), make_meta())
    assert writer.event_count == 0
    assert writer.reserve(10) == 0
    assert writer.reserve(4) == 10
    assert writer.event_count == 14


def test_file_index_is_the_round_robin_mapping(tmpdir):
    writer = ms.LHEMultiFileWriter(paths(tmpdir, 3), make_meta())
    assert [writer.file_index(i) for i in range(7)] == [0, 1, 2, 0, 1, 2, 0]
    assert writer.file_count == 3


def test_chunked_writes_match_one_at_a_time(tmpdir):
    """What combine_to_lhe does: claim a run of positions, format the chunk
    into one string per output file, hand the strings over out of order."""
    directory_a = os.path.join(tmpdir, "chunked")
    directory_b = os.path.join(tmpdir, "serial")
    os.makedirs(directory_a)
    os.makedirs(directory_b)
    chunked_targets = paths(directory_a, 4)
    serial_targets = paths(directory_b, 4)

    writer = ms.LHEMultiFileWriter(chunked_targets, make_meta())
    chunks = []
    index = 0
    for size in (10, 7, 10, 3):
        first = writer.reserve(size)
        parts = ["" for _ in range(writer.file_count)]
        for i in range(size):
            parts[writer.file_index(first + i)] += _format(make_event(index))
            index += 1
        chunks.append(parts)
    # ... handed back in a different order than they were claimed, which is what
    # a thread pool does
    for parts in [chunks[2], chunks[0], chunks[3], chunks[1]]:
        for file, part in enumerate(parts):
            writer.write_string(file, part)
    del writer

    serial = ms.LHEMultiFileWriter(serial_targets, make_meta())
    for i in range(30):
        serial.write(make_event(i))
    del serial

    for chunked, expected in zip(chunked_targets, serial_targets):
        assert sorted(event_ids(chunked)) == sorted(event_ids(expected))


def _format(event):
    """Render one LHEEvent the way LHEFileWriter.write would, via a throwaway
    single-file writer -- there is no direct binding for LHEEvent.format_to."""
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "one.lhe")
        writer = ms.LHEFileWriter(path, ms.LHEMeta())
        writer.write(event)
        del writer
        with open(path) as fsock:
            text = fsock.read()
    body = re.search(r"(<event>\n.*?</event>\n)", text, re.S)
    assert body is not None
    return body.group(1)


# --------------------------------------------------------------------------
# Degenerate cases
# --------------------------------------------------------------------------


def test_one_file_is_the_ordinary_single_file_case(tmpdir):
    targets = paths(tmpdir, 1)
    writer = ms.LHEMultiFileWriter(targets, make_meta())
    for i in range(5):
        writer.write(make_event(i))
    del writer

    assert event_ids(targets[0]) == [0, 1, 2, 3, 4]
    with open(targets[0]) as fsock:
        text = fsock.read()
    assert text.count("<init>") == 1


def test_no_files_is_rejected(tmpdir):
    with pytest.raises(ValueError):
        ms.LHEMultiFileWriter([], make_meta())
