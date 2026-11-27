"""Gzip-compressed LHE output written through LHEFileWriter's gzip_device."""

import gzip

import pytest
from madspace_testing import ms_device

import madspace as ms


def test_gzip_device_without_compressor_raises(tmp_path):
    with pytest.raises(ValueError, match="no gzip compressor"):
        ms.LHEFileWriter(
            str(tmp_path / "events.lhe.gz"), ms.LHEMeta(), gzip_device=ms.cpu_device()
        )


needs_gzip_device = pytest.mark.skipif(
    not ms_device().supports_gzip(), reason="test device has no gzip compressor"
)


@needs_gzip_device
def test_gzip_algorithm_out_of_range_raises(tmp_path):
    with pytest.raises(ValueError, match="algorithm must be 0 to 5"):
        ms.LHEFileWriter(
            str(tmp_path / "events.lhe.gz"),
            ms.LHEMeta(),
            gzip_device=ms_device(),
            gzip_algorithm=6,
        )


@needs_gzip_device
@pytest.mark.parametrize("algorithm", [2, 5])
def test_gzip_device_output_decompresses_to_plain_lhe(tmp_path, algorithm):
    """More than one 64 MiB flush, so the file holds many gzip members."""
    block = "".join(
        f"<event>\n 2 1 +{i:.10e} 9.1e+01 7.8e-03 1.2e-01\n</event>\n"
        for i in range(1000)
    )
    blocks = [block.replace("<event>", f"<event> {j}") for j in range(1400)]
    plain_path, gzip_path = tmp_path / "events.lhe", tmp_path / "events.lhe.gz"
    for path, device in ((plain_path, None), (gzip_path, ms_device())):
        writer = ms.LHEFileWriter(
            str(path), ms.LHEMeta(), gzip_device=device, gzip_algorithm=algorithm
        )
        for text in blocks:
            writer.write_string(text)
        writer.close()

    assert plain_path.stat().st_size > 64 << 20
    assert gzip_path.stat().st_size < plain_path.stat().st_size / 3
    with gzip.open(gzip_path, "rb") as compressed:
        assert compressed.read() == plain_path.read_bytes()
