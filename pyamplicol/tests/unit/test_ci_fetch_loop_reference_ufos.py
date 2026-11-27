# SPDX-License-Identifier: 0BSD
"""``tools/ci/fetch_loop_reference_ufos.py``: pinned downloads fail closed."""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from tools.ci import fetch_loop_reference_ufos as fetcher


def _source(tmp_path: Path, data: bytes = b"reference bytes\n") -> tuple[str, str]:
    path = tmp_path / "upstream" / "file"
    path.parent.mkdir()
    path.write_bytes(data)
    return path.as_uri(), hashlib.sha256(data).hexdigest()


def test_a_matching_download_is_kept(tmp_path):
    url, digest = _source(tmp_path)
    target = fetcher.fetch(url, digest, tmp_path / "cache" / "file")
    assert target.read_bytes() == b"reference bytes\n"


def test_a_wrong_hash_fails_and_leaves_no_file(tmp_path):
    url, _ = _source(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(fetcher.FetchError, match="sha256"):
        fetcher.fetch(url, "0" * 64, cache / "file")
    assert list(cache.iterdir()) == []


def test_a_cached_file_is_checked_and_fetched_again(tmp_path):
    url, digest = _source(tmp_path)
    target = tmp_path / "cache" / "file"
    target.parent.mkdir()
    target.write_bytes(b"tampered\n")
    assert fetcher.fetch(url, digest, target).read_bytes() == b"reference bytes\n"
    # A matching cached file is used without the source.
    assert fetcher.fetch("file:///nonexistent", digest, target) == target
    # A tampered cache with a changed upstream fails.
    target.write_bytes(b"tampered\n")
    (tmp_path / "upstream" / "file").write_bytes(b"changed upstream\n")
    with pytest.raises(fetcher.FetchError, match="sha256"):
        fetcher.fetch(url, digest, target)


def test_an_unreachable_source_fails(tmp_path):
    with pytest.raises(fetcher.FetchError):
        fetcher.fetch(
            (tmp_path / "missing").as_uri(), "0" * 64, tmp_path / "cache" / "file"
        )


def test_a_pinned_archive_with_a_link_is_refused(tmp_path):
    archive = tmp_path / "model.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("model/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/etc/passwd"
        tar.addfile(info)
        data = b"x = 1\n"
        info = tarfile.TarInfo("model/__pycache__/x.pyc")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(fetcher.FetchError, match="not a file"):
        fetcher.extract(archive, tmp_path / "out")


def test_the_python3_conversion_makes_sibling_imports_relative(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "__init__.py").write_text("import particles\nimport function_library\n")
    (model / "object_library.py").write_text(
        "import cmath\nfor k, v in d.iteritems():\n    raise UFOError, 'bad'\n"
    )
    (model / "particles.py").write_text(
        "from object_library import Particle\nimport parameters as Param\n"
    )
    (model / "parameters.py").write_text("import cmath\n")
    (model / "function_library.py").write_text("")
    card = tmp_path / "write_param_card.py"
    card.write_text("def f():\n    from parameters import all_parameters\n")
    fetcher.convert_python3(model, card)
    assert (model / "__init__.py").read_text() == (
        "from . import object_library \nfrom . import particles\n"
        "from . import function_library\n"
    )
    assert (model / "object_library.py").read_text() == (
        'import cmath\nfor k, v in d.items():\n    raise UFOError("bad")\n'
    )
    assert (model / "particles.py").read_text() == (
        "from .object_library import Particle\nfrom . import parameters as Param\n"
    )
    assert (model / "write_param_card.py").read_text() == (
        "def f():\n    from .parameters import all_parameters\n"
    )


def test_main_fails_closed_without_writing_the_environment(tmp_path, monkeypatch):
    url, _ = _source(tmp_path)
    monkeypatch.setattr(fetcher, "PATCH_URL", url)
    monkeypatch.setattr(fetcher, "UFODB_ARCHIVES", {})
    environment = tmp_path / "github-env"
    code = fetcher.main([str(tmp_path / "out"), "--github-env", str(environment)])
    assert code == 1
    assert not environment.exists()
