import hashlib
import io
from http.client import IncompleteRead
from urllib.error import HTTPError, URLError

import pytest

from myotrace import source_integrity as module


def source(payload=b"movie"):
    return {"url": "https://example.invalid/movie.tif", "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest()}


def test_valid_cache_is_rehashed_and_reused_without_network(tmp_path, monkeypatch):
    path = tmp_path / "movie.tif"
    path.write_bytes(b"movie")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: pytest.fail("unexpected download"))
    assert module.verified_source(source(), path)["reused"] is True


def test_corrupt_cache_replaced_only_by_verified_bytes(tmp_path, monkeypatch):
    path = tmp_path / "movie.tif"
    path.write_bytes(b"bad cache")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"movie"))
    assert module.verified_source(source(), path)["reused"] is False
    assert path.read_bytes() == b"movie"
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("payload", [b"mov", b"wrong"])
def test_bad_size_or_hash_never_replaces_cache(tmp_path, monkeypatch, payload):
    path = tmp_path / "movie.tif"
    path.write_bytes(b"retained invalid cache")
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(payload))
    with pytest.raises(ValueError, match="expected.*observed"):
        module.verified_source(source(), path)
    assert path.read_bytes() == b"retained invalid cache"
    assert list(tmp_path.glob("*.part")) == []


@pytest.mark.parametrize("error", [URLError("interrupted transfer"), IncompleteRead(b"mo", 3)])
def test_interrupted_download_is_not_cached_and_transient_failure_retries(tmp_path, monkeypatch, error):
    class Interrupted(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise error
            return super().read(2)

    responses = iter([Interrupted(b"movie"), io.BytesIO(b"movie")])
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda *a, **k: next(responses))
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    path = tmp_path / "movie.tif"
    assert module.verified_source(source(), path)["reused"] is False
    assert path.read_bytes() == b"movie"
    assert list(tmp_path.glob("*.part")) == []


def test_permanent_http_failure_is_not_retried_or_cached(tmp_path, monkeypatch):
    calls = []

    def missing(*args, **kwargs):
        calls.append(1)
        raise HTTPError(source()["url"], 404, "missing", {}, None)

    monkeypatch.setattr(module.urllib.request, "urlopen", missing)
    with pytest.raises(HTTPError):
        module.verified_source(source(), tmp_path / "movie.tif")
    assert len(calls) == 1
    assert list(tmp_path.iterdir()) == []
