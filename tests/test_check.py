import http.client
import io
import json
import socket
import urllib.error
import zlib

import numpy as np
import pytest

import run_build
from pipeline import verify as V
from pipeline.build import build
from pipeline.grid import Source
from tests.conftest import make_tif


def _tif(tmp_path):
    arr = np.zeros((240, 240), np.float32)   # lat 46..44, lon -122..-120 at 30″
    arr[10, 20] = 10.0
    tif = tmp_path / "patch.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    return tif


def _site(tmp_path):
    out = tmp_path / "site"
    build(Source(str(_tif(tmp_path))), str(out), 2024, "patch.tif", 50, 40, log=lambda *_: None)
    return out


def _check(out):
    return V.check_site(str(out), full_world=False, lat_top=50, lat_bottom=40)


def test_a_good_site_passes(tmp_path):
    assert _check(_site(tmp_path)) == []


def test_missing_manifest_fails(tmp_path):
    out = _site(tmp_path)
    (out / "manifest.json").unlink()
    assert _check(out) == ["manifest.json is missing"]


def test_listed_but_missing_tile_fails(tmp_path):
    out = _site(tmp_path)
    (out / "fine" / "N44W122.bin").unlink()
    assert any("listed but missing" in f for f in _check(out))


def test_truncated_or_wrong_size_tile_fails(tmp_path):
    out = _site(tmp_path)
    p = out / "fine" / "N44W122.bin"
    p.write_bytes(p.read_bytes()[:10])
    assert any("not a 240x240" in f for f in _check(out))
    p.write_bytes(zlib.compress(b"\x01" * 100))
    assert any("not a 240x240" in f for f in _check(out))


def test_stray_tile_fails(tmp_path):
    out = _site(tmp_path)
    (out / "fine" / "N10E010.bin").write_bytes(zlib.compress(b"\x01" * 240 * 240))
    assert any("not in the manifest" in f for f in _check(out))


def test_manifest_the_app_would_refuse_fails(tmp_path):
    out = _site(tmp_path)
    m = json.loads((out / "manifest.json").read_text())
    m["encoding"]["scale"] = 71.0
    m["fine"]["keys"].append("bogus")
    (out / "manifest.json").write_text(json.dumps(m))
    fails = _check(out)
    assert any("encoding" in f for f in fails) and any("malformed" in f for f in fails)
    (out / "manifest.json").write_text("{not json")
    assert "does not parse" in _check(out)[0]


def test_full_world_floor_catches_a_small_build(tmp_path):
    out = _site(tmp_path)
    fails = V.check_site(str(out), full_world=True)
    assert any("only 1 fine tiles" in f for f in fails)


def test_known_place_inside_the_extent_is_checked(tmp_path, monkeypatch):
    out = _site(tmp_path)
    monkeypatch.setattr(V, "KNOWN_PLACES", [("lit patch should be dark", 46.0 - 10.5 / 120,
                                              -122.0 + 20.5 / 120, '<', 1)])
    assert any("lit patch" in f for f in _check(out))


# --- run_build: a failed check never produces <out> ---------------------------

def test_build_publishes_only_after_checks(tmp_path):
    out = tmp_path / "build"
    rc = run_build.main(["--src", str(_tif(tmp_path)), "--year", "2024", "--out", str(out),
                         "--lat-top", "50", "--lat-bottom", "40"])
    assert rc == 0 and (out / "manifest.json").is_file()
    assert not (tmp_path / "build.building").exists()
    assert run_build.main(["--check", str(out)]) == 1   # full-world floors: a patch is not a world


def test_failed_check_leaves_out_untouched(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "KNOWN_PLACES", [("lit patch should be dark", 46.0 - 10.5 / 120,
                                              -122.0 + 20.5 / 120, '<', 1)])
    out = tmp_path / "build"
    rc = run_build.main(["--src", str(_tif(tmp_path)), "--year", "2024", "--out", str(out),
                         "--lat-top", "50", "--lat-bottom", "40"])
    assert rc == 1
    assert not out.exists()
    assert (tmp_path / "build.building" / "manifest.json").is_file()


def test_leftover_staging_dir_refuses(tmp_path):
    (tmp_path / "build.building").mkdir()
    with pytest.raises(SystemExit):
        run_build.main(["--src", str(_tif(tmp_path)), "--year", "2024",
                        "--out", str(tmp_path / "build"), "--lat-top", "50", "--lat-bottom", "40"])


# --- http_reader: retries what is transient, never mistakes it for absent -----

class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _opener(script):
    calls = []

    def op(url, timeout):
        calls.append(url)
        r = script[len(calls) - 1]
        if isinstance(r, BaseException):
            raise r
        return _Resp(r)
    return op, calls


def _http(code):
    return urllib.error.HTTPError("u", code, "x", {}, None)


def test_http_reader_retries_dropped_connections_and_timeouts():
    op, calls = _opener([ConnectionResetError(), socket.timeout(), urllib.error.URLError("dns"),
                         _http(503), http.client.IncompleteRead(b"ab", 10), b"ok"])
    read = V.http_reader("https://x/", attempts=6, opener=op, sleep=lambda s: None)
    assert read("manifest.json") == b"ok" and len(calls) == 6 and calls[0] == "https://x/manifest.json"


def test_http_reader_404_is_absent_without_retry():
    op, calls = _opener([_http(404)])
    assert V.http_reader("https://x", opener=op, sleep=lambda s: None)("fine/N00E000.bin") is None
    assert len(calls) == 1


def test_http_reader_gives_up_loudly():
    op, calls = _opener([socket.timeout()] * 3)
    with pytest.raises(RuntimeError):
        V.http_reader("https://x", attempts=3, opener=op, sleep=lambda s: None)("manifest.json")
    assert len(calls) == 3


def test_check_site_over_http_samples_tiles(tmp_path):
    out = _site(tmp_path)
    local = V.local_reader(str(out))
    served = []

    def op(url, timeout):
        served.append(url)
        b = local(url.split("https://x/")[1])
        if b is None:
            raise _http(404)
        return _Resp(b)
    read = V.http_reader("https://x", opener=op, sleep=lambda s: None)
    assert V.check_site(read, full_world=False, lat_top=50, lat_bottom=40, sample=5) == []
    (out / "coarse" / "N40W130.bin").unlink()   # a half-finished publish
    assert any("listed but missing" in f
               for f in V.check_site(read, full_world=False, lat_top=50, lat_bottom=40, sample=5))
