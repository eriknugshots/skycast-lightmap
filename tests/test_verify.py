import numpy as np

from pipeline.build import build
from pipeline.grid import Source
from pipeline.verify import verify
from tests.conftest import make_tif


def _site(tmp_path):
    arr = np.zeros((240, 240), np.float32)
    arr[10, 20] = 10.0   # lat 46..44, lon -122..-120 at 30″ (whole-pixel origin)
    tif = tmp_path / "patch.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    out = tmp_path / "site"
    build(Source(str(tif)), str(out), 2024, "patch.tif", 50, 40, log=lambda *_: None)
    return out


def test_verify_passes_a_true_check(tmp_path):
    out = _site(tmp_path)
    lat, lon = 46.0 - 10.5 / 120, -122.0 + 20.5 / 120   # inside cell (10, 20)
    assert verify(str(out), [("lit patch", lat, lon, '>', 1)]) == []


def test_verify_fails_a_false_check(tmp_path):
    out = _site(tmp_path)
    lat, lon = 46.0 - 10.5 / 120, -122.0 + 20.5 / 120
    assert verify(str(out), [("too bright", lat, lon, '>', 1000)]) != []


def test_verify_treats_an_unlisted_tile_as_dark(tmp_path):
    out = _site(tmp_path)
    failures = verify(str(out), [("elsewhere", 10.0, 10.0, '<', 0.001)])
    assert failures == []
