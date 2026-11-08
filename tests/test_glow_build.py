import json
import math
import shutil
import zlib

import numpy as np
import pytest

from pipeline import verify as V
from pipeline.build import build
from pipeline.glow import GLOW_R0, GLOW_SCALE, decode_glow, encode_glow
from pipeline.grid import Source
from tests.conftest import make_tif, tile


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """One lit 2°x2° tile (N44W122, every cell 1000 nW) inside a 50..40 build."""
    d = tmp_path_factory.mktemp("glowsite")
    arr = np.full((240, 240), 1000.0, np.float32)
    tif = d / "block.tif"
    make_tif(tif, -122.0, 46.0, 1 / 120, arr)
    out = d / "site"
    m, _ = build(Source(str(tif)), str(out), 2024, "block.tif", 50, 40, log=lambda *_: None, glow=True)
    return out, m


def test_build_writes_glow(site):
    out, m = site
    g = m["glow"]
    assert json.loads((out / "manifest.json").read_text())["glow"] == g
    assert g["encoding"] == {"scale": GLOW_SCALE, "r0": GLOW_R0}
    assert g["fine"]["deg"] == 2 and g["fine"]["cells"] == 240
    assert g["coarse"]["deg"] == 10 and g["coarse"]["cells"] == 240
    keys = set(g["fine"]["keys"])
    # the lit tile and its neighbours (all within 300 km of its cells) are listed
    for k in ("N44W122", "N46W122", "N42W122", "N48W122", "N44W124", "N44W120"):
        assert k in keys, k
    # a tile 6 degrees of longitude away (>400 km) is not
    assert "N44W114" not in keys and "N44W128" not in keys
    for k in keys:
        assert tile(out / "glow" / "fine" / f"{k}.bin").shape == (240, 240)
    assert set(g["fine"]["keys"]) == {p.name[:-4] for p in (out / "glow" / "fine").iterdir()}
    assert set(g["coarse"]["keys"]) == {p.name[:-4] for p in (out / "glow" / "coarse").iterdir()}


def test_coarse_cell_is_the_mean_of_its_fine_cells(site):
    out, m = site
    fine = tile(out / "glow" / "fine" / "N44W122.bin")
    coarse = tile(out / "glow" / "coarse" / "N40W130.bin")
    # coarse row 96 + 10, col 192 + 10 covers fine rows 50..54, cols 50..54 of N44W122
    mean_r = decode_glow(fine[50:55, 50:55]).mean()
    got = int(coarse[96 + 10, 192 + 10])
    assert abs(got - int(encode_glow(mean_r))) <= 1   # the fine bytes are themselves quantised


def test_glow_only_adds_to_an_existing_build(site, tmp_path):
    out, m = site
    plain = tmp_path / "plain"
    shutil.copytree(out, plain, ignore=shutil.ignore_patterns("glow"))
    mm = json.loads((plain / "manifest.json").read_text())
    del mm["glow"]
    (plain / "manifest.json").write_text(json.dumps(mm))
    import run_build
    assert run_build.main(["--glow-only", str(plain), "--lat-top", "50", "--lat-bottom", "40"]) == 0
    assert json.loads((plain / "manifest.json").read_text())["glow"] == m["glow"]
    assert run_build.main(["--glow-only", str(plain), "--lat-top", "50", "--lat-bottom", "40"]) != 0  # glow/ exists


def _check(out):
    return V.check_site(str(out), full_world=False, lat_top=50, lat_bottom=40, require_glow=True)


def test_a_good_glow_site_passes(site):
    assert _check(site[0]) == []


def test_verify_glow_missing_tile(site, tmp_path):
    out = tmp_path / "s"
    shutil.copytree(site[0], out)
    (out / "glow" / "fine" / "N44W122.bin").unlink()
    fails = _check(out)
    assert any("glow/fine/N44W122.bin" in f and "missing" in f for f in fails)


def test_verify_glow_encoding(site, tmp_path):
    out = tmp_path / "s"
    shutil.copytree(site[0], out)
    m = json.loads((out / "manifest.json").read_text())
    m["glow"]["encoding"]["scale"] = GLOW_SCALE + 1
    (out / "manifest.json").write_text(json.dumps(m))
    assert any("glow encoding" in f for f in _check(out))


def test_verify_glow_stray_and_wrong_size(site, tmp_path):
    out = tmp_path / "s"
    shutil.copytree(site[0], out)
    (out / "glow" / "fine" / "N10E010.bin").write_bytes(zlib.compress(b"\x01" * 240 * 240))
    assert any("glow" in f and "not in the manifest" in f for f in _check(out))
    (out / "glow" / "fine" / "N10E010.bin").unlink()
    (out / "glow" / "coarse" / "N40W130.bin").write_bytes(zlib.compress(b"\x01" * 100))
    assert any("glow/coarse/N40W130.bin" in f and "not a 240x240" in f for f in _check(out))


def test_missing_glow_block_fails_when_required(site, tmp_path):
    out = tmp_path / "s"
    shutil.copytree(site[0], out)
    m = json.loads((out / "manifest.json").read_text())
    del m["glow"]
    (out / "manifest.json").write_text(json.dumps(m))
    assert any("no glow block" in f for f in _check(out))
    assert V.check_site(str(out), full_world=False, lat_top=50, lat_bottom=40) == []


def test_glow_pins_read_the_decoded_fine_glow_cell(tmp_path):
    lat, lon, r = 44.06, -121.31, 0.05
    mag = 21.9 - 2.5 * math.log10(1 + r)
    row = int((46 - lat) * 120)
    col = int((lon + 122) * 120)
    b = np.zeros((240, 240), np.uint8)
    b[row, col] = encode_glow(r)
    (tmp_path / "glow" / "fine").mkdir(parents=True)
    (tmp_path / "glow" / "fine" / "N44W122.bin").write_bytes(zlib.compress(b.tobytes()))
    read = V.local_reader(str(tmp_path))
    assert V.check_glow_pins(read, [("here", lat, lon, mag)]) == []
    fails = V.check_glow_pins(read, [("here", lat, lon, mag + 0.5)])
    assert len(fails) == 1 and "here" in fails[0]
    # an unlisted glow tile is r = 0: mag 21.9
    assert V.check_glow_pins(read, [("far", 10.0, 10.0, 21.9)]) == []


def test_glow_only_refuses_the_site_directory():
    import os
    import run_build
    site_dir = os.path.join(os.path.dirname(os.path.abspath(run_build.__file__)), "site")
    assert run_build.glow_only(site_dir, 80, -70) == 1
